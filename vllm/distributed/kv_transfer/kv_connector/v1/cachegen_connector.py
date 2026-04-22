# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CacheGenConnector: file-backed KV transfer connector with optional CacheGen.

This connector is intentionally simple and benchmark-oriented:
- `cachegen_enabled=False`: stores raw KV tensors and loads them back.
- `cachegen_enabled=True`: compresses extracted KV tensors with CacheGen,
  stores compressed payloads, then decodes on load.

The connector reports per-request transfer stats through worker metadata and
returns them to the client via `kv_transfer_params` in `request_finished`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.cachegen_payload_utils import (
    build_transfer_stats,
    empty_transfer_stats,
    load_payload_from_bytes,
    merge_transfer_stats,
    serialize_payload_to_bytes,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.utils.hashing import safe_hash
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    request_id: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    mm_hashes: list[str]

    @staticmethod
    def make_meta(
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> "ReqMeta":
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            request_id=request_id,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
        )


@dataclass
class CacheGenConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(
                request_id=request_id,
                token_ids=token_ids,
                block_ids=block_ids,
                block_size=block_size,
                is_store=is_store,
                mm_hashes=mm_hashes,
            )
        )


@dataclass
class CacheGenConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    request_stats: dict[str, dict[str, float | int | bool]]

    def aggregate(
        self,
        other: "KVConnectorWorkerMetadata",
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, CacheGenConnectorWorkerMetadata)
        merged = dict(self.request_stats)
        for req_id, other_stats in other.request_stats.items():
            merged[req_id] = merge_transfer_stats(
                merged.get(req_id, empty_transfer_stats()),
                other_stats,
            )
        return CacheGenConnectorWorkerMetadata(request_stats=merged)


class CacheGenConnector(KVConnectorBase_V1):
    _FORMAT_VERSION = 1

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Request] = {}
        self._storage_path = Path(
            self._kv_transfer_config.get_from_extra_config("shared_storage_path", "/tmp")
        )
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._cachegen_enabled = bool(
            self._kv_transfer_config.get_from_extra_config("cachegen_enabled", False)
        )
        self._bandwidth_mbps = float(
            self._kv_transfer_config.get_from_extra_config("bandwidth_mbps", 1000.0)
        )
        self._compression_level = int(
            self._kv_transfer_config.get_from_extra_config("compression_level", 3)
        )
        self._kv_cache_dtype = str(vllm_config.cache_config.cache_dtype)

        if self._bandwidth_mbps <= 0:
            raise ValueError("bandwidth_mbps must be positive")

        self._encoder = None
        self._decoder = None
        self._encoded_chunk_cls = None
        self._warned_nonstandard_shape = False

        # Worker -> scheduler: stats collected in this worker process.
        self._worker_request_stats: dict[str, dict[str, float | int | bool]] = {}
        # Scheduler-side stats merged from worker metadata and used by request_finished.
        self._scheduler_request_stats: dict[str, dict[str, float | int | bool]] = {}

        if self._cachegen_enabled:
            self._ensure_cachegen_modules()

        logger.info(
            "CacheGenConnector initialized: role=%s cachegen_enabled=%s "
            "kv_cache_dtype=%s bandwidth_mbps=%.3f storage=%s",
            role,
            self._cachegen_enabled,
            self._kv_cache_dtype,
            self._bandwidth_mbps,
            str(self._storage_path),
        )

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        # This connector runs Python-side file I/O and transforms in load/save hooks.
        return True

    def _ensure_cachegen_modules(self) -> None:
        if self._encoder is not None and self._decoder is not None:
            return

        try:
            from decoder.decoder import CacheGenDecoder
            from encoder.encoder import CacheGenEncoder, EncodedChunk
        except Exception as exc:  # pragma: no cover - import environment dependent
            raise RuntimeError(
                "cachegen_enabled=True requires importing local CacheGen modules "
                "(encoder/decoder). Ensure the benchmark runs from this repo root."
            ) from exc

        self._encoder = CacheGenEncoder(compression_level=self._compression_level)
        self._decoder = CacheGenDecoder()
        self._encoded_chunk_cls = EncodedChunk

    def _network_delay(self, num_bytes: int) -> float:
        return (num_bytes * 8.0) / (self._bandwidth_mbps * 1_000_000.0)

    @staticmethod
    def _to_encoder_input(canonical_kv: torch.Tensor) -> torch.Tensor:
        # CacheGen encoder expects [num_layers, 2, num_heads, seq, head_dim].
        return canonical_kv.unsqueeze(0).unsqueeze(2).contiguous()

    @staticmethod
    def _from_decoder_output(cachegen_tensor: torch.Tensor) -> torch.Tensor:
        # Inverse of _to_encoder_input: [1, 2, 1, seq, hidden] -> [2, seq, hidden]
        return cachegen_tensor[0, :, 0].contiguous()

    def _cachegen_supported(self, kv_cache: torch.Tensor) -> bool:
        return kv_cache.ndim >= 2 and (kv_cache.shape[0] == 2 or kv_cache.shape[1] == 2)

    @staticmethod
    def _canonicalize_for_cachegen(kv_cache: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Convert extracted KV to canonical [2, seq, hidden] for CacheGen."""
        original_shape = tuple(kv_cache.shape)
        if kv_cache.ndim >= 2 and kv_cache.shape[0] == 2:
            seq_len = kv_cache.shape[1]
            canonical = kv_cache.reshape(2, seq_len, -1).contiguous()
            return canonical, {
                "layout": "kv_first",
                "original_shape": original_shape,
            }
        if kv_cache.ndim >= 2 and kv_cache.shape[1] == 2:
            seq_len = kv_cache.shape[0]
            permuted = kv_cache.permute(1, 0, *range(2, kv_cache.ndim)).contiguous()
            canonical = permuted.reshape(2, seq_len, -1)
            return canonical, {
                "layout": "seq_first",
                "original_shape": original_shape,
            }
        raise ValueError(f"Unsupported KV shape for CacheGen: {original_shape}")

    @staticmethod
    def _restore_from_cachegen(
        canonical_kv: torch.Tensor,
        shape_meta: dict[str, Any],
        raw_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Restore canonical [2, seq, hidden] tensor to original extracted layout."""
        layout = shape_meta["layout"]
        original_shape = tuple(shape_meta["original_shape"])
        if layout == "kv_first":
            restored = canonical_kv.reshape(original_shape)
            return restored.to(raw_dtype).contiguous()

        if layout == "seq_first":
            seq_len = original_shape[0]
            trailing = original_shape[2:]
            with_kv_axis = canonical_kv.reshape((2, seq_len, *trailing))
            restored = with_kv_axis.permute(1, 0, *range(2, with_kv_axis.ndim))
            return restored.to(raw_dtype).contiguous()

        raise ValueError(f"Unsupported CacheGen payload layout: {layout!r}")

    @staticmethod
    def _extract_kv_from_layer(
        layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
        block_size: int,
    ) -> torch.Tensor:
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = layer.shape[0], layer.shape[1]
            return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
        if isinstance(attn_metadata, TritonAttentionMetadata):
            block_idxs = slot_mapping // block_size
            offsets = slot_mapping % block_size
            return layer[block_idxs, :, offsets]
        num_pages, page_size = layer.shape[1], layer.shape[2]
        return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]

    @staticmethod
    def _inject_kv_into_layer(
        dst_kv_cache_layer: torch.Tensor,
        src_kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
        block_size: int,
    ) -> None:
        dst_shape = dst_kv_cache_layer.shape
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages = dst_shape[0]
            page_size = dst_shape[1]
            dst = dst_kv_cache_layer.reshape(num_pages * page_size, -1)
            dst[slot_mapping, ...] = src_kv_cache
            return

        if isinstance(attn_metadata, TritonAttentionMetadata):
            block_idxs = slot_mapping // block_size
            offsets = slot_mapping % block_size
            dst_kv_cache_layer[block_idxs, :, offsets] = src_kv_cache
            return

        num_pages = dst_shape[1]
        page_size = dst_shape[2]
        dst = dst_kv_cache_layer.reshape(2, num_pages * page_size, -1)
        dst[:, slot_mapping, ...] = src_kv_cache

    def _serialize_cachegen_payload(self, kv_cache_cpu: torch.Tensor) -> dict[str, Any]:
        self._ensure_cachegen_modules()
        assert self._encoder is not None

        canonical_kv, shape_meta = self._canonicalize_for_cachegen(kv_cache_cpu)
        raw_tensor_bytes = int(kv_cache_cpu.numel() * kv_cache_cpu.element_size())
        prepared = self._to_encoder_input(canonical_kv)
        encoded_chunks = self._encoder.encode(prepared)
        chunks_payload = [
            {
                "data": chunk.data,
                "scales": chunk.scales.cpu(),
                "original_dtype": chunk.original_dtype,
                "original_shape": tuple(chunk.original_shape),
            }
            for chunk in encoded_chunks
        ]

        return {
            "format_version": self._FORMAT_VERSION,
            "kv_format": "cachegen",
            "shape_meta": shape_meta,
            "raw_dtype": kv_cache_cpu.dtype,
            "raw_tensor_bytes": raw_tensor_bytes,
            "raw_bytes": raw_tensor_bytes,
            "chunks": chunks_payload,
        }

    def _deserialize_cachegen_payload(
        self,
        payload: dict[str, Any],
    ) -> tuple[torch.Tensor, float]:
        self._ensure_cachegen_modules()
        assert self._decoder is not None
        assert self._encoded_chunk_cls is not None

        decode_start = time.perf_counter()
        encoded_chunks = [
            self._encoded_chunk_cls(
                data=chunk["data"],
                scales=chunk["scales"],
                original_dtype=chunk["original_dtype"],
                original_shape=tuple(chunk["original_shape"]),
            )
            for chunk in payload["chunks"]
        ]
        decoded = self._decoder.decode(encoded_chunks)
        canonical_kv = self._from_decoder_output(decoded)

        shape_meta = payload.get("shape_meta")
        if shape_meta is None:
            # Backward compatibility with older payloads from early local versions.
            shape_meta = {
                "layout": "kv_first",
                "original_shape": tuple(payload["raw_shape"]),
            }
        kv_cache_cpu = self._restore_from_cachegen(
            canonical_kv,
            shape_meta=shape_meta,
            raw_dtype=payload["raw_dtype"],
        )
        decode_time = time.perf_counter() - decode_start
        return kv_cache_cpu, decode_time

    def _serialize_raw_payload(self, kv_cache_cpu: torch.Tensor) -> dict[str, Any]:
        raw_tensor_bytes = int(kv_cache_cpu.numel() * kv_cache_cpu.element_size())
        return {
            "format_version": self._FORMAT_VERSION,
            "kv_format": "raw",
            "raw_tensor_bytes": raw_tensor_bytes,
            "raw_bytes": raw_tensor_bytes,
            "raw_tensor": kv_cache_cpu,
        }

    def _save_layer_payload(self, filename: Path, kv_cache_cpu: torch.Tensor) -> None:
        if self._cachegen_enabled and self._cachegen_supported(kv_cache_cpu):
            payload = self._serialize_cachegen_payload(kv_cache_cpu)
        else:
            if self._cachegen_enabled and not self._warned_nonstandard_shape:
                logger.warning(
                    "CacheGenConnector saw non-standard KV shape %s; "
                    "falling back to raw transfer for this model/layout.",
                    tuple(kv_cache_cpu.shape),
                )
                self._warned_nonstandard_shape = True
            payload = self._serialize_raw_payload(kv_cache_cpu)
            payload["raw_fallback"] = bool(self._cachegen_enabled)

        filename.parent.mkdir(parents=True, exist_ok=True)
        blob = serialize_payload_to_bytes(payload)
        filename.write_bytes(blob)

    def _load_layer_payload(
        self,
        filename: Path,
    ) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
        blob = filename.read_bytes()
        bytes_to_transfer = len(blob)
        payload = load_payload_from_bytes(blob)
        kv_format = payload.get("kv_format", "raw")
        raw_tensor_bytes = int(payload.get("raw_tensor_bytes", payload.get("raw_bytes", 0)))

        if kv_format == "raw":
            kv_cache_cpu = payload["raw_tensor"].contiguous()
            delay = self._network_delay(bytes_to_transfer)
            time.sleep(delay)
            stats = build_transfer_stats(
                raw_tensor_bytes=raw_tensor_bytes,
                transmitted_bytes=bytes_to_transfer,
                network_time=delay,
                decode_time=0.0,
                cachegen_applied=False,
                raw_fallback_layers=1 if payload.get("raw_fallback", False) else 0,
            )
            return kv_cache_cpu, stats

        if kv_format != "cachegen":
            raise ValueError(f"Unsupported kv_format={kv_format!r} in {filename}")

        delay = self._network_delay(bytes_to_transfer)
        time.sleep(delay)

        kv_cache_cpu, decode_time = self._deserialize_cachegen_payload(payload)
        stats = build_transfer_stats(
            raw_tensor_bytes=raw_tensor_bytes,
            transmitted_bytes=bytes_to_transfer,
            network_time=delay,
            decode_time=decode_time,
            cachegen_applied=True,
            raw_fallback_layers=0,
        )
        return kv_cache_cpu, stats

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, CacheGenConnectorMetadata)

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("CacheGenConnector.start_load_kv called with no attn metadata")
            return

        for req in metadata.requests:
            if req.is_store:
                continue

            req_stats = empty_transfer_stats()
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue

                layer_attn_metadata = (
                    attn_metadata[layer_name] if isinstance(attn_metadata, dict) else attn_metadata
                )

                filename = self._generate_filename(layer_name, req.token_ids, req.mm_hashes)
                kv_cache_cpu, layer_stats = self._load_layer_payload(filename)
                req_stats = merge_transfer_stats(req_stats, layer_stats)

                kv_cache_device = kv_cache_cpu.to(
                    device=kv_cache_layer.device,
                    dtype=kv_cache_layer.dtype,
                    non_blocking=False,
                )
                self._inject_kv_into_layer(
                    kv_cache_layer,
                    kv_cache_device,
                    req.slot_mapping,
                    layer_attn_metadata,
                    self._block_size,
                )

            self._worker_request_stats[req.request_id] = merge_transfer_stats(
                self._worker_request_stats.get(req.request_id, empty_transfer_stats()),
                req_stats,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, CacheGenConnectorMetadata)

        for req in connector_metadata.requests:
            if not req.is_store or req.slot_mapping.numel() == 0:
                continue

            kv_cache = self._extract_kv_from_layer(
                kv_layer,
                req.slot_mapping,
                attn_metadata,
                self._block_size,
            )
            kv_cache_cpu = kv_cache.detach().to("cpu").contiguous()

            filename = self._generate_filename(layer_name, req.token_ids, req.mm_hashes)
            self._save_layer_payload(filename, kv_cache_cpu)

    def wait_for_save(self):
        return

    def build_connector_worker_meta(self) -> CacheGenConnectorWorkerMetadata | None:
        if not self._worker_request_stats:
            return None

        meta = CacheGenConnectorWorkerMetadata(request_stats=self._worker_request_stats)
        self._worker_request_stats = {}
        return meta

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if not self._found_match_for_request(request):
            return 0, False

        token_ids = request.prompt_token_ids or []
        num_tokens_to_check = align_to_block_size(len(token_ids), self._block_size)
        return max(0, num_tokens_to_check - num_computed_tokens), False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = CacheGenConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]

            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=mm_hashes,
                )
                total_need_load += 1
            elif not self._found_match_for_prompt(token_ids, mm_hashes):
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=True,
                    mm_hashes=mm_hashes,
                )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids
            if not resumed_from_preemption or req_id not in self._requests_need_load:
                continue

            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]

            request = self._requests_need_load[req_id]
            total_tokens = num_computed_tokens + num_new_tokens
            token_ids = request.all_token_ids[:total_tokens]

            assert new_block_ids is not None
            block_ids = new_block_ids[0]

            meta.add_request(
                request_id=req_id,
                token_ids=token_ids,
                block_ids=block_ids,
                block_size=self._block_size,
                is_store=False,
                mm_hashes=[f.identifier for f in request.mm_features],
            )
            total_need_load += 1

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, CacheGenConnectorWorkerMetadata):
            return

        for req_id, stats in meta.request_stats.items():
            self._scheduler_request_stats[req_id] = merge_transfer_stats(
                self._scheduler_request_stats.get(req_id, empty_transfer_stats()),
                stats,
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        stats = self._scheduler_request_stats.pop(request.request_id, empty_transfer_stats())
        transmitted_bytes = int(stats["transmitted_bytes"])
        raw_tensor_bytes = int(stats["raw_tensor_bytes"])
        ratio = (
            raw_tensor_bytes / transmitted_bytes
            if raw_tensor_bytes > 0 and transmitted_bytes > 0
            else 1.0
        )

        kv_transfer_params = {
            "raw_tensor_bytes": raw_tensor_bytes,
            "transmitted_bytes": transmitted_bytes,
            "transport_ratio": float(ratio),
            "raw_bytes": raw_tensor_bytes,
            "compressed_bytes": transmitted_bytes,
            "compression_ratio": float(ratio),
            "network_time": float(stats["network_time"]),
            "decode_time": float(stats["decode_time"]),
            "cachegen_enabled": self._cachegen_enabled,
            "cachegen_applied": bool(stats["cachegen_applied"]),
            "raw_fallback_layers": int(stats["raw_fallback_layers"]),
            "kv_cache_dtype": self._kv_cache_dtype,
            "bandwidth_mbps": float(self._bandwidth_mbps),
        }
        return False, kv_transfer_params

    # ==============================
    # Helper functions
    # ==============================

    def _found_match_for_request(self, request: "Request") -> bool:
        return self._found_match_for_prompt(
            list(request.prompt_token_ids or []),
            [f.identifier for f in request.mm_features],
        )

    def _found_match_for_prompt(
        self,
        prompt_token_ids: list[int],
        mm_hashes: list[str],
    ) -> bool:
        num_tokens_to_check = align_to_block_size(len(prompt_token_ids), self._block_size)
        if num_tokens_to_check == 0:
            return False

        folder = self._generate_foldername(
            torch.tensor(prompt_token_ids)[:num_tokens_to_check],
            mm_hashes,
            create_folder=False,
        )
        return folder.exists() and any(folder.glob("*.pt"))

    def _generate_foldername(
        self,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
        create_folder: bool = False,
    ) -> Path:
        token_bytes = token_ids.numpy().tobytes()
        if mm_hashes:
            token_bytes += "-".join(mm_hashes).encode("utf-8")

        input_ids_hash = safe_hash(token_bytes, usedforsecurity=False).hexdigest()
        folder = self._storage_path / input_ids_hash
        if create_folder:
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _generate_filename(
        self,
        layer_name: str,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
    ) -> Path:
        safe_layer = layer_name.replace("/", "__").replace(os.sep, "__")
        folder = self._generate_foldername(token_ids, mm_hashes=mm_hashes, create_folder=True)
        return folder / f"{safe_layer}.pt"


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    """Return the largest block-aligned cached prefix below ``num_tokens``.

    The v1 scheduler requires a connector-backed prefill request to still have
    some local work left to schedule. Matching the entire prompt can drive the
    scheduler into a ``num_new_tokens == 0`` assertion on the first measured
    pass, so we intentionally reserve the final partial/full block for local
    compute and only advertise the strict prefix as externally loadable.
    """
    if num_tokens <= 1:
        return 0
    return ((num_tokens - 1) // block_size) * block_size
