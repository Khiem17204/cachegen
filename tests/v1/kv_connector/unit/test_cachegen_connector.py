# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.cachegen_connector import (
    CacheGenConnector,
    ReqMeta,
    align_to_block_size,
)
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

pytestmark = pytest.mark.skip_global_cleanup


def test_align_to_block_size_reserves_local_work() -> None:
    assert align_to_block_size(0, 16) == 0
    assert align_to_block_size(1, 16) == 0
    assert align_to_block_size(16, 16) == 0
    assert align_to_block_size(17, 16) == 16
    assert align_to_block_size(2048, 16) == 2032


def test_req_meta_uses_strict_prefix_alignment() -> None:
    meta = ReqMeta.make_meta(
        request_id="req-1",
        token_ids=list(range(2048)),
        block_ids=list(range(128)),
        block_size=16,
        is_store=False,
        mm_hashes=[],
    )

    assert meta.token_ids.numel() == 2032
    assert meta.slot_mapping.numel() == 2032


def test_req_meta_caps_tokens_to_available_blocks() -> None:
    meta = ReqMeta.make_meta(
        request_id="req-1",
        token_ids=list(range(4096)),
        block_ids=list(range(128)),
        block_size=16,
        is_store=True,
        mm_hashes=[],
    )

    assert meta.token_ids.numel() == 2048
    assert meta.slot_mapping.numel() == 2048


def test_build_connector_meta_accumulates_chunked_prefill_store(tmp_path) -> None:
    request_id = "req-1"
    prompt_token_ids = list(range(4096))

    connector = CacheGenConnector(
        create_vllm_config(
            kv_connector="CacheGenConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
        ),
        KVConnectorRole.SCHEDULER,
    )

    first_step = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=request_id,
                prompt_token_ids=prompt_token_ids,
                mm_features=[],
                sampling_params=None,
                pooling_params=None,
                block_ids=(list(range(128)),),
                num_computed_tokens=0,
                lora_request=None,
            )
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request_id: 2048},
        total_num_scheduled_tokens=2048,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    first_meta = connector.build_connector_meta(first_step)
    assert first_meta.requests == []
    assert request_id in connector._chunked_prefill_stores

    second_step = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[request_id],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[(list(range(128, 256)),)],
            num_computed_tokens=[2048],
            num_output_tokens=[0],
        ),
        num_scheduled_tokens={request_id: 2048},
        total_num_scheduled_tokens=2048,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    second_meta = connector.build_connector_meta(second_step)

    assert len(second_meta.requests) == 1
    req = second_meta.requests[0]
    assert req.is_store is True
    assert req.token_ids.numel() == 4080
    assert req.slot_mapping.numel() == 4080
    assert request_id not in connector._chunked_prefill_stores
