"""Deterministic wire serialization for CacheGen encoded chunks."""

from __future__ import annotations

import struct
from typing import Final

import torch

from .encoder import EncodedChunk

MAGIC: Final[bytes] = b"CGEN1"
_HEADER = struct.Struct("<5sI")
_CHUNK_PREFIX = struct.Struct("<BB")
_U32 = struct.Struct("<I")

_DTYPE_TO_CODE: Final[dict[torch.dtype, int]] = {
    torch.float16: 1,
    torch.float32: 2,
    torch.bfloat16: 3,
}
_CODE_TO_DTYPE: Final[dict[int, torch.dtype]] = {
    code: dtype for dtype, code in _DTYPE_TO_CODE.items()
}


def serialize_encoded_chunks(encoded_chunks: list[EncodedChunk]) -> bytes:
    """Serialize chunks into the benchmark's deterministic ``CGEN1`` payload."""
    payload = bytearray()
    payload.extend(_HEADER.pack(MAGIC, len(encoded_chunks)))

    for chunk in encoded_chunks:
        dtype_code = _dtype_to_code(chunk.original_dtype)
        shape = tuple(int(dim) for dim in chunk.original_shape)
        if len(shape) > 255:
            raise ValueError("EncodedChunk rank must fit in uint8")

        scale_bytes = (
            chunk.scales.detach()
            .cpu()
            .contiguous()
            .to(dtype=torch.float16)
            .numpy()
            .tobytes()
        )
        data = bytes(chunk.data)

        payload.extend(_CHUNK_PREFIX.pack(dtype_code, len(shape)))
        for dim in shape:
            if dim < 0:
                raise ValueError(f"EncodedChunk shape cannot contain negative dim: {shape}")
            payload.extend(_U32.pack(dim))
        payload.extend(_U32.pack(len(scale_bytes)))
        payload.extend(scale_bytes)
        payload.extend(_U32.pack(len(data)))
        payload.extend(data)

    return bytes(payload)


def deserialize_encoded_chunks(payload: bytes) -> list[EncodedChunk]:
    """Deserialize a ``CGEN1`` payload back into ``EncodedChunk`` objects."""
    reader = _Reader(payload)
    magic, chunk_count = _HEADER.unpack(reader.read(_HEADER.size))
    if magic != MAGIC:
        raise ValueError("Invalid CacheGen payload magic")

    chunks: list[EncodedChunk] = []
    for _ in range(chunk_count):
        dtype_code, rank = _CHUNK_PREFIX.unpack(reader.read(_CHUNK_PREFIX.size))
        original_dtype = _code_to_dtype(dtype_code)
        shape = tuple(_U32.unpack(reader.read(_U32.size))[0] for _ in range(rank))

        scale_len = _U32.unpack(reader.read(_U32.size))[0]
        scale_bytes = reader.read(scale_len)
        scale_shape = shape[:-1] + (1,)
        scales = torch.frombuffer(bytearray(scale_bytes), dtype=torch.float16).reshape(
            scale_shape
        )

        data_len = _U32.unpack(reader.read(_U32.size))[0]
        data = reader.read(data_len)
        chunks.append(
            EncodedChunk(
                data=data,
                scales=scales.contiguous(),
                original_dtype=original_dtype,
                original_shape=shape,
            )
        )

    reader.assert_finished()
    return chunks


def _dtype_to_code(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_TO_CODE[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported original dtype for CacheGen wire format: {dtype}") from exc


def _code_to_dtype(code: int) -> torch.dtype:
    try:
        return _CODE_TO_DTYPE[code]
    except KeyError as exc:
        raise ValueError(f"Unsupported CacheGen wire dtype code: {code}") from exc


class _Reader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def read(self, size: int) -> bytes:
        end = self._offset + size
        if end > len(self._payload):
            raise ValueError("Truncated CacheGen payload")
        chunk = self._payload[self._offset : end]
        self._offset = end
        return chunk

    def assert_finished(self) -> None:
        if self._offset != len(self._payload):
            raise ValueError("CacheGen payload contains trailing bytes")
