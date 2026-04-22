# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.cachegen_connector import (
    ReqMeta,
    align_to_block_size,
)

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
