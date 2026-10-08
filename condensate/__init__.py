"""Reference implementation of the condensate selector and paired survival metrics."""

from .selector import (
    BLOCK_SIZE,
    WINDOW,
    block_count,
    block_ranges,
    build_decode_mask,
    select_blocks_kv_group,
    should_refresh,
)

__all__ = [
    "BLOCK_SIZE",
    "WINDOW",
    "block_count",
    "block_ranges",
    "build_decode_mask",
    "select_blocks_kv_group",
    "should_refresh",
]
