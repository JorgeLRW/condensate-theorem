"""CPU-only checks of the selector and the omitted-mass identity from the paper."""

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condensate.selector import (  # noqa: E402
    BLOCK_SIZE,
    WINDOW,
    block_count,
    block_ranges,
    build_decode_mask,
    select_blocks_kv_group,
    should_refresh,
)


class BlockBudgetTests(unittest.TestCase):
    def test_support_levels_map_to_paper_block_counts(self):
        self.assertEqual(block_count(97), 2)
        self.assertEqual(block_count(193), 8)
        self.assertEqual(block_count(385), 20)
        self.assertEqual(block_count(769), 44)

    def test_nominal_support_is_anchor_plus_window_plus_blocks(self):
        for support in (97, 193, 385, 769):
            used = 1 + WINDOW + block_count(support) * BLOCK_SIZE
            self.assertEqual(used, support)

    def test_block_ranges_cover_distant_region_exactly(self):
        window_start, ranges = block_ranges(2048)
        self.assertEqual(window_start, 2048 - WINDOW)
        self.assertEqual(ranges[0][0], 1)
        self.assertEqual(ranges[-1][1], window_start)
        self.assertEqual(len(ranges), 124)
        for (_, end), (start, _) in zip(ranges, ranges[1:]):
            self.assertEqual(end, start)
        self.assertTrue(all(end - start <= BLOCK_SIZE for start, end in ranges))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.query_heads, self.kv_heads, self.head_dim = 14, 2, 64
        self.length = 512
        self.window_start, self.ranges = block_ranges(self.length)

    def test_group_heads_share_one_selection(self):
        queries = torch.randn(self.query_heads, self.head_dim)
        keys = torch.randn(self.kv_heads, self.length, self.head_dim)
        selected = select_blocks_kv_group(queries, keys, self.ranges, count=4)
        self.assertEqual(len(selected), self.query_heads)
        for head in range(1, 7):
            self.assertEqual(selected[head], selected[0])
        for head in range(8, 14):
            self.assertEqual(selected[head], selected[7])
        self.assertEqual(len(selected[0]), 4)

    def test_planted_block_is_selected_for_its_group(self):
        group = self.query_heads // self.kv_heads
        queries = torch.randn(self.query_heads, self.head_dim)
        keys = 0.01 * torch.randn(self.kv_heads, self.length, self.head_dim)
        target_block = 10
        start, end = self.ranges[target_block]
        group_query = queries[:group].mean(dim=0)
        keys[0, start:end] = 5.0 * group_query / group_query.norm()
        selected = select_blocks_kv_group(queries, keys, self.ranges, count=2)
        self.assertIn(target_block, selected[0])
        self.assertIn(target_block, selected[group - 1])

    def test_zero_budget_selects_nothing(self):
        queries = torch.randn(self.query_heads, self.head_dim)
        keys = torch.randn(self.kv_heads, self.length, self.head_dim)
        self.assertEqual(select_blocks_kv_group(queries, keys, self.ranges, count=0),
                         [[] for _ in range(self.query_heads)])


class RefreshTests(unittest.TestCase):
    def test_reuse_one_refreshes_every_step(self):
        self.assertTrue(all(should_refresh(step, 1, True) for step in range(1, 10)))

    def test_reuse_four_refreshes_at_one_based_steps_1_5_9(self):
        refreshed = [step for step in range(1, 11) if should_refresh(step, 4, True)]
        self.assertEqual(refreshed, [1, 5, 9])

    def test_first_step_always_refreshes(self):
        self.assertTrue(should_refresh(3, 8, False))


class MaskTests(unittest.TestCase):
    def test_unmasked_keys_per_head_match_anchor_window_and_selected_blocks(self):
        total = 300
        window_start, ranges = block_ranges(total)
        selected = [[0], [0, 1], [], [5, 6, 7]] + [[2]] * 10
        mask = build_decode_mask(selected, ranges, window_start, total, torch.float32, "cpu")
        self.assertEqual(mask.shape, (1, 14, 1, total))
        dropped = torch.finfo(torch.float32).min
        for head, blocks in enumerate(selected):
            kept = (mask[0, head, 0] == 0).nonzero().flatten().tolist()
            expected = {0} | set(range(window_start, total))
            for index in blocks:
                start, end = ranges[index]
                expected |= set(range(start, end))
            self.assertEqual(set(kept), expected)
            self.assertTrue(torch.all(mask[0, head, 0][mask[0, head, 0] != 0] == dropped))


class OmittedMassIdentityTests(unittest.TestCase):
    """Paper identity: with omitted set D and eps = a(D), o_full = (1-eps) o_C + eps o_D, so
    o_sparse - o_full = eps (o_C - o_D) and ||o_sparse - o_full|| <= 2 eps V_max."""

    def test_identity_and_bound_on_random_attention(self):
        gen = torch.Generator().manual_seed(1)
        for _ in range(50):
            n, d = 64, 16
            scores = torch.randn(n, generator=gen, dtype=torch.float64) * 3
            values = torch.randn(n, d, generator=gen, dtype=torch.float64)
            attn = torch.softmax(scores, dim=0)
            omitted = torch.rand(n, generator=gen, dtype=torch.float64) < 0.4
            omitted[0] = True
            kept = ~omitted

            eps = attn[omitted].sum()
            o_full = attn @ values
            o_c = (attn[kept, None] * values[kept]).sum(0) / attn[kept].sum()
            o_d = (attn[omitted, None] * values[omitted]).sum(0) / attn[omitted].sum()
            o_sparse = o_c

            delta = o_sparse - o_full
            self.assertTrue(torch.allclose(o_full, (1 - eps) * o_c + eps * o_d, atol=1e-12))
            self.assertTrue(torch.allclose(delta, eps * (o_c - o_d), atol=1e-12))

            v_max = values.norm(dim=1).max()
            self.assertLessEqual(delta.norm().item(), (2 * eps * v_max).item() + 1e-12)


if __name__ == "__main__":
    unittest.main()
