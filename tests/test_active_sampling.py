from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lanc.active_sampling import (
    build_rx_metadata,
    count_from_ratio,
    select_proposed_rx_ids,
    select_random_rx_ids,
)
from lanc.evaluation import coverage_quality_tables


class ActiveSamplingTest(unittest.TestCase):
    def test_count_from_ratio_keeps_small_positive_budget(self) -> None:
        self.assertEqual(count_from_ratio(total=100, ratio=0.02), 2)
        self.assertEqual(count_from_ratio(total=30, ratio=0.01), 1)
        self.assertEqual(count_from_ratio(total=30, ratio=0.0), 0)

    def test_build_rx_metadata_collapses_pair_rows_to_uav_points(self) -> None:
        pair_df = pd.DataFrame(
            {
                "rx_id": ["a", "a", "b"],
                "rx_col": [0.0, 0.0, 16.0],
                "rx_row": [0.0, 0.0, 0.0],
                "rx_height_m": [100.0, 100.0, 110.0],
                "building_density_nearby": [0.2, 0.4, 0.6],
                "path_building_ratio_simple": [0.1, 0.3, 0.5],
                "local_building_height": [20.0, 20.0, 30.0],
                "height_clearance": [80.0, 80.0, 80.0],
            }
        )

        metadata = build_rx_metadata(pair_df)

        self.assertEqual(list(metadata["rx_id"]), ["a", "b"])
        self.assertAlmostEqual(float(metadata.loc[metadata["rx_id"] == "a", "building_density_nearby"].iloc[0]), 0.3)
        self.assertAlmostEqual(float(metadata.loc[metadata["rx_id"] == "a", "path_building_ratio_simple"].iloc[0]), 0.2)

    def test_random_selection_excludes_already_selected_points(self) -> None:
        candidates = pd.DataFrame({"rx_id": ["a", "b", "c", "d"]})

        selected = select_random_rx_ids(candidates, already_selected={"b"}, budget=3, rng=np.random.default_rng(7))

        self.assertEqual(len(selected), 3)
        self.assertNotIn("b", selected)
        self.assertEqual(len(set(selected)), 3)

    def test_proposed_selection_prefers_weak_signal_and_building_complexity(self) -> None:
        candidates = pd.DataFrame(
            {
                "rx_id": ["strong_clean", "weak_complex", "sampled", "weak_clean"],
                "rx_col": [0.0, 16.0, 32.0, 48.0],
                "rx_row": [0.0, 0.0, 0.0, 0.0],
                "rx_height_m": [100.0, 100.0, 100.0, 100.0],
                "signal_pred_norm": [0.90, 0.20, 0.10, 0.25],
                "building_density_nearby": [0.0, 0.9, 1.0, 0.0],
                "path_building_ratio_simple": [0.0, 0.8, 1.0, 0.0],
            }
        )

        selected = select_proposed_rx_ids(
            candidates,
            already_selected={"sampled"},
            budget=1,
            rng=np.random.default_rng(11),
        )

        self.assertEqual(selected, ["weak_complex"])

    def test_sparse_unmeasured_grid_keeps_ssim_finite(self) -> None:
        coverage_map = pd.DataFrame(
            {
                "rx_id": ["a", "b", "c"],
                "rx_col": [0.0, 16.0, 0.0],
                "rx_row": [0.0, 0.0, 16.0],
                "rx_height_m": [100.0, 100.0, 100.0],
                "signal_true_norm": [0.7, 0.6, 0.5],
                "signal_pred_norm": [0.68, 0.61, 0.52],
            }
        )

        overall, by_height = coverage_quality_tables(coverage_map)

        self.assertTrue(np.isfinite(float(overall["ssim"].iloc[0])))
        self.assertTrue(np.isfinite(float(by_height["ssim"].iloc[0])))


if __name__ == "__main__":
    unittest.main()
