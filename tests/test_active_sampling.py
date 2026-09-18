from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lanc.active_sampling import (
    build_proposed_v3_diagnostics,
    build_rx_metadata,
    candidate_pool_with_scores,
    compute_gradient_score,
    count_from_ratio,
    proposed_v3_coverage_budget,
    select_building_only_rx_ids,
    select_proposed_v2_rx_ids,
    select_proposed_v3_rx_ids,
    select_proposed_rx_ids,
    select_random_rx_ids,
    select_uniform_grid_rx_ids,
    select_weak_only_rx_ids,
)
from lanc.evaluation import build_candidate_pool, build_coverage_map, coverage_quality_tables


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

    def test_weak_only_selection_prefers_lowest_predicted_signal(self) -> None:
        candidates = pd.DataFrame(
            {
                "rx_id": ["strong", "weakest", "middle", "sampled"],
                "signal_pred_norm": [0.9, 0.1, 0.4, 0.0],
            }
        )

        selected = select_weak_only_rx_ids(candidates, already_selected={"sampled"}, budget=2)

        self.assertEqual(selected, ["weakest", "middle"])

    def test_building_only_selection_prefers_most_complex_environment(self) -> None:
        candidates = pd.DataFrame(
            {
                "rx_id": ["plain", "dense", "blocked", "sampled"],
                "building_density_nearby": [0.0, 0.9, 0.1, 1.0],
                "path_building_ratio_simple": [0.0, 0.2, 0.95, 1.0],
            }
        )

        selected = select_building_only_rx_ids(candidates, already_selected={"sampled"}, budget=2)

        self.assertEqual(selected, ["dense", "blocked"])

    def test_uniform_grid_selection_spreads_points_away_from_existing_samples(self) -> None:
        rows = []
        for row in [0.0, 100.0, 200.0]:
            for col in [0.0, 100.0, 200.0]:
                rows.append({"rx_id": f"r{int(row)}_c{int(col)}", "rx_row": row, "rx_col": col, "rx_height_m": 100.0})
        candidates = pd.DataFrame(rows)

        selected = select_uniform_grid_rx_ids(
            candidates,
            already_selected={"r100_c100"},
            budget=4,
            rng=np.random.default_rng(5),
        )

        selected_points = candidates[candidates["rx_id"].isin(selected)]
        self.assertEqual(len(selected), 4)
        self.assertGreaterEqual(float(selected_points["rx_col"].max() - selected_points["rx_col"].min()), 200.0)
        self.assertGreaterEqual(float(selected_points["rx_row"].max() - selected_points["rx_row"].min()), 200.0)

    def test_gradient_score_is_higher_near_prediction_discontinuity(self) -> None:
        rows = []
        for row in [0.0, 16.0, 32.0]:
            for col in [0.0, 16.0, 32.0]:
                rows.append(
                    {
                        "rx_id": f"r{int(row)}_c{int(col)}",
                        "rx_row": row,
                        "rx_col": col,
                        "rx_height_m": 100.0,
                        "signal_pred_norm": 0.1 if col < 16.0 else 0.9,
                    }
                )
        candidates = pd.DataFrame(rows)

        gradient = compute_gradient_score(candidates)
        edge_score = float(gradient[candidates["rx_col"] == 16.0].mean())
        flat_score = float(gradient[candidates["rx_col"] == 32.0].mean())

        self.assertGreater(edge_score, flat_score)

    def test_proposed_v2_selection_does_not_repeat_sampled_points(self) -> None:
        candidates = pd.DataFrame(
            {
                "rx_id": ["sampled", "edge_weak", "clean"],
                "rx_col": [0.0, 16.0, 32.0],
                "rx_row": [0.0, 0.0, 0.0],
                "rx_height_m": [100.0, 100.0, 100.0],
                "signal_pred_norm": [0.1, 0.2, 0.8],
                "building_density_nearby": [1.0, 0.8, 0.0],
                "path_building_ratio_simple": [1.0, 0.7, 0.0],
            }
        )

        selected = select_proposed_v2_rx_ids(
            candidates,
            already_selected={"sampled"},
            budget=2,
            rng=np.random.default_rng(17),
            variant="proposed_v2_a",
        )

        self.assertNotIn("sampled", selected)
        self.assertEqual(len(selected), len(set(selected)))

    def test_proposed_v3_coverage_budget_splits_coverage_and_exploitation(self) -> None:
        self.assertEqual(proposed_v3_coverage_budget(10, "proposed_v3_rho04"), 4)
        self.assertEqual(proposed_v3_coverage_budget(10, "proposed_v3_rho05"), 5)
        self.assertEqual(proposed_v3_coverage_budget(10, "proposed_v3_rho06"), 6)
        self.assertEqual(proposed_v3_coverage_budget(0, "proposed_v3"), 0)

    def test_proposed_v3_selection_balances_heights_and_avoids_repeats(self) -> None:
        rows = []
        for height_idx, height in enumerate([100.0, 110.0, 120.0, 130.0]):
            for row in range(8):
                for col in range(8):
                    rows.append(
                        {
                            "rx_id": f"h{int(height)}_r{row}_c{col}",
                            "rx_row": float(row),
                            "rx_col": float(col),
                            "rx_height_m": height,
                            "signal_pred_norm": 0.1 + 0.02 * height_idx + 0.001 * (row + col),
                            "building_density_nearby": (row + col) / 14.0,
                            "path_building_ratio_simple": row / 7.0,
                            "prediction_spread_norm": (row + height_idx) / 20.0,
                        }
                    )
        candidates = pd.DataFrame(rows)

        selected = select_proposed_v3_rx_ids(
            candidates,
            already_selected=set(),
            budget=8,
            rng=np.random.default_rng(23),
            variant="proposed_v3",
        )
        diagnostics = build_proposed_v3_diagnostics(
            candidates,
            selected,
            variant="proposed_v3",
            round_budget=8,
        )

        self.assertEqual(len(selected), 8)
        self.assertEqual(len(set(selected)), 8)
        self.assertEqual(diagnostics["total_budget"], 8)
        self.assertEqual(diagnostics["actual_selected_count"], 8)
        self.assertEqual(diagnostics["coverage_quota"], 4)
        self.assertEqual(diagnostics["exploitation_quota"], 4)
        self.assertEqual(diagnostics["unique_height_count"], 4)
        self.assertGreaterEqual(diagnostics["unique_spatial_strata"], 4)
        height_counts = [diagnostics[f"height_{height}_count"] for height in [100, 110, 120, 130]]
        self.assertLessEqual(max(height_counts) - min(height_counts), 1)

        scored_pool = candidate_pool_with_scores(candidates, selected, strategy="proposed_v3")
        self.assertEqual(len(scored_pool), len(candidates))
        self.assertTrue(np.isfinite(scored_pool["value_score"].to_numpy(dtype=float)).all())

    def test_candidate_pool_contains_finite_prediction_spread(self) -> None:
        pair_df = pd.DataFrame(
            {
                "rx_id": ["a", "a", "a", "b"],
                "rx_col": [0.0, 0.0, 0.0, 16.0],
                "rx_row": [0.0, 0.0, 0.0, 0.0],
                "rx_height_m": [100.0, 100.0, 100.0, 100.0],
                "config_id": ["c1", "c2", "c3", "c4"],
                "tx_id": [1, 2, 3, 4],
                "yaw_deg": [0.0, 90.0, 180.0, 270.0],
                "signal_norm": [0.2, 0.4, 0.6, 0.8],
                "signal_encoded": [51.0, 102.0, 153.0, 204.0],
            }
        )

        coverage = build_coverage_map(pair_df, pred_norm=np.array([0.1, 0.2, 0.6, 0.8]))
        pool = build_candidate_pool(coverage)
        spread_a = float(pool.loc[pool["rx_id"] == "a", "prediction_spread_norm"].iloc[0])

        self.assertTrue(np.isfinite(spread_a))
        self.assertGreater(spread_a, 0.0)

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
