from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lanc.run_paper_experiments import (
    BatchConfig,
    build_experiment_specs,
    command_for_spec,
    filter_existing_specs,
    load_existing_completed_keys,
)
from lanc.summarize_paper_results import (
    build_budget_curve_summary,
    build_strategy_mean_std,
    build_zero_shot_vs_active_final,
)


class PaperExperimentWorkflowTest(unittest.TestCase):
    def test_stability_suite_builds_three_seed_specs(self) -> None:
        config = BatchConfig(
            suite="stability",
            source_run=Path("runs/source"),
            target_farm_root=Path("13"),
            target_scene_id=13,
        )

        specs = build_experiment_specs(config)

        self.assertEqual([spec.seed for spec in specs], [42, 2024, 3407])
        self.assertTrue(all(spec.suite == "stability" for spec in specs))
        self.assertTrue(all(spec.initial_ratio == 0.05 for spec in specs))
        self.assertTrue(all(spec.budget_ratio == 0.02 for spec in specs))
        self.assertIn("proposed_v2_j", specs[0].strategies)
        self.assertNotIn("building_only", specs[0].strategies)

    def test_budget_suite_builds_three_budget_specs_with_two_strategies(self) -> None:
        config = BatchConfig(
            suite="budget",
            source_run=Path("runs/source"),
            target_farm_root=Path("13"),
            target_scene_id=13,
        )

        specs = build_experiment_specs(config)

        self.assertEqual([(spec.initial_ratio, spec.budget_ratio) for spec in specs], [(0.02, 0.01), (0.05, 0.02), (0.10, 0.02)])
        self.assertTrue(all(spec.strategies == ("random", "proposed_v2_j") for spec in specs))
        self.assertTrue(all(spec.seed == 42 for spec in specs))

    def test_command_for_spec_uses_module_entry_and_strategy_list(self) -> None:
        spec = build_experiment_specs(
            BatchConfig(
                suite="budget",
                source_run=Path("runs/source"),
                target_farm_root=Path("13"),
                target_scene_id=13,
            )
        )[0]

        command = command_for_spec(spec, python_executable="python")

        self.assertEqual(command[:3], ["python", "-m", "lanc.run_active_experiment"])
        self.assertIn("--target-scene-id", command)
        self.assertIn("13", command)
        self.assertIn("--strategies", command)
        self.assertEqual(command[-2:], ["random", "proposed_v2_j"])

    def test_skip_existing_filters_completed_matching_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "paper_batch_old"
            batch.mkdir()
            config = BatchConfig(
                suite="stability",
                source_run=Path("runs/source"),
                target_farm_root=Path("13"),
                target_scene_id=13,
            )
            specs = build_experiment_specs(config)
            pd.DataFrame(
                [
                    {
                        "config_key": specs[0].config_key,
                        "status": "completed",
                        "run_dir": "runs/active_scene11_to_scene13_example",
                    }
                ]
            ).to_csv(batch / "batch_manifest.csv", index=False)

            existing = load_existing_completed_keys(root)
            filtered = filter_existing_specs(specs, existing)

            self.assertNotIn(specs[0].config_key, {spec.config_key for spec in filtered})
            self.assertEqual(len(filtered), len(specs) - 1)

    def test_strategy_mean_std_ignores_building_only_for_main_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = _make_active_run(root / "run_a", scene_id=13, rmse_random=5.0, rmse_proposed=4.5)
            run_b = _make_active_run(root / "run_b", scene_id=13, rmse_random=6.0, rmse_proposed=5.5)
            manifest = root / "batch_manifest.csv"
            pd.DataFrame(
                [
                    {"suite": "stability", "status": "completed", "run_dir": str(run_a)},
                    {"suite": "stability", "status": "completed", "run_dir": str(run_b)},
                ]
            ).to_csv(manifest, index=False)

            summary = build_strategy_mean_std(manifest, exclude_strategies={"building_only"})

            self.assertEqual(set(summary["strategy"]), {"random", "proposed_v2_j"})
            proposed = summary[summary["strategy"] == "proposed_v2_j"].iloc[0]
            self.assertAlmostEqual(float(proposed["rmse_encoded_mean"]), 5.0)
            self.assertEqual(int(proposed["run_count"]), 2)

    def test_budget_curve_summary_groups_by_budget_and_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = _make_active_run(root / "budget_a", scene_id=13, rmse_random=5.0, rmse_proposed=4.7)
            run_b = _make_active_run(root / "budget_b", scene_id=13, rmse_random=4.8, rmse_proposed=4.2)
            manifest = root / "batch_manifest.csv"
            pd.DataFrame(
                [
                    {
                        "suite": "budget",
                        "status": "completed",
                        "run_dir": str(run_a),
                        "initial_ratio": 0.02,
                        "budget_ratio": 0.01,
                    },
                    {
                        "suite": "budget",
                        "status": "completed",
                        "run_dir": str(run_b),
                        "initial_ratio": 0.10,
                        "budget_ratio": 0.02,
                    },
                ]
            ).to_csv(manifest, index=False)

            summary = build_budget_curve_summary(manifest)

            self.assertEqual(len(summary), 4)
            self.assertIn("final_sampled_ratio", summary.columns)
            self.assertTrue((summary["rmse_encoded_mean"] > 0).all())

    def test_zero_shot_vs_active_final_combines_scene_and_batch_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene12 = _make_active_run(root / "scene12", scene_id=12, rmse_random=4.8, rmse_proposed=4.4)
            scene13 = _make_active_run(root / "scene13", scene_id=13, rmse_random=5.1, rmse_proposed=5.0)
            manifest = root / "batch_manifest.csv"
            pd.DataFrame(
                [{"suite": "stability", "status": "completed", "run_dir": str(scene13)}]
            ).to_csv(manifest, index=False)

            summary = build_zero_shot_vs_active_final(scene12, scene13, manifest, "proposed_v2_j")

            self.assertIn("scene12_development", set(summary["experiment_group"]))
            self.assertIn("scene13_single_run", set(summary["experiment_group"]))
            self.assertIn("scene13_stability_mean", set(summary["experiment_group"]))
            self.assertTrue((summary["rmse_improvement"] >= 0).all())


def _make_active_run(path: Path, scene_id: int, rmse_random: float, rmse_proposed: float) -> Path:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        f'{{"target": {{"scene_id": {scene_id}}}}}',
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "rmse_encoded": rmse_proposed + 0.5,
                "psnr_db": 33.0,
                "ssim": 0.95,
                "mean_error_encoded": -5.0,
            }
        ]
    ).to_csv(path / "zero_shot_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy": "proposed_v2_j",
                "round": 5,
                "sampled_ratio": 0.15,
                "rmse_encoded": rmse_proposed,
                "psnr_db": 35.0,
                "ssim": 0.97,
                "mean_error_encoded": -4.5,
                "rmse_rank": 1,
            },
            {
                "strategy": "random",
                "round": 5,
                "sampled_ratio": 0.15,
                "rmse_encoded": rmse_random,
                "psnr_db": 34.0,
                "ssim": 0.96,
                "mean_error_encoded": -4.9,
                "rmse_rank": 2,
            },
            {
                "strategy": "building_only",
                "round": 5,
                "sampled_ratio": 0.15,
                "rmse_encoded": rmse_proposed - 0.1,
                "psnr_db": 35.5,
                "ssim": 0.98,
                "mean_error_encoded": -4.4,
                "rmse_rank": 0,
            },
        ]
    ).to_csv(path / "strategy_comparison_summary.csv", index=False)
    pd.DataFrame(
        [
            {"strategy": "random", "round": 0, "rmse_encoded": rmse_random + 0.4, "psnr_db": 33.0, "ssim": 0.95},
            {"strategy": "random", "round": 5, "rmse_encoded": rmse_random, "psnr_db": 34.0, "ssim": 0.96},
            {"strategy": "proposed_v2_j", "round": 0, "rmse_encoded": rmse_proposed + 0.5, "psnr_db": 33.0, "ssim": 0.95},
            {"strategy": "proposed_v2_j", "round": 5, "rmse_encoded": rmse_proposed, "psnr_db": 35.0, "ssim": 0.97},
        ]
    ).to_csv(path / "active_round_metrics.csv", index=False)
    return path
