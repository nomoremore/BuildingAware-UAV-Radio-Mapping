from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare active sampling runs across FARM target scenes.")
    parser.add_argument("--scene12-run", type=Path, required=True, help="Active sampling run directory for scene 12.")
    parser.add_argument("--scene13-run", type=Path, required=True, help="Active sampling run directory for scene 13.")
    parser.add_argument(
        "--main-strategy",
        default="proposed_v2_j",
        help="Frozen proposed strategy used as the final method in the paper-style summary.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / f"paper_summary_scene12_scene13_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scene_runs = [args.scene12_run, args.scene13_run]
    strategy_summary = _build_strategy_summary(scene_runs)
    strategy_summary.to_csv(run_dir / "scene12_vs_scene13_strategy_summary.csv", index=False, encoding="utf-8-sig")

    zero_vs_active = _build_zero_vs_active_summary(scene_runs, args.main_strategy)
    zero_vs_active.to_csv(run_dir / "zero_shot_vs_active_summary.csv", index=False, encoding="utf-8-sig")

    _plot_metric_bars(strategy_summary, run_dir / "rmse_psnr_ssim_comparison.png")
    _plot_zero_vs_active(zero_vs_active, run_dir / "zero_shot_vs_active_rmse.png")
    _plot_active_curves(scene_runs, run_dir / "active_curve_rmse_scene12_scene13.png", "rmse_encoded", "Encoded RMSE")

    manifest = {
        "scene12_run": str(args.scene12_run),
        "scene13_run": str(args.scene13_run),
        "main_strategy": args.main_strategy,
        "outputs": [
            "scene12_vs_scene13_strategy_summary.csv",
            "zero_shot_vs_active_summary.csv",
            "rmse_psnr_ssim_comparison.png",
            "zero_shot_vs_active_rmse.png",
            "active_curve_rmse_scene12_scene13.png",
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Summary directory: {run_dir.resolve()}")
    print("Final strategy summary:")
    print(
        strategy_summary[
            ["scene_id", "strategy", "round", "rmse_encoded", "psnr_db", "ssim", "mean_error_encoded", "rmse_rank"]
        ].to_string(index=False)
    )
    print("Zero-shot vs active summary:")
    print(
        zero_vs_active[
            [
                "scene_id",
                "main_strategy",
                "zero_shot_rmse_encoded",
                "active_rmse_encoded",
                "rmse_improvement",
                "rmse_improvement_pct",
            ]
        ].to_string(index=False)
    )


def _build_strategy_summary(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        summary_path = run_dir / "strategy_comparison_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing strategy summary: {summary_path}")
        df = pd.read_csv(summary_path)
        df.insert(0, "scene_id", _read_scene_id(run_dir))
        df.insert(1, "run_dir", str(run_dir))
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["scene_id", "rmse_rank"]).reset_index(drop=True)


def _build_zero_vs_active_summary(run_dirs: list[Path], main_strategy: str) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        scene_id = _read_scene_id(run_dir)
        zero = pd.read_csv(run_dir / "zero_shot_metrics.csv").iloc[0].to_dict()
        active = pd.read_csv(run_dir / "strategy_comparison_summary.csv")
        if main_strategy not in set(active["strategy"].astype(str)):
            raise ValueError(f"{main_strategy!r} not found in {run_dir / 'strategy_comparison_summary.csv'}")
        main = active[active["strategy"].astype(str) == main_strategy].iloc[0].to_dict()
        rmse_improvement = float(zero["rmse_encoded"]) - float(main["rmse_encoded"])
        rows.append(
            {
                "scene_id": scene_id,
                "run_dir": str(run_dir),
                "main_strategy": main_strategy,
                "zero_shot_rmse_encoded": float(zero["rmse_encoded"]),
                "active_rmse_encoded": float(main["rmse_encoded"]),
                "rmse_improvement": rmse_improvement,
                "rmse_improvement_pct": rmse_improvement / max(float(zero["rmse_encoded"]), 1e-12) * 100.0,
                "zero_shot_psnr_db": float(zero["psnr_db"]),
                "active_psnr_db": float(main["psnr_db"]),
                "psnr_gain_db": float(main["psnr_db"]) - float(zero["psnr_db"]),
                "zero_shot_ssim": float(zero["ssim"]),
                "active_ssim": float(main["ssim"]),
                "ssim_gain": float(main["ssim"]) - float(zero["ssim"]),
                "zero_shot_mean_error_encoded": float(zero["mean_error_encoded"]),
                "active_mean_error_encoded": float(main["mean_error_encoded"]),
            }
        )
    return pd.DataFrame(rows).sort_values("scene_id").reset_index(drop=True)


def _plot_metric_bars(strategy_summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metrics = [
        ("rmse_encoded", "Encoded RMSE", True),
        ("psnr_db", "PSNR (dB)", False),
        ("ssim", "SSIM", False),
    ]
    scenes = sorted(strategy_summary["scene_id"].unique())
    for ax, (metric, ylabel, lower_is_better) in zip(axes, metrics, strict=True):
        pivot = strategy_summary.pivot(index="strategy", columns="scene_id", values=metric)
        # scene 12 是策略开发集，可能包含更多 proposed_v2 候选；跨场景图只画共同策略，避免把测试集图画乱。
        pivot = pivot.dropna(axis=0, how="any")
        # 按两个场景的平均 RMSE 排序，让不同指标图里的策略顺序保持一致，方便论文阅读。
        rmse_order = (
            strategy_summary.pivot(index="strategy", columns="scene_id", values="rmse_encoded")
            .dropna(axis=0, how="any")
            .mean(axis=1)
            .sort_values(ascending=True)
            .index
        )
        pivot = pivot.loc[rmse_order]
        x = np.arange(len(pivot.index))
        width = 0.8 / max(len(scenes), 1)
        for idx, scene_id in enumerate(scenes):
            offset = (idx - (len(scenes) - 1) / 2) * width
            ax.bar(x + offset, pivot[scene_id], width=width, label=f"scene {scene_id}")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        direction = "lower is better" if lower_is_better else "higher is better"
        ax.set_title(f"{ylabel} ({direction})")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_zero_vs_active(summary: pd.DataFrame, output_path: Path) -> None:
    scenes = [f"scene {scene_id}" for scene_id in summary["scene_id"]]
    x = np.arange(len(summary))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - width / 2, summary["zero_shot_rmse_encoded"], width=width, label="zero-shot", color="#9ecae1")
    ax.bar(x + width / 2, summary["active_rmse_encoded"], width=width, label=summary["main_strategy"].iloc[0], color="#3182bd")
    ax.set_ylabel("Encoded RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_title("Zero-shot vs active fine-tuning")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_active_curves(run_dirs: list[Path], output_path: Path, metric: str, ylabel: str) -> None:
    fig, axes = plt.subplots(1, len(run_dirs), figsize=(7.2 * len(run_dirs), 4.8), sharey=True)
    if len(run_dirs) == 1:
        axes = [axes]
    common_strategies = _common_strategies(run_dirs)
    for ax, run_dir in zip(axes, run_dirs, strict=True):
        metrics = pd.read_csv(run_dir / "active_round_metrics.csv")
        metrics = metrics[metrics["strategy"].astype(str).isin(common_strategies)]
        for strategy, part in metrics.groupby("strategy"):
            ordered = part.sort_values("round")
            ax.plot(ordered["round"], ordered[metric], marker="o", label=strategy)
        ax.set_title(f"scene {_read_scene_id(run_dir)}")
        ax.set_xlabel("Active round")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _read_scene_id(run_dir: Path) -> int:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = manifest.get("target", {})
        if "scene_id" in target:
            return int(target["scene_id"])
    scene_tokens = []
    for part in run_dir.name.split("_"):
        if part.startswith("scene") and part[5:].isdigit():
            scene_tokens.append(int(part[5:]))
    if scene_tokens:
        return scene_tokens[-1]
    raise ValueError(f"Cannot infer scene id from {run_dir}")


def _common_strategies(run_dirs: list[Path]) -> set[str]:
    common: set[str] | None = None
    for run_dir in run_dirs:
        metrics = pd.read_csv(run_dir / "active_round_metrics.csv")
        strategies = set(metrics["strategy"].astype(str))
        common = strategies if common is None else common & strategies
    return common or set()


if __name__ == "__main__":
    main()
