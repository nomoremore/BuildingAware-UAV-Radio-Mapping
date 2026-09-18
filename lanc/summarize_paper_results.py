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


METRICS = ("rmse_encoded", "psnr_db", "ssim", "mean_error_encoded")
DEFAULT_MAIN_EXCLUDES: set[str] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready tables and figures from active sampling runs.")
    parser.add_argument("--scene12-run", type=Path, required=True)
    parser.add_argument("--scene13-run", type=Path, required=True)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--main-strategy", default="proposed_v3")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / f"paper_final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_strategies = set(DEFAULT_MAIN_EXCLUDES)
    strategy_summary = build_strategy_mean_std(args.batch_manifest, exclude_strategies=exclude_strategies)
    budget_summary = build_budget_curve_summary(args.batch_manifest, exclude_strategies=exclude_strategies)
    zero_vs_active = build_zero_shot_vs_active_final(
        args.scene12_run,
        args.scene13_run,
        args.batch_manifest,
        args.main_strategy,
    )

    strategy_summary.to_csv(output_dir / "strategy_mean_std_scene13.csv", index=False, encoding="utf-8-sig")
    budget_summary.to_csv(output_dir / "budget_curve_summary.csv", index=False, encoding="utf-8-sig")
    zero_vs_active.to_csv(output_dir / "zero_shot_vs_active_final.csv", index=False, encoding="utf-8-sig")

    _plot_strategy_rmse(strategy_summary, output_dir / "strategy_rmse_mean_std.png")
    _plot_strategy_psnr_ssim(strategy_summary, output_dir / "strategy_psnr_ssim_mean_std.png")
    _plot_budget_curve(budget_summary, output_dir / "sampling_budget_curve.png")
    _plot_zero_vs_active(zero_vs_active, output_dir / "final_zero_shot_vs_active.png")
    _write_manifest(output_dir, args, exclude_strategies)

    print(f"Paper summary directory: {output_dir.resolve()}")
    print("Strategy mean/std summary:")
    print(strategy_summary.to_string(index=False))
    print("Zero-shot vs active final summary:")
    print(zero_vs_active.to_string(index=False))


def build_strategy_mean_std(
    batch_manifest: Path,
    exclude_strategies: set[str] | None = None,
) -> pd.DataFrame:
    if exclude_strategies is None:
        exclude_strategies = set(DEFAULT_MAIN_EXCLUDES)
    run_rows = _completed_manifest_rows(batch_manifest, suite="stability")
    metric_rows = []
    for _, row in run_rows.iterrows():
        run_dir = Path(str(row["run_dir"]))
        summary = _read_strategy_summary(run_dir)
        summary["run_dir"] = str(run_dir)
        summary["seed"] = int(row["seed"]) if "seed" in row and pd.notna(row["seed"]) else np.nan
        metric_rows.append(summary)
    if not metric_rows:
        return _empty_strategy_mean_std()

    combined = pd.concat(metric_rows, ignore_index=True)
    combined = _filter_excluded_strategies(combined, exclude_strategies)
    # 每个 seed 的最低 RMSE 记为一次胜出，用于报告跨随机种子的逐 seed 胜率。
    combined["rmse_is_win"] = combined.groupby("run_dir")["rmse_encoded"].transform(
        lambda values: np.isclose(values, values.min())
    )
    grouped = combined.groupby("strategy", as_index=False).agg(
        run_count=("run_dir", "nunique"),
        rmse_win_rate=("rmse_is_win", "mean"),
        rmse_encoded_mean=("rmse_encoded", "mean"),
        rmse_encoded_std=("rmse_encoded", "std"),
        psnr_db_mean=("psnr_db", "mean"),
        psnr_db_std=("psnr_db", "std"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", "std"),
        mean_error_encoded_mean=("mean_error_encoded", "mean"),
        mean_error_encoded_std=("mean_error_encoded", "std"),
    )
    grouped = grouped.fillna(0.0).sort_values("rmse_encoded_mean").reset_index(drop=True)
    grouped["rmse_rank_mean"] = np.arange(1, len(grouped) + 1)
    return grouped


def build_budget_curve_summary(
    batch_manifest: Path,
    exclude_strategies: set[str] | None = None,
) -> pd.DataFrame:
    if exclude_strategies is None:
        exclude_strategies = set(DEFAULT_MAIN_EXCLUDES)
    run_rows = _completed_manifest_rows(batch_manifest, suite="budget")
    metric_rows = []
    for _, row in run_rows.iterrows():
        run_dir = Path(str(row["run_dir"]))
        summary = _read_strategy_summary(run_dir)
        summary["run_dir"] = str(run_dir)
        summary["initial_ratio"] = float(row["initial_ratio"])
        summary["budget_ratio"] = float(row["budget_ratio"])
        summary["final_sampled_ratio"] = summary.get(
            "sampled_ratio",
            summary["initial_ratio"] + float(row.get("rounds", 5)) * summary["budget_ratio"],
        )
        metric_rows.append(summary)
    if not metric_rows:
        return _empty_budget_summary()

    combined = pd.concat(metric_rows, ignore_index=True)
    combined = _filter_excluded_strategies(combined, exclude_strategies)
    grouped = combined.groupby(["initial_ratio", "budget_ratio", "strategy"], as_index=False).agg(
        run_count=("run_dir", "nunique"),
        final_sampled_ratio=("final_sampled_ratio", "mean"),
        rmse_encoded_mean=("rmse_encoded", "mean"),
        rmse_encoded_std=("rmse_encoded", "std"),
        psnr_db_mean=("psnr_db", "mean"),
        psnr_db_std=("psnr_db", "std"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", "std"),
        mean_error_encoded_mean=("mean_error_encoded", "mean"),
        mean_error_encoded_std=("mean_error_encoded", "std"),
    )
    return grouped.fillna(0.0).sort_values(["strategy", "final_sampled_ratio"]).reset_index(drop=True)


def build_zero_shot_vs_active_final(
    scene12_run: Path,
    scene13_run: Path,
    batch_manifest: Path,
    main_strategy: str,
) -> pd.DataFrame:
    rows = [
        _single_run_zero_vs_active(scene12_run, main_strategy, "scene12_development"),
        _single_run_zero_vs_active(scene13_run, main_strategy, "scene13_single_run"),
    ]
    stability_rows = _completed_manifest_rows(batch_manifest, suite="stability")
    if not stability_rows.empty:
        rows.append(_batch_zero_vs_active(stability_rows, main_strategy, "scene13_stability_mean"))
    return pd.DataFrame(rows)


def _completed_manifest_rows(batch_manifest: Path, suite: str | None = None) -> pd.DataFrame:
    if not batch_manifest.exists():
        raise FileNotFoundError(f"Missing batch manifest: {batch_manifest}")
    manifest = pd.read_csv(batch_manifest)
    if manifest.empty:
        return manifest
    if "status" not in manifest.columns or "run_dir" not in manifest.columns:
        raise ValueError("batch_manifest.csv must contain status and run_dir columns.")
    rows = manifest[manifest["status"].astype(str) == "completed"].copy()
    rows = rows[rows["run_dir"].notna() & (rows["run_dir"].astype(str) != "")]
    if suite is not None and "suite" in rows.columns:
        rows = rows[rows["suite"].astype(str) == suite]
    return rows.reset_index(drop=True)


def _read_strategy_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "strategy_comparison_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing strategy comparison summary: {path}")
    summary = pd.read_csv(path)
    required = {"strategy", *METRICS}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return summary


def _read_zero_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "zero_shot_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing zero-shot metrics: {path}")
    row = pd.read_csv(path).iloc[0].to_dict()
    return {metric: float(row[metric]) for metric in METRICS}


def _single_run_zero_vs_active(run_dir: Path, main_strategy: str, group_name: str) -> dict[str, float | int | str]:
    zero = _read_zero_metrics(run_dir)
    summary = _read_strategy_summary(run_dir)
    active = _main_strategy_row(summary, main_strategy)
    row = _zero_vs_active_payload(zero, active, main_strategy)
    row.update({"experiment_group": group_name, "scene_id": _read_scene_id(run_dir), "run_count": 1})
    return row


def _batch_zero_vs_active(rows: pd.DataFrame, main_strategy: str, group_name: str) -> dict[str, float | int | str]:
    zero_metrics = []
    active_metrics = []
    scene_ids = []
    for _, manifest_row in rows.iterrows():
        run_dir = Path(str(manifest_row["run_dir"]))
        zero_metrics.append(_read_zero_metrics(run_dir))
        active_metrics.append(_main_strategy_row(_read_strategy_summary(run_dir), main_strategy))
        scene_ids.append(_read_scene_id(run_dir))
    zero_mean = {metric: float(np.mean([row[metric] for row in zero_metrics])) for metric in METRICS}
    active_mean = {metric: float(np.mean([row[metric] for row in active_metrics])) for metric in METRICS}
    payload = _zero_vs_active_payload(zero_mean, active_mean, main_strategy)
    payload.update(
        {
            "experiment_group": group_name,
            "scene_id": int(scene_ids[0]) if scene_ids else -1,
            "run_count": int(len(rows)),
        }
    )
    return payload


def _zero_vs_active_payload(
    zero: dict[str, float],
    active: dict[str, float],
    main_strategy: str,
) -> dict[str, float | str]:
    rmse_improvement = zero["rmse_encoded"] - active["rmse_encoded"]
    return {
        "main_strategy": main_strategy,
        "zero_shot_rmse_encoded": zero["rmse_encoded"],
        "active_rmse_encoded": active["rmse_encoded"],
        "rmse_improvement": rmse_improvement,
        "rmse_improvement_pct": rmse_improvement / max(zero["rmse_encoded"], 1e-12) * 100.0,
        "zero_shot_psnr_db": zero["psnr_db"],
        "active_psnr_db": active["psnr_db"],
        "psnr_gain_db": active["psnr_db"] - zero["psnr_db"],
        "zero_shot_ssim": zero["ssim"],
        "active_ssim": active["ssim"],
        "ssim_gain": active["ssim"] - zero["ssim"],
        "zero_shot_mean_error_encoded": zero["mean_error_encoded"],
        "active_mean_error_encoded": active["mean_error_encoded"],
    }


def _main_strategy_row(summary: pd.DataFrame, main_strategy: str) -> dict[str, float]:
    rows = summary[summary["strategy"].astype(str) == main_strategy]
    if rows.empty:
        raise ValueError(f"Strategy {main_strategy!r} not found in strategy_comparison_summary.csv")
    raw = rows.iloc[0].to_dict()
    return {metric: float(raw[metric]) for metric in METRICS}


def _filter_excluded_strategies(df: pd.DataFrame, exclude_strategies: set[str] | None) -> pd.DataFrame:
    if not exclude_strategies:
        return df
    return df[~df["strategy"].astype(str).isin(exclude_strategies)].reset_index(drop=True)


def _plot_strategy_rmse(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if summary.empty:
        _draw_empty_plot(ax, "No stability runs found")
    else:
        x = np.arange(len(summary))
        ax.bar(x, summary["rmse_encoded_mean"], yerr=summary["rmse_encoded_std"], color="#4c78a8", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["strategy"], rotation=25, ha="right")
        ax.set_ylabel("Encoded RMSE")
        ax.set_title("Scene 13 strategy stability")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_strategy_psnr_ssim(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    if summary.empty:
        _draw_empty_plot(ax1, "No stability runs found")
    else:
        x = np.arange(len(summary))
        width = 0.38
        ax1.bar(x - width / 2, summary["psnr_db_mean"], yerr=summary["psnr_db_std"], width=width, color="#59a14f", capsize=3, label="PSNR")
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary["strategy"], rotation=25, ha="right")
        ax1.grid(axis="y", alpha=0.25)
        ax2 = ax1.twinx()
        ax2.bar(x + width / 2, summary["ssim_mean"], yerr=summary["ssim_std"], width=width, color="#f28e2b", capsize=3, label="SSIM")
        ax2.set_ylabel("SSIM")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        ax1.set_title("Scene 13 strategy PSNR/SSIM stability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_budget_curve(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if summary.empty:
        _draw_empty_plot(ax, "No budget runs found")
    else:
        for strategy, part in summary.groupby("strategy"):
            ordered = part.sort_values("final_sampled_ratio")
            ax.errorbar(
                ordered["final_sampled_ratio"] * 100.0,
                ordered["rmse_encoded_mean"],
                yerr=ordered["rmse_encoded_std"],
                marker="o",
                capsize=3,
                label=strategy,
            )
        ax.set_xlabel("Final sampled UAV points (%)")
        ax.set_ylabel("Encoded RMSE")
        ax.set_title("Sampling budget curve")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_zero_vs_active(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if summary.empty:
        _draw_empty_plot(ax, "No zero-shot summary found")
    else:
        x = np.arange(len(summary))
        width = 0.36
        ax.bar(x - width / 2, summary["zero_shot_rmse_encoded"], width=width, label="zero-shot", color="#9ecae1")
        ax.bar(x + width / 2, summary["active_rmse_encoded"], width=width, label=summary["main_strategy"].iloc[0], color="#3182bd")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["experiment_group"], rotation=20, ha="right")
        ax.set_ylabel("Encoded RMSE")
        ax.set_title("Zero-shot vs active fine-tuning")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _draw_empty_plot(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _write_manifest(output_dir: Path, args: argparse.Namespace, exclude_strategies: set[str]) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scene12_run": str(args.scene12_run),
        "scene13_run": str(args.scene13_run),
        "batch_manifest": str(args.batch_manifest),
        "main_strategy": args.main_strategy,
        "excluded_from_main_plots": sorted(exclude_strategies),
        "outputs": [
            "strategy_mean_std_scene13.csv",
            "budget_curve_summary.csv",
            "zero_shot_vs_active_final.csv",
            "strategy_rmse_mean_std.png",
            "strategy_psnr_ssim_mean_std.png",
            "sampling_budget_curve.png",
            "final_zero_shot_vs_active.png",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    return -1


def _empty_strategy_mean_std() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "strategy",
            "run_count",
            "rmse_win_rate",
            "rmse_encoded_mean",
            "rmse_encoded_std",
            "psnr_db_mean",
            "psnr_db_std",
            "ssim_mean",
            "ssim_std",
            "mean_error_encoded_mean",
            "mean_error_encoded_std",
            "rmse_rank_mean",
        ]
    )


def _empty_budget_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "initial_ratio",
            "budget_ratio",
            "strategy",
            "run_count",
            "final_sampled_ratio",
            "rmse_encoded_mean",
            "rmse_encoded_std",
            "psnr_db_mean",
            "psnr_db_std",
            "ssim_mean",
            "ssim_std",
            "mean_error_encoded_mean",
            "mean_error_encoded_std",
        ]
    )


if __name__ == "__main__":
    main()
