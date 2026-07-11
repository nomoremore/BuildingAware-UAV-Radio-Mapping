from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lanc.active_sampling import (
    attach_rx_metadata,
    candidate_pool_with_scores,
    count_from_ratio,
    mark_known_labels,
    select_proposed_rx_ids,
    select_random_rx_ids,
    split_pairs_by_rx_id,
)
from lanc.evaluation import (
    build_candidate_pool,
    build_coverage_map,
    coverage_quality_tables,
    plot_coverage_3d_maps,
)
from lanc.farm_data import FarmPreprocessConfig, build_farm_lanc_pairs
from lanc.features import (
    add_farm_lanc_features,
    feature_sanity_checks,
    get_branch_columns,
    inverse_target,
    prepare_matrix_pack,
    transform_with_pack,
)
from lanc.models import BuildingAwareLANC, BuildingAwareLANCV2
from lanc.training import TrainConfig, make_device, predict_lanc_model, set_seed, train_lanc_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-scene active sampling for FARM-adapted LANC.")
    parser.add_argument("--source-run", type=Path, required=True, help="Scene 11 pretraining run directory.")
    parser.add_argument("--target-farm-root", type=Path, required=True, help="Target FARM scene root, e.g. .\\12.")
    parser.add_argument("--target-scene-id", type=int, default=12)
    parser.add_argument("--frequency", default="freq35")
    parser.add_argument("--antenna-pattern", default="pattern_120")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--initial-ratio", type=float, default=0.05)
    parser.add_argument("--budget-ratio", type=float, default=0.02)
    parser.add_argument("--strategies", nargs="+", choices=("random", "proposed"), default=["random", "proposed"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=("lanc_v1", "lanc_v2"), default=None)
    parser.add_argument("--epochs", type=int, default=8, help="Fine-tuning epochs per active round.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--quick", action="store_true", help="Use tx0-tx9 and one yaw per tx in target scene.")
    parser.add_argument("--max-tx", type=int, default=None)
    parser.add_argument("--one-yaw-per-tx", action="store_true")
    parser.add_argument("--save-pairs", action="store_true", help="Save target_pairs.csv; disabled by default to save disk.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.max_tx = args.max_tx or 10
        args.one_yaw_per_tx = True
        args.rounds = min(args.rounds, 2)
        args.epochs = min(args.epochs, 3)
        args.batch_size = min(args.batch_size, 1024)

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = make_device(prefer_gpu=not args.cpu)
    source_manifest = _read_json(args.source_run / "manifest.json")
    model_kind = args.model or source_manifest.get("model", {}).get("kind", "lanc_v2")
    model_name = "BuildingAwareLANCV2" if model_kind == "lanc_v2" else "BuildingAwareLANC"
    source_model_path = args.source_run / f"{model_name}.pt"
    if not source_model_path.exists():
        raise FileNotFoundError(f"Missing source model checkpoint: {source_model_path}")

    run_dir = (
        args.output_root
        / f"active_scene{source_manifest.get('preprocess', {}).get('scene_id', 'source')}"
        f"_to_scene{args.target_scene_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading source run and rebuilding source scalers...")
    source_pairs, source_pack = _load_source_pack(args.source_run, model_kind)
    branch_dims = {branch: len(columns) for branch, columns in get_branch_columns(model_kind).items()}
    source_state = torch.load(source_model_path, map_location="cpu")

    print("Building target scene point dataset...")
    preprocess_config = FarmPreprocessConfig(
        farm_root=args.target_farm_root,
        scene_id=args.target_scene_id,
        frequency=args.frequency,
        antenna_pattern=args.antenna_pattern,
        stride=args.stride,
        max_tx=args.max_tx,
        one_yaw_per_tx=args.one_yaw_per_tx,
    )
    target_pairs = build_farm_lanc_pairs(preprocess_config)
    target_pairs = add_farm_lanc_features(target_pairs)
    feature_sanity_checks(target_pairs)
    if args.save_pairs:
        target_pairs.to_csv(run_dir / "target_pairs.csv", index=False, encoding="utf-8-sig")

    print("Running zero-shot prediction on target scene...")
    base_model = _build_model(model_kind, branch_dims, args.hidden_dim, args.depth)
    base_model.load_state_dict(source_state)
    zero_shot_coverage = _predict_coverage(base_model, target_pairs, source_pack, device)
    zero_shot_coverage.to_csv(run_dir / "coverage_map_zero_shot.csv", index=False, encoding="utf-8-sig")
    zero_metrics = _write_metric_files(zero_shot_coverage, run_dir, "zero_shot", selected_rx_ids=set())
    plot_coverage_3d_maps(zero_shot_coverage, run_dir, prefix="zero_shot_")

    base_candidate_pool = attach_rx_metadata(build_candidate_pool(zero_shot_coverage), target_pairs)
    base_candidate_pool.to_csv(run_dir / "candidate_pool_zero_shot.csv", index=False, encoding="utf-8-sig")
    all_rx_ids = base_candidate_pool["rx_id"].astype(str).tolist()
    initial_budget = count_from_ratio(len(all_rx_ids), args.initial_ratio)
    per_round_budget = count_from_ratio(len(all_rx_ids), args.budget_ratio)
    initial_selected = select_random_rx_ids(base_candidate_pool, set(), initial_budget, rng)

    metric_rows: list[dict[str, float | int | str]] = []
    selected_rows: list[dict[str, float | int | str]] = []

    for strategy in args.strategies:
        print(f"Running active sampling strategy: {strategy}")
        model = _build_model(model_kind, branch_dims, args.hidden_dim, args.depth)
        model.load_state_dict(source_state)
        selected_rx_ids = list(initial_selected)
        current_candidate_pool = base_candidate_pool.copy()
        current_coverage = zero_shot_coverage

        for rx_id in selected_rx_ids:
            selected_rows.append({"strategy": strategy, "round": 0, "rx_id": rx_id, "selection_phase": "initial"})

        for round_id in range(0, args.rounds + 1):
            if round_id > 0:
                if strategy == "random":
                    new_rx_ids = select_random_rx_ids(current_candidate_pool, selected_rx_ids, per_round_budget, rng)
                else:
                    new_rx_ids = select_proposed_rx_ids(current_candidate_pool, selected_rx_ids, per_round_budget, rng)
                selected_rx_ids.extend(new_rx_ids)
                for rx_id in new_rx_ids:
                    selected_rows.append(
                        {"strategy": strategy, "round": round_id, "rx_id": rx_id, "selection_phase": "active"}
                    )

            if selected_rx_ids:
                _fine_tune_on_selected_points(
                    model=model,
                    target_pairs=target_pairs,
                    selected_rx_ids=selected_rx_ids,
                    source_pack=source_pack,
                    config=TrainConfig(
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        patience=max(2, min(5, args.epochs)),
                        min_delta=1e-6,
                    ),
                    device=device,
                    output_dir=run_dir,
                    model_name=f"fine_tuned_{model_name}_{strategy}",
                    rng=rng,
                )

            current_coverage = _predict_coverage(model, target_pairs, source_pack, device)
            current_candidate_pool = attach_rx_metadata(build_candidate_pool(current_coverage), target_pairs)
            current_candidate_pool = candidate_pool_with_scores(current_candidate_pool, selected_rx_ids)

            prefix = f"after_{strategy}" if round_id == args.rounds else f"round{round_id}_{strategy}"
            coverage_path = run_dir / f"coverage_map_{prefix}.csv"
            if round_id == args.rounds or round_id == 0:
                current_coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
            if round_id == args.rounds:
                plot_coverage_3d_maps(current_coverage, run_dir, prefix=f"{prefix}_")
                torch.save(model.state_dict(), run_dir / f"fine_tuned_{model_name}_{strategy}.pt")

            metrics = _coverage_metrics_for_unmeasured(current_coverage, selected_rx_ids)
            metrics.update(
                {
                    "strategy": strategy,
                    "round": round_id,
                    "sampled_rx_count": len(set(selected_rx_ids)),
                    "sampled_ratio": len(set(selected_rx_ids)) / max(len(all_rx_ids), 1),
                    "new_budget_rx_count": 0 if round_id == 0 else per_round_budget,
                    "eval_scope": "unmeasured_rx",
                }
            )
            metric_rows.append(metrics)

    active_metrics = pd.DataFrame(metric_rows)
    active_metrics.to_csv(run_dir / "active_round_metrics.csv", index=False, encoding="utf-8-sig")
    selected_points = _selected_points_table(pd.DataFrame(selected_rows), base_candidate_pool)
    selected_points.to_csv(run_dir / "active_selected_rx_points.csv", index=False, encoding="utf-8-sig")
    _plot_active_curves(active_metrics, run_dir / "active_curve_rmse.png", "rmse_encoded", "Encoded RMSE")
    _plot_active_curves(active_metrics, run_dir / "active_curve_psnr_ssim.png", "psnr_db", "PSNR (dB)")
    _plot_active_curves(active_metrics, run_dir / "active_curve_ssim.png", "ssim", "SSIM")

    manifest = {
        "source_run": str(args.source_run),
        "source_model_path": str(source_model_path),
        "target": asdict(preprocess_config),
        "model_kind": model_kind,
        "model_name": model_name,
        "seed": args.seed,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "zero_shot_metrics": zero_metrics,
        "active": {
            "strategies": args.strategies,
            "rounds": args.rounds,
            "initial_ratio": args.initial_ratio,
            "budget_ratio": args.budget_ratio,
            "initial_budget_rx_count": initial_budget,
            "per_round_budget_rx_count": per_round_budget,
            "evaluation": "Metrics are computed on unmeasured rx_id points after each active round.",
        },
        "rows": {
            "source_pairs": len(source_pairs),
            "target_pairs": len(target_pairs),
            "target_rx_points": len(all_rx_ids),
        },
        "note": "FARM target labels are hidden until selected; revealed labels simulate UAV measurements.",
    }
    _write_json(run_dir / "manifest.json", manifest)

    print(f"Run directory: {run_dir.resolve()}")
    print(active_metrics[["strategy", "round", "sampled_rx_count", "rmse_encoded", "psnr_db", "ssim"]].to_string(index=False))


def _load_source_pack(source_run: Path, model_kind: str):
    pairs_path = source_run / "farm_lanc_pairs.csv"
    splits_path = source_run / "splits.json"
    if not pairs_path.exists() or not splits_path.exists():
        raise FileNotFoundError("source-run must contain farm_lanc_pairs.csv and splits.json.")
    source_pairs = pd.read_csv(pairs_path)
    if "distance" not in source_pairs.columns:
        source_pairs = add_farm_lanc_features(source_pairs)
    feature_sanity_checks(source_pairs)
    splits = _read_json(splits_path)
    train_df = source_pairs[source_pairs["tx_id"].isin(splits["train_tx"])].reset_index(drop=True)
    val_df = source_pairs[source_pairs["tx_id"].isin(splits["val_tx"])].reset_index(drop=True)
    test_df = source_pairs[source_pairs["tx_id"].isin(splits["test_tx"])].reset_index(drop=True)
    return source_pairs, prepare_matrix_pack(train_df, val_df, test_df, model_kind)


def _build_model(model_kind: str, branch_dims: dict[str, int], hidden_dim: int, depth: int):
    model_cls = BuildingAwareLANCV2 if model_kind == "lanc_v2" else BuildingAwareLANC
    return model_cls(
        geometry_dim=branch_dims["geometry"],
        frequency_dim=branch_dims["frequency"],
        antenna_dim=branch_dims["antenna"],
        building_dim=branch_dims["building"],
        hidden_dim=hidden_dim,
        depth=depth,
    )


def _predict_coverage(model, target_pairs: pd.DataFrame, source_pack, device: torch.device) -> pd.DataFrame:
    # 目标场景只能使用源场景拟合的 scaler，否则预训练权重的输入/输出尺度会错位。
    target_x = transform_with_pack(target_pairs, source_pack)
    pred_scaled = predict_lanc_model(model, target_x, device)
    pred_norm = inverse_target(source_pack, pred_scaled)
    return build_coverage_map(target_pairs, pred_norm)


def _fine_tune_on_selected_points(
    model,
    target_pairs: pd.DataFrame,
    selected_rx_ids: list[str],
    source_pack,
    config: TrainConfig,
    device: torch.device,
    output_dir: Path,
    model_name: str,
    rng: np.random.Generator,
) -> None:
    selected_pairs, _ = split_pairs_by_rx_id(target_pairs, selected_rx_ids)
    if selected_pairs.empty:
        return
    train_df, val_df = _split_selected_pairs_for_fine_tuning(selected_pairs, rng)
    x_train = transform_with_pack(train_df, source_pack)
    x_val = transform_with_pack(val_df, source_pack)
    y_train = source_pack.target_scaler.transform(train_df[["signal_norm"]]).astype("float32")
    y_val = source_pack.target_scaler.transform(val_df[["signal_norm"]]).astype("float32")
    train_lanc_model(model, x_train, y_train, x_val, y_val, config, device, output_dir, model_name)


def _split_selected_pairs_for_fine_tuning(
    selected_pairs: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rx_ids = np.array(sorted(selected_pairs["rx_id"].astype(str).unique()))
    if len(rx_ids) < 5:
        return selected_pairs.reset_index(drop=True), selected_pairs.reset_index(drop=True)
    val_count = max(1, int(round(len(rx_ids) * 0.2)))
    val_ids = set(rng.choice(rx_ids, size=val_count, replace=False).tolist())
    val_mask = selected_pairs["rx_id"].astype(str).isin(val_ids)
    train_df = selected_pairs[~val_mask].reset_index(drop=True)
    val_df = selected_pairs[val_mask].reset_index(drop=True)
    if train_df.empty:
        train_df = selected_pairs.reset_index(drop=True)
    if val_df.empty:
        val_df = train_df
    return train_df, val_df


def _write_metric_files(
    coverage_map: pd.DataFrame,
    run_dir: Path,
    prefix: str,
    selected_rx_ids: set[str],
) -> dict[str, float | str]:
    metrics = _coverage_metrics_for_unmeasured(coverage_map, selected_rx_ids)
    pd.DataFrame([metrics]).to_csv(run_dir / f"{prefix}_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics


def _coverage_metrics_for_unmeasured(
    coverage_map: pd.DataFrame,
    selected_rx_ids: list[str] | set[str],
) -> dict[str, float | str]:
    selected = {str(rx_id) for rx_id in selected_rx_ids}
    eval_map = coverage_map[~coverage_map["rx_id"].astype(str).isin(selected)]
    if eval_map.empty:
        eval_map = coverage_map
    overall, _ = coverage_quality_tables(eval_map)
    row = overall.iloc[0].to_dict()
    row["eval_rx_count"] = int(len(eval_map))
    return row


def _selected_points_table(selected_rows: pd.DataFrame, candidate_pool: pd.DataFrame) -> pd.DataFrame:
    if selected_rows.empty:
        return selected_rows
    columns = [
        "rx_id",
        "rx_col",
        "rx_row",
        "rx_height_m",
        "signal_true_norm",
        "signal_pred_norm",
        "building_density_nearby",
        "path_building_ratio_simple",
    ]
    return selected_rows.merge(candidate_pool[columns], on="rx_id", how="left")


def _plot_active_curves(metrics: pd.DataFrame, output_path: Path, metric_name: str, ylabel: str) -> None:
    plt.figure(figsize=(7, 4.5))
    for strategy, part in metrics.groupby("strategy"):
        ordered = part.sort_values("round")
        plt.plot(ordered["round"], ordered[metric_name], marker="o", label=strategy)
    plt.xlabel("Active round")
    plt.ylabel(ylabel)
    plt.title(f"Active sampling curve: {ylabel}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    serializable = json.loads(json.dumps(data, ensure_ascii=False, default=str))
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
