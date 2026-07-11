from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lanc.evaluation import (
    build_candidate_pool,
    build_coverage_map,
    coverage_quality_tables,
    plot_coverage_3d_maps,
    plot_coverage_slice_set,
    plot_metric_summary,
    plot_training_curves,
    signal_metrics,
)
from lanc.farm_data import FarmPreprocessConfig, build_farm_lanc_pairs, split_by_tx_id
from lanc.features import (
    BUILDING_FEATURES,
    add_farm_lanc_features,
    feature_sanity_checks,
    inverse_target,
    prepare_matrix_pack,
    transform_with_pack,
)
from lanc.models import BuildingAwareLANC
from lanc.training import TrainConfig, make_device, predict_lanc_model, set_seed, train_lanc_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FARM scene 11 adapted Building-aware LANC experiment.")
    parser.add_argument("--farm-root", type=Path, default=Path("11"), help="Path containing tx_positions_512.csv.")
    parser.add_argument("--scene-id", type=int, default=11)
    parser.add_argument("--frequency", default="freq35")
    parser.add_argument("--antenna-pattern", default="pattern_120")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--quick", action="store_true", help="Use tx0-tx9 and one yaw per tx for a smoke test.")
    parser.add_argument("--max-tx", type=int, default=None)
    parser.add_argument("--one-yaw-per-tx", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.max_tx = args.max_tx or 10
        args.one_yaw_per_tx = True
        args.epochs = min(args.epochs, 20)
        args.batch_size = min(args.batch_size, 2048)
        args.hidden_dim = min(args.hidden_dim, 96)

    set_seed(args.seed)
    device = make_device(prefer_gpu=not args.cpu)
    run_dir = args.output_root / f"farm_lanc_scene{args.scene_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    preprocess_config = FarmPreprocessConfig(
        farm_root=args.farm_root,
        scene_id=args.scene_id,
        frequency=args.frequency,
        antenna_pattern=args.antenna_pattern,
        stride=args.stride,
        max_tx=args.max_tx,
        one_yaw_per_tx=args.one_yaw_per_tx,
    )
    train_config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)

    print("Building FARM-LANC point dataset...")
    pairs = build_farm_lanc_pairs(preprocess_config)
    pairs = add_farm_lanc_features(pairs)
    feature_sanity_checks(pairs)

    train_df, val_df, test_df, splits = split_by_tx_id(pairs)
    _split_checks(splits)

    # 保存点级样本，后续主动选点和微调可以直接复用同一张表。
    pairs.to_csv(run_dir / "farm_lanc_pairs.csv", index=False, encoding="utf-8-sig")
    (run_dir / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

    pack = prepare_matrix_pack(train_df, val_df, test_df)
    model = BuildingAwareLANC(
        geometry_dim=pack.x_train["geometry"].shape[1],
        frequency_dim=pack.x_train["frequency"].shape[1],
        antenna_dim=pack.x_train["antenna"].shape[1],
        building_dim=pack.x_train["building"].shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )

    result = train_lanc_model(
        model=model,
        x_train=pack.x_train,
        y_train=pack.y_train,
        x_val=pack.x_val,
        y_val=pack.y_val,
        config=train_config,
        device=device,
        output_dir=run_dir,
    )

    test_pred_scaled = predict_lanc_model(model, pack.x_test, device)
    test_pred_norm = inverse_target(pack, test_pred_scaled)
    metrics = signal_metrics(test_df["signal_norm"].to_numpy(), test_pred_norm)
    metrics.update(
        {
            "model": "BuildingAwareLANC",
            "best_val_loss": result.best_val_loss,
            "epochs_ran": result.epochs_ran,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    )
    predictions_test = test_df[
        [
            "rx_id",
            "config_id",
            "tx_id",
            "yaw_deg",
            "rx_col",
            "rx_row",
            "rx_height_m",
            "signal_encoded",
            "signal_norm",
        ]
    ].copy()
    predictions_test["signal_pred_norm"] = np.clip(test_pred_norm, 0.0, 1.0)
    predictions_test["signal_pred_encoded"] = predictions_test["signal_pred_norm"] * 255.0
    predictions_test["abs_error_norm"] = np.abs(predictions_test["signal_pred_norm"] - predictions_test["signal_norm"])
    predictions_test["abs_error_encoded"] = predictions_test["abs_error_norm"] * 255.0
    predictions_test.to_csv(run_dir / "predictions_test.csv", index=False, encoding="utf-8-sig")

    # 全点预测用于形成直观三维无线信号地图；这是后续主动选点的候选空间。
    full_x = transform_with_pack(pairs, pack)
    full_pred_scaled = predict_lanc_model(model, full_x, device)
    full_pred_norm = inverse_target(pack, full_pred_scaled)
    coverage_map = build_coverage_map(pairs, full_pred_norm)
    coverage_map.to_csv(run_dir / "coverage_map_pred.csv", index=False, encoding="utf-8-sig")
    coverage_map[
        [
            "rx_id",
            "rx_col",
            "rx_row",
            "rx_height_m",
            "serving_config_true",
            "serving_tx_true",
            "serving_yaw_true",
            "signal_true_norm",
            "signal_true_encoded",
        ]
    ].to_csv(run_dir / "coverage_map_true.csv", index=False, encoding="utf-8-sig")
    coverage_map[
        [
            "rx_id",
            "rx_col",
            "rx_row",
            "rx_height_m",
            "signal_pred_norm",
            "signal_true_norm",
            "signal_error_norm",
            "abs_error_norm",
            "signal_error_encoded",
            "abs_error_encoded",
        ]
    ].to_csv(run_dir / "coverage_map_error.csv", index=False, encoding="utf-8-sig")
    build_candidate_pool(coverage_map).to_csv(run_dir / "candidate_pool.csv", index=False, encoding="utf-8-sig")

    overall_coverage_metrics, height_coverage_metrics = coverage_quality_tables(coverage_map)
    overall_coverage_metrics.to_csv(run_dir / "coverage_metrics.csv", index=False, encoding="utf-8-sig")
    height_coverage_metrics.to_csv(run_dir / "coverage_metrics_by_height.csv", index=False, encoding="utf-8-sig")
    metrics.update(
        {
            "coverage_nmse": float(overall_coverage_metrics.loc[0, "nmse"]),
            "coverage_rmse_norm": float(overall_coverage_metrics.loc[0, "rmse_norm"]),
            "coverage_rmse_encoded": float(overall_coverage_metrics.loc[0, "rmse_encoded"]),
            "coverage_psnr_db": float(overall_coverage_metrics.loc[0, "psnr_db"]),
            "coverage_ssim": float(overall_coverage_metrics.loc[0, "ssim"]),
        }
    )
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    plot_training_curves(result.history, run_dir / "training_curves.png")
    slice_paths = plot_coverage_slice_set(coverage_map, run_dir)
    map_3d_paths = plot_coverage_3d_maps(coverage_map, run_dir)
    plot_metric_summary(overall_coverage_metrics, height_coverage_metrics, run_dir / "coverage_metrics_summary.png")

    manifest = {
        "seed": args.seed,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "preprocess": {
            "farm_root": str(args.farm_root),
            "scene_id": args.scene_id,
            "frequency": args.frequency,
            "antenna_pattern": args.antenna_pattern,
            "heights_m": list(preprocess_config.heights_m),
            "stride": args.stride,
            "max_tx": args.max_tx,
            "one_yaw_per_tx": args.one_yaw_per_tx,
        },
        "train": train_config.__dict__,
        "rows": {
            "pairs": len(pairs),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "coverage_points": len(coverage_map),
        },
        "splits": splits,
        "building_features": BUILDING_FEATURES,
        "plots": [path.name for path in slice_paths]
        + [path.name for path in map_3d_paths]
        + ["coverage_metrics_summary.png", "training_curves.png"],
        "note": "FARM uint8 labels are predicted as normalized encoded signal, not converted to dBm.",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Run directory: {run_dir.resolve()}")
    print(
        pd.DataFrame([metrics])[
            [
                "model",
                "signal_norm_mae",
                "encoded_mae",
                "coverage_nmse",
                "coverage_rmse_encoded",
                "coverage_psnr_db",
                "coverage_ssim",
                "epochs_ran",
                "device",
            ]
        ].to_string(index=False)
    )


def _split_checks(splits: dict[str, list[int]]) -> None:
    train = set(splits["train_tx"])
    val = set(splits["val_tx"])
    test = set(splits["test_tx"])
    if train & val or train & test or val & test:
        raise ValueError("Split leakage check failed: tx_id overlaps between train/val/test.")
    if not train or not val or not test:
        raise ValueError("Split check failed: train/val/test must all be non-empty.")


if __name__ == "__main__":
    main()
