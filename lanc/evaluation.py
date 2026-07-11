from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def signal_metrics(y_true_norm: np.ndarray, y_pred_norm: np.ndarray) -> dict[str, float]:
    error = y_pred_norm - y_true_norm
    signal_norm_mae = float(np.mean(np.abs(error)))
    signal_norm_rmse = float(np.sqrt(np.mean(error * error)))
    return {
        "signal_norm_mae": signal_norm_mae,
        "signal_norm_rmse": signal_norm_rmse,
        "encoded_mae": signal_norm_mae * 255.0,
        "encoded_rmse": signal_norm_rmse * 255.0,
    }


def coverage_quality_tables(coverage_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_height_rows = []
    for height_m, layer in coverage_map.groupby("rx_height_m"):
        by_height_rows.append(_quality_row(layer, scope=f"z{int(height_m)}", height_m=float(height_m)))
    by_height = pd.DataFrame(by_height_rows)

    overall = _quality_row(coverage_map, scope="overall", height_m=np.nan, compute_ssim=False)
    overall["ssim"] = float(by_height["ssim"].mean())
    return pd.DataFrame([overall]), by_height


def build_coverage_map(pair_df: pd.DataFrame, pred_norm: np.ndarray) -> pd.DataFrame:
    temp = pair_df[
        [
            "rx_id",
            "rx_col",
            "rx_row",
            "rx_height_m",
            "config_id",
            "tx_id",
            "yaw_deg",
            "signal_norm",
            "signal_encoded",
        ]
    ].copy()
    temp["signal_pred_norm"] = np.clip(pred_norm, 0.0, 1.0)
    temp["signal_pred_encoded"] = temp["signal_pred_norm"] * 255.0

    # 多发射机/多朝向融合：同一个三维接收点取预测信号最强的配置，形成区域级三维无线信号图。
    pred_idx = temp.groupby("rx_id")["signal_pred_norm"].idxmax()
    pred_map = temp.loc[pred_idx].rename(
        columns={
            "config_id": "serving_config_pred",
            "tx_id": "serving_tx_pred",
            "yaw_deg": "serving_yaw_pred",
            "signal_norm": "signal_true_for_pred_config_norm",
            "signal_encoded": "signal_true_for_pred_config_encoded",
        }
    )

    true_idx = temp.groupby("rx_id")["signal_norm"].idxmax()
    true_map = temp.loc[true_idx, ["rx_id", "config_id", "tx_id", "yaw_deg", "signal_norm", "signal_encoded"]].rename(
        columns={
            "config_id": "serving_config_true",
            "tx_id": "serving_tx_true",
            "yaw_deg": "serving_yaw_true",
            "signal_norm": "signal_true_norm",
            "signal_encoded": "signal_true_encoded",
        }
    )
    merged = pred_map.merge(true_map, on="rx_id", how="left")
    merged["signal_error_norm"] = merged["signal_pred_norm"] - merged["signal_true_norm"]
    merged["abs_error_norm"] = np.abs(merged["signal_error_norm"])
    merged["signal_error_encoded"] = merged["signal_error_norm"] * 255.0
    merged["abs_error_encoded"] = merged["abs_error_norm"] * 255.0
    columns = [
        "rx_id",
        "rx_col",
        "rx_row",
        "rx_height_m",
        "serving_config_pred",
        "serving_tx_pred",
        "serving_yaw_pred",
        "serving_config_true",
        "serving_tx_true",
        "serving_yaw_true",
        "signal_pred_norm",
        "signal_pred_encoded",
        "signal_true_norm",
        "signal_true_encoded",
        "signal_true_for_pred_config_norm",
        "signal_true_for_pred_config_encoded",
        "signal_error_norm",
        "abs_error_norm",
        "signal_error_encoded",
        "abs_error_encoded",
    ]
    return merged[columns].sort_values(["rx_height_m", "rx_row", "rx_col"]).reset_index(drop=True)


def build_candidate_pool(coverage_map: pd.DataFrame) -> pd.DataFrame:
    pool = coverage_map[
        ["rx_id", "rx_col", "rx_row", "rx_height_m", "signal_true_norm", "signal_pred_norm"]
    ].copy()
    pool["label_known"] = False
    pool["uncertainty"] = 0.0
    pool["value_score"] = 0.0
    return pool[
        [
            "rx_id",
            "rx_col",
            "rx_row",
            "rx_height_m",
            "label_known",
            "signal_true_norm",
            "signal_pred_norm",
            "uncertainty",
            "value_score",
        ]
    ]


def plot_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (scaled target)")
    plt.title("Building-aware LANC training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_coverage_slice_set(coverage_map: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    signal_vmin, signal_vmax = _color_limits(
        np.concatenate(
            [
                coverage_map["signal_pred_norm"].to_numpy(),
                coverage_map["signal_true_norm"].to_numpy(),
            ]
        )
    )
    error_vmin, error_vmax = _error_limits(coverage_map["abs_error_norm"].to_numpy())
    for z_m in sorted(coverage_map["rx_height_m"].unique()):
        paths.append(
            _plot_one_slice(
                coverage_map,
                output_dir / f"coverage_slice_pred_z{int(z_m)}.png",
                z_m,
                "signal_pred_norm",
                "Predicted normalized signal",
                f"Predicted coverage slice z={int(z_m)}m",
                "viridis",
                signal_vmin,
                signal_vmax,
            )
        )
        paths.append(
            _plot_one_slice(
                coverage_map,
                output_dir / f"coverage_slice_true_z{int(z_m)}.png",
                z_m,
                "signal_true_norm",
                "True normalized signal",
                f"FARM true coverage slice z={int(z_m)}m",
                "viridis",
                signal_vmin,
                signal_vmax,
            )
        )
        paths.append(
            _plot_one_slice(
                coverage_map,
                output_dir / f"coverage_slice_abs_error_z{int(z_m)}.png",
                z_m,
                "abs_error_norm",
                "Absolute error (normalized)",
                f"Absolute error slice z={int(z_m)}m",
                "magma",
                error_vmin,
                error_vmax,
            )
        )
    return paths


def plot_coverage_3d_maps(coverage_map: pd.DataFrame, output_dir: Path) -> list[Path]:
    signal_vmin, signal_vmax = _color_limits(
        np.concatenate(
            [
                coverage_map["signal_pred_norm"].to_numpy(),
                coverage_map["signal_true_norm"].to_numpy(),
            ]
        )
    )
    error_vmin, error_vmax = _error_limits(coverage_map["abs_error_norm"].to_numpy())
    specs = [
        (
            "signal_pred_norm",
            output_dir / "coverage_3d_pred.png",
            "Predicted 3D radio map",
            "Predicted normalized signal",
            "viridis",
            signal_vmin,
            signal_vmax,
        ),
        (
            "signal_true_norm",
            output_dir / "coverage_3d_true.png",
            "FARM true 3D radio map",
            "True normalized signal",
            "viridis",
            signal_vmin,
            signal_vmax,
        ),
        (
            "abs_error_norm",
            output_dir / "coverage_3d_abs_error.png",
            "3D absolute prediction error",
            "Absolute error (normalized)",
            "magma",
            error_vmin,
            error_vmax,
        ),
    ]
    paths = []
    for value_column, output_path, title, colorbar_label, cmap, vmin, vmax in specs:
        _plot_coverage_3d_scatter(
            coverage_map,
            output_path,
            value_column,
            title,
            colorbar_label,
            cmap,
            vmin,
            vmax,
        )
        paths.append(output_path)
    return paths


def plot_metric_summary(
    overall_metrics: pd.DataFrame,
    height_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    overall = overall_metrics.iloc[0]
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.15])

    ax_table = fig.add_subplot(gs[0])
    ax_table.axis("off")
    table_data = [
        ["NMSE ↓", f"{overall['nmse']:.6f}"],
        ["RMSE ↓", f"{overall['rmse_norm']:.6f}"],
        ["Encoded RMSE ↓", f"{overall['rmse_encoded']:.3f}"],
        ["PSNR ↑", f"{overall['psnr_db']:.2f} dB"],
        ["SSIM ↑", f"{overall['ssim']:.4f}"],
    ]
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Metric", "Overall coverage map"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.35)
    ax_table.set_title("Coverage-map quality metrics", pad=10)

    ax = fig.add_subplot(gs[1])
    ax.plot(height_metrics["height_m"], height_metrics["rmse_norm"], marker="o", label="RMSE ↓")
    ax.set_xlabel("height (m)")
    ax.set_ylabel("RMSE")
    ax2 = ax.twinx()
    ax2.plot(height_metrics["height_m"], height_metrics["ssim"], marker="s", color="#d95f02", label="SSIM ↑")
    ax2.set_ylabel("SSIM")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_one_slice(
    coverage_map: pd.DataFrame,
    output_path: Path,
    z_m: float,
    value_column: str,
    colorbar_label: str,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> Path:
    layer = coverage_map[coverage_map["rx_height_m"] == z_m]
    pivot = layer.pivot(index="rx_row", columns="rx_col", values=value_column)
    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        pivot.to_numpy(),
        origin="lower",
        extent=[
            float(pivot.columns.min()),
            float(pivot.columns.max()),
            float(pivot.index.min()),
            float(pivot.index.max()),
        ],
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, label=colorbar_label)
    plt.xlabel("x grid")
    plt.ylabel("y grid")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _plot_coverage_3d_scatter(
    coverage_map: pd.DataFrame,
    output_path: Path,
    value_column: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    sample = coverage_map.copy()
    if len(sample) > 12000:
        sample = sample.sample(12000, random_state=42)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        sample["rx_col"],
        sample["rx_row"],
        sample["rx_height_m"],
        c=sample[value_column],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=10,
        alpha=0.78,
    )
    ax.set_xlabel("x grid")
    ax.set_ylabel("y grid")
    ax.set_zlabel("height (m)")
    ax.set_title(title)
    ax.view_init(elev=24, azim=-58)
    fig.colorbar(scatter, ax=ax, shrink=0.72, label=colorbar_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _quality_row(
    df: pd.DataFrame,
    scope: str,
    height_m: float,
    compute_ssim: bool = True,
) -> dict[str, float | str]:
    true = df["signal_true_norm"].to_numpy(dtype=float)
    pred = df["signal_pred_norm"].to_numpy(dtype=float)
    error = pred - true
    mse = float(np.mean(error * error))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    denom = float(np.sum(true * true))
    nmse = float(np.sum(error * error) / denom) if denom > 0 else float("nan")
    psnr = float("inf") if mse == 0 else float(20.0 * np.log10(1.0 / rmse))
    ssim = _ssim_for_layer(df) if compute_ssim else float("nan")
    return {
        "scope": scope,
        "height_m": height_m,
        "nmse": nmse,
        "mae_norm": mae,
        "rmse_norm": rmse,
        "mae_encoded": mae * 255.0,
        "rmse_encoded": rmse * 255.0,
        "psnr_db": psnr,
        "ssim": ssim,
    }


def _ssim_for_layer(df: pd.DataFrame) -> float:
    true = df.pivot(index="rx_row", columns="rx_col", values="signal_true_norm").to_numpy(dtype=float)
    pred = df.pivot(index="rx_row", columns="rx_col", values="signal_pred_norm").to_numpy(dtype=float)
    data_range = 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_true = float(true.mean())
    mu_pred = float(pred.mean())
    var_true = float(((true - mu_true) ** 2).mean())
    var_pred = float(((pred - mu_pred) ** 2).mean())
    cov = float(((true - mu_true) * (pred - mu_pred)).mean())
    numerator = (2.0 * mu_true * mu_pred + c1) * (2.0 * cov + c2)
    denominator = (mu_true * mu_true + mu_pred * mu_pred + c1) * (var_true + var_pred + c2)
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _color_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0, 1.0
    vmin = float(np.quantile(finite, 0.02))
    vmax = float(np.quantile(finite, 0.98))
    if vmax <= vmin:
        center = float(finite.mean())
        return max(0.0, center - 0.05), min(1.0, center + 0.05)
    return max(0.0, vmin), min(1.0, vmax)


def _error_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0, 0.05
    vmax = float(np.quantile(finite, 0.98))
    if vmax <= 0:
        vmax = float(finite.max())
    return 0.0, max(vmax, 1e-6)
