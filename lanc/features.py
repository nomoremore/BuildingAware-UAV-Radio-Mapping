from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


GEOMETRY_FEATURES_V1 = ["log_distance", "theta_v_deg", "rx_height_m"]
GEOMETRY_FEATURES_V2 = [
    "log_distance",
    "distance",
    "theta_v_deg",
    "rx_height_m",
    "tx_row",
    "tx_col",
    "tx_height_m",
    "rx_row",
    "rx_col",
]
FREQUENCY_FEATURES = ["frequency_ghz"]
ANTENNA_FEATURES_V1 = ["delta_theta_h_sin", "delta_theta_h_cos", "antenna_pattern_id"]
ANTENNA_FEATURES_V2 = [
    "delta_theta_h_sin",
    "delta_theta_h_cos",
    "yaw_sin",
    "yaw_cos",
    "antenna_pattern_id",
]
BUILDING_FEATURES = [
    "rx_is_building",
    "local_building_height",
    "height_clearance",
    "building_density_nearby",
    "path_building_ratio_simple",
]
BRANCH_COLUMNS_BY_MODEL = {
    "lanc_v1": {
        "geometry": GEOMETRY_FEATURES_V1,
        "frequency": FREQUENCY_FEATURES,
        "antenna": ANTENNA_FEATURES_V1,
        "building": BUILDING_FEATURES,
    },
    "lanc_v2": {
        "geometry": GEOMETRY_FEATURES_V2,
        "frequency": FREQUENCY_FEATURES,
        "antenna": ANTENNA_FEATURES_V2,
        "building": BUILDING_FEATURES,
    },
}


@dataclass
class MatrixPack:
    x_train: dict[str, np.ndarray]
    x_val: dict[str, np.ndarray]
    x_test: dict[str, np.ndarray]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    target_scaler: StandardScaler
    feature_scalers: dict[str, StandardScaler]
    branch_columns: dict[str, list[str]]


def add_farm_lanc_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dx = out["rx_col"].to_numpy(dtype=np.float32) - out["tx_col"].to_numpy(dtype=np.float32)
    dy = out["rx_row"].to_numpy(dtype=np.float32) - out["tx_row"].to_numpy(dtype=np.float32)
    dz = out["rx_height_m"].to_numpy(dtype=np.float32) - out["tx_height_m"].to_numpy(dtype=np.float32)
    horizontal_distance = np.maximum(np.sqrt(dx * dx + dy * dy), 1.0)
    distance = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1.0)

    # FARM 版专家特征压缩：只保留可由 Tx/Rx 几何、频率和 yaw 得到的核心物理特征。
    theta_h_deg = np.degrees(np.arctan2(dx, dy)).astype(np.float32)
    theta_v_deg = np.degrees(np.arctan2(dz, horizontal_distance)).astype(np.float32)
    delta_h = wrap_angle_deg(theta_h_deg - out["yaw_deg"].to_numpy(dtype=np.float32))

    out["dx"] = dx
    out["dy"] = dy
    out["dz"] = dz
    out["distance"] = distance
    out["log_distance"] = np.log10(distance)
    out["theta_h_deg"] = theta_h_deg
    out["theta_v_deg"] = theta_v_deg
    out["delta_theta_h_deg"] = delta_h
    out["delta_theta_h_sin"] = np.sin(np.radians(delta_h))
    out["delta_theta_h_cos"] = np.cos(np.radians(delta_h))
    # V2 仍保持四分支结构，只把绝对朝向作为天线分支的补充条件。
    out["yaw_sin"] = np.sin(np.radians(out["yaw_deg"].to_numpy(dtype=np.float32)))
    out["yaw_cos"] = np.cos(np.radians(out["yaw_deg"].to_numpy(dtype=np.float32)))
    return out


def wrap_angle_deg(angle: np.ndarray) -> np.ndarray:
    return ((angle + 180.0) % 360.0 - 180.0).astype(np.float32)


def prepare_matrix_pack(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_kind: str,
) -> MatrixPack:
    branch_columns = get_branch_columns(model_kind)
    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(train_df[["signal_norm"]]).astype("float32")
    y_val = target_scaler.transform(val_df[["signal_norm"]]).astype("float32")
    y_test = target_scaler.transform(test_df[["signal_norm"]]).astype("float32")

    feature_scalers: dict[str, StandardScaler] = {}
    train_parts: dict[str, np.ndarray] = {}
    val_parts: dict[str, np.ndarray] = {}
    test_parts: dict[str, np.ndarray] = {}
    for branch, columns in branch_columns.items():
        scaler = StandardScaler()
        train_parts[branch] = scaler.fit_transform(train_df[columns]).astype("float32")
        val_parts[branch] = scaler.transform(val_df[columns]).astype("float32")
        test_parts[branch] = scaler.transform(test_df[columns]).astype("float32")
        feature_scalers[branch] = scaler

    return MatrixPack(
        train_parts,
        val_parts,
        test_parts,
        y_train,
        y_val,
        y_test,
        target_scaler,
        feature_scalers,
        branch_columns,
    )


def transform_with_pack(df: pd.DataFrame, pack: MatrixPack) -> dict[str, np.ndarray]:
    return {
        branch: pack.feature_scalers[branch].transform(df[columns]).astype("float32")
        for branch, columns in pack.branch_columns.items()
    }


def inverse_target(pack: MatrixPack, y_scaled: np.ndarray) -> np.ndarray:
    return pack.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)


def get_branch_columns(model_kind: str) -> dict[str, list[str]]:
    if model_kind not in BRANCH_COLUMNS_BY_MODEL:
        raise ValueError(f"Unknown model kind: {model_kind}")
    return {branch: list(columns) for branch, columns in BRANCH_COLUMNS_BY_MODEL[model_kind].items()}


def feature_sanity_checks(df: pd.DataFrame) -> None:
    numeric_cols = [
        "distance",
        "log_distance",
        "yaw_sin",
        "yaw_cos",
        "theta_v_deg",
        "delta_theta_h_deg",
        "signal_norm",
        "signal_encoded",
        *BUILDING_FEATURES,
    ]
    if not np.isfinite(df[numeric_cols].to_numpy()).all():
        raise ValueError("Feature sanity check failed: found NaN or infinity.")
    if not df["delta_theta_h_deg"].between(-180.0, 180.0).all():
        raise ValueError("Angle sanity check failed: delta_theta_h_deg is outside [-180, 180].")
    if not df["signal_norm"].between(0.0, 1.0).all():
        raise ValueError("Label sanity check failed: signal_norm is outside [0, 1].")
