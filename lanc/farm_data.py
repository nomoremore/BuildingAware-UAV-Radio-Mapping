from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HEIGHT_LEVELS_M = np.array(
    [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
        105,
        110,
        115,
        120,
        125,
        130,
        135,
        140,
        145,
        150,
    ],
    dtype=np.float32,
)

PATTERN_IDS = {"iso": 0, "pattern_30": 1, "pattern_60": 2, "pattern_120": 3}
NPY_RE = re.compile(r"tx(?P<tx_id>\d+)_yaw(?P<yaw>\d+)\.npy$")


@dataclass(frozen=True)
class FarmPreprocessConfig:
    farm_root: Path
    scene_id: int = 11
    frequency: str = "freq35"
    antenna_pattern: str = "pattern_120"
    heights_m: tuple[int, ...] = (100, 110, 120, 130)
    stride: int = 16
    max_tx: int | None = None
    one_yaw_per_tx: bool = False
    local_window_radius: int = 2
    path_samples: int = 16


def build_farm_lanc_pairs(config: FarmPreprocessConfig) -> pd.DataFrame:
    tx_positions = _load_tx_positions(config.farm_root)
    radio_files = _select_radio_files(config)
    if not radio_files:
        raise FileNotFoundError(
            f"No npy files found under {config.farm_root} for "
            f"{config.frequency}/{config.antenna_pattern}."
        )

    first_volume = np.load(radio_files[0]["path"], mmap_mode="r")
    _validate_radio_volume(first_volume, radio_files[0]["path"])

    # FARM 的 value==0 表示建筑物占据区域；这里先从第一个 ARM 近似提取三维建筑先验。
    building_height_map = _building_height_map(first_volume)
    rx_rows = np.arange(0, 512, config.stride, dtype=np.int32)
    rx_cols = np.arange(0, 512, config.stride, dtype=np.int32)
    grid_cols, grid_rows = np.meshgrid(rx_cols, rx_rows)
    flat_rows = grid_rows.reshape(-1)
    flat_cols = grid_cols.reshape(-1)
    density = _local_building_density(
        building_height_map,
        flat_rows,
        flat_cols,
        config.local_window_radius,
    )
    local_heights = building_height_map[flat_rows, flat_cols].astype(np.float32)

    height_to_index = {int(h): idx for idx, h in enumerate(HEIGHT_LEVELS_M.tolist())}
    requested_heights = []
    for height_m in config.heights_m:
        if int(height_m) not in height_to_index:
            raise ValueError(f"Unsupported FARM height {height_m}; available heights are {HEIGHT_LEVELS_M}.")
        requested_heights.append((int(height_m), height_to_index[int(height_m)]))

    pieces: list[pd.DataFrame] = []
    for file_info in radio_files:
        tx_id = int(file_info["tx_id"])
        tx = tx_positions.loc[tx_id]
        volume = np.load(file_info["path"], mmap_mode="r")
        _validate_radio_volume(volume, file_info["path"])

        for height_m, height_index in requested_heights:
            labels = volume[0, height_index, flat_rows, flat_cols].astype(np.float32)
            is_building = labels == 0.0
            if bool(is_building.all()):
                continue

            valid_rows = flat_rows[~is_building]
            valid_cols = flat_cols[~is_building]
            valid_labels = labels[~is_building]
            valid_local_heights = local_heights[~is_building]
            valid_density = density[~is_building]
            path_ratio = _path_building_ratio(
                building_height_map,
                tx_row=float(tx["row"]),
                tx_col=float(tx["col"]),
                tx_height_m=float(tx["height"]),
                rx_rows=valid_rows.astype(np.float32),
                rx_cols=valid_cols.astype(np.float32),
                rx_height_m=float(height_m),
                samples=config.path_samples,
            )

            # 每一行样本对应“一个发射机配置 -> 一个三维接收点”的点级监督学习样本。
            part = pd.DataFrame(
                {
                    "scene_id": config.scene_id,
                    "config_id": [
                        f"scene{config.scene_id}_tx{tx_id}_yaw{int(file_info['yaw'])}"
                    ]
                    * len(valid_labels),
                    "tx_id": tx_id,
                    "yaw_deg": float(file_info["yaw"]),
                    "frequency_code": config.frequency,
                    "frequency_ghz": _parse_frequency_ghz(config.frequency),
                    "antenna_pattern": config.antenna_pattern,
                    "antenna_pattern_id": PATTERN_IDS.get(config.antenna_pattern, -1),
                    "tx_row": float(tx["row"]),
                    "tx_col": float(tx["col"]),
                    "tx_height_m": float(tx["height"]),
                    "rx_row": valid_rows.astype(np.float32),
                    "rx_col": valid_cols.astype(np.float32),
                    "rx_height_m": float(height_m),
                    "rx_is_building": 0.0,
                    "local_building_height": valid_local_heights,
                    "height_clearance": float(height_m) - valid_local_heights,
                    "building_density_nearby": valid_density,
                    "path_building_ratio_simple": path_ratio,
                    "signal_encoded": valid_labels,
                    "signal_norm": valid_labels / 255.0,
                }
            )
            part["rx_id"] = (
                "r"
                + part["rx_row"].astype(int).astype(str)
                + "_c"
                + part["rx_col"].astype(int).astype(str)
                + "_z"
                + part["rx_height_m"].astype(int).astype(str)
            )
            pieces.append(part)

    if not pieces:
        raise ValueError("FARM preprocessing produced no valid samples after filtering building voxels.")
    return pd.concat(pieces, ignore_index=True)


def split_by_tx_id(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    tx_ids = sorted(int(tx_id) for tx_id in df["tx_id"].unique())
    if len(tx_ids) < 3:
        raise ValueError("Need at least 3 tx_id values to build train/val/test splits.")

    if max(tx_ids) >= 99 and len(tx_ids) >= 100:
        train_ids = [tx for tx in tx_ids if tx <= 79]
        val_ids = [tx for tx in tx_ids if 80 <= tx <= 89]
        test_ids = [tx for tx in tx_ids if tx >= 90]
    else:
        n_train = max(1, int(round(len(tx_ids) * 0.8)))
        n_val = max(1, int(round(len(tx_ids) * 0.1)))
        if n_train + n_val >= len(tx_ids):
            n_train = max(1, len(tx_ids) - 2)
            n_val = 1
        train_ids = tx_ids[:n_train]
        val_ids = tx_ids[n_train : n_train + n_val]
        test_ids = tx_ids[n_train + n_val :]

    splits = {"train_tx": train_ids, "val_tx": val_ids, "test_tx": test_ids}
    train_df = df[df["tx_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["tx_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["tx_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df, splits


def _load_tx_positions(farm_root: Path) -> pd.DataFrame:
    csv_path = farm_root / "tx_positions_512.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing transmitter position file: {csv_path}")
    tx_positions = pd.read_csv(csv_path).set_index("tx_id")
    expected_cols = {"row", "col", "height"}
    if not expected_cols.issubset(tx_positions.columns):
        raise ValueError(f"{csv_path} must contain columns {sorted(expected_cols)}.")
    return tx_positions


def _select_radio_files(config: FarmPreprocessConfig) -> list[dict[str, object]]:
    pattern_dir = _scene_dir(config.farm_root, config.scene_id) / config.frequency / config.antenna_pattern
    if not pattern_dir.exists():
        raise FileNotFoundError(f"Missing FARM radio map directory: {pattern_dir}")

    rows = []
    for path in sorted(pattern_dir.glob("*.npy")):
        match = NPY_RE.match(path.name)
        if not match:
            continue
        tx_id = int(match.group("tx_id"))
        if config.max_tx is not None and tx_id >= config.max_tx:
            continue
        rows.append({"path": path, "tx_id": tx_id, "yaw": int(match.group("yaw"))})

    rows.sort(key=lambda item: (int(item["tx_id"]), int(item["yaw"])))
    if not config.one_yaw_per_tx:
        return rows

    selected: list[dict[str, object]] = []
    seen: set[int] = set()
    for item in rows:
        tx_id = int(item["tx_id"])
        if tx_id in seen:
            continue
        selected.append(item)
        seen.add(tx_id)
    return selected


def _scene_dir(farm_root: Path, scene_id: int) -> Path:
    nested = farm_root / str(scene_id)
    if nested.exists():
        return nested
    return farm_root


def _validate_radio_volume(volume: np.ndarray, path: Path) -> None:
    if volume.shape != (1, 30, 512, 512):
        raise ValueError(f"{path} has shape {volume.shape}; expected (1, 30, 512, 512).")


def _building_height_map(volume: np.ndarray) -> np.ndarray:
    building_mask = volume[0] == 0
    heights = HEIGHT_LEVELS_M.reshape(-1, 1, 1)
    return np.where(building_mask, heights, 0.0).max(axis=0).astype(np.float32)


def _local_building_density(
    building_height_map: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    radius: int,
) -> np.ndarray:
    footprint = (building_height_map > 0).astype(np.float32)
    padded = np.pad(footprint, radius, mode="constant")
    out = np.empty(len(rows), dtype=np.float32)
    side = 2 * radius + 1
    area = float(side * side)
    for idx, (row, col) in enumerate(zip(rows, cols, strict=True)):
        rr = int(row) + radius
        cc = int(col) + radius
        out[idx] = float(padded[rr - radius : rr + radius + 1, cc - radius : cc + radius + 1].sum() / area)
    return out


def _path_building_ratio(
    building_height_map: np.ndarray,
    tx_row: float,
    tx_col: float,
    tx_height_m: float,
    rx_rows: np.ndarray,
    rx_cols: np.ndarray,
    rx_height_m: float,
    samples: int,
) -> np.ndarray:
    if len(rx_rows) == 0:
        return np.empty(0, dtype=np.float32)

    blocked = np.zeros(len(rx_rows), dtype=np.float32)
    # 粗略 LOS 采样：沿 Tx-Rx 连线取若干点，统计建筑高度超过连线高度的比例。
    for t in np.linspace(0.0, 1.0, max(samples, 2), dtype=np.float32):
        rows = np.clip(np.rint(tx_row + (rx_rows - tx_row) * t).astype(np.int32), 0, 511)
        cols = np.clip(np.rint(tx_col + (rx_cols - tx_col) * t).astype(np.int32), 0, 511)
        line_height = tx_height_m + (rx_height_m - tx_height_m) * float(t)
        blocked += (building_height_map[rows, cols] >= line_height).astype(np.float32)
    return blocked / float(max(samples, 2))


def _parse_frequency_ghz(frequency: str) -> float:
    if not frequency.startswith("freq"):
        raise ValueError(f"Invalid FARM frequency code: {frequency}")
    return float(frequency.replace("freq", "")) / 10.0
