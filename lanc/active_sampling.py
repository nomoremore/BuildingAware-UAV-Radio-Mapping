from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


ENVIRONMENT_COLUMNS = [
    "building_density_nearby",
    "path_building_ratio_simple",
    "local_building_height",
    "height_clearance",
]


def count_from_ratio(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0:
        return 0
    return max(1, int(round(total * ratio)))


def build_rx_metadata(pair_df: pd.DataFrame) -> pd.DataFrame:
    agg_spec: dict[str, str] = {
        "rx_col": "first",
        "rx_row": "first",
        "rx_height_m": "first",
    }
    for column in ENVIRONMENT_COLUMNS:
        if column in pair_df.columns:
            agg_spec[column] = "mean"

    metadata = pair_df.groupby("rx_id", as_index=False).agg(agg_spec)
    return metadata.sort_values(["rx_height_m", "rx_row", "rx_col"]).reset_index(drop=True)


def attach_rx_metadata(candidate_pool: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    metadata = build_rx_metadata(pair_df)
    meta_cols = ["rx_id"] + [column for column in ENVIRONMENT_COLUMNS if column in metadata.columns]
    out = candidate_pool.merge(metadata[meta_cols], on="rx_id", how="left")
    for column in ENVIRONMENT_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = out[column].fillna(0.0)
    return out


def select_random_rx_ids(
    candidates: pd.DataFrame,
    already_selected: Iterable[str],
    budget: int,
    rng: np.random.Generator,
) -> list[str]:
    available = _available_candidates(candidates, already_selected)
    if budget <= 0 or available.empty:
        return []
    count = min(int(budget), len(available))
    chosen = rng.choice(available["rx_id"].to_numpy(dtype=str), size=count, replace=False)
    return [str(rx_id) for rx_id in chosen]


def select_proposed_rx_ids(
    candidates: pd.DataFrame,
    already_selected: Iterable[str],
    budget: int,
    rng: np.random.Generator,
) -> list[str]:
    selected = [str(rx_id) for rx_id in already_selected]
    newly_selected: list[str] = []
    if budget <= 0:
        return newly_selected

    for _ in range(int(budget)):
        available = _available_candidates(candidates, selected)
        if available.empty:
            break
        scored = _score_proposed_candidates(available, selected, candidates)
        max_score = scored["value_score"].max()
        tied = scored[np.isclose(scored["value_score"], max_score)]
        pick = str(tied.sample(1, random_state=int(rng.integers(0, 2**31 - 1)))["rx_id"].iloc[0])
        selected.append(pick)
        newly_selected.append(pick)
    return newly_selected


def mark_known_labels(candidate_pool: pd.DataFrame, selected_rx_ids: Iterable[str]) -> pd.DataFrame:
    out = candidate_pool.copy()
    selected = {str(rx_id) for rx_id in selected_rx_ids}
    out["label_known"] = out["rx_id"].astype(str).isin(selected)
    return out


def split_pairs_by_rx_id(pair_df: pd.DataFrame, selected_rx_ids: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = {str(rx_id) for rx_id in selected_rx_ids}
    mask = pair_df["rx_id"].astype(str).isin(selected)
    return pair_df[mask].reset_index(drop=True), pair_df[~mask].reset_index(drop=True)


def candidate_pool_with_scores(candidate_pool: pd.DataFrame, selected_rx_ids: Iterable[str]) -> pd.DataFrame:
    out = candidate_pool.copy()
    selected = {str(rx_id) for rx_id in selected_rx_ids}
    out["label_known"] = out["rx_id"].astype(str).isin(selected)
    scored = _score_proposed_candidates(out, selected, out)
    out["value_score"] = scored["value_score"].to_numpy()
    return out


def _available_candidates(candidates: pd.DataFrame, already_selected: Iterable[str]) -> pd.DataFrame:
    selected = {str(rx_id) for rx_id in already_selected}
    mask = ~candidates["rx_id"].astype(str).isin(selected)
    return candidates[mask].reset_index(drop=True)


def _score_proposed_candidates(
    candidates: pd.DataFrame,
    selected_rx_ids: Iterable[str],
    all_candidates: pd.DataFrame,
) -> pd.DataFrame:
    out = candidates.copy()
    weak_score = 1.0 - _normalize_01(out["signal_pred_norm"].to_numpy(dtype=float), invert=False)
    density = out.get("building_density_nearby", pd.Series(0.0, index=out.index)).to_numpy(dtype=float)
    path_ratio = out.get("path_building_ratio_simple", pd.Series(0.0, index=out.index)).to_numpy(dtype=float)
    building_score = 0.5 * _normalize_01(density) + 0.5 * _normalize_01(path_ratio)
    diversity_score = _diversity_score(out, selected_rx_ids, all_candidates)

    # proposed 策略保持简单：弱覆盖点优先，同时偏向建筑复杂、与已测点分散的位置。
    out["value_score"] = 0.45 * weak_score + 0.35 * building_score + 0.20 * diversity_score
    return out


def _normalize_01(values: np.ndarray, invert: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    finite = np.where(np.isfinite(values), values, np.nan)
    min_v = np.nanmin(finite)
    max_v = np.nanmax(finite)
    if not np.isfinite(min_v) or not np.isfinite(max_v) or max_v <= min_v:
        normalized = np.zeros_like(values, dtype=float)
    else:
        normalized = (values - min_v) / (max_v - min_v)
    normalized = np.clip(np.nan_to_num(normalized, nan=0.0), 0.0, 1.0)
    if invert:
        return 1.0 - normalized
    return normalized


def _diversity_score(
    candidates: pd.DataFrame,
    selected_rx_ids: Iterable[str],
    all_candidates: pd.DataFrame,
) -> np.ndarray:
    if candidates.empty:
        return np.array([], dtype=float)
    coord_columns = ["rx_col", "rx_row", "rx_height_m"]
    selected = {str(rx_id) for rx_id in selected_rx_ids}
    selected_df = all_candidates[all_candidates["rx_id"].astype(str).isin(selected)]
    if selected_df.empty:
        return np.full(len(candidates), 0.5, dtype=float)

    all_coords = all_candidates[coord_columns].to_numpy(dtype=float)
    scale = np.maximum(all_coords.max(axis=0) - all_coords.min(axis=0), 1.0)
    cand_coords = candidates[coord_columns].to_numpy(dtype=float) / scale
    selected_coords = selected_df[coord_columns].to_numpy(dtype=float) / scale
    diff = cand_coords[:, None, :] - selected_coords[None, :, :]
    min_dist = np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)
    return _normalize_01(min_dist)
