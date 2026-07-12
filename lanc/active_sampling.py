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
PROPOSED_V2_WEIGHTS = {
    "proposed_v2_a": {"weak": 0.30, "building": 0.20, "diversity": 0.35, "gradient": 0.15},
    "proposed_v2_b": {"weak": 0.25, "building": 0.15, "diversity": 0.45, "gradient": 0.15},
    "proposed_v2_c": {"weak": 0.20, "building": 0.20, "diversity": 0.40, "gradient": 0.20},
    "proposed_v2_d": {"weak": 0.05, "building": 0.75, "diversity": 0.15, "gradient": 0.05},
    "proposed_v2_e": {"weak": 0.00, "building": 0.85, "diversity": 0.10, "gradient": 0.05},
    "proposed_v2_f": {"weak": 0.00, "building": 0.70, "diversity": 0.25, "gradient": 0.05},
    "proposed_v2_g": {"weak": 0.00, "building": 0.95, "diversity": 0.05, "gradient": 0.00},
    "proposed_v2_h": {"weak": 0.00, "building": 0.90, "diversity": 0.05, "gradient": 0.05},
    "proposed_v2_i": {"weak": 0.00, "building": 0.98, "diversity": 0.02, "gradient": 0.00},
    "proposed_v2_j": {"weak": 0.00, "building": 0.70, "diversity": 0.25, "gradient": 0.05},
}
STRATEGY_CHOICES = (
    "random",
    "uniform_grid",
    "weak_only",
    "building_only",
    "proposed",
    *PROPOSED_V2_WEIGHTS.keys(),
)


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


def select_strategy_rx_ids(
    candidates: pd.DataFrame,
    already_selected: Iterable[str],
    budget: int,
    rng: np.random.Generator,
    strategy: str,
) -> list[str]:
    if strategy == "random":
        return select_random_rx_ids(candidates, already_selected, budget, rng)
    if strategy == "uniform_grid":
        return select_uniform_grid_rx_ids(candidates, already_selected, budget, rng)
    if strategy == "weak_only":
        return select_weak_only_rx_ids(candidates, already_selected, budget)
    if strategy == "building_only":
        return select_building_only_rx_ids(candidates, already_selected, budget)
    if strategy == "proposed":
        return select_proposed_rx_ids(candidates, already_selected, budget, rng)
    if strategy in PROPOSED_V2_WEIGHTS:
        return select_proposed_v2_rx_ids(candidates, already_selected, budget, rng, strategy)
    raise ValueError(f"Unknown active sampling strategy: {strategy}")


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


def select_weak_only_rx_ids(
    candidates: pd.DataFrame,
    already_selected: Iterable[str],
    budget: int,
) -> list[str]:
    available = _available_candidates(candidates, already_selected)
    if budget <= 0 or available.empty:
        return []
    scored = available.copy()
    scored["value_score"] = 1.0 - _normalize_01(scored["signal_pred_norm"].to_numpy(dtype=float))
    return _top_scored_rx_ids(scored, budget)


def select_building_only_rx_ids(
    candidates: pd.DataFrame,
    already_selected: Iterable[str],
    budget: int,
) -> list[str]:
    available = _available_candidates(candidates, already_selected)
    if budget <= 0 or available.empty:
        return []
    scored = available.copy()
    scored["value_score"] = _building_density_score(scored)
    return _top_scored_rx_ids(scored, budget)


def select_uniform_grid_rx_ids(
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
        scored = available.copy()
        if selected:
            scored["value_score"] = _diversity_score(available, selected, candidates)
        else:
            coords = _normalized_coords(available)
            center = np.full(coords.shape[1], 0.5, dtype=float)
            scored["value_score"] = _normalize_01(np.sqrt(np.sum((coords - center) ** 2, axis=1)))
        pick = _pick_one_top_scored(scored, rng)
        selected.append(pick)
        newly_selected.append(pick)
    return newly_selected


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
        pick = _pick_one_top_scored(scored, rng)
        selected.append(pick)
        newly_selected.append(pick)
    return newly_selected


def select_proposed_v2_rx_ids(
    candidates: pd.DataFrame,
    already_selected: Iterable[str],
    budget: int,
    rng: np.random.Generator,
    variant: str,
) -> list[str]:
    if variant not in PROPOSED_V2_WEIGHTS:
        raise ValueError(f"Unknown proposed_v2 variant: {variant}")
    if variant == "proposed_v2_j":
        return _select_top_building_diverse_rx_ids(candidates, already_selected, budget, rng)
    selected = [str(rx_id) for rx_id in already_selected]
    newly_selected: list[str] = []
    if budget <= 0:
        return newly_selected

    for _ in range(int(budget)):
        available = _available_candidates(candidates, selected)
        if available.empty:
            break
        scored = _score_proposed_v2_candidates(available, selected, candidates, variant)
        pick = _pick_one_top_scored(scored, rng)
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


def candidate_pool_with_scores(
    candidate_pool: pd.DataFrame,
    selected_rx_ids: Iterable[str],
    strategy: str = "proposed",
) -> pd.DataFrame:
    out = candidate_pool.copy()
    selected = {str(rx_id) for rx_id in selected_rx_ids}
    out["label_known"] = out["rx_id"].astype(str).isin(selected)
    if strategy == "weak_only":
        out["value_score"] = 1.0 - _normalize_01(out["signal_pred_norm"].to_numpy(dtype=float))
    elif strategy == "building_only":
        out["value_score"] = _building_density_score(out)
    elif strategy == "uniform_grid":
        out["value_score"] = _diversity_score(out, selected, out)
    elif strategy in PROPOSED_V2_WEIGHTS:
        if strategy == "proposed_v2_j":
            out["value_score"] = _score_top_building_diverse_candidates(out, selected, out)["value_score"].to_numpy()
        else:
            out["value_score"] = _score_proposed_v2_candidates(out, selected, out, strategy)["value_score"].to_numpy()
    else:
        out["value_score"] = _score_proposed_candidates(out, selected, out)["value_score"].to_numpy()
    return out


def compute_gradient_score(candidates: pd.DataFrame) -> np.ndarray:
    if candidates.empty:
        return np.array([], dtype=float)
    gradient = np.zeros(len(candidates), dtype=float)
    indexed = candidates.reset_index(drop=False).rename(columns={"index": "_source_index"})

    for _, layer in indexed.groupby("rx_height_m"):
        pivot = layer.pivot(index="rx_row", columns="rx_col", values="signal_pred_norm").sort_index().sort_index(axis=1)
        values = pivot.to_numpy(dtype=float)
        if values.size == 0:
            continue
        fill_value = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
        values = np.nan_to_num(values, nan=fill_value)
        if values.shape[0] == 1 and values.shape[1] == 1:
            magnitude = np.zeros_like(values, dtype=float)
        elif values.shape[0] == 1:
            magnitude = np.abs(np.gradient(values, axis=1))
        elif values.shape[1] == 1:
            magnitude = np.abs(np.gradient(values, axis=0))
        else:
            grad_y, grad_x = np.gradient(values)
            magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)

        mag_df = pd.DataFrame(magnitude, index=pivot.index, columns=pivot.columns)
        source_indices = layer.set_index(["rx_row", "rx_col"])["_source_index"]
        for (row, col), source_index in source_indices.items():
            gradient[int(source_index)] = float(mag_df.loc[row, col])

    return _normalize_01(gradient)


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
    weak_score = 1.0 - _normalize_01(out["signal_pred_norm"].to_numpy(dtype=float))
    building_score = _building_score(out)
    diversity_score = _diversity_score(out, selected_rx_ids, all_candidates)
    # 旧版 proposed：弱覆盖 + 建筑复杂度 + 空间分散，用于和 proposed_v2 做消融对比。
    out["value_score"] = 0.45 * weak_score + 0.35 * building_score + 0.20 * diversity_score
    return out


def _score_proposed_v2_candidates(
    candidates: pd.DataFrame,
    selected_rx_ids: Iterable[str],
    all_candidates: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    out = candidates.copy()
    weights = PROPOSED_V2_WEIGHTS[variant]
    weak_score = 1.0 - _normalize_01(out["signal_pred_norm"].to_numpy(dtype=float))
    building_score = _building_score(out)
    diversity_score = _diversity_score(out, selected_rx_ids, all_candidates)
    gradient_score = compute_gradient_score(out)
    # 新版策略加入覆盖突变项，并提高空间分散权重，避免采样点扎堆在弱覆盖建筑区。
    out["value_score"] = (
        weights["weak"] * weak_score
        + weights["building"] * building_score
        + weights["diversity"] * diversity_score
        + weights["gradient"] * gradient_score
    )
    return out


def _select_top_building_diverse_rx_ids(
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
        scored = _score_top_building_diverse_candidates(available, selected, candidates)
        pick = _pick_one_top_scored(scored, rng)
        selected.append(pick)
        newly_selected.append(pick)
    return newly_selected


def _score_top_building_diverse_candidates(
    candidates: pd.DataFrame,
    selected_rx_ids: Iterable[str],
    all_candidates: pd.DataFrame,
) -> pd.DataFrame:
    out = candidates.copy()
    building_score = _building_score(out)
    if len(building_score) >= 5:
        threshold = float(np.quantile(building_score, 0.80))
        top_mask = building_score >= threshold
    else:
        top_mask = np.ones(len(out), dtype=bool)
    diversity_score = _diversity_score(out, selected_rx_ids, all_candidates)
    gradient_score = compute_gradient_score(out)

    # proposed_v2_j 先聚焦建筑复杂候选，再用空间分散和覆盖突变做排序。
    raw_score = 0.70 * building_score + 0.25 * diversity_score + 0.05 * gradient_score
    out["value_score"] = np.where(top_mask, raw_score, raw_score - 1.0)
    return out


def _building_score(candidates: pd.DataFrame) -> np.ndarray:
    density = candidates.get("building_density_nearby", pd.Series(0.0, index=candidates.index)).to_numpy(dtype=float)
    path_ratio = candidates.get("path_building_ratio_simple", pd.Series(0.0, index=candidates.index)).to_numpy(dtype=float)
    return 0.5 * _normalize_01(density) + 0.5 * _normalize_01(path_ratio)


def _building_density_score(candidates: pd.DataFrame) -> np.ndarray:
    density = candidates.get("building_density_nearby", pd.Series(0.0, index=candidates.index)).to_numpy(dtype=float)
    return _normalize_01(density)


def _top_scored_rx_ids(scored: pd.DataFrame, budget: int) -> list[str]:
    if scored.empty or budget <= 0:
        return []
    ordered = scored.sort_values(["value_score", "rx_id"], ascending=[False, True])
    return [str(rx_id) for rx_id in ordered["rx_id"].head(int(budget)).tolist()]


def _pick_one_top_scored(scored: pd.DataFrame, rng: np.random.Generator) -> str:
    max_score = scored["value_score"].max()
    tied = scored[np.isclose(scored["value_score"], max_score)]
    return str(tied.sample(1, random_state=int(rng.integers(0, 2**31 - 1)))["rx_id"].iloc[0])


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
    selected = {str(rx_id) for rx_id in selected_rx_ids}
    selected_df = all_candidates[all_candidates["rx_id"].astype(str).isin(selected)]
    if selected_df.empty:
        return np.full(len(candidates), 0.5, dtype=float)

    scale = _coord_scale(all_candidates)
    cand_coords = candidates[["rx_col", "rx_row", "rx_height_m"]].to_numpy(dtype=float) / scale
    selected_coords = selected_df[["rx_col", "rx_row", "rx_height_m"]].to_numpy(dtype=float) / scale
    diff = cand_coords[:, None, :] - selected_coords[None, :, :]
    min_dist = np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)
    return _normalize_01(min_dist)


def _normalized_coords(candidates: pd.DataFrame) -> np.ndarray:
    coords = candidates[["rx_col", "rx_row", "rx_height_m"]].to_numpy(dtype=float)
    mins = coords.min(axis=0)
    scale = np.maximum(coords.max(axis=0) - mins, 1.0)
    return (coords - mins) / scale


def _coord_scale(candidates: pd.DataFrame) -> np.ndarray:
    all_coords = candidates[["rx_col", "rx_row", "rx_height_m"]].to_numpy(dtype=float)
    return np.maximum(all_coords.max(axis=0) - all_coords.min(axis=0), 1.0)
