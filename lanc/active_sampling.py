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
PROPOSED_V3_RHOS = {
    "proposed_v3": 0.50,
    "proposed_v3_rho04": 0.40,
    "proposed_v3_rho05": 0.50,
    "proposed_v3_rho06": 0.60,
}
# proposed_v3 的建筑利用配额评分固定为论文中预定义的权重，避免在 scene 13 上继续调参。
PROPOSED_V3_EXPLOITATION_WEIGHTS = {
    "building": 0.50,
    "diversity": 0.20,
    "prediction_spread": 0.15,
    "gradient": 0.15,
}
STRATEGY_CHOICES = (
    "random",
    "uniform_grid",
    "weak_only",
    "building_only",
    "proposed",
    *PROPOSED_V2_WEIGHTS.keys(),
    *PROPOSED_V3_RHOS.keys(),
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
    if strategy in PROPOSED_V3_RHOS:
        return select_proposed_v3_rx_ids(candidates, already_selected, budget, rng, strategy)
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


def proposed_v3_coverage_budget(budget: int, variant: str = "proposed_v3") -> int:
    if variant not in PROPOSED_V3_RHOS:
        raise ValueError(f"Unknown proposed_v3 variant: {variant}")
    if budget <= 0:
        return 0
    rho = PROPOSED_V3_RHOS[variant]
    return max(1, min(int(budget), int(np.ceil(float(budget) * rho))))


def select_proposed_v3_rx_ids(
    candidates: pd.DataFrame,
    already_selected: Iterable[str],
    budget: int,
    rng: np.random.Generator,
    variant: str = "proposed_v3",
) -> list[str]:
    """建筑感知、空间均衡的主动采样。

    前半部分按高度和水平空间分层补齐覆盖，后半部分再利用建筑复杂度、
    多配置预测分歧和覆盖梯度选择信息量更高的点。这样既保留建筑场景先验，
    也避免 proposed_v2_j 在低采样预算下把点集中到少数高建筑区域。
    """
    if variant not in PROPOSED_V3_RHOS:
        raise ValueError(f"Unknown proposed_v3 variant: {variant}")

    selected = [str(rx_id) for rx_id in already_selected]
    newly_selected: list[str] = []
    if budget <= 0:
        return newly_selected

    coverage_budget = proposed_v3_coverage_budget(budget, variant)
    for _ in range(coverage_budget):
        available = _available_candidates(candidates, selected)
        if available.empty:
            break
        scored = _score_proposed_v3_coverage_candidates(available, selected, candidates)
        pick = _pick_one_top_scored(scored, rng)
        selected.append(pick)
        newly_selected.append(pick)

    for _ in range(max(0, int(budget) - coverage_budget)):
        available = _available_candidates(candidates, selected)
        if available.empty:
            break
        scored = _score_proposed_v3_exploitation_candidates(available, selected, candidates)
        pick = _pick_one_top_scored(scored, rng)
        selected.append(pick)
        newly_selected.append(pick)
    return newly_selected


def build_proposed_v3_diagnostics(
    candidates: pd.DataFrame,
    selected_rx_ids: Iterable[str],
    variant: str = "proposed_v3",
    round_budget: int | None = None,
) -> dict[str, float | int]:
    """汇总 proposed_v3 单轮选点诊断，便于解释覆盖配额是否按预期执行。"""
    if variant not in PROPOSED_V3_RHOS:
        raise ValueError(f"Unknown proposed_v3 variant: {variant}")

    selected_order = [str(rx_id) for rx_id in selected_rx_ids]
    selected = set(selected_order)
    selected_df = candidates[candidates["rx_id"].astype(str).isin(selected)].copy()
    actual_selected_count = int(len(selected_df))
    total_budget = actual_selected_count if round_budget is None else max(0, int(round_budget))
    coverage_quota = proposed_v3_coverage_budget(total_budget, variant)
    exploitation_quota = max(0, total_budget - coverage_quota)
    diagnostics: dict[str, float | int] = {
        "total_budget": total_budget,
        "actual_selected_count": actual_selected_count,
        "coverage_quota": coverage_quota,
        "exploitation_quota": exploitation_quota,
        "unique_height_count": 0,
        "unique_spatial_strata": 0,
        "mean_building_density_nearby": 0.0,
        "mean_path_building_ratio_simple": 0.0,
        "mean_prediction_spread_norm": 0.0,
        "mean_signal_pred_norm": 0.0,
        "mean_value_score": 0.0,
        "max_value_score": 0.0,
    }
    if selected_df.empty:
        return diagnostics

    binned = _spatial_strata(selected_df, bounds=_spatial_bounds(candidates))
    diagnostics["unique_height_count"] = int(selected_df["rx_height_m"].nunique())
    diagnostics["unique_spatial_strata"] = int(
        binned[["rx_height_m", "spatial_row_bin", "spatial_col_bin"]].drop_duplicates().shape[0]
    )
    diagnostics["mean_building_density_nearby"] = float(
        selected_df.get("building_density_nearby", pd.Series(0.0, index=selected_df.index)).mean()
    )
    diagnostics["mean_path_building_ratio_simple"] = float(
        selected_df.get("path_building_ratio_simple", pd.Series(0.0, index=selected_df.index)).mean()
    )
    diagnostics["mean_prediction_spread_norm"] = float(_prediction_spread(selected_df).mean())
    diagnostics["mean_signal_pred_norm"] = float(selected_df["signal_pred_norm"].mean())
    value_score = selected_df.get("value_score", pd.Series(0.0, index=selected_df.index)).to_numpy(dtype=float)
    diagnostics["mean_value_score"] = float(np.nan_to_num(value_score, nan=0.0).mean())
    diagnostics["max_value_score"] = float(np.nan_to_num(value_score, nan=0.0).max())

    height_counts = selected_df.groupby("rx_height_m").size().to_dict()
    for height, count in height_counts.items():
        diagnostics[f"height_{int(round(float(height)))}_count"] = int(count)
    return diagnostics


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
    elif strategy in PROPOSED_V3_RHOS:
        # 候选池需要给全部高度层写评分；真正选点时再限制到当前最少覆盖的高度层。
        out["value_score"] = _score_proposed_v3_exploitation_candidates(
            out,
            selected,
            out,
            restrict_to_least_height=False,
        )["value_score"].to_numpy()
    else:
        out["value_score"] = _score_proposed_candidates(out, selected, out)["value_score"].to_numpy()
    return out


def compute_gradient_score(candidates: pd.DataFrame) -> np.ndarray:
    if candidates.empty:
        return np.array([], dtype=float)
    gradient = np.zeros(len(candidates), dtype=float)
    # 过滤器可能保留原始 DataFrame 的稀疏索引，这里显式重建 0..n-1
    # 位置索引，避免后续把原索引写回长度较小的 gradient 数组时报越界。
    indexed = candidates.reset_index(drop=True)
    indexed["_source_index"] = np.arange(len(indexed), dtype=int)

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


def _score_proposed_v3_coverage_candidates(
    candidates: pd.DataFrame,
    selected_rx_ids: Iterable[str],
    all_candidates: pd.DataFrame,
) -> pd.DataFrame:
    out = candidates.copy()
    preferred_height = _least_represented_height(out, selected_rx_ids, all_candidates)
    out = out[out["rx_height_m"] == preferred_height].copy()
    if out.empty:
        return out

    binned = _spatial_strata(all_candidates)
    selected = {str(rx_id) for rx_id in selected_rx_ids}
    selected_bins = binned[binned["rx_id"].astype(str).isin(selected)]
    selected_counts = selected_bins.groupby(["rx_height_m", "spatial_row_bin", "spatial_col_bin"]).size()

    # 分区边界必须由完整候选池决定；否则先按高度过滤后，局部子集会把不同分区重新拉伸。
    out = _spatial_strata(out, bounds=_spatial_bounds(all_candidates))
    out["stratum_count"] = [
        int(selected_counts.get((float(row.rx_height_m), int(row.spatial_row_bin), int(row.spatial_col_bin)), 0))
        for row in out.itertuples()
    ]
    min_count = int(out["stratum_count"].min())
    eligible = out[out["stratum_count"] == min_count].copy()
    eligible["value_score"] = _diversity_score(eligible, selected_rx_ids, all_candidates)
    return eligible


def _score_proposed_v3_exploitation_candidates(
    candidates: pd.DataFrame,
    selected_rx_ids: Iterable[str],
    all_candidates: pd.DataFrame,
    restrict_to_least_height: bool = True,
) -> pd.DataFrame:
    if restrict_to_least_height:
        preferred_height = _least_represented_height(candidates, selected_rx_ids, all_candidates)
        out = candidates[candidates["rx_height_m"] == preferred_height].copy()
    else:
        out = candidates.copy()
    if out.empty:
        return out

    building_score = _building_score(out)
    diversity_score = _diversity_score(out, selected_rx_ids, all_candidates)
    spread_score = _normalize_01(_prediction_spread(out))
    gradient_score = compute_gradient_score(out)
    # 建筑先验仍是核心，但加入空间、多配置分歧和覆盖梯度后，低预算下不会只扎堆于建筑高密度点。
    out["value_score"] = (
        PROPOSED_V3_EXPLOITATION_WEIGHTS["building"] * building_score
        + PROPOSED_V3_EXPLOITATION_WEIGHTS["diversity"] * diversity_score
        + PROPOSED_V3_EXPLOITATION_WEIGHTS["prediction_spread"] * spread_score
        + PROPOSED_V3_EXPLOITATION_WEIGHTS["gradient"] * gradient_score
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


def _prediction_spread(candidates: pd.DataFrame) -> np.ndarray:
    if "prediction_spread_norm" in candidates.columns:
        values = candidates["prediction_spread_norm"].to_numpy(dtype=float)
    elif "uncertainty" in candidates.columns:
        values = candidates["uncertainty"].to_numpy(dtype=float)
    else:
        values = np.zeros(len(candidates), dtype=float)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _spatial_bounds(candidates: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """返回候选池在水平坐标上的全局边界，保证所有轮次使用同一套 8x8 分区。"""
    bounds: dict[str, tuple[float, float]] = {}
    for column in ("rx_row", "rx_col"):
        values = candidates[column].to_numpy(dtype=float)
        min_v = float(np.nanmin(values)) if len(values) else 0.0
        max_v = float(np.nanmax(values)) if len(values) else 0.0
        if not np.isfinite(min_v) or not np.isfinite(max_v) or max_v <= min_v:
            bounds[column] = (0.0, 0.0)
        else:
            bounds[column] = (min_v, max_v)
    return bounds


def _spatial_strata(
    candidates: pd.DataFrame,
    bins: int = 8,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    out = candidates.copy()
    for column in ("rx_row", "rx_col"):
        values = out[column].to_numpy(dtype=float)
        if bounds is not None and column in bounds:
            min_v, max_v = bounds[column]
        else:
            min_v = float(np.nanmin(values)) if len(values) else 0.0
            max_v = float(np.nanmax(values)) if len(values) else 0.0
        if not np.isfinite(min_v) or not np.isfinite(max_v) or max_v <= min_v:
            out[f"spatial_{column[3:]}_bin"] = 0
        else:
            bin_values = np.floor((values - min_v) / (max_v - min_v) * bins).astype(int)
            out[f"spatial_{column[3:]}_bin"] = np.clip(bin_values, 0, bins - 1)
    return out


def _least_represented_height(
    candidates: pd.DataFrame,
    selected_rx_ids: Iterable[str],
    all_candidates: pd.DataFrame,
) -> float:
    available_heights = sorted(float(value) for value in candidates["rx_height_m"].dropna().unique())
    if not available_heights:
        raise ValueError("proposed_v3 requires at least one candidate height.")

    selected = {str(rx_id) for rx_id in selected_rx_ids}
    selected_df = all_candidates[all_candidates["rx_id"].astype(str).isin(selected)]
    selected_counts = selected_df.groupby("rx_height_m").size().to_dict()

    min_count = min(int(selected_counts.get(height, 0)) for height in available_heights)
    preferred = [height for height in available_heights if int(selected_counts.get(height, 0)) == min_count]
    return float(preferred[0])


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
