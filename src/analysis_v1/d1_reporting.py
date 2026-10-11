"""Report fitted inner tuning choices and source availability; never select models."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .d1_features import PROGRAM_FEATURES, WEATHER_FEATURES
from .reporting import save, style

SCOPE = "inner_event_balanced_mae_training_only_not_outer_test"
PROFILE = "minimum_inner_mae_over_other_prespecified_grid_parameters"
FEATURE_LABELS = {
    "planned_drone": "預定無人機",
    "planned_fireworks_minutes": "預定煙火分鐘",
    "planned_fireworks_start_hour": "預定煙火開始",
    "planned_program_minutes": "預定整體節目分鐘",
    "forecast_temperature_c": "預報溫度",
    "forecast_precip_probability": "預報降雨機率",
    "forecast_wind_speed_m_s": "預報風速",
    "forecast_cloud_cover_pct": "預報雲量",
}


def _parameters(value):
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("Tuning parameters must be a JSON object")
    if not {"max_depth", "n_estimators", "min_child_weight", "reg_lambda"} <= set(parsed):
        raise ValueError("XGBoost tuning parameters are incomplete")
    return parsed


def tuning_tables(choices, folds):
    """Match each fitted outer model to its exact historical parent train_end."""
    required = {"model", "params", "train_end", "inner_event_mae"}
    if not required <= set(choices.columns):
        raise ValueError("Tuning choices are missing required columns")
    candidates = choices.loc[choices.model.eq("xgboost")].copy()
    candidates["train_end"] = pd.to_datetime(candidates.train_end, errors="raise")
    candidates["inner_event_mae"] = pd.to_numeric(candidates.inner_event_mae, errors="raise")
    if not np.isfinite(candidates.inner_event_mae).all() or candidates.inner_event_mae.lt(0).any():
        raise ValueError("Tuning MAE must be finite and nonnegative")
    candidates["parsed_params"] = candidates.params.map(_parameters)
    surfaces, selected = [], []
    for fold in folds:
        year, end = int(fold["outer_year"]), pd.Timestamp(fold["train_end"])
        if end.year >= year:
            raise ValueError("Tuning train_end must precede the outer test year")
        part = candidates.loc[candidates.train_end.eq(end)].copy()
        if part.empty:
            raise ValueError(f"Missing inner tuning candidates for train_end={end.date()}")
        part["param_key"] = part.parsed_params.map(lambda x: json.dumps(x, sort_keys=True))
        if part.param_key.duplicated().any():
            raise ValueError("Duplicate XGBoost candidate in the same parent training set")
        for name in ("max_depth", "n_estimators", "min_child_weight", "reg_lambda"):
            part[name] = part.parsed_params.map(lambda params: params[name])
        chosen = _parameters(fold["base_models"]["xgboost"]["params"])
        matching = part.loc[part.param_key.eq(json.dumps(chosen, sort_keys=True))]
        if len(matching) != 1:
            raise ValueError("Fitted XGBoost parameters are absent from the corresponding inner search")
        winner = matching.iloc[0]
        if not np.isclose(winner.inner_event_mae, part.inner_event_mae.min(), rtol=0, atol=1e-8):
            raise ValueError("Fitted XGBoost parameters do not minimize the logged inner MAE")
        selected.append({"outer_year": year, "train_end": end.strftime("%Y-%m-%d"),
                         **chosen, "inner_event_mae": float(winner.inner_event_mae),
                         "candidate_count": len(part), "selection_scope": SCOPE})
        for (depth, trees), cell in part.groupby(["max_depth", "n_estimators"], sort=True):
            best = cell.loc[cell.inner_event_mae.idxmin()]
            surfaces.append({"outer_year": year, "train_end": end.strftime("%Y-%m-%d"),
                             "max_depth": int(depth), "n_estimators": int(trees),
                             "inner_event_mae": float(best.inner_event_mae),
                             "best_min_child_weight": best.min_child_weight,
                             "best_reg_lambda": best.reg_lambda,
                             "profiled_candidate_count": len(cell),
                             "selected_cell": depth == chosen["max_depth"] and trees == chosen["n_estimators"],
                             "profile_definition": PROFILE, "selection_scope": SCOPE})
    if not surfaces:
        raise ValueError("No outer folds to report")
    return pd.DataFrame(surfaces), pd.DataFrame(selected)


def coverage_table(audit):
    """Coverage counts evidence slots, not stations or the numerical feature value."""
    required = {"event_date", "hour", "feature", "eligible"}
    if audit.empty or not required <= set(audit.columns):
        raise ValueError("D1 coverage requires a nonempty complete audit")
    frame = audit.copy()
    frame["event_date"] = pd.to_datetime(frame.event_date, errors="raise")
    if frame.event_date.isna().any() or not frame.event_date.eq(frame.event_date.dt.normalize()).all():
        raise ValueError("D1 coverage requires event calendar dates")
    frame["eligible"] = frame.eligible.astype(str).str.lower().map({"true": True, "false": False,
                                                                   "1": True, "0": False})
    if frame.eligible.isna().any():
        raise ValueError("D1 eligibility must be an explicit boolean")
    if frame.duplicated(["event_date", "hour", "feature"]).any():
        raise ValueError("D1 coverage has duplicate evidence slots")
    if not set(frame.feature) <= set(FEATURE_LABELS):
        raise ValueError("D1 coverage contains unapproved features")
    rows = []
    for date in sorted(frame.event_date.unique()):
        for feature in frame.feature.drop_duplicates():
            part = frame.loc[frame.event_date.eq(date) & frame.feature.eq(feature)]
            expected = 6 if feature in WEATHER_FEATURES else 1
            if len(part) != expected:
                raise ValueError("D1 coverage requires all 6 weather hours or one program slot per event")
            if feature in WEATHER_FEATURES and set(part.hour) != set(range(18, 24)):
                raise ValueError("D1 weather coverage requires hours 18..23 exactly")
            if feature in PROGRAM_FEATURES and part.hour.notna().any():
                raise ValueError("D1 program coverage requires event-level slots")
            known = int(part.eligible.sum())
            rows.append({"event_date": pd.Timestamp(date).strftime("%Y-%m-%d"), "feature": feature,
                         "known_slots": known, "expected_slots": expected,
                         "available_fraction": known / expected,
                         "meaning": "verified_before_D1_cutoff_fraction_not_feature_value"})
    return pd.DataFrame(rows)


def build_tuning_figures(choices, folds, output, d1_audit=None):
    output = Path(output)
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    surface, selected = tuning_tables(choices, folds)
    surface.to_csv(tables / "xgb_tuning_surface.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(tables / "xgb_selected_params.csv", index=False, encoding="utf-8-sig")
    style()
    years = sorted(surface.outer_year.unique())
    fig, axes = plt.subplots(1, len(years), figsize=(5.3 * len(years), 6), squeeze=False,
                             constrained_layout=True)
    low, high = surface.inner_event_mae.min(), surface.inner_event_mae.max()
    for ax, year in zip(axes.flat, years):
        part = surface.loc[surface.outer_year.eq(year)]
        matrix = part.pivot(index="max_depth", columns="n_estimators", values="inner_event_mae")
        row = selected.loc[selected.outer_year.eq(year)].iloc[0]
        im = ax.imshow(matrix, cmap="YlOrBr", aspect="auto", vmin=low, vmax=high)
        for i, depth in enumerate(matrix.index):
            for j, trees in enumerate(matrix.columns):
                value = matrix.loc[depth, trees]
                ax.text(j, i, f"{value:.1f}" if pd.notna(value) else "未搜尋",
                        ha="center", va="center", color="white" if value > low + .65 * (high-low) else "black")
                if depth == row.max_depth and trees == row.n_estimators:
                    ax.add_patch(Rectangle((j-.46, i-.46), .92, .92, fill=False, edgecolor="#AD2626", lw=2.5))
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set(title=f"{year} 年模型｜訓練截至 {row.train_end}", ylabel="樹深 max_depth",
               xlabel=f"樹數 n_estimators\n紅框：選中深度 {int(row.max_depth)}、{int(row.n_estimators)} 棵\n"
                       f"min_child_weight={row.min_child_weight:g}，λ={row.reg_lambda:g}；MAE={row.inner_event_mae:.1f}")
    fig.colorbar(im, ax=list(axes.flat), shrink=.8, label="內層活動等權 MAE（人次／站／小時）")
    fig.suptitle("XGBoost 如何調參：只比較較早年度的內層驗證\n每格對葉節點限制及正則化候選取最低 MAE；不是外層測試誤差")
    save(fig, figures / "xgb_tuning_heatmap.png")
    if d1_audit is not None:
        coverage = coverage_table(d1_audit)
        coverage.to_csv(tables / "d1_feature_coverage.csv", index=False, encoding="utf-8-sig")
        features = d1_audit.feature.drop_duplicates().tolist()
        matrix = coverage.pivot(index="event_date", columns="feature", values="available_fraction")[features]
        slots = coverage.set_index(["event_date", "feature"])
        fig, ax = plt.subplots(figsize=(max(8, len(features) * 1.35), max(5, len(matrix) * .34 + 2.4)),
                               constrained_layout=True)
        im = ax.imshow(matrix, cmap="Greys", aspect="auto", vmin=0, vmax=1)
        for i, date in enumerate(matrix.index):
            for j, feature in enumerate(features):
                row = slots.loc[(date, feature)]
                ax.text(j, i, f"{int(row.known_slots)}/{int(row.expected_slots)}", ha="center", va="center",
                        fontsize=9, color="white" if row.available_fraction > .55 else "black")
        ax.set_xticks(range(len(features)), [FEATURE_LABELS[f] for f in features], rotation=28, ha="right")
        ax.set_yticks(range(len(matrix)), matrix.index)
        ax.set_title("D−1 18:00 前可核實的輸入覆蓋率\n節目每場 1 格、天氣每場 6 個目標小時；0 代表未知，不代表無人機為無或降雨為零")
        fig.colorbar(im, ax=ax, ticks=[0, .5, 1], shrink=.8, label="已核實時段／應有時段")
        save(fig, figures / "d1_feature_coverage.png")
    return {"surface": surface, "selected": selected}
