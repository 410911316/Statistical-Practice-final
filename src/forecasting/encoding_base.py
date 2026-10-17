"""Training-only numeric and categorical transformations."""
from __future__ import annotations
import numpy as np
import pandas as pd

CORE_NUMERIC = ("baseline_hourly_56d",)


PLANNED_END_CANDIDATES = {
    "planned_end_0_60_beimen", "planned_end_0_60_daqiaotou",
    "planned_end_0_60_shuanglian", "planned_end_0_60_minquan_west",
    "planned_end_60_120_beimen", "planned_end_60_120_daqiaotou",
    "planned_end_60_120_shuanglian", "planned_end_60_120_minquan_west",
}


MISSING_ALLOWED_CANDIDATES = {
    "planned_drone", "planned_fireworks_minutes", "planned_fireworks_start_hour",
    "planned_program_minutes", "forecast_temperature_c", "forecast_precip_probability",
    "forecast_wind_speed_m_s", "forecast_cloud_cover_pct",
} | PLANNED_END_CANDIDATES


ALLOWED_CANDIDATES = {"is_weekend", "day_of_year", "year_index"} | MISSING_ALLOWED_CANDIDATES


class Encoder:
    """Fit numeric scaling, reference levels and constants on this train only."""

    def fit(self, frame: pd.DataFrame, groups: tuple[str, ...], station_hour_interaction: bool = False,
            include_historical_baseline: bool = True):
        if not set(groups) <= ALLOWED_CANDIDATES:
            raise ValueError("Unapproved or post-event feature in model")
        self.groups = groups
        self.station_hour_interaction = station_hour_interaction
        self.include_historical_baseline = include_historical_baseline
        self.numeric = (list(CORE_NUMERIC) if include_historical_baseline else []) + list(groups)
        # Only independently approved D-1 fields may be unknown. Their source
        # timing is checked by the loader; this encoder must never use holdout
        # values to learn an imputation, scale, or missingness effect.
        self.imputation_values = {}
        self.imputation_audit = {}
        self.missing_indicator_numeric = [name for name in self.numeric
                                          if name in MISSING_ALLOWED_CANDIDATES]
        self.all_missing_numeric = []
        for name in self.missing_indicator_numeric:
            values = pd.to_numeric(frame[name], errors="raise")
            known = values.dropna()
            self._validate_optional_values(name, known)
            if known.empty:
                value, strategy = 0.0, "all_training_values_unknown_placeholder_dropped"
                self.all_missing_numeric.append(name)
            elif name == "planned_drone":
                value, strategy = float(known.mode().iloc[0]), "training_mode_tie_choose_zero"
            else:
                value, strategy = float(known.median()), "training_median"
            self.imputation_values[name] = value
            self.imputation_audit[name] = {
                "strategy": strategy, "fill_value": value,
                "training_known_rows": int(values.notna().sum()),
                "training_unknown_rows": int(values.isna().sum()),
                "scope": "training_fold_only_unknown_is_not_observed_zero",
            }
        values = self._numeric_values(frame)
        self.mean = values.mean()
        self.scale = values.std(ddof=0).replace(0, 1)
        # Dummy/binary columns stay on their 0/1 scale, as specified in the PDF.
        if "is_weekend" in self.numeric:
            self.mean["is_weekend"], self.scale["is_weekend"] = 0.0, 1.0
        if "planned_drone" in self.numeric:
            self.mean["planned_drone"], self.scale["planned_drone"] = 0.0, 1.0
        self.categories = {c: sorted(frame[c].unique()) for c in ("station", "hour")}
        self.all_names = self.numeric + [f"{name}:missing" for name in self.missing_indicator_numeric]
        self.all_names += [f"{c}={v}" for c, cats in self.categories.items() for v in cats[1:]]
        if self.station_hour_interaction:
            # Treatment coding retains both main effects. The reference station
            # and hour are learned from this training subset, never the holdout.
            self.all_names += [f"station={s}:hour={h}"
                               for s in self.categories["station"][1:]
                               for h in self.categories["hour"][1:]]
        raw = self._raw(frame)
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite model input; missing values are not zero")
        self.keep = np.ptp(raw, axis=0) > 1e-12
        self.names = [n for n, keep in zip(self.all_names, self.keep) if keep]
        self.dropped_constants = [n for n, keep in zip(self.all_names, self.keep) if not keep]
        return self

    @staticmethod
    def _validate_optional_values(name, known):
        if not np.isfinite(known.to_numpy(float)).all():
            raise ValueError(f"Nonfinite approved feature {name}; infinity is not missing")
        if name == "planned_drone" and not known.isin([0, 1]).all():
            raise ValueError("planned_drone must be 0, 1, or unknown")
        if name in PLANNED_END_CANDIDATES and not known.between(0, 1).all():
            raise ValueError("Planned show-end hourly overlap must be within [0, 1] or unknown")

    def _numeric_values(self, frame):
        values = frame[self.numeric].copy()
        # Old joblib encoders have no imputation attributes; their design and
        # predictions retain the original strictly complete-input behavior.
        for name, fill in getattr(self, "imputation_values", {}).items():
            column = pd.to_numeric(values[name], errors="raise")
            self._validate_optional_values(name, column.dropna())
            values[name] = column.fillna(fill)
        return values

    def _raw(self, frame):
        arrays = [((self._numeric_values(frame) - self.mean) / self.scale).to_numpy(float)]
        arrays.extend([frame[name].isna().to_numpy(float)[:, None]
                       for name in getattr(self, "missing_indicator_numeric", [])])
        for col, cats in self.categories.items():
            arrays.extend([(frame[col].to_numpy() == val).astype(float)[:, None] for val in cats[1:]])
        # Missingness flags are optional for purely structural designs.
        if getattr(self, "station_hour_interaction", False):
            arrays.extend([((frame.station.to_numpy() == station) &
                            (frame.hour.to_numpy() == hour)).astype(float)[:, None]
                           for station in self.categories["station"][1:]
                           for hour in self.categories["hour"][1:]])
        return np.column_stack(arrays)

    def transform(self, frame):
        # Stations/hours are structural; unknown structural levels are invalid.
        for col, cats in self.categories.items():
            if not set(frame[col].unique()) <= set(cats):
                raise ValueError(f"Unseen {col} category; cannot silently forecast a new station/hour")
        x = self._raw(frame)[:, self.keep]
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite prediction features")
        return x
