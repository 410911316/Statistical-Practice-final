"""V13 isolated encoders: version-screened baseline and signed activity residuals.
Archived v1/v5/v7/v11 files and their fitted objects are unchanged.
"""
import numpy as np
import pandas as pd
from analysis_v1.models import ALLOWED_CANDIDATES, MISSING_ALLOWED_CANDIDATES, CORE_NUMERIC
from analysis_v5 import models as v5
from analysis_v7 import models as v7


BINARY = {"planned_drone", "planned_finale", "planned_enhanced_nonfinale"}
EXTRA = {"planned_finale", "planned_enhanced_nonfinale", "available_baseline", "unverified_google_prev_month"}
ALLOWED = ALLOWED_CANDIDATES | EXTRA
OPTIONAL = MISSING_ALLOWED_CANDIDATES | EXTRA

class Encoder(v5.Encoder):
    def fit(self, frame: pd.DataFrame, groups: tuple[str, ...], station_hour_interaction: bool = True,
            include_historical_baseline: bool = False):
        if include_historical_baseline:
            raise ValueError("Use the separately labelled retrospective baseline sensitivity")
        if not station_hour_interaction or len(set(groups)) != len(groups):
            raise ValueError("Need unique groups and station-hour interactions")
        if not set(groups) <= ALLOWED:
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
                                          if name in OPTIONAL]
        self.all_missing_numeric = []
        for name in self.missing_indicator_numeric:
            values = pd.to_numeric(frame[name], errors="raise")
            known = values.dropna()
            self._validate_optional_values(name, known)
            if known.empty:
                value, strategy = 0.0, "all_training_values_unknown_placeholder_dropped"
                self.all_missing_numeric.append(name)
            elif name in BINARY:
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
        for name in BINARY & set(self.numeric):
            self.mean[name], self.scale[name] = 0.0, 1.0
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
        raw = self._raw(frame)
        constants = set(self.dropped_constants)

        def priority(index):
            name = self.all_names[index]
            if name.startswith(("station=", "hour=")):
                return (1 if ":hour=" in name else 0, index)
            return (3 if name.endswith(":missing") else 2, index)

        candidates = sorted((i for i, name in enumerate(self.all_names)
                             if name not in constants), key=priority)
        # Unit-norm columns make the numerical tolerance independent of units.
        current = np.ones((len(frame), 1), dtype=float) / np.sqrt(len(frame))
        self.rank_tolerance = float(10 * np.finfo(float).eps * max(raw.shape[0], raw.shape[1] + 1))
        self.selected_indices, self.dropped_dependencies = [], []
        for index in candidates:
            column = raw[:, index]
            scaled = column / np.linalg.norm(column)
            trial = np.column_stack([current, scaled])
            if np.linalg.matrix_rank(trial, tol=self.rank_tolerance) == trial.shape[1]:
                self.selected_indices.append(index)
                current = trial
            else:
                self.dropped_dependencies.append({
                    "column": self.all_names[index],
                    "reason": "exact_linear_dependency_with_intercept_and_prior_training_columns",
                    "scope": "training_design_only_same_restriction_for_all_four_models",
                })
        self.names = [self.all_names[i] for i in self.selected_indices]
        self.keep = np.array([i in self.selected_indices for i in range(len(self.all_names))])
        self.rank = int(current.shape[1])
        self.column_selection_audit = {
            "priority": ["intercept", "station_and_hour_main_effects", "station_hour_products",
                         "approved_numeric_in_candidate_order", "missing_indicators"],
            "rank_tolerance_on_unit_norm_columns": self.rank_tolerance,
            "dropped_constants": self.dropped_constants,
            "dropped_dependencies": self.dropped_dependencies,
            "uses_y": False,
        }
        return self


    @staticmethod
    def _validate_optional_values(name, known):
        v5.Encoder._validate_optional_values(name, known)
        if name in BINARY and not known.isin([0,1]).all():
            raise ValueError("Event-type flags must be 0, 1 or unknown")
        if name == "available_baseline" and (known < 0).any():
            raise ValueError("Negative baseline")
        if name == "unverified_google_prev_month" and not known.between(0,100).all():
            raise ValueError("Google index outside 0-100")

class GAMEncoder(v7.Encoder, Encoder):
    def fit(self, frame, groups, config=None):
        config = config or {}
        if int(config.get("gam_min_known_events", 6)) != 6:
            raise ValueError("V7 fixes a minimum of six known training events")
        binary = tuple(config.get("gam_binary_groups", ["is_weekend"]))
        if binary != ("is_weekend",):
            raise ValueError("V7 prespecifies is_weekend as its support-gate exemption")
        groups = tuple(groups)
        self.requested_groups = groups
        self.continuous_audit, effective = [], []
        for name in groups:
            if name in binary:
                effective.append(name)
                continue
            values = pd.to_numeric(frame[name], errors="raise")
            known = values.notna()
            if not np.isfinite(values.loc[known].to_numpy(float)).all():
                raise ValueError(f"Nonfinite GAM candidate {name}")
            n_events = int(frame.loc[known, "event_date"].nunique())
            accepted = n_events >= 6
            if accepted:
                effective.append(name)
            self.continuous_audit.append({
                "column": name, "known_training_events": n_events,
                "known_training_rows": int(known.sum()),
                "distinct_known_training_values": int(values.loc[known].nunique()),
                "training_min_raw": float(values.loc[known].min()) if known.any() else None,
                "training_max_raw": float(values.loc[known].max()) if known.any() else None,
                "support_gate_passed": accepted,
                "omission_reason": None if accepted else "fewer_than_six_known_training_events",
                "gate_uses_y": False,
            })
        Encoder.fit(self, frame, tuple(effective))
        dependencies = {item["column"] for item in self.dropped_dependencies}
        for item in self.continuous_audit:
            name = item["column"]
            item["active"] = name in self.names
            if item["support_gate_passed"] and not item["active"]:
                item["omission_reason"] = ("training_constant" if name in self.dropped_constants else
                    "training_exact_rank_dependency" if name in dependencies else "not_in_encoded_design")
            item["missing_indicator_active"] = f"{name}:missing" in self.names
        self.column_selection_audit["scope"] = "GAM_specific_train_only_subset; legacy_design_unchanged"
        self.column_selection_audit["support_gate"] = self.continuous_audit
        return self

# The fitting path below accepts signed increments. Archived estimators reject
# negative count targets and clip predictions, so they cannot be reused as-is.
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from pygam import LinearGAM, l, s
from analysis_v13.data import require_features


@dataclass
class Fitted:
    model: str
    mode: str
    encoder: object
    estimator: object
    params: dict
    certificate: dict

    def predict_components(self, frame):
        require_features(frame)
        raw=np.asarray(self.estimator.predict(self.encoder.transform(frame)),float)
        if not np.isfinite(raw).all():raise ValueError('Nonfinite prediction')
        total=raw+frame.available_baseline.to_numpy(float) if self.mode=='increment' else raw
        return raw,np.maximum(total,0)

    def predict(self, frame):return self.predict_components(frame)[1]


def fit_model(frame, model, mode, groups, params, config):
    require_features(frame)
    if mode not in ('core','direct','increment') or model not in ('ridge','gam','xgboost'):
        raise ValueError('Unknown model/mode')
    if (mode=='core') != ('available_baseline' not in groups):
        raise ValueError('Core excludes baseline; direct and increment share it')
    encoder=GAMEncoder().fit(frame,tuple(groups),config) if model=='gam' else Encoder().fit(frame,tuple(groups))
    x=encoder.transform(frame);y=frame.y.to_numpy(float)
    if not len(y) or not np.isfinite(y).all() or (y<0).any():raise ValueError('Invalid total-count labels')
    target=y-frame.available_baseline.to_numpy(float) if mode=='increment' else y
    certificate={}
    if model=='ridge':
        estimator=Ridge(alpha=len(frame)*float(params['lambda']),solver='svd').fit(x,target)
    elif model=='xgboost':
        estimator=XGBRegressor(objective='reg:squarederror',tree_method='hist',n_jobs=1,
            random_state=config['seed'],**{**config['xgb_fixed'],**params}).fit(x,target)
    else:
        support={a['column']:a for a in encoder.continuous_audit}
        terms=None;audit=[];lam=float(params['lambda']);alpha=float(params['ridge_alpha']);k=int(params['n_splines'])
        if min(lam,alpha)<=0 or k not in (4,5,7):raise ValueError('Invalid GAM complexity')
        for i,name in enumerate(encoder.names):
            item=support.get(name,{})
            smooth=name in v7.SMOOTH_CANDIDATES and item.get('known_training_events',0)>=6 and item.get('distinct_known_training_values',0)>=8
            term=(s(i,n_splines=k,spline_order=3,lam=[lam,len(frame)*alpha],penalties=['derivative','l2'],
                    basis='ps',edge_knots=[float(x[:,i].min()),float(x[:,i].max())]) if smooth else
                  l(i,lam=len(frame)*alpha,penalties='l2'))
            terms=term if terms is None else terms+term
            audit.append(dict(column=name,kind='smooth' if smooth else 'linear',ridge_penalty=len(frame)*alpha,
                              **{'lambda':lam if smooth else 0}))
        if terms is None:raise ValueError('No GAM design')
        estimator=LinearGAM(terms,fit_intercept=True,max_iter=config.get('gam_max_iter',1000),
                            tol=config.get('gam_tol',1e-6)).fit(x,target)
        certificate=v7.certify_gam(estimator,x,target,audit)
    certificate.update(negative_training_increments=int((target<0).sum()) if mode=='increment' else 0,
                       signed_increment_not_clipped=True,encoder_columns=encoder.names)
    return Fitted(model,mode,encoder,estimator,params,certificate)
