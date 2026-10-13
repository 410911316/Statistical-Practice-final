"""Extend GAM basis sizes without changing archived v7/v8 implementations."""
import numpy as np
from pygam import LinearGAM, l, s

from analysis_v5 import models as v5
from analysis_v7 import models as v7


class GAMFitted(v7.GAMFitted):
    def info(self):
        result = super().info()
        if result["smooth_columns"]:
            result["effective_parameters"] = ["lambda", "ridge_alpha", "n_splines"]
        result["version"] = "v11_variable_basis_with_v7_numerical_certificates"
        return result


def fit_one(frame, name, groups, params, config, encoder=None):
    if name != "gam":
        return v5.fit_one(frame, name, groups, params, config, encoder=encoder)
    if config.get("include_historical_baseline", False):
        raise ValueError("Unverified historical baseline is excluded")
    encoder = v7.Encoder().fit(frame, groups, config) if encoder is None else encoder
    if not isinstance(encoder, v7.Encoder) or tuple(groups) != encoder.requested_groups:
        raise ValueError("GAM needs a matching training-only encoder")
    k, lam, alpha = params["n_splines"], float(params["lambda"]), float(params["ridge_alpha"])
    if k not in (4,5,7) or not np.isfinite([lam,alpha]).all() or min(lam,alpha) <= 0:
        raise ValueError("Invalid prespecified GAM grid")
    x, y = encoder.transform(frame), frame.y.to_numpy(float)
    if not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid training truth")
    terms, audit = None, []
    support_map = {a["column"]:a for a in encoder.continuous_audit}
    ridge_penalty = len(frame)*alpha
    for i, column in enumerate(encoder.names):
        support = support_map.get(column,{})
        known, distinct = support.get("known_training_events"), support.get("distinct_known_training_values")
        smooth = column in v7.SMOOTH_CANDIDATES and known >= 6 and distinct >= 8
        lo, hi = float(x[:,i].min()), float(x[:,i].max())
        term = (s(i,n_splines=k,spline_order=3,lam=[lam,ridge_penalty],
                  penalties=["derivative","l2"],basis="ps",edge_knots=[lo,hi])
                if smooth else l(i,lam=ridge_penalty,penalties="l2"))
        terms = term if terms is None else terms+term
        audit.append(dict(column=column,feature_index=i,kind="smooth" if smooth else "linear",
                          reason="training_support_passed_for_smoothing" if smooth else "supported_linear_or_structural_term_with_ridge",
                          known_training_events=known,distinct_known_training_values=distinct,
                          training_min_raw=support.get("training_min_raw"),training_max_raw=support.get("training_max_raw"),
                          training_min_encoded=lo,training_max_encoded=hi,n_splines=k if smooth else None,
                          spline_order=3 if smooth else None,lambda_=lam if smooth else 0.,
                          ridge_alpha=alpha,ridge_penalty=ridge_penalty,
                          penalty="second_order_coefficient_difference_plus_full_coefficient_l2" if smooth else "coefficient_l2",
                          edge_knots_encoded=[lo,hi] if smooth else None,uses_y_for_term_or_knot_selection=False))
        audit[-1]["lambda"] = audit[-1].pop("lambda_")
    if terms is None:
        raise ValueError("No usable GAM design")
    estimator = LinearGAM(terms,fit_intercept=True,max_iter=config.get("gam_max_iter",1000),
                          tol=config.get("gam_tol",1e-6)).fit(x,y)
    estimator.v7_terms_audit_ = audit
    certificate = v7.certify_gam(estimator,x,y,audit)
    design = np.column_stack([np.ones(len(x)),x])
    return GAMFitted("gam",encoder,estimator,
                     {**params,"spline_order":3,"ridge_penalty":ridge_penalty,"n_train_rows":len(frame),
                      "lambda_scale":"unnormalized_sum_SSE_no_n_multiplier",
                      "ridge_alpha_scale":"n_rows_times_alpha_coefficient_L2",
                      "max_iter":estimator.max_iter,"tol":estimator.tol},
                     int(np.linalg.matrix_rank(design)),design.shape[1],terms_audit=audit,certificate=certificate)

