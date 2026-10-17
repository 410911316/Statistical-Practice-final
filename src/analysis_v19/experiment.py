"""Remove stations before preprocessing; preserve the archived loss conventions."""
import numpy as np
from analysis_v16.experiment import check_panel, select_train, fit_cohort
from analysis_v14.experiment import Fitted, estimator, WINDOW
from analysis_v12.models import Encoder


def training_weights(train, stations, multiplier):
    check_panel(train, stations)
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError('Invalid training multiplier')
    frame = train.reset_index(drop=True)
    weights = np.ones(len(frame))
    audit = []
    for date, event in frame.groupby('event_date'):
        north = event.loc[event.station.eq('北門')]
        values = north[WINDOW].to_numpy(float)
        unknown = np.isnan(values).all()
        if unknown:
            overlap = np.zeros(len(north))
        else:
            if (not np.isfinite(values).all() or (values < 0).any()
                    or (values > 1).any() or (values.sum(axis=1) > 1+1e-8).any()):
                raise ValueError('Partial or invalid planned window')
            overlap = values.sum(axis=1)
        weights[north.index] = 1+(multiplier-1)*overlap
        weights[event.index] *= len(event)/weights[event.index].sum()
        audit.append(dict(event_date=str(date.date()), unknown_end=bool(unknown),
                          rows=len(event), weight_sum=float(weights[event.index].sum()),
                          north_weight=float(weights[north.index].sum())))
    return weights, audit


def fit(frame, year, stations, method, groups, params, seed):
    if method in ('ridge', 'xgboost_w1'):
        model = 'ridge' if method == 'ridge' else 'xgboost'
        fitted, train = fit_cohort(frame, year, stations, model, 'none', groups, params, seed)
        return fitted, train, []
    if method != 'xgboost_w4':
        raise ValueError('Unknown prespecified method')
    train = select_train(frame, year, stations)
    weights, audit = training_weights(train, stations, 4)
    encoder = Encoder().fit(train, tuple(groups))
    settings = dict(params)
    trees = settings.pop('n_estimators')
    model = estimator(settings, trees, {'seed': seed}).fit(
        encoder.transform(train), train.y, sample_weight=weights)
    return Fitted(encoder, model, dict(params), 4), train, audit
