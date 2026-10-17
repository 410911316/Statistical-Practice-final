"""Matched station-count experiment including the currently published weighted XGB."""
from pathlib import Path
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from threadpoolctl import threadpool_limits
from analysis_v19.experiment import fit
from analysis_v16.experiment import check_panel, common_metrics
from analysis_v14.experiment import KEYS, WINDOW
from analysis_v11.evaluation import metrics
from analysis_v13.data import sha
from run_analysis_v8 import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'result/analysis_v19'
PRIVATE = ROOT/'.local/analysis_v19'


def run():
    cfg = json.loads((ROOT/'configs/analysis_v19.json').read_text('utf8'))
    old = {v: json.loads((ROOT/f'result/analysis_v{v}/manifest.json').read_text('utf8')) for v in (14, 16)}
    signature = {}
    for v, record in old.items():
        for path, digest in record['source_hashes'].items():
            if sha(ROOT/path) != digest:
                raise ValueError(f'Archived source changed: {path}')
        if sha(ROOT/f'.local/analysis_v{v}/predictions.csv') != record['prediction_sha256']:
            raise ValueError(f'Archived V{v} predictions changed')
        signature.update(record['source_hashes'])
    assert old[14]['groups'] == old[16]['groups']
    groups = old[14]['groups']
    full = pd.read_csv(ROOT/cfg['input'], parse_dates=['event_date'])
    # Discard all lag, post-event and unrequested fields before the experiment.
    frame = full[list(dict.fromkeys(KEYS+['year', 'y']+groups))].copy()
    check_panel(frame, cfg['cohorts']['four'])
    references = {v: pd.read_csv(ROOT/f'.local/analysis_v{v}/predictions.csv', parse_dates=['event_date']) for v in old}
    archived_fits = {14: json.loads((ROOT/'result/analysis_v14/models.json').read_text('utf8')),
                     16: json.loads((ROOT/'result/analysis_v16/fits.json').read_text('utf8'))}
    (OUT/'tables').mkdir(parents=True, exist_ok=True)
    (PRIVATE/'models').mkdir(parents=True, exist_ok=True)
    frame.to_csv(PRIVATE/'analysis_input.csv', index=False, encoding='utf-8-sig')
    fits, blocks, controls, correlations = [], [], [], []
    with threadpool_limits(limits=1):
        for year in cfg['outer_years']:
            valid = frame.loc[frame.year.eq(year)&frame.station.isin(cfg['evaluation_stations'])].sort_values(KEYS).reset_index(drop=True)
            check_panel(valid, cfg['evaluation_stations'])
            block = valid[KEYS+['year', 'y', *WINDOW]].copy()
            for cohort, stations in cfg['cohorts'].items():
                for method in cfg['methods']:
                    if method == 'xgboost_w4':
                        params = next(r['params'] for r in archived_fits[14] if r['year']==year and r['method']=='frozen_w4')
                    else:
                        model = 'ridge' if method=='ridge' else 'xgboost'
                        params = next(r['params'] for r in archived_fits[16] if r['year']==year and r['cohort']=='four' and r['model']==model and r['variant']=='none')
                    fitted, train, weights = fit(frame, year, stations, method, groups, params, cfg['seed'])
                    name = f'{cohort}_{method}'
                    block[name] = fitted.predict(valid).astype(float)
                    base = fitted.encoder if method=='xgboost_w4' else fitted.encoder.base
                    assert set(base.categories['station'])==set(stations)
                    if method=='xgboost_w4' and cohort=='four':
                        ref = references[14].loc[references[14].year.eq(year)].sort_values(KEYS)
                        actual = block.loc[block.station.eq('北門')].sort_values(KEYS)
                        column = 'frozen_w4'
                    elif method!='xgboost_w4' and cohort in ('four', 'three'):
                        ref = references[16].loc[references[16].year.eq(year)&references[16].station.isin(cfg['evaluation_stations'])].sort_values(KEYS)
                        actual = block.sort_values(KEYS)
                        column = f'{cohort}_{model}_none'
                    else:
                        ref = None
                    if ref is not None:
                        assert actual[KEYS].reset_index(drop=True).equals(ref[KEYS].reset_index(drop=True))
                        difference = float(np.max(abs(actual[name].to_numpy()-ref[column].to_numpy())))
                        if difference > 1e-6:
                            raise ValueError(f'Control not reproduced: {name} {year}: {difference}')
                        controls.append(dict(year=year, method=name, max_difference=difference))
                    path = PRIVATE/'models'/f'{year}_{name}.joblib'
                    joblib.dump(fitted, path)
                    fits.append(dict(year=year, cohort=cohort, model=method, method=name,
                        stations=stations, params=params, train_rows=len(train), train_events=train.event_date.nunique(),
                        train_end=str(train.event_date.max().date()), feature_names=fitted.encoder.names,
                        ridge_alpha=float(fitted.estimator.alpha) if method=='ridge' else None,
                        weight_audit=weights, path=path.relative_to(ROOT).as_posix(), sha256=sha(path)))
            blocks.append(block)
            past = frame.loc[frame.year.lt(year)].copy()
            past['increment'] = past.y-past.retrospective_baseline_56d
            for field in ('y', 'increment'):
                totals = past.groupby(['event_date', 'station'])[field].sum(min_count=6).unstack('station')
                for station in ['大橋頭站', '雙連', '民權西路']:
                    pair = totals[['北門', station]].dropna()
                    correlations.append(dict(before_year=year, field=field, station=station,
                        events=len(pair), spearman=float(pair.corr(method='spearman').iloc[0, 1])))
            print(f'Completed {year}: three cohorts x three settings', flush=True)
    pred = pd.concat(blocks, ignore_index=True)
    pred.to_csv(PRIVATE/'predictions.csv', index=False, encoding='utf-8-sig')
    tables = {name: [] for name in ('summary', 'annual', 'events', 'stations', 'common_two')}
    north = pred.loc[pred.station.eq('北門')]
    for cohort in cfg['cohorts']:
        for model in cfg['methods']:
            method = f'{cohort}_{model}'
            info = dict(method=method, cohort=cohort, model=model)
            tables['summary'].append(dict(**info, **metrics(north, north[method])))
            for year, group in north.groupby('year'):
                tables['annual'].append(dict(**info, year=year, **metrics(group, group[method])))
            for date, group in north.groupby('event_date'):
                tables['events'].append(dict(**info, event_date=str(date.date()), **metrics(group, group[method])))
            for station, group in pred.groupby('station'):
                tables['stations'].append(dict(**info, station=station, **common_metrics(group, group[method])))
            tables['common_two'].append(dict(**info, **common_metrics(pred, pred[method])))
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(OUT/'tables'/f'{name}.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(correlations).to_csv(OUT/'tables/past_event_correlations.csv', index=False, encoding='utf-8-sig')
    peaks = north.loc[north.groupby('event_date').y.idxmax()].copy()
    peaks.to_csv(OUT/'tables/actual_peak_predictions.csv', index=False, encoding='utf-8-sig')
    own = [Path(__file__), ROOT/'configs/analysis_v19.json', *sorted((ROOT/'src/analysis_v19').glob('*.py')),
           PRIVATE/'analysis_input.csv']
    for v in (14, 16):
        own.extend([ROOT/f'result/analysis_v{v}/manifest.json', ROOT/f'.local/analysis_v{v}/predictions.csv',
                    ROOT/f'result/analysis_v{v}'/('models.json' if v==14 else 'fits.json')])
    signature.update({p.relative_to(ROOT).as_posix(): sha(p) for p in own})
    write_json(OUT/'fits.json', fits)
    write_json(OUT/'manifest.json', dict(config=cfg, groups=groups, source_hashes=signature,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__),
        prediction_sha256=sha(PRIVATE/'predictions.csv'), reference_checks=controls,
        input_events=frame.event_date.nunique(), north_test_rows=len(north), common_test_rows=len(pred),
        fits=len(fits), deployed=False, independent_unseen_test=False, new_tuning=False))
    print(pd.DataFrame(tables['summary'])[['method', 'mae', 'rmse', 'peak_abs', 'cost_2']].round(2).to_string(index=False))


if __name__=='__main__':
    run()
