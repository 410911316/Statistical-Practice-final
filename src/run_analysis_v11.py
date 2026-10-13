"""Run frozen dispatch-oriented tuning; cached predictions have strict source hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from analysis_v5.models import Encoder
from analysis_v7.models import Encoder as GAMEncoder
from analysis_v11.models import fit_one
from analysis_v11.evaluation import KEYS, WINDOW, costs, metrics, select_candidate, window_weights
from compare_fusion_losses import load_verified
from run_analysis_v8 import write_json

ROOT=Path(__file__).resolve().parents[1]
MODELS=("ridge","gam","xgboost")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_grid(root,frame,base,config,variant,model,reuse,progress=print):
    private=root/".local/analysis_v11"/variant/model
    private.mkdir(parents=True,exist_ok=True)
    code=[root/"src/run_analysis_v11.py",root/"src/analysis_v11/models.py",root/"src/analysis_v11/evaluation.py",
          root/"src/analysis_v5/models.py",root/"src/analysis_v7/models.py"]
    signature={str(p.relative_to(root)):digest(p) for p in
               [*code,root/"configs/analysis_v11.json",root/f"configs/analysis_v8_{variant}.json",
                root/f".local/analysis_v8/{variant}/analysis_input.csv"]}
    pieces,predictions,folds,files=[],[],[],[]
    grid=config["grids"][model]
    groups=tuple(base["candidate_groups"])
    for year in sorted(frame.year.unique()):
        train=frame.loc[frame.year<year].copy()
        if train.event_date.nunique()<config["candidate_prediction_start_min_train_events"]:continue
        valid=frame.loc[frame.year.eq(year)&frame.station.eq("北門")].sort_values(KEYS).reset_index(drop=True)
        cache,audit_path=private/f"{year}.npz",private/f"{year}.json"
        expected={"signature":signature,"validation_year":int(year),"train_end":str(train.event_date.max().date())}
        if reuse and cache.exists() and audit_path.exists():
            record=json.loads(audit_path.read_text("utf-8"))
            if record["source"]!=expected or digest(cache)!=record["sha256"]:
                raise ValueError("Changed v11 cache; rerun without --reuse-candidates")
            matrix=np.load(cache)["predictions"]
            if matrix.shape!=(len(valid),len(grid)):raise ValueError("Wrong cached grid shape")
            progress(f"{variant} {model} {year}: verified cache",flush=True)
        else:
            encoder=(GAMEncoder().fit(train,groups,base) if model=="gam" else Encoder().fit(train,groups))
            matrix=np.empty((len(valid),len(grid)))
            worst_certificate=0.
            progress(f"{variant} {model} {year}: {len(train)} rows, {len(grid)} candidates",flush=True)
            for index,params in enumerate(grid):
                fitted=fit_one(train,model,groups,params,base,encoder)
                matrix[:,index]=fitted.predict(valid)
                if model=="gam":
                    worst_certificate=max(worst_certificate,fitted.certificate["relative_normal_equation_residual"])
            if not np.isfinite(matrix).all() or (matrix<0).any():raise ValueError("Invalid grid prediction")
            np.savez_compressed(cache,predictions=matrix)
            record={"source":expected,"sha256":digest(cache),"train_rows":len(train),
                    "train_events":int(train.event_date.nunique()),"train_stations":sorted(train.station.unique()),
                    "validation_events":int(valid.event_date.nunique()),"validation_start":str(valid.event_date.min().date()),
                    "candidate_count":len(grid),"max_gam_normal_equation_residual":worst_certificate}
            write_json(audit_path,record)
        pieces.append(valid[KEYS+["year","y",*WINDOW]])
        predictions.append(matrix)
        folds.append({"variant":variant,"model":model,**record})
        files.extend([cache,audit_path])
    return pd.concat(pieces,ignore_index=True),np.vstack(predictions),folds,files


def run(reuse=False):
    root=ROOT
    config=json.loads((root/"configs/analysis_v11.json").read_text("utf-8"))
    shared,_,_,manifests,sources=load_verified(root)
    output,private=root/"result/analysis_v11",root/".local/analysis_v11"
    (output/"tables").mkdir(parents=True,exist_ok=True)
    selections,score_rows,annual_rows,event_rows,folds,source_files=[],[],[],[],[],[]
    frames={}
    saved={}
    for variant in config["variants"]:
        base=json.loads((root/f"configs/analysis_v8_{variant}.json").read_text("utf-8-sig"))
        source=pd.read_csv(root/f".local/analysis_v8/{variant}/analysis_input.csv",parse_dates=["event_date"])
        source=source.sort_values(KEYS).reset_index(drop=True)
        outer=source.loc[source.year.isin(config["outer_years"])&source.station.eq("北門")].reset_index(drop=True)
        old=shared[variant].loc[lambda d:d.station.eq("北門")].sort_values(KEYS).reset_index(drop=True)
        assert outer[KEYS].equals(old[KEYS]) and np.array_equal(outer.y,old.y)
        frame=outer[KEYS+["year","y",*WINDOW]].copy()
        for model in MODELS:
            frame["old_"+model]=old[model].to_numpy()
            timeline,matrix,records,files=predict_grid(root,source,base,config,variant,model,reuse)
            folds.extend(records);source_files.extend(files)
            for year in config["outer_years"]:
                before=timeline.year.lt(year).to_numpy()
                current=timeline.year.eq(year).to_numpy()
                past=timeline.loc[before].reset_index(drop=True)
                groups=tuple(base["candidate_groups"])
                for grid_name,count in (("original",config["old_grid_sizes"][model]),("expanded",len(config["grids"][model]))):
                    for ratio in config["ratios"]:
                        best,loss,losses=select_candidate(past,matrix[before],year,ratio,count)
                        name=f"{grid_name}_r{ratio}_{model}"
                        frame.loc[frame.year.eq(year),name]=matrix[current,best]
                        _,audit=window_weights(past)
                        selections.append({"variant":variant,"outer_year":year,"grid":grid_name,"ratio":ratio,"model":model,
                                           "candidate_id":best,"candidate_count":count,"params":config["grids"][model][best],
                                           "inner_cost":loss,"inner_events":int(past.event_date.nunique()),
                                           "inner_start":str(past.event_date.min().date()),"inner_end":str(past.event_date.max().date()),
                                           "inner_fallback_events":sum(x["fallback"] is not None for x in audit),
                                           "all_candidate_costs":losses.tolist()})
                        # Save each distinct selected model once, refitted only on past years.
                        key=(variant,year,model,best)
                        if key not in saved:
                            fitted=fit_one(source.loc[source.year<year],model,groups,config["grids"][model][best],base)
                            prediction=fitted.predict(outer.loc[outer.year.eq(year)])
                            np.testing.assert_allclose(prediction,matrix[current,best],atol=1e-7,rtol=0)
                            saved[key]=fitted
        names=[c for c in frame if c.startswith(("old_","original_","expanded_"))]
        for name in names:
            grid="old" if name.startswith("old_") else name.split("_")[0]
            ratio=0 if grid=="old" else int(name.split("_")[1][1:])
            model=name.split("_")[-1]
            common={"variant":variant,"method":name,"model":model,"grid":grid,"ratio":ratio}
            score_rows.append({**common,**metrics(frame,frame[name])})
            for year,f in frame.groupby("year"):
                annual_rows.append({**common,"year":int(year),**metrics(f.reset_index(drop=True),f[name])})
            for date,f in frame.groupby("event_date"):
                event_rows.append({**common,"event_date":str(date.date()),**metrics(f.reset_index(drop=True),f[name])})
        frame.to_csv(private/f"{variant}_predictions.csv",index=False,encoding="utf-8-sig")
        frames[variant]=frame
        source_files.append(private/f"{variant}_predictions.csv")
    joblib.dump(saved,private/"selected_models.joblib")
    source_files.append(private/"selected_models.joblib")
    scores=pd.DataFrame(score_rows)
    for name,data in [("summary",scores),("annual",pd.DataFrame(annual_rows)),("events",pd.DataFrame(event_rows))]:
        data.to_csv(output/"tables"/f"{name}.csv",index=False,encoding="utf-8-sig")
    write_json(output/"selections.json",selections)
    write_json(output/"folds.json",folds)
    files=[root/"configs/analysis_v11.json",Path(__file__),root/"src/analysis_v11/models.py",root/"src/analysis_v11/evaluation.py"]
    write_json(output/"run_manifest.json",{"config":config,"d1_deployable":False,"d1_limitations":manifests["both"]["data"]["audit"],
               "source_files":sources,"source_and_output_hashes":[{"path":p.relative_to(root).as_posix(),"sha256":digest(p)} for p in [*files,*source_files]],
               "base_training_rows":504,"events":21,"outer_events":9,
               "candidate_grid_fits":sum(x["candidate_count"] for x in folds),
               "distinct_selected_models":len(saved),"status":"completed_exploratory_repeated_outer_events"})
    print(scores.loc[(scores.variant=="both")&((scores.grid=="old")|((scores.grid=="expanded")&(scores.ratio==2))),
                     ["method","mae","rmse","cost_2","window_under","window_over"]].to_string(index=False),flush=True)
    print("Complete:",output,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-candidates",action="store_true")
    run(parser.parse_args().reuse_candidates)

