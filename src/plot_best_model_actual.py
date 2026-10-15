"""Draw archived actual/predicted counts; no training or retrospective deployment choice."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis_v1.reporting import style,save
from analysis_v13.data import sha
from analysis_v11.evaluation import metrics
from run_analysis_v8 import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"result/model_diagnostics_20261014"
KEYS=["event_date","station","hour"]


def event_table(frame,column):
    rows=[]
    for date,g in frame.groupby("event_date"):
        g=g.sort_values("hour")
        peak=g.loc[g.y.idxmax()]
        own_peak=g.loc[g[column].idxmax()]
        rows.append(dict(event_date=str(date.date()),actual_mean=g.y.mean(),predicted_mean=g[column].mean(),
            actual_total=g.y.sum(),predicted_total=g[column].sum(),actual_peak_hour=int(peak.hour),
            actual_peak=float(peak.y),prediction_at_actual_peak=float(peak[column]),
            peak_error=float(peak[column]-peak.y),peak_error_percent=100*(peak[column]-peak.y)/peak.y,
            predicted_peak_hour=int(own_peak.hour),predicted_peak=float(own_peak[column]),
            hourly_mae=float(np.mean(abs(g[column]-g.y))),hourly_rmse=float(np.sqrt(np.mean((g[column]-g.y)**2)))))
    return pd.DataFrame(rows)


def bars(events,title,path):
    fig,axes=plt.subplots(2,1,figsize=(14,9),constrained_layout=True)
    x=np.arange(len(events));w=.35
    for ax,actual,pred,subtitle in [(axes[0],"actual_mean","predicted_mean","各場六時段平均：不是整場總人數"),
                                   (axes[1],"actual_peak","prediction_at_actual_peak","各場實際最高時段：同一時段比較，不是各自取最大")]:
        a=ax.bar(x-w/2,events[actual],w,color="#434a46",label="實際")
        b=ax.bar(x+w/2,events[pred],w,color="#689884",label="模型預測")
        ax.bar_label(a,fmt="%.0f",padding=3,fontsize=9)
        ax.bar_label(b,fmt="%.0f",padding=3,fontsize=9)
        ax.set(xticks=x,xticklabels=events.event_date.str[2:],ylabel="人次／來源時段",
               title=subtitle,ylim=(0,max(events[actual].max(),events[pred].max())*1.29))
        ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
        if actual=="actual_peak":
            for j,r in events.iterrows():
                h=max(r.actual_peak,r.prediction_at_actual_peak)+max(events.actual_peak)*.075
                ax.text(j,h,f"差 {r.peak_error:+,.0f}\n({r.peak_error_percent:+.1f}%)",
                        ha="center",fontsize=9,color="#a1664b" if r.peak_error<0 else "#52758c")
    fig.suptitle(title+"\n2024–2026 相同九場北門｜負差＝低估，正差＝高估；反覆使用的開發驗證",fontsize=14)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc="outside lower center",ncol=2)
    save(fig,path)


def run():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"figures").mkdir(exist_ok=True);(OUT/"tables").mkdir(exist_ok=True)
    sources={}
    for version in [14,16]:
        path=ROOT/f".local/analysis_v{version}/predictions.csv"
        manifest_path=ROOT/f"result/analysis_v{version}/manifest.json"
        m=json.loads(manifest_path.read_text("utf8"))
        assert sha(path)==m["prediction_sha256"]
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
        sources[manifest_path.relative_to(ROOT).as_posix()]=sha(manifest_path)
    a=pd.read_csv(ROOT/".local/analysis_v14/predictions.csv",parse_dates=["event_date"]).sort_values(KEYS)
    b=pd.read_csv(ROOT/".local/analysis_v16/predictions.csv",parse_dates=["event_date"])
    b=b.loc[b.station.eq("北門")].sort_values(KEYS)
    assert len(a)==len(b)==54 and a.event_date.nunique()==9
    assert a[KEYS].reset_index(drop=True).equals(b[KEYS].reset_index(drop=True))
    np.testing.assert_array_equal(a.y.to_numpy(),b.y.to_numpy())
    recent=[]
    # Rank only whole-model retrospective candidates in V14–V18, NOT per-event cherry picking.
    for version in [14,15,16,18]:
        path=ROOT/f"result/analysis_v{version}/tables/summary.csv"
        s=pd.read_csv(path)
        for r in s.to_dict("records"):
            recent.append(dict(version=version,method=r["method"],model=r.get("model","xgboost"),
                               mae=r["mae"],peak_abs=r["peak_abs"],cost_2=r["cost_2"]))
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    rank=pd.DataFrame(recent).sort_values("mae")
    assert rank.iloc[0].version==14 and rank.iloc[0].method=="frozen_w4"
    s3=pd.read_csv(ROOT/"result/analysis_v16/tables/summary.csv")
    assert s3.loc[s3.cohort.eq("three")].sort_values("mae").iloc[0].method=="three_xgboost_none"
    rank.to_csv(OUT/"tables/recent_candidate_ranking.csv",index=False,encoding="utf-8-sig")
    style()
    summary=[]
    for f,col,name,title in [(a,"frozen_w4","best_four","近期可比實驗最低 MAE 440.28｜四站 XGBoost＋散場訓練 4 倍"),
                             (b,"three_xgboost_none","current_three","目前三站基準 MAE 491.32｜三站 XGBoost")]:
        events=event_table(f,col)
        events.to_csv(OUT/f"tables/{name}_events.csv",index=False,encoding="utf-8-sig")
        bars(events,title,OUT/f"figures/{name}_actual.png")
        score=metrics(f,f[col])
        residual=f[col].to_numpy()-f.y.to_numpy()
        j=int(np.argmax(abs(residual)));worst=f.iloc[j]
        summary.append(dict(model=name,source_column=col,**score,
            actual_mean=float(f.y.mean()),predicted_mean=float(f[col].mean()),
            max_absolute_error=float(abs(residual[j])),worst_event=str(worst.event_date.date()),
            worst_hour=int(worst.hour),worst_actual=float(worst.y),worst_prediction=float(worst[col]),
            within_500=int(np.sum(abs(residual)<=500)),within_1000=int(np.sum(abs(residual)<=1000)),rows=len(f)))
    pd.DataFrame(summary).to_csv(OUT/"tables/summary.csv",index=False,encoding="utf-8-sig")

    fig,axes=plt.subplots(3,3,figsize=(14,9.6),constrained_layout=True,sharey=True)
    for ax,(date,g) in zip(axes.flat,a.groupby("event_date")):
        ax.bar(g.hour-.18,g.y,.35,color="#434a46",label="實際")
        ax.bar(g.hour+.18,g.frozen_w4,.35,color="#689884",label="模型預測")
        ax.set(title=f"{date.date()}｜MAE {np.mean(abs(g.frozen_w4-g.y)):.0f}",
               xticks=range(18,24),xlabel="原始資料來源時段",ylabel="北門人次",ylim=(0,12000))
        ax.grid(axis="y",alpha=.14);ax.set_axisbelow(True)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc="outside lower center",ncol=2)
    fig.suptitle("最低 MAE 候選：每場逐時實際／預測長條圖\n四站 XGBoost＋散場訓練 4 倍；平均誤差低不代表最大人潮都抓到",fontsize=14)
    save(fig,OUT/"figures/best_four_hourly.png")

    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True,sharey=True)
    edges=np.arange(-3000,3500,500)
    for ax,f,col,label in [(axes[0],a,"frozen_w4","最低 MAE 候選（四站）"),
                           (axes[1],b,"three_xgboost_none","目前三站基準")]:
        error=f[col].to_numpy()-f.y.to_numpy()
        counts,_=np.histogram(error,bins=edges)
        assert counts.sum()==54
        ax.hist(error,bins=edges,color="#689884",edgecolor="white")
        ax.axvline(0,color="#333333",ls="--")
        ax.set(title=label,xlabel="預測−實際（負＝低估；正＝高估）",ylabel="來源時段筆數",
               xticks=np.arange(-3000,3500,1000))
        ax.grid(axis="y",alpha=.15)
    fig.suptitle("預測誤差直方圖｜每個模型 54 筆；同場各時段不視為獨立活動",fontsize=13)
    save(fig,OUT/"figures/error_histogram.png")

    input_path=ROOT/".local/analysis_v15/analysis_input.csv"
    frame=pd.read_csv(input_path,parse_dates=["event_date"])
    sources[input_path.relative_to(ROOT).as_posix()]=sha(input_path)
    oof_path=ROOT/".local/analysis_v18/temporal_oof.csv"
    oof=pd.read_csv(oof_path,parse_dates=["event_date"])
    tail=[]
    for year in [2024,2025,2026]:
        train=frame.loc[frame.year.lt(year)&frame.station.eq("北門")]
        z=oof.loc[oof.year.lt(year)&oof.model.eq("xgboost")].copy()
        z["residual"]=z.y-z.base_prediction
        u=float(z.residual.quantile(.9));above=z.loc[z.residual>u]
        tail.append(dict(prediction_year=year,training_event_maxima=train.event_date.nunique(),
                         past_oof_events=z.event_date.nunique(),oof_rows=len(z),
                         illustrative_residual_p90=u,exceeding_rows=len(above),
                         exceeding_events=above.event_date.nunique()))
    pd.DataFrame(tail).to_csv(OUT/"tables/evt_support_audit.csv",index=False,encoding="utf-8-sig")
    sources[oof_path.relative_to(ROOT).as_posix()]=sha(oof_path)
    sources[Path(__file__).relative_to(ROOT).as_posix()]=sha(Path(__file__))
    write_json(OUT/"chart_audit.json",dict(passed=True,sources=sources,rows_per_model=54,events=9,
        matched_keys_and_truth=True,rank_scope="saved full-model candidates V14,V15,V16,V18",
        selection_is_retrospective=True,models_retrained=False,deployed=False,summary=summary,evt_support=tail))
    print(json.dumps(dict(summary=summary,evt_support=tail),ensure_ascii=False,indent=2))


if __name__=="__main__":run()
