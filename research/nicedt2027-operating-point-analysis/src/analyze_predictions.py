#!/usr/bin/env python3
"""Reproduce metrics and bootstrap intervals from saved crack-classification predictions.
This re-analyzes stored predictions; it does not retrain or rerun model inference.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score,
    average_precision_score, brier_score_loss)

REQUIRED = {"relative_path","label","group_id","surface","probability","prediction","threshold"}
METRICS = ["accuracy","balanced_accuracy","precision","recall","specificity","f1","mcc","roc_auc","average_precision","brier"]

def validate(df):
    missing = REQUIRED - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty or df[list(REQUIRED)].isna().any().any(): raise ValueError("Empty data or missing required values")
    if df.relative_path.duplicated().any(): raise ValueError("relative_path must be unique")
    if not set(df.label.unique()).issubset({0,1}) or not set(df.prediction.unique()).issubset({0,1}):
        raise ValueError("Labels and predictions must be binary")
    if not np.isfinite(df.probability).all() or ((df.probability<0)|(df.probability>1)).any():
        raise ValueError("Probabilities must be finite and within [0,1]")
    if df.threshold.nunique()!=1: raise ValueError("Expected one fixed threshold")
    if not np.array_equal(df.prediction.to_numpy(), (df.probability.to_numpy() >= df.threshold.to_numpy()).astype(int)):
        raise ValueError("Prediction column is inconsistent with probability >= threshold")
    if df.groupby("group_id").surface.nunique().max()!=1: raise ValueError("A group maps to multiple surfaces")
    basename=df.relative_path.str.rsplit("/", n=1).str[-1]
    prefix=pd.to_numeric(basename.str.extract(r"^(\d+)")[0], errors="coerce")
    if prefix.isna().any() or not np.array_equal(prefix.to_numpy(dtype=np.int64),df.group_id.to_numpy(dtype=np.int64)):
        raise ValueError("group_id does not match numeric filename prefix")
    return {"rows":int(len(df)), "unique_paths":int(df.relative_path.nunique()),
        "groups":int(df.group_id.nunique()), "labels":{str(k):int(v) for k,v in df.label.value_counts().sort_index().items()},
        "surface_patches":{str(k):int(v) for k,v in df.surface.value_counts().items()},
        "surface_groups":{str(k):int(v) for k,v in df.groupby("surface").group_id.nunique().items()},
        "threshold":float(df.threshold.iloc[0]),
        "caveat":"group_id is derived from numeric filename prefixes; original photographs and model checkpoint are not independently verified."}

def metric_values(df, weights=None):
    y=df.label.to_numpy(dtype=int); pred=df.prediction.to_numpy(dtype=int); prob=df.probability.to_numpy(dtype=float)
    w=np.ones(len(df), dtype=float) if weights is None else np.asarray(weights,dtype=float)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1],sample_weight=w).ravel()
    sens=tp/(tp+fn) if tp+fn else float("nan")
    spec=tn/(tn+fp) if tn+fp else float("nan")
    try: auc=float(roc_auc_score(y,prob,sample_weight=w))
    except ValueError: auc=float("nan")
    try: ap=float(average_precision_score(y,prob,sample_weight=w))
    except ValueError: ap=float("nan")
    return {"accuracy":float(accuracy_score(y,pred,sample_weight=w)),
        "balanced_accuracy":float((sens+spec)/2),
        "precision":float(precision_score(y,pred,sample_weight=w,zero_division=0)),
        "recall":float(recall_score(y,pred,sample_weight=w,zero_division=0)),
        "specificity":float(spec),
        "f1":float(f1_score(y,pred,sample_weight=w,zero_division=0)),
        "mcc":float(matthews_corrcoef(y,pred,sample_weight=w)),
        "roc_auc":auc,"average_precision":ap,
        "brier":float(brier_score_loss(y,prob,sample_weight=w))}

def bootstrap(df, n=2000, seed=20261019, unit="group", stratify_surface=True):
    rng=np.random.default_rng(seed); values={m:np.empty(n) for m in METRICS}
    groups=df.group_id.to_numpy()
    unique=np.unique(groups)
    group_to_rows={g:np.flatnonzero(groups==g) for g in unique}
    if unit=="patch":
        strata={"all":np.arange(len(df))}
        inverse=None
    else:
        strata={s:np.array(g.group_id.unique()) for s,g in df.groupby("surface")} if stratify_surface else {"all":unique}
    for b in range(n):
        if unit=="patch":
            selected=rng.choice(len(df), size=len(df), replace=True)
            w=np.bincount(selected,minlength=len(df)).astype(float)
        else:
            counts={}
            for surface, ids in strata.items():
                drawn=rng.choice(ids,size=len(ids),replace=True)
                u,c=np.unique(drawn,return_counts=True)
                for gid,count in zip(u,c): counts[gid]=counts.get(gid,0)+int(count)
            w=np.array([counts.get(g,0) for g in groups],dtype=float)
        m=metric_values(df,w)
        for name in METRICS: values[name][b]=m[name]
    return {name:(float(np.nanquantile(arr,.025)),float(np.nanquantile(arr,.975))) for name,arr in values.items()}

def run(input_csv, outdir, figdir, n=2000, seed=20261019):
    df=pd.read_csv(input_csv); audit=validate(df)
    outdir=Path(outdir); figdir=Path(figdir); outdir.mkdir(parents=True,exist_ok=True); figdir.mkdir(parents=True,exist_ok=True)
    point=metric_values(df); ci=bootstrap(df,n,seed,"group",True)
    majority=float((df.label==0).mean())
    rows=[{"metric":m,"estimate":point[m],"ci_lower":ci[m][0],"ci_upper":ci[m][1],"bootstrap_replicates":n,"seed":seed,"resampling_unit":"filename-prefix group, stratified by surface"} for m in METRICS]
    rows.extend([
        {"metric":"majority_class_accuracy_baseline","estimate":majority,"ci_lower":np.nan,"ci_upper":np.nan,"bootstrap_replicates":0,"seed":"","resampling_unit":"observed label prevalence"},
        {"metric":"accuracy_margin_over_majority_baseline_pp","estimate":(point["accuracy"]-majority)*100,"ci_lower":np.nan,"ci_upper":np.nan,"bootstrap_replicates":0,"seed":"","resampling_unit":"percentage points"}])
    pd.DataFrame(rows).to_csv(outdir/"overall_metrics.csv",index=False)
    surface_rows=[]
    for i,(surface,g) in enumerate(df.groupby("surface")):
        p=metric_values(g); ci_s=bootstrap(g,n,seed+i+1,"group",False)
        surface_rows.append({"surface":surface,"patches":len(g),"source_groups":g.group_id.nunique(),"recall":p["recall"],"recall_ci_low":ci_s["recall"][0],"recall_ci_high":ci_s["recall"][1],"specificity":p["specificity"],"specificity_ci_low":ci_s["specificity"][0],"specificity_ci_high":ci_s["specificity"][1],"balanced_accuracy":p["balanced_accuracy"],"roc_auc":p["roc_auc"],"average_precision":p["average_precision"]})
    surface=pd.DataFrame(surface_rows).sort_values("surface"); surface.to_csv(outdir/"surface_metrics.csv",index=False)
    patch_ci=bootstrap(df,n,seed+100,"patch",False)
    compare=[]
    for m in ["accuracy","balanced_accuracy","precision","recall","specificity","f1","mcc","brier"]:
        pw=patch_ci[m][1]-patch_ci[m][0]; gw=ci[m][1]-ci[m][0]
        compare.append({"metric":m,"estimate":point[m],"patch_ci_low":patch_ci[m][0],"patch_ci_high":patch_ci[m][1],"patch_ci_width":pw,"source_group_ci_low":ci[m][0],"source_group_ci_high":ci[m][1],"source_group_ci_width":gw,"width_ratio_source_to_patch":gw/pw if pw else np.nan})
    pd.DataFrame(compare).to_csv(outdir/"patch_vs_source_bootstrap_comparison.csv",index=False)
    report={**audit,"metrics":point,"majority_class_accuracy_baseline":majority,
        "accuracy_margin_over_majority_baseline_pp":(point["accuracy"]-majority)*100,
        "bootstrap":{"replicates":n,"seed":seed,"CI":"percentile 95%","primary_unit":"filename-prefix group"},
        "interpretation_limit":"Saved-prediction analysis only: no model retraining/inference; group IDs are filename-prefix proxies and require provenance confirmation."}
    (outdir/"integrity_and_analysis_report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    (outdir/"reproduction_config.json").write_text(json.dumps({"bootstrap_replicates":n,"seed":seed,"patch_seed":seed+100,"CI":"percentile 95%"},indent=2))
    try:
        import matplotlib.pyplot as plt
        order=["Deck","Pavement","Wall"]; s=surface.set_index("surface").loc[order]
        y=s.recall.to_numpy()*100; lo=(s.recall-s.recall_ci_low).to_numpy()*100; hi=(s.recall_ci_high-s.recall).to_numpy()*100
        fig,ax=plt.subplots(figsize=(7,4)); bars=ax.bar(order,y,yerr=np.vstack([lo,hi]),capsize=6)
        for i,(bar,v) in enumerate(zip(bars,y)): ax.text(bar.get_x()+bar.get_width()/2,v+hi[i]+1.5,f"{v:.1f}%",ha="center",fontsize=9)
        ax.set_ylabel("Crack recall (%)"); ax.set_title(f"Recall at frozen threshold ({df.threshold.iloc[0]:.2f})")
        ax.set_ylim(0,max((y+hi).max()+8,50)); ax.grid(axis="y",alpha=.25); fig.tight_layout()
        fig.savefig(figdir/"surface_recall_with_group_bootstrap_ci.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    except ImportError:
        print("matplotlib missing; CSV results were still written.")
    print(json.dumps(report,indent=2)); print(f"Results written to {outdir}")
    return report

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="data/sdnet2018_external_predictions_sanitized.csv")
    ap.add_argument("--output-dir",default="results"); ap.add_argument("--figure-dir",default="figures")
    ap.add_argument("--bootstrap",type=int,default=2000); ap.add_argument("--seed",type=int,default=20261019)
    a=ap.parse_args()
    if a.bootstrap<100: ap.error("--bootstrap must be at least 100")
    run(a.input,a.output_dir,a.figure_dir,a.bootstrap,a.seed)
