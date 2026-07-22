import glob, os, json
import numpy as np
try:
    import pandas as pd
except Exception as e:
    print("no pandas", e); raise

DATASETS = sorted(glob.glob("data/data/*/*"))
for ds in DATASETS:
    meta = os.path.join(ds,"meta")
    info_p = os.path.join(meta,"info.json")
    n_ep=None; n_frames=None
    if os.path.exists(info_p):
        info=json.load(open(info_p))
        n_ep=info.get("total_episodes"); n_frames=info.get("total_frames")
    # episodes.jsonl
    ep_p=os.path.join(meta,"episodes.jsonl")
    if n_ep is None and os.path.exists(ep_p):
        n_ep=sum(1 for _ in open(ep_p))
    # tasks
    tasks_p=os.path.join(meta,"tasks.jsonl")
    tasks=[]
    if os.path.exists(tasks_p):
        for line in open(tasks_p):
            try: tasks.append(json.loads(line).get("task"))
            except: pass
    # one parquet
    pq = sorted(glob.glob(os.path.join(ds,"data","**","*.parquet"), recursive=True))
    cols=None; tipstats=None; nnan=None; ntot=None
    if pq:
        df=pd.read_parquet(pq[0])
        cols=list(df.columns)
        if "next.tip_target_dist" in df.columns:
            d=np.asarray(df["next.tip_target_dist"].tolist(),dtype=float).ravel()
            valid=d[~np.isnan(d)]
            ntot=len(d); nnan=int(np.isnan(d).sum())
            if len(valid):
                tipstats=(float(valid.min()),float(np.median(valid)),float(valid.max()))
    print(f"\n### {ds}")
    print(f"  episodes={n_ep} frames={n_frames} parquet_files~={len(pq)}")
    print(f"  tasks({len(tasks)}): {tasks[:3]}")
    if cols:
        interesting=[c for c in cols if any(k in c for k in ('tip_target','progress','action','state','annotation'))]
        print(f"  key cols: {interesting}")
    if tipstats:
        print(f"  tip_target_dist (m): min={tipstats[0]:.3f} med={tipstats[1]:.3f} max={tipstats[2]:.3f}  nan={nnan}/{ntot}")
