import glob, os, csv, math
import numpy as np

def load_csv(path):
    rows=[]
    with open(path) as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def fnum(x):
    try:
        v=float(x)
        if math.isnan(v): return None
        return v
    except: return None

def summarize(path):
    rows=load_csv(path)
    if not rows: return None
    # final row metrics
    def col(name):
        return [fnum(r.get(name)) for r in rows]
    dta=[v for v in col("dist_target_actual") if v is not None]
    dtc=[v for v in col("dist_target_cmd") if v is not None]
    tel2=[v for v in col("track_err_l2") if v is not None]
    rel_l2=[]
    for r in rows:
        ex,ey,ez=fnum(r.get("rel_err_x")),fnum(r.get("rel_err_y")),fnum(r.get("rel_err_z"))
        if None not in (ex,ey,ez):
            rel_l2.append(math.sqrt(ex*ex+ey*ey+ez*ez))
    rotdeg=[v for v in col("rel_err_rot_deg") if v is not None]
    # final target distance = last non-nan dist_target_actual
    final_dta = dta[-1] if dta else None
    min_dta = min(dta) if dta else None
    return dict(
        n=len(rows),
        final_dist_target=final_dta,
        min_dist_target=min_dta,
        median_dist_target=(float(np.median(dta)) if dta else None),
        track_err_l2_mean=(float(np.mean(tel2)) if tel2 else None),
        rel_err_l2_mean=(float(np.mean(rel_l2)) if rel_l2 else None),
        rel_err_l2_median=(float(np.median(rel_l2)) if rel_l2 else None),
        rot_err_deg_mean=(float(np.mean(rotdeg)) if rotdeg else None),
        rot_err_deg_median=(float(np.median(rotdeg)) if rotdeg else None),
    )

def mm(x):
    return f"{x*1000:.1f}mm" if isinstance(x,float) else "  -  "
def dd(x):
    return f"{x:.1f}deg" if isinstance(x,float) else "  -  "

all_csv=sorted(glob.glob("vla_eval_results/**/*.csv", recursive=True))
# group by folder
from collections import defaultdict
groups=defaultdict(list)
for p in all_csv:
    folder=os.path.dirname(p) or "(root)"
    groups[folder].append(p)

for folder in sorted(groups):
    print(f"\n===== {folder}  ({len(groups[folder])} csv) =====")
    print(f"{'scenario':<42} {'finalDist':>9} {'minDist':>8} {'medDist':>8} {'trackL2':>8} {'relErrL2':>9} {'rotErr':>7}")
    agg=defaultdict(list)
    for p in sorted(groups[folder]):
        s=summarize(p)
        if not s: 
            continue
        name=os.path.basename(p).replace("vla_metrics_","").replace("vla_planner_","").replace(".csv","")
        name=name[:40]
        print(f"{name:<42} {mm(s['final_dist_target']):>9} {mm(s['min_dist_target']):>8} {mm(s['median_dist_target']):>8} {mm(s['track_err_l2_mean']):>8} {mm(s['rel_err_l2_mean']):>9} {dd(s['rot_err_deg_mean']):>7}")
        for k,v in s.items():
            if isinstance(v,float): agg[k].append(v)
    if agg:
        print("-"*95)
        print(f"{'MEAN over scenarios':<42} {mm(np.mean(agg['final_dist_target'])):>9} {mm(np.mean(agg['min_dist_target'])):>8} {mm(np.mean(agg['median_dist_target'])):>8} {mm(np.mean(agg['track_err_l2_mean'])):>8} {mm(np.mean(agg['rel_err_l2_mean'])):>9} {dd(np.mean(agg['rot_err_deg_mean'])):>7}")
