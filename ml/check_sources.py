import json
import numpy as np
from pathlib import Path

root = Path(r"D:\sonar project\seafloor-anomaly")
data = json.loads((root / "deploy_data" / "index" / "test_ranked.json").read_text())
recs = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))

SCORES = {"autoencoder": "ae_score", "brightness": "baseline_score", "patchcore": "pc_score"}

def prefix(r):
    return str(r["tile_id"]).split("_")[0]

def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq

groups = {}
for r in recs:
    groups.setdefault(prefix(r), []).append(r)

print("== Median score per source (0..1 scale) ==")
print(f"{'source':<8}{'n':>5}   {'autoenc':>8}{'bright':>8}{'patchc':>8}")
for p, rs in sorted(groups.items()):
    meds = [np.median([x[f] for x in rs]) for f in SCORES.values()]
    print(f"{p:<8}{len(rs):>5}   " + "".join(f"{m:>8.2f}" for m in meds))

normals = {p: rs for p, rs in groups.items() if p in ("pipe", "bg")}
debris = {p: rs for p, rs in groups.items() if p not in ("pipe", "bg")}

print("\n== AUROC: each debris source vs normal tiles (0.5 = chance, 1.0 = perfect) ==")
for name, key in SCORES.items():
    print(f"\n{name}")
    print(f"   {'debris source':<10}{'vs all normal':>15}{'vs pipe only':>15}{'vs bg only':>13}")
    for dp, drs in sorted(debris.items()):
        d = [x[key] for x in drs]
        row = [auroc(d, [x[key] for rs in normals.values() for x in rs])]
        for np_ in ("pipe", "bg"):
            row.append(auroc(d, [x[key] for x in normals.get(np_, [])]) if np_ in normals else float("nan"))
        print(f"   {dp:<10}{row[0]:>15.3f}{row[1]:>15.3f}{row[2]:>13.3f}")