"""
ml/run_grouped_experiment.py

Clean experiment on the source-grouped split (ml/data/tiles_grouped):
  1. trains the convolutional autoencoder on NORMAL train tiles (validated on normal valid tiles)
  2. builds the PatchCore-style memory bank from the same normal train tiles
  3. scores every test tile with: autoencoder, brightness, PatchCore, and AE+PatchCore average
  4. reports AUROC / AUPRC with cluster-bootstrap confidence intervals (whole source images are
     resampled, because tiles from one image are not independent), per-source AUROC, precision
     at the top of the ranking, review effort saved, and the rare-debris simulation.
Outputs go to ml/data/grouped_run/ (scores CSV, results JSON, models).
"""

import csv
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

from models.conv_ae import ConvAutoencoder

ROOT = Path(__file__).resolve().parent / "data"
TILES = ROOT / "tiles_grouped"
OUT = ROOT / "grouped_run"
EPOCHS, BATCH, LR = 15, 64, 1e-3
BANK_SIZE = 20000
BOOT = 500                       # bootstrap resamples
FRACS = [0.05, 0.10, 0.20, 0.30]
RECALLS = [0.80, 0.90]
RARE_RATES = [0.10, 0.05, 0.02]
RARE_TRIALS = 300
SEED = 0

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- data
def group_key(orig_split, stem):
    if stem.startswith(("synth_", "mine_")):
        return f"{orig_split}|{stem}"
    if stem.startswith("wreckR_"):
        return re.sub(r"\.rf\.[0-9a-fA-F]+$", "", stem)
    return re.sub(r"(_y\d+)?_x\d+$", "", stem)


def load_split(split):
    imgs, labels, prefixes, groups, names = [], [], [], [], []
    for label, cls in ((0, "normal"), (1, "anomalous")):
        for p in sorted((TILES / split / cls).glob("*.png")):
            im = Image.open(p).convert("L")
            if im.size != (64, 64):
                im = im.resize((64, 64), Image.LANCZOS)
            imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
            orig, stem = p.stem.split("__", 1)
            labels.append(label); prefixes.append(stem.split("_")[0])
            groups.append(group_key(orig, stem)); names.append(p.name)
    return {"x": np.stack(imgs), "y": np.array(labels), "prefix": np.array(prefixes),
            "group": np.array(groups), "name": names}


train, valid, test = load_split("train"), load_split("valid"), load_split("test")
x_train = train["x"][train["y"] == 0]
x_valid = valid["x"][valid["y"] == 0]
print(f"Normal train tiles: {len(x_train)}, normal valid tiles: {len(x_valid)}, test tiles: {len(test['y'])} "
      f"({int(test['y'].sum())} debris, {int((test['y'] == 0).sum())} normal)")


# ---------------------------------------------------------------- 1. autoencoder
t0 = time.time()
model = ConvAutoencoder().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
tr = torch.from_numpy(x_train)[:, None]
va = torch.from_numpy(x_valid)[:, None]
best, best_state = float("inf"), None
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(tr))
    for i in range(0, len(tr), BATCH):
        b = tr[perm[i:i + BATCH]].to(device)
        opt.zero_grad()
        loss_fn(model(b), b).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        v = sum(loss_fn(model(va[i:i + BATCH].to(device)), va[i:i + BATCH].to(device)).item() * len(va[i:i + BATCH])
                for i in range(0, len(va), BATCH)) / len(va)
    if v < best:
        best = v
        best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    print(f"  AE epoch {epoch:2d}/{EPOCHS}  val_loss {v:.5f}")
model.load_state_dict(best_state); model.eval()
torch.save(best_state, OUT / "conv_ae_grouped.pt")
print(f"Autoencoder trained in {time.time() - t0:.0f} s (best val_loss {best:.5f})")


@torch.no_grad()
def ae_scores(x):
    out = []
    k = max(1, int(0.10 * 64 * 64))
    for i in range(0, len(x), 256):
        b = torch.from_numpy(x[i:i + 256])[:, None].to(device)
        err = ((model(b) - b) ** 2).flatten(1)
        out.append(err.topk(k, dim=1).values.mean(1).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- 2. PatchCore-style detector
net = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def patch_features(gray):                     # gray: (B,64,64) numpy 0..1
    x = torch.from_numpy(gray)[:, None].repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
    x = (x - MEAN) / STD
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    x = net.layer1(x)
    f2 = net.layer2(x)
    f3 = F.interpolate(net.layer3(f2), size=f2.shape[-2:], mode="bilinear", align_corners=False)
    f = F.avg_pool2d(torch.cat([f2, f3], dim=1), 3, stride=1, padding=1)
    return f.flatten(2).transpose(1, 2)       # (B,256,384)


t0 = time.time()
per_tile = -(-BANK_SIZE // len(x_train))
bank_parts = []
for i in range(0, len(x_train), 64):
    feats = patch_features(x_train[i:i + 64])
    for tile_feats in feats:
        bank_parts.append(tile_feats[torch.randperm(tile_feats.shape[0])[:per_tile]])
bank = torch.cat(bank_parts)[:BANK_SIZE].contiguous()
torch.save(bank, OUT / "patchcore_bank_grouped.pt")
print(f"PatchCore memory bank {tuple(bank.shape)} built in {time.time() - t0:.0f} s")


def pc_scores(x):
    out = []
    for i in range(0, len(x), 32):
        for tile_feats in patch_features(x[i:i + 32]):
            out.append(float(torch.cdist(tile_feats, bank).min(dim=1).values.max()))
    return np.array(out)


# ---------------------------------------------------------------- 3. score the test tiles
t0 = time.time()
y = test["y"]
ae = ae_scores(test["x"])
bright = test["x"].reshape(len(y), -1).max(axis=1)
pc = pc_scores(test["x"])
mm = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-12)
SCORES = {"autoencoder": ae, "brightness": bright, "patchcore": pc, "ae+patchcore": (mm(ae) + mm(pc)) / 2}
print(f"Test tiles scored in {time.time() - t0:.0f} s")

with open(OUT / "test_scores.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tile", "source", "group", "label", "autoencoder", "brightness", "patchcore"])
    for i in range(len(y)):
        w.writerow([test["name"][i], test["prefix"][i], test["group"][i], int(y[i]), ae[i], bright[i], pc[i]])


# ---------------------------------------------------------------- 4. metrics
def auroc(yy, s):
    n1 = int(yy.sum()); n0 = len(yy) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    return (ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auprc(yy, s):
    s = s + rng.random(len(s)) * 1e-9             # random tie-breaking
    order = np.argsort(-s)
    ys = yy[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return prec[ys == 1].sum() / max(1, ys.sum())


def ranking_stats(yy, s, trials=50):
    n, n1 = len(yy), int(yy.sum())
    prec, eff = np.zeros(len(FRACS)), np.zeros(len(RECALLS))
    for _ in range(trials):
        cum = np.cumsum(yy[np.lexsort((rng.random(n), -s))])
        for j, fr in enumerate(FRACS):
            k = max(1, round(n * fr)); prec[j] += cum[k - 1] / k
        for j, t in enumerate(RECALLS):
            k = int(np.argmax(cum >= t * n1)) + 1; eff[j] += 1 - k / n
    return prec / trials, eff / trials


def cluster_bootstrap(mask, fns):
    """fns: {name: f(indices)->value}. Resamples whole source images among tiles in `mask`."""
    idx = np.flatnonzero(mask)
    by_group = defaultdict(list)
    for i in idx:
        by_group[test["group"][i]].append(i)
    arrs = [np.array(v) for v in by_group.values()]
    draws = {k: [] for k in fns}
    for _ in range(BOOT):
        pick = rng.integers(0, len(arrs), len(arrs))
        ind = np.concatenate([arrs[j] for j in pick])
        for k, fn in fns.items():
            draws[k].append(fn(ind))
    return {k: np.array(v) for k, v in draws.items()}


def ci(a):
    return np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)


results = {"n_test": int(len(y)), "n_debris": int(y.sum())}
all_mask = np.ones(len(y), dtype=bool)

# --- overall AUROC / AUPRC with CIs, and paired differences against PatchCore
fns = {}
for m, s in SCORES.items():
    fns[f"auroc|{m}"] = (lambda ind, s=s: auroc(y[ind], s[ind]))
    fns[f"auprc|{m}"] = (lambda ind, s=s: auprc(y[ind], s[ind]))
d = cluster_bootstrap(all_mask, fns)
print("\n=== OVERALL on the clean test split (62% debris; cluster bootstrap over source images, 95% CI) ===")
print(f"{'method':<15}{'AUROC':>8}  {'95% CI':<16}{'AUPRC':>8}  {'95% CI':<16}")
overall = {}
for m, s in SCORES.items():
    a, p = auroc(y, s), auprc(y, s)
    overall[m] = {"auroc": a, "auroc_ci": ci(d[f"auroc|{m}"]), "auprc": p, "auprc_ci": ci(d[f"auprc|{m}"])}
    print(f"{m:<15}{a:>8.3f}  [{overall[m]['auroc_ci'][0]:.3f}, {overall[m]['auroc_ci'][1]:.3f}]   "
          f"{p:>8.3f}  [{overall[m]['auprc_ci'][0]:.3f}, {overall[m]['auprc_ci'][1]:.3f}]")
results["overall"] = overall

print("\nPaired AUROC difference (PatchCore minus other), 95% CI:")
for other in ("autoencoder", "brightness"):
    diff = d["auroc|patchcore"] - d[f"auroc|{other}"]
    lo, hi = ci(diff)
    flag = "PatchCore better" if lo > 0 else ("not significant" if hi > 0 else "PatchCore worse")
    print(f"   vs {other:<12} {np.mean(diff):+.3f}  [{lo:+.3f}, {hi:+.3f}]  {flag}")

# --- per-source AUROC (each debris source against ALL normal tiles)
print("\n=== PER-SOURCE AUROC (debris source vs all normal tiles), 95% CI ===")
sources = sorted(set(test["prefix"][y == 1]))
per_source = {}
for src in sources:
    mask = (y == 0) | (test["prefix"] == src)
    d = cluster_bootstrap(mask, {m: (lambda ind, s=s: auroc(y[ind], s[ind])) for m, s in SCORES.items()})
    n_img = len(set(test["group"][test["prefix"] == src]))
    print(f"\n{src}  ({int((test['prefix'] == src).sum())} tiles from {n_img} source images)")
    per_source[src] = {}
    for m, s in SCORES.items():
        a = auroc(y[mask], s[mask]); lo, hi = ci(d[m])
        per_source[src][m] = {"auroc": a, "ci": (lo, hi)}
        print(f"   {m:<15}{a:>7.3f}  [{lo:.3f}, {hi:.3f}]")
results["per_source"] = per_source

# --- ranking quality at the top and review effort
print("\n=== PRECISION IN TOP K% OF THE RANKING and REVIEW EFFORT SAVED (62% debris) ===")
print(f"{'method':<15}" + "".join(f"{'top ' + str(int(f * 100)) + '%':>9}" for f in FRACS)
      + "".join(f"{'skip@' + str(int(t * 100)) + '%':>10}" for t in RECALLS))
rank = {}
for m, s in SCORES.items():
    p, e = ranking_stats(y, s)
    rank[m] = {"precision": p.tolist(), "effort_saved": e.tolist()}
    print(f"{m:<15}" + "".join(f"{v:>9.1%}" for v in p) + "".join(f"{v:>10.1%}" for v in e))
results["ranking"] = rank

# --- rare-debris simulation: keep all normal tiles, thin debris tiles, 300 random draws
print("\n=== RARE DEBRIS: precision in top 5% / top 10% (mean of 300 random draws) ===")
normal_idx, debris_idx = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
rare = {}
for rate in RARE_RATES:
    nd = max(1, round(rate / (1 - rate) * len(normal_idx)))
    print(f"\ndebris rate {rate:.0%}  ({nd} debris tiles + {len(normal_idx)} normal tiles)")
    rare[rate] = {}
    res = {m: np.zeros((RARE_TRIALS, 2)) for m in SCORES}
    for t in range(RARE_TRIALS):
        chosen = np.concatenate([normal_idx, rng.choice(debris_idx, nd, replace=False)])
        jitter = rng.random(len(chosen)) * 1e-9
        for m, s in SCORES.items():
            cum = np.cumsum(y[chosen][np.argsort(-(s[chosen] + jitter))])
            for j, fr in enumerate((0.05, 0.10)):
                k = max(1, round(len(chosen) * fr)); res[m][t, j] = cum[k - 1] / k
    for m in SCORES:
        mean = res[m].mean(0)
        rare[rate][m] = mean.tolist()
        print(f"   {m:<15} top 5%: {mean[0]:.1%}   top 10%: {mean[1]:.1%}")
results["rare_debris"] = {str(k): v for k, v in rare.items()}

(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
print(f"\nSaved scores and results to {OUT}")