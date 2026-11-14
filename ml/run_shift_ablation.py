"""
ml/run_shift_ablation.py

Three extra experiments on the source-grouped split (run run_grouped_experiment.py first):
  PART 1  Memory-bank size ablation for the PatchCore-style detector (1000 / 5000 / 20000 patches)
  PART 2  Honest AE + PatchCore combination: each score is z-normalised with the VALIDATION
          normal tiles only (no test statistics), equal weights, fixed in advance
  PART 3  Seabed-shift test: train on ONE normal source (pipe or bg) and measure false alarms
          on the OTHER, unseen normal source
"""

import csv
import random
import re
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
BOOT = 500
SEED = 0

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- data (same rules as before)
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
y = test["y"]
sources = sorted(set(test["prefix"][y == 1]))


# ---------------------------------------------------------------- detectors
def train_ae(x_tr, x_va):
    model = ConvAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    tr, va = torch.from_numpy(x_tr)[:, None], torch.from_numpy(x_va)[:, None]
    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), BATCH):
            b = tr[perm[i:i + BATCH]].to(device)
            opt.zero_grad(); loss_fn(model(b), b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = sum(loss_fn(model(va[i:i + BATCH].to(device)), va[i:i + BATCH].to(device)).item() * len(va[i:i + BATCH])
                    for i in range(0, len(va), BATCH)) / len(va)
        if v < best:
            best = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model


@torch.no_grad()
def ae_scores(model, x):
    out, k = [], max(1, int(0.10 * 64 * 64))
    for i in range(0, len(x), 256):
        b = torch.from_numpy(x[i:i + 256])[:, None].to(device)
        out.append(((model(b) - b) ** 2).flatten(1).topk(k, dim=1).values.mean(1).cpu().numpy())
    return np.concatenate(out)


net = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def patch_features(gray):
    x = torch.from_numpy(gray)[:, None].repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
    x = (x - MEAN) / STD
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    x = net.layer1(x)
    f2 = net.layer2(x)
    f3 = F.interpolate(net.layer3(f2), size=f2.shape[-2:], mode="bilinear", align_corners=False)
    f = F.avg_pool2d(torch.cat([f2, f3], dim=1), 3, stride=1, padding=1)
    return f.flatten(2).transpose(1, 2)


def build_bank(x, size):
    per_tile = -(-size // len(x))
    parts = []
    for i in range(0, len(x), 64):
        for tile_feats in patch_features(x[i:i + 64]):
            parts.append(tile_feats[torch.randperm(tile_feats.shape[0])[:per_tile]])
    return torch.cat(parts)[:size].contiguous()


def pc_scores(x, banks):
    """Score every tile against each bank in `banks`; returns array (len(banks), len(x))."""
    out = np.zeros((len(banks), len(x)))
    i = 0
    for s in range(0, len(x), 32):
        for tile_feats in patch_features(x[s:s + 32]):
            for b, bank in enumerate(banks):
                out[b, i] = float(torch.cdist(tile_feats, bank).min(dim=1).values.max())
            i += 1
    return out


# ---------------------------------------------------------------- metrics
def auroc(yy, s):
    n1 = int(yy.sum()); n0 = len(yy) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    return (ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auprc(yy, s):
    s = s + rng.random(len(s)) * 1e-9
    ys = yy[np.argsort(-s)]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return prec[ys == 1].sum() / max(1, ys.sum())


def cluster_bootstrap(mask, fns):
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


# ---------------------------------------------------------------- load what run_grouped_experiment saved
rows = list(csv.DictReader(open(OUT / "test_scores.csv")))
assert [r["tile"] for r in rows] == test["name"], "test_scores.csv does not match the tile order"
ae_t = np.array([float(r["autoencoder"]) for r in rows])
pc_t = np.array([float(r["patchcore"]) for r in rows])
bank_full = torch.load(OUT / "patchcore_bank_grouped.pt")
ae_model = ConvAutoencoder().to(device)
ae_model.load_state_dict(torch.load(OUT / "conv_ae_grouped.pt", map_location=device)); ae_model.eval()
x_val_norm = valid["x"][valid["y"] == 0]


# ================================================================ PART 1: memory-bank size
print("=== PART 1: PatchCore memory-bank size (AUROC / AUPRC on the clean test split) ===")
perm = torch.randperm(len(bank_full))
sizes = [1000, 5000]
subs = [bank_full[perm[:n]].contiguous() for n in sizes]
sc = pc_scores(test["x"], subs)
print(f"{'bank size':<11}{'AUROC':>8}{'AUPRC':>8}   " + "".join(f"{s:>9}" for s in sources) + "   (per-source AUROC)")
for n, s in list(zip(sizes, sc)) + [(len(bank_full), pc_t)]:
    per = []
    for src in sources:
        mask = (y == 0) | (test["prefix"] == src)
        per.append(auroc(y[mask], s[mask]))
    print(f"{n:<11}{auroc(y, s):>8.3f}{auprc(y, s):>8.3f}   " + "".join(f"{v:>9.3f}" for v in per))


# ================================================================ PART 2: honest combination
print("\n=== PART 2: AE + PatchCore, z-scored with VALIDATION normal tiles only (equal weights) ===")
ae_v = ae_scores(ae_model, x_val_norm)
pc_v = pc_scores(x_val_norm, [bank_full])[0]
z = lambda s, ref: (s - ref.mean()) / (ref.std() + 1e-12)
ens = (z(ae_t, ae_v) + z(pc_t, pc_v)) / 2
cands = {"patchcore": pc_t, "ae+patchcore (val-normalised)": ens}
all_mask = np.ones(len(y), dtype=bool)
d = cluster_bootstrap(all_mask, {**{f"auroc|{k}": (lambda ind, s=s: auroc(y[ind], s[ind])) for k, s in cands.items()},
                                 **{f"auprc|{k}": (lambda ind, s=s: auprc(y[ind], s[ind])) for k, s in cands.items()}})
print(f"{'method':<32}{'AUROC':>8}  {'95% CI':<16}{'AUPRC':>8}  {'95% CI'}")
for k, s in cands.items():
    print(f"{k:<32}{auroc(y, s):>8.3f}  [{ci(d[f'auroc|{k}'])[0]:.3f}, {ci(d[f'auroc|{k}'])[1]:.3f}]   "
          f"{auprc(y, s):>8.3f}  [{ci(d[f'auprc|{k}'])[0]:.3f}, {ci(d[f'auprc|{k}'])[1]:.3f}]")
diff = d["auroc|ae+patchcore (val-normalised)"] - d["auroc|patchcore"]
lo, hi = ci(diff)
print(f"\nPaired AUROC difference (combination minus PatchCore alone): {np.mean(diff):+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
      + ("combination better" if lo > 0 else "not significant" if hi > 0 else "combination worse"))
print("\nPer-source AUROC (point estimate):")
for src in sources:
    mask = (y == 0) | (test["prefix"] == src)
    print(f"   {src:<8} patchcore {auroc(y[mask], pc_t[mask]):.3f}   combination {auroc(y[mask], ens[mask]):.3f}")


# ================================================================ PART 3: seabed-shift test
print("\n=== PART 3: seabed shift. Train on ONE normal source, test on that source AND the other ===")
print("Threshold = 95th percentile of validation-normal scores from the training source (so about 5% false alarms are expected on")
print("unseen tiles of the SAME source). 'other' = normal tiles of the source the model never saw.\n")
print(f"{'trained on':<11}{'detector':<13}{'false alarm: same':>19}{'false alarm: other':>20}{'debris found':>14}{'AUROC vs same':>15}{'AUROC vs other':>16}")
for src in ("pipe", "bg"):
    other = "bg" if src == "pipe" else "pipe"
    xtr = train["x"][(train["y"] == 0) & (train["prefix"] == src)]
    xva = valid["x"][(valid["y"] == 0) & (valid["prefix"] == src)]
    m = train_ae(xtr, xva)
    bank = build_bank(xtr, BANK_SIZE)
    dets = {"autoencoder": (ae_scores(m, xva), ae_scores(m, test["x"])),
            "patchcore": (pc_scores(xva, [bank])[0], pc_scores(test["x"], [bank])[0])}
    same = (y == 0) & (test["prefix"] == src)
    oth = (y == 0) & (test["prefix"] == other)
    deb = y == 1
    for name, (sv, st) in dets.items():
        thr = np.percentile(sv, 95)
        a_same = auroc(np.r_[np.zeros(same.sum()), np.ones(deb.sum())], np.r_[st[same], st[deb]])
        a_oth = auroc(np.r_[np.zeros(oth.sum()), np.ones(deb.sum())], np.r_[st[oth], st[deb]])
        print(f"{src:<11}{name:<13}{(st[same] > thr).mean():>19.1%}{(st[oth] > thr).mean():>20.1%}"
              f"{(st[deb] > thr).mean():>14.1%}{a_same:>15.3f}{a_oth:>16.3f}")
print("\nDone.")