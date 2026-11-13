"""
ml/patchcore_scores.py

PatchCore-style detector using a PRETRAINED ResNet18 (no training on sonar needed).

1. Take every normal TRAIN tile, run it through ResNet18, and keep mid-level feature
   "patches" (layer2 + layer3). Each patch describes a small region of seabed.
2. Randomly keep a subset of those patches: the "memory bank" of what normal looks like.
   (Real PatchCore uses a smarter coreset. Random is simpler and good enough here.)
3. For each TEST tile, every patch is compared to its nearest normal patch.
   Tile score = the distance of its MOST unusual patch (max). Far from anything normal = suspicious.

Output: ml/data/scores/patchcore_test.csv  (tile_id, label, score)
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parent.parent          # seafloor-anomaly
TRAIN_NORMAL = ROOT / "ml" / "data" / "tiles_drishti" / "train" / "normal"
TEST_DIR = ROOT / "deploy_data" / "tiles" / "test"     # has normal/ and anomalous/
OUT_CSV = ROOT / "ml" / "data" / "scores" / "patchcore_test.csv"
BANK_PATH = ROOT / "ml" / "data" / "patchcore_bank.pt"

BANK_SIZE = 20000       # patches kept in memory bank (more = slower but more faithful)
INPUT_SIZE = 128        # tiles are 64x64; upscaling gives the CNN a finer feature grid
SEED = 0

random.seed(SEED)
torch.manual_seed(SEED)
device = "cpu"

# ---- Feature extractor: ResNet18 up to layer3, frozen ----
net = resnet18(weights=ResNet18_Weights.DEFAULT).eval().to(device)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def patch_features(batch: torch.Tensor) -> torch.Tensor:
    """batch: (B,3,H,W) in 0..1  ->  (B, num_patches, 384) patch descriptors."""
    x = F.interpolate(batch, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
    x = (x - MEAN) / STD
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    x = net.layer1(x)
    f2 = net.layer2(x)                       # (B,128,16,16)
    f3 = net.layer3(f2)                      # (B,256,8,8)
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([f2, f3], dim=1)           # (B,384,16,16)
    f = F.avg_pool2d(f, 3, stride=1, padding=1)   # blend each patch with its neighbours
    return f.flatten(2).transpose(1, 2)      # (B,256,384)


def load_batch(paths):
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((64, 64), Image.LANCZOS)
        imgs.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1))
    return torch.stack(imgs)


def batches(paths, size=64):
    for i in range(0, len(paths), size):
        yield paths[i:i + size]


# ---- 1. Build the memory bank from normal training tiles ----
train_paths = sorted(TRAIN_NORMAL.glob("*.png"))
print(f"Normal train tiles found: {len(train_paths)} in {TRAIN_NORMAL}")
if not train_paths:
    raise SystemExit("No training tiles found. Check TRAIN_NORMAL at the top of this script.")

all_patches = []
for chunk in batches(train_paths):
    all_patches.append(patch_features(load_batch(chunk)).reshape(-1, 384))
all_patches = torch.cat(all_patches)
print(f"Total normal patches: {len(all_patches)}")
keep = torch.randperm(len(all_patches))[:BANK_SIZE]
bank = all_patches[keep].contiguous()
torch.save(bank, BANK_PATH)
print(f"Memory bank: {tuple(bank.shape)} saved to {BANK_PATH}")

# ---- 2. Score the test tiles ----
test_items = []
for label_name, label in (("normal", 0), ("anomalous", 1)):
    for p in sorted((TEST_DIR / label_name).glob("*.png")):
        test_items.append((p, label))
print(f"Test tiles: {len(test_items)}")

rows = []
for chunk in batches(test_items, 32):
    feats = patch_features(load_batch([p for p, _ in chunk]))     # (B,256,384)
    for (p, label), tile_feats in zip(chunk, feats):
        d = torch.cdist(tile_feats, bank)                         # (256, BANK_SIZE)
        nearest = d.min(dim=1).values                             # distance to closest normal patch
        rows.append((p.stem, label, float(nearest.max())))        # most unusual patch

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tile_id", "label", "score"])
    w.writerows(rows)
print(f"Saved {len(rows)} scores to {OUT_CSV}")

# ---- 3. Quick precision check (all 700 test tiles, ~65% debris, so compare methods, not raw numbers) ----
rows.sort(key=lambda r: -r[2])
labels = np.array([r[1] for r in rows])
cum = np.cumsum(labels)
print("\nPatchCore-style precision on the full test set:")
for frac in (0.05, 0.10, 0.20, 0.30):
    k = round(len(rows) * frac)
    print(f"  top {int(frac*100):>2}% ({k} tiles): precision {cum[k-1]/k:.1%}, recall {cum[k-1]/labels.sum():.1%}")