"""
ml/make_grouped_split.py

Pools every tile from the dataset's train/valid/test folders and re-assigns them to a NEW
train/valid/test split BY SOURCE IMAGE, so no source image (and no near-duplicate view of it)
appears in two splits. The original folders are not modified; copies go to ml/data/tiles_grouped.

Group rules (what counts as "the same source image"):
  pipe_/bg_/wreckA_ : the name without its tile position (_yNNN_xNNN or _xNNN)
  wreckR_           : the name without the Roboflow augmentation hash (.rf.<hash>)
  synth_/mine_      : numbered afresh in every original split, so each tile is its own group
Split is stratified per source (pipe, bg, wreckA, wreckR, mine, synth), about 70/10/20 by tile count.
"""

import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

SEED = 42
FRACTIONS = {"test": 0.20, "valid": 0.10}      # the rest goes to train
ROOT = Path(__file__).resolve().parent / "data"
SRC = ROOT / "tiles_drishti"
DST = ROOT / "tiles_grouped"


def group_key(orig_split, stem):
    if stem.startswith(("synth_", "mine_")):
        return f"{orig_split}|{stem}"
    if stem.startswith("wreckR_"):
        return re.sub(r"\.rf\.[0-9a-fA-F]+$", "", stem)
    return re.sub(r"(_y\d+)?_x\d+$", "", stem)


# 1. Collect every tile with its original split, class folder and group
tiles = []   # (path, orig_split, class_name, prefix, group)
for split in ("train", "valid", "test"):
    for cls in ("normal", "anomalous"):
        for p in sorted((SRC / split / cls).glob("*.png")):
            tiles.append((p, split, cls, p.stem.split("_")[0], group_key(split, p.stem)))
print(f"Pooled {len(tiles)} tiles")

# 2. Assign whole groups to splits, separately for each source prefix
by_prefix = defaultdict(lambda: defaultdict(list))
for t in tiles:
    by_prefix[t[3]][t[4]].append(t)

rng = random.Random(SEED)
assignment = {}   # group -> new split
for prefix, groups in sorted(by_prefix.items()):
    total = sum(len(v) for v in groups.values())
    keys = sorted(groups)
    rng.shuffle(keys)
    counts = {"test": 0, "valid": 0}
    for k in keys:
        size = len(groups[k])
        if counts["test"] < FRACTIONS["test"] * total:
            assignment[k] = "test"; counts["test"] += size
        elif counts["valid"] < FRACTIONS["valid"] * total:
            assignment[k] = "valid"; counts["valid"] += size
        else:
            assignment[k] = "train"

# 3. Copy into the new folders (names prefixed with the original split to avoid collisions)
if DST.exists():
    shutil.rmtree(DST)
for split in ("train", "valid", "test"):
    for cls in ("normal", "anomalous"):
        (DST / split / cls).mkdir(parents=True, exist_ok=True)

report = defaultdict(int)
group_sets = defaultdict(set)
for p, orig, cls, prefix, g in tiles:
    new = assignment[g]
    shutil.copy2(p, DST / new / cls / f"{orig}__{p.name}")
    report[(new, cls, prefix)] += 1
    group_sets[new].add(g)

# 4. Report
print("\nTiles per split / class / source:")
for key in sorted(report):
    print(f"   {key[0]:<6}{key[1]:<10}{key[2]:<8}{report[key]:>6}")

print("\nTotals:")
for split in ("train", "valid", "test"):
    n_norm = sum(v for (s, c, _), v in report.items() if s == split and c == "normal")
    n_anom = sum(v for (s, c, _), v in report.items() if s == split and c == "anomalous")
    print(f"   {split:<6} normal {n_norm:>5}   anomalous {n_anom:>5}   debris rate {n_anom / max(1, n_norm + n_anom):.0%}   groups {len(group_sets[split])}")

overlap = (group_sets["train"] & group_sets["test"]) | (group_sets["train"] & group_sets["valid"]) | (group_sets["valid"] & group_sets["test"])
print(f"\nSource groups shared between splits: {len(overlap)}  (must be 0)")
print(f"Saved to {DST}")