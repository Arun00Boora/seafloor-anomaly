import re
from collections import defaultdict
from pathlib import Path

DST = Path(__file__).resolve().parent / "data" / "tiles_grouped"

def group_key(orig_split, stem):
    if stem.startswith(("synth_", "mine_")):
        return f"{orig_split}|{stem}"
    if stem.startswith("wreckR_"):
        return re.sub(r"\.rf\.[0-9a-fA-F]+$", "", stem)
    return re.sub(r"(_y\d+)?_x\d+$", "", stem)

groups = defaultdict(set)
tiles = defaultdict(int)
for split in ("train", "valid", "test"):
    for cls in ("normal", "anomalous"):
        for p in (DST / split / cls).glob("*.png"):
            orig, stem = p.stem.split("__", 1)
            prefix = stem.split("_")[0]
            groups[(split, prefix)].add(group_key(orig, stem))
            tiles[(split, prefix)] += 1

print(f"{'split':<7}{'source':<9}{'tiles':>7}{'source images':>15}")
for key in sorted(groups):
    print(f"{key[0]:<7}{key[1]:<9}{tiles[key]:>7}{len(groups[key]):>15}")