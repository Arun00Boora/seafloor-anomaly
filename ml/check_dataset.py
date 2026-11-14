import re
from collections import defaultdict
from pathlib import Path

root = Path(r"D:\sonar project\seafloor-anomaly")
base = root / "ml" / "data" / "tiles_drishti"
test_dir = root / "deploy_data" / "tiles" / "test"

def source(stem):
    return re.sub(r"(_y\d+)?_x\d+$", "", stem)

# Collect (split, class) -> list of tile stems
groups = defaultdict(list)
for split in sorted(p for p in base.iterdir() if p.is_dir()):
    for f in split.rglob("*.png"):
        groups[(split.name, f.parent.name)].append(f.stem)
for f in test_dir.rglob("*.png"):
    groups[("test", f.parent.name)].append(f.stem)

print("\n== Tile name prefixes per split and class (prefix = text before first underscore) ==")
for (split, cls), stems in sorted(groups.items()):
    prefixes = defaultdict(list)
    for s in stems:
        prefixes[s.split("_")[0]].append(s)
    print(f"\n{split} / {cls}  ({len(stems)} tiles)")
    for pre, items in sorted(prefixes.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"   {pre:<20} {len(items):>5}   e.g. {items[0]}")

train = {source(s) for (sp, c), v in groups.items() if sp == "train" for s in v}
test = {source(s) for (sp, c), v in groups.items() if sp == "test" for s in v}
both = train & test
print("\n== Source-image overlap ==")
print("source images in train:", len(train))
print("source images in test: ", len(test))
print("in BOTH:               ", len(both), sorted(both)[:6])