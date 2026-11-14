import re
from pathlib import Path

root = Path(r"D:\sonar project\seafloor-anomaly")
train_dir = root / "ml" / "data" / "tiles_drishti" / "train"
test_dir = root / "deploy_data" / "tiles" / "test"

def source(stem):
    # Strip the tile position suffix: "..._y0_x320" or "..._x3000" -> source image name
    return re.sub(r"(_y\d+)?_x\d+$", "", stem)

train = {source(p.stem) for p in train_dir.rglob("*.png")}
test = {source(p.stem) for p in test_dir.rglob("*.png")}
both = train & test
print("source images in train:", len(train))
print("source images in test: ", len(test))
print("source images in BOTH: ", len(both))
print("examples in both:", sorted(both)[:8])