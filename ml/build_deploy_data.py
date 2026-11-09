"""
build_deploy_data.py

Packages everything the deployed backend needs into ../deploy_data/, so the
server does not need PyTorch or the raw dataset:

  deploy_data/tiles/test/{normal,anomalous}/*.png   the 700 test tiles
  deploy_data/index/test_ranked.json                ranked scores + true labels
  deploy_data/heatmaps/<tile_id>.png                pre-rendered error heatmaps

Run from the ml/ folder:   python build_deploy_data.py
Re-run it whenever the model is retrained or the ranked index changes.
"""

import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.conv_ae import ConvAutoencoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_TILES = PROJECT_ROOT / "ml" / "data" / "tiles_drishti" / "test"
SRC_INDEX = PROJECT_ROOT / "data" / "index" / "test_ranked.json"
CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "conv_ae.pt"
OUT = PROJECT_ROOT / "deploy_data"


def error_to_rgba(err: np.ndarray) -> np.ndarray:
    """Same coloring as backend/app/routers/heatmap.py: clear where the error is
    low, teal -> amber -> red as it rises, scaled to this tile's own maximum."""
    v = err / (err.max() + 1e-8)
    stops = [0.0, 0.5, 1.0]
    r = np.interp(v, stops, [45, 251, 255])
    g = np.interp(v, stops, [212, 191, 122])
    b = np.interp(v, stops, [191, 36, 107])
    a = (v ** 1.5) * 0.9 * 255
    return np.stack([r, g, b, a], axis=-1).astype(np.uint8)


def main():
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()

    (OUT / "index").mkdir(parents=True, exist_ok=True)
    (OUT / "heatmaps").mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC_INDEX, OUT / "index" / "test_ranked.json")

    count = 0
    for subfolder in ("normal", "anomalous"):
        dest = OUT / "tiles" / "test" / subfolder
        dest.mkdir(parents=True, exist_ok=True)

        for path in sorted((SRC_TILES / subfolder).glob("*.png")):
            shutil.copy(path, dest / path.name)

            img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            x = torch.from_numpy(img)[None, None]  # shape [1, 1, 64, 64]
            with torch.no_grad():
                recon = model(x)
            err = ((recon - x) ** 2)[0, 0].numpy()  # per-pixel squared error

            Image.fromarray(error_to_rgba(err), "RGBA").save(
                OUT / "heatmaps" / f"{path.stem}.png"
            )
            count += 1

    print(f"Wrote {count} tiles and {count} heatmaps to {OUT}")


if __name__ == "__main__":
    main()