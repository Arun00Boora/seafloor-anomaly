"""
app/pc_scorer.py

Runs the PatchCore-style ONNX model on 64x64 tiles using onnxruntime only (no PyTorch).

Tiles are scored a few at a time: the scorer compares every patch of a tile with a 20000-patch
memory bank, which costs roughly 80 MB per 4 tiles. Chunking keeps peak memory low enough for
Render's free 512 MB plan.

Self-test:  cd backend ; python -m app.pc_scorer
"""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_DIR = Path(__file__).resolve().parents[2] / "deploy_data" / "model"
CHUNK = 4  # tiles per model call

_session = None
_cal = None


def _load():
    global _session, _cal
    if _session is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        _session = ort.InferenceSession(
            str(MODEL_DIR / "patchcore.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        _cal = json.loads((MODEL_DIR / "pc_calibration.json").read_text())
    return _session, _cal


def score_tiles(tiles: np.ndarray) -> np.ndarray:
    """tiles: (N, 3, 64, 64) float32 in 0..1  ->  (N,) raw anomaly scores (higher = more unusual)."""
    session, _ = _load()
    out = []
    for i in range(0, len(tiles), CHUNK):
        out.append(session.run(None, {"tile": tiles[i:i + CHUNK].astype(np.float32)})[0])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def normalize(raw: np.ndarray) -> np.ndarray:
    """Scale raw scores to 0..1 using the min/max seen on the test set (clipped)."""
    _, cal = _load()
    span = max(cal["max"] - cal["min"], 1e-9)
    return np.clip((raw - cal["min"]) / span, 0.0, 1.0)


if __name__ == "__main__":
    from PIL import Image

    root = Path(__file__).resolve().parents[2]
    index_path = root / "deploy_data" / "index" / "test_ranked.json"
    data = json.loads(index_path.read_text())
    records = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))
    sample = records[:4] + records[-4:]          # most and least suspicious tiles
    files = {p.stem: p for p in (root / "deploy_data" / "tiles" / "test").rglob("*.png")}

    tiles = np.stack([
        np.asarray(Image.open(files[str(r["tile_id"])]).convert("RGB").resize((64, 64), Image.LANCZOS),
                   dtype=np.float32).transpose(2, 0, 1) / 255.0
        for r in sample
    ])
    t0 = time.time()
    raw = score_tiles(tiles)
    print(f"Scored {len(tiles)} tiles in {time.time() - t0:.2f} s")
    for r, s in zip(sample, raw):
        print(f"tile {r['tile_id']}: onnx {s:.4f} | index pc_score {r['pc_score']:.4f}")
    print("Normalized sample:", np.round(normalize(raw), 2))