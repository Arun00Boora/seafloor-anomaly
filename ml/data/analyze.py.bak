"""
routers/analyze.py

POST /analyze  (multipart form: file, tile_size)

Takes any sonar image, cuts it into square tiles, shrinks each tile to 64x64 (the
size the autoencoder was trained on), and scores every tile the same way the main
pipeline does. Returns the image, a full-size error heatmap, and the ranked tiles.
Runs on ONNX, so the server does not need PyTorch.
"""

import base64
import io
import json

import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from . import tiles as tiles_router

router = APIRouter(tags=["analyze"])

MODEL_DIR = tiles_router.PROJECT_ROOT / "deploy_data" / "model"
MAX_BYTES = 8 * 1024 * 1024   # reject uploads larger than 8 MB
MAX_SIDE = 2048               # larger images are shrunk to keep memory small
TOP_FRACTION = 0.10           # AE score = mean of the worst 10% pixel errors

_session = None
_calibration = None


def get_model():
    global _session, _calibration
    if _session is None:
        _session = ort.InferenceSession(
            str(MODEL_DIR / "conv_ae.onnx"), providers=["CPUExecutionProvider"]
        )
        _calibration = json.loads((MODEL_DIR / "calibration.json").read_text())
    return _session, _calibration


def normalize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Same min-max scaling as the ranked index, clipped to 0..1 for new data."""
    return np.clip((values - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def png_base64(array: np.ndarray, mode: str) -> str:
    buf = io.BytesIO()
    Image.fromarray(array, mode).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def error_to_rgba(err: np.ndarray) -> np.ndarray:
    """Clear where error is low, teal -> amber -> red as it rises (scaled to the image max)."""
    v = err / (err.max() + 1e-8)
    stops = [0.0, 0.5, 1.0]
    r = np.interp(v, stops, [45, 251, 255])
    g = np.interp(v, stops, [212, 191, 122])
    b = np.interp(v, stops, [191, 36, 107])
    a = (v ** 1.5) * 0.9 * 255
    return np.stack([r, g, b, a], axis=-1).astype(np.uint8)


@router.post("/analyze")
def analyze(file: UploadFile = File(...), tile_size: int = Form(640)):
    if not 64 <= tile_size <= 1024:
        raise HTTPException(status_code=400, detail="tile_size must be between 64 and 1024")

    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 8 MB")
    try:
        gray = Image.open(io.BytesIO(data)).convert("L")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="That file is not a readable image")

    resized = False
    if max(gray.size) > MAX_SIDE:
        f = MAX_SIDE / max(gray.size)
        gray = gray.resize((round(gray.width * f), round(gray.height * f)), Image.LANCZOS)
        resized = True

    arr = np.asarray(gray, dtype=np.float32) / 255.0
    H, W = arr.shape
    rows = -(-H // tile_size)   # ceiling division
    cols = -(-W // tile_size)
    padded = np.pad(arr, ((0, rows * tile_size - H), (0, cols * tile_size - W)), mode="edge")

    # Cut into tiles and shrink each to 64x64, the scale the model was trained on.
    tiles64, positions = [], []
    for r in range(rows):
        for c in range(cols):
            patch = padded[r * tile_size:(r + 1) * tile_size, c * tile_size:(c + 1) * tile_size]
            small = Image.fromarray((patch * 255).astype(np.uint8)).resize((64, 64), Image.LANCZOS)
            tiles64.append(np.asarray(small, dtype=np.float32) / 255.0)
            positions.append((r, c))
    x = np.stack(tiles64)[:, None]   # shape [N, 1, 64, 64]

    session, cal = get_model()
    recon = session.run(None, {"tile": x})[0]
    err = ((recon - x) ** 2)[:, 0]   # per-pixel squared error, [N, 64, 64]

    # Scores, computed exactly like the training pipeline.
    k = int(64 * 64 * TOP_FRACTION)
    ae_raw = np.partition(err.reshape(len(err), -1), -k, axis=1)[:, -k:].mean(axis=1)
    base_raw = x.reshape(len(x), -1).max(axis=1)
    ae = normalize(ae_raw, cal["ae_min"], cal["ae_max"])
    base = normalize(base_raw, cal["baseline_min"], cal["baseline_max"])
    ensemble = (ae + base) / 2

    # Stitch the per-tile error maps into one full-size heatmap.
    grid = err.reshape(rows, cols, 64, 64).transpose(0, 2, 1, 3).reshape(rows * 64, cols * 64)
    full = Image.fromarray(grid.astype(np.float32), mode="F").resize(
        (cols * tile_size, rows * tile_size), Image.BILINEAR
    )
    full_err = np.asarray(full, dtype=np.float32)[:H, :W]

    ranked = sorted(range(len(positions)), key=lambda i: ensemble[i], reverse=True)
    return {
        "width": W,
        "height": H,
        "tile_size": tile_size,
        "rows": rows,
        "cols": cols,
        "was_resized": resized,
        "image": png_base64((arr * 255).astype(np.uint8), "L"),
        "heatmap": png_base64(error_to_rgba(full_err), "RGBA"),
        "tiles": [
            {
                "rank": n + 1,
                "row": positions[i][0],
                "col": positions[i][1],
                "x": positions[i][1] * tile_size,
                "y": positions[i][0] * tile_size,
                "ensemble": float(ensemble[i]),
                "ae": float(ae[i]),
                "baseline": float(base[i]),
            }
            for n, i in enumerate(ranked)
        ],
    }