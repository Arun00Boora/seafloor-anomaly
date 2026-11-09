"""
routers/heatmap.py

GET /tiles/{tile_id}/heatmap -> transparent RGBA PNG showing where the
autoencoder failed to reconstruct the tile.

Uses the pre-rendered file in deploy_data/heatmaps/ when it exists (this is what
the deployed site does, with no PyTorch needed). Otherwise it computes the
heatmap live, which needs torch and the trained checkpoint.
"""

import io
import importlib.util

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from . import tiles as tiles_router  # reuse its tile_id -> image path lookup

router = APIRouter(prefix="/tiles", tags=["heatmap"])

PROJECT_ROOT = tiles_router.PROJECT_ROOT
STATIC_HEATMAPS = PROJECT_ROOT / "deploy_data" / "heatmaps"
CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "conv_ae.pt"
MODEL_FILE = PROJECT_ROOT / "ml" / "models" / "conv_ae.py"

_model = None
_cache: dict[str, bytes] = {}


def get_model():
    """Load the trained autoencoder once, on first live request."""
    global _model
    if _model is None:
        import torch  # imported here so the deployed server never needs it

        spec = importlib.util.spec_from_file_location("conv_ae_module", MODEL_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = module.ConvAutoencoder()
        model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
        model.eval()
        _model = model
    return _model


def render_live(path) -> bytes:
    import numpy as np
    import torch
    from PIL import Image

    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    x = torch.from_numpy(img)[None, None]
    with torch.no_grad():
        recon = get_model()(x)
    err = ((recon - x) ** 2)[0, 0].numpy()

    v = err / (err.max() + 1e-8)
    stops = [0.0, 0.5, 1.0]
    r = np.interp(v, stops, [45, 251, 255])
    g = np.interp(v, stops, [212, 191, 122])
    b = np.interp(v, stops, [191, 36, 107])
    a = (v ** 1.5) * 0.9 * 255
    rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


@router.get("/{tile_id}/heatmap")
def get_tile_heatmap(tile_id: str):
    static_file = STATIC_HEATMAPS / f"{tile_id}.png"
    if static_file.exists():
        return FileResponse(static_file, media_type="image/png")

    path = tiles_router._TILE_ID_TO_IMAGE_PATH.get(tile_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Tile image not found")

    if tile_id not in _cache:
        _cache[tile_id] = render_live(path)
    return Response(_cache[tile_id], media_type="image/png")