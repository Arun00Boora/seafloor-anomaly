"""
routers/tiles.py

Endpoints:
    GET  /tiles              -> paginated ranked tile list (most anomalous first)
    GET  /tiles/{tile_id}/image -> serves the actual tile PNG
    POST /tiles/{tile_id}/flag  -> analyst marks a tile reviewed
    GET  /tiles/{tile_id}/flag  -> fetch existing flag for a tile, if any
"""

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Flag
from ..schemas import TileOut, FlagIn, FlagOut

router = APIRouter(prefix="/tiles", tags=["tiles"])

# ---- Config: paths resolved relative to this file's location, not the
# current working directory, so the server works regardless of which
# folder you launch uvicorn from. ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # .../seafloor-anomaly
# The deployed server uses the packaged deploy_data folder; if it is missing,
# fall back to the working data folders used during development.
DEPLOY_DATA = PROJECT_ROOT / "deploy_data"
if (DEPLOY_DATA / "index" / "test_ranked.json").exists():
    RANKED_INDEX_PATH = DEPLOY_DATA / "index" / "test_ranked.json"
    TILES_ROOT = DEPLOY_DATA / "tiles" / "test"
else:
    RANKED_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "test_ranked.json"
    TILES_ROOT = PROJECT_ROOT / "ml" / "data" / "tiles_drishti" / "test"

_RANKED_TILES = []
_TILE_ID_TO_RECORD = {}
_TILE_ID_TO_IMAGE_PATH = {}


def load_index():
    global _RANKED_TILES, _TILE_ID_TO_RECORD, _TILE_ID_TO_IMAGE_PATH

    with open(RANKED_INDEX_PATH, "r") as f:
        _RANKED_TILES = json.load(f)
    _TILE_ID_TO_RECORD = {t["tile_id"]: t for t in _RANKED_TILES}

    _TILE_ID_TO_IMAGE_PATH = {}
    for subfolder in ("normal", "anomalous"):
        for path in (TILES_ROOT / subfolder).glob("*.png"):
            _TILE_ID_TO_IMAGE_PATH[path.stem] = path


load_index()


@router.get("", response_model=list[TileOut])
def list_tiles(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Return a page of the ranked tile list, most anomalous first."""
    page = _RANKED_TILES[offset: offset + limit]

    tile_ids = [t["tile_id"] for t in page]
    flags = db.query(Flag).filter(Flag.tile_id.in_(tile_ids)).all()
    flag_map = {f.tile_id: f.status for f in flags}

    return [
        TileOut(
            tile_id=t["tile_id"],
            rank=t["rank"],
            ensemble_score=t["ensemble_score"],
            ae_score=t["ae_score"],
            baseline_score=t["baseline_score"],
            flag_status=flag_map.get(t["tile_id"]),
        )
        for t in page
    ]


@router.get("/{tile_id}/image")
def get_tile_image(tile_id: str):
    """Serve the raw tile PNG for display in the frontend."""
    path = _TILE_ID_TO_IMAGE_PATH.get(tile_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Tile image not found")
    return FileResponse(path, media_type="image/png")


@router.post("/{tile_id}/flag", response_model=FlagOut)
def create_flag(tile_id: str, flag_in: FlagIn, db: Session = Depends(get_db)):
    """Create or update a flag for a tile (upsert -- an analyst can re-flag)."""
    if tile_id not in _TILE_ID_TO_RECORD:
        raise HTTPException(status_code=404, detail="Tile not found")

    existing = db.query(Flag).filter(Flag.tile_id == tile_id).first()
    if existing:
        existing.status = flag_in.status
        existing.note = flag_in.note
        flag = existing
    else:
        flag = Flag(tile_id=tile_id, status=flag_in.status, note=flag_in.note)
        db.add(flag)

    db.commit()
    db.refresh(flag)
    return flag


@router.get("/{tile_id}/flag", response_model=FlagOut)
def get_flag(tile_id: str, db: Session = Depends(get_db)):
    flag = db.query(Flag).filter(Flag.tile_id == tile_id).first()
    if flag is None:
        raise HTTPException(status_code=404, detail="No flag for this tile yet")
    return flag