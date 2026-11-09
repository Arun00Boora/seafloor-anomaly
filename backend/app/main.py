"""
main.py

FastAPI app entry point. Run with:
    uvicorn app.main:app --reload

(run this command from inside the `backend/` folder, so Python can find
the `app` package)
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .routers import tiles, heatmap, evaluation

Base.metadata.create_all(bind=engine)  # creates flags.db / the flags table if not present

app = FastAPI(title="Seafloor Anomaly Review API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tiles.router)
app.include_router(heatmap.router)
app.include_router(evaluation.router)


@app.get("/")
def root():
    return {"status": "ok", "message": "Seafloor Anomaly Review API"}