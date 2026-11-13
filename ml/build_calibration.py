"""
build_calibration.py

Saves the min/max raw scores of the test set, so the deployed server can put the
scores of uploaded images on the same 0-1 scale as the ranked index.
Then checks that this reproduces the normalized scores already in the ranked index.

Run from the ml/ folder:   python build_calibration.py
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES = PROJECT_ROOT / "data" / "scores"
INDEX = PROJECT_ROOT / "deploy_data" / "index" / "test_ranked.json"
OUT = PROJECT_ROOT / "deploy_data" / "model" / "calibration.json"


def load_scores(path):
    """Return {tile_id: raw score} from one of the score CSVs."""
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        key = next(k for k in reader.fieldnames if "score" in k)
        for row in reader:
            stem = Path(row["filepath"].replace("\\", "/")).stem
            rows[stem] = float(row[key])
    return rows


def main():
    ae = load_scores(SCORES / "test_ae_scores.csv")
    base = load_scores(SCORES / "test_baseline_scores.csv")

    calibration = {
        "ae_min": min(ae.values()), "ae_max": max(ae.values()),
        "baseline_min": min(base.values()), "baseline_max": max(base.values()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(calibration, indent=2))
    print(f"Saved {OUT}")
    print(calibration)

    # Sanity check against the ranked index (its scores are already normalized).
    records = json.loads(INDEX.read_text())[:3]
    for rec in records:
        tid = rec["tile_id"]
        ae_n = (ae[tid] - calibration["ae_min"]) / (calibration["ae_max"] - calibration["ae_min"])
        b_n = (base[tid] - calibration["baseline_min"]) / (calibration["baseline_max"] - calibration["baseline_min"])
        print(f"{tid}: ae {ae_n:.4f} vs index {rec['ae_score']:.4f} | "
              f"baseline {b_n:.4f} vs index {rec['baseline_score']:.4f}")


if __name__ == "__main__":
    main()