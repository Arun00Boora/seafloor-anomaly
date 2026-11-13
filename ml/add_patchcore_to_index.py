"""
ml/add_patchcore_to_index.py

Adds a "pc_score" (PatchCore-style score) to each tile in deploy_data/index/test_ranked.json,
matching tiles by tile_id. Makes a backup first. Safe to re-run.
"""

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "deploy_data" / "index" / "test_ranked.json"
SCORES = ROOT / "ml" / "data" / "scores" / "patchcore_test.csv"

scores = {}
with open(SCORES, newline="") as f:
    for row in csv.DictReader(f):
        scores[row["tile_id"]] = float(row["score"])

data = json.loads(INDEX.read_text())
# The index may be a plain list, or a dict holding the list. Handle both.
if isinstance(data, list):
    records, holder = data, None
else:
    holder = next(k for k, v in data.items() if isinstance(v, list))
    records = data[holder]

print("Keys on one record:", sorted(records[0].keys()))

matched = 0
for r in records:
    tid = str(r.get("tile_id", r.get("id", "")))
    if tid in scores:
        r["pc_score"] = scores[tid]
        matched += 1

print(f"Matched {matched} of {len(records)} tiles ({len(scores)} PatchCore scores available)")
if matched != len(records):
    print("Some tiles did not match. Example index id:", records[0].get("tile_id", records[0].get("id")),
          "| example CSV id:", next(iter(scores)))
    raise SystemExit("Not saved: ids do not match. Send me this output.")

backup = INDEX.with_suffix(".json.bak")
if not backup.exists():
    shutil.copy(INDEX, backup)
INDEX.write_text(json.dumps(data))
print("Saved with pc_score. Backup at", backup)