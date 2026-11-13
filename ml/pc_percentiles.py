import csv, json
import numpy as np

root = r"D:\sonar project\seafloor-anomaly"
rows = list(csv.DictReader(open(root + r"\ml\data\scores\patchcore_test.csv")))
cal = json.load(open(root + r"\deploy_data\model\pc_calibration.json"))

def scaled(label):
    return np.array([(float(r["score"]) - cal["min"]) / (cal["max"] - cal["min"])
                     for r in rows if r["label"] == label])

normal, debris = scaled("0"), scaled("1")
print("NORMAL tiles (", len(normal), ") p50/p90/p95/p99:", np.round(np.percentile(normal, [50, 90, 95, 99]), 2))
print("DEBRIS tiles (", len(debris), ") p10/p25/p50/p75:", np.round(np.percentile(debris, [10, 25, 50, 75]), 2))