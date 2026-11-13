"""
ml/export_patchcore_onnx.py

Packs the PatchCore-style scorer into ONE ONNX file:
  input : (B, 3, 64, 64) float32 tile, values 0..1 (RGB)
  output: (B,) anomaly score = distance of the tile's most unusual patch to the normal memory bank

The memory bank (20000 x 384) is baked into the file as constants, so the server needs only
onnxruntime. Distances use |a-b|^2 = |a|^2 + |b|^2 - 2 a.b, which is plain matrix maths that
ONNX supports well (torch.cdist is not reliably exportable).

Also writes deploy_data/model/pc_calibration.json (min/max of test scores, for 0..1 scaling).
"""

import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parent.parent
BANK_PATH = ROOT / "ml" / "data" / "patchcore_bank.pt"
SCORES_CSV = ROOT / "ml" / "data" / "scores" / "patchcore_test.csv"
TEST_DIR = ROOT / "deploy_data" / "tiles" / "test"
OUT_ONNX = ROOT / "deploy_data" / "model" / "patchcore.onnx"
OUT_CAL = ROOT / "deploy_data" / "model" / "pc_calibration.json"
INPUT_SIZE = 128


class PatchCoreScorer(nn.Module):
    def __init__(self, bank: torch.Tensor):
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("bank", bank)                       # (N, 384)
        self.register_buffer("bank_sq", (bank ** 2).sum(dim=1))  # (N,)

    def forward(self, x):
        n = self.net
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer1(x)
        f2 = n.layer2(x)                                          # (B,128,16,16)
        f3 = F.interpolate(n.layer3(f2), size=(16, 16), mode="bilinear", align_corners=False)
        f = F.avg_pool2d(torch.cat([f2, f3], dim=1), 3, stride=1, padding=1)
        b = f.shape[0]
        f = f.flatten(2).transpose(1, 2).reshape(-1, 384)         # (B*256, 384)
        d2 = (f ** 2).sum(dim=1, keepdim=True) + self.bank_sq.unsqueeze(0) - 2.0 * f @ self.bank.t()
        nearest = d2.min(dim=1).values.clamp(min=0).sqrt()        # distance to closest normal patch
        return nearest.reshape(b, -1).max(dim=1).values           # most unusual patch per tile


bank = torch.load(BANK_PATH)
model = PatchCoreScorer(bank).eval()

OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)
dummy = torch.rand(2, 3, 64, 64)
torch.onnx.export(
    model, dummy, str(OUT_ONNX),
    input_names=["tile"], output_names=["score"],
    dynamic_axes={"tile": {0: "batch"}, "score": {0: "batch"}},
    opset_version=17, dynamo=False,
)
print(f"Exported {OUT_ONNX} ({OUT_ONNX.stat().st_size / 1e6:.1f} MB)")

# ---- Parity check: ONNX output vs the scores we computed earlier with plain PyTorch ----
csv_scores = {r["tile_id"]: float(r["score"]) for r in csv.DictReader(open(SCORES_CSV))}
paths = {p.stem: p for p in TEST_DIR.rglob("*.png")}
ids = sorted(csv_scores)[:12]

def load(p):
    im = Image.open(p).convert("RGB").resize((64, 64), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0

sess = ort.InferenceSession(str(OUT_ONNX), providers=["CPUExecutionProvider"])
batch = np.stack([load(paths[i]) for i in ids])
onnx_scores = sess.run(None, {"tile": batch})[0]
expected = np.array([csv_scores[i] for i in ids])
print("Max difference vs earlier PyTorch scores:", float(np.abs(onnx_scores - expected).max()))
print("Score range in this sample:", float(expected.min()), "to", float(expected.max()))

allv = np.array(list(csv_scores.values()))
OUT_CAL.write_text(json.dumps({"min": float(allv.min()), "max": float(allv.max())}))
print("Calibration saved:", OUT_CAL)