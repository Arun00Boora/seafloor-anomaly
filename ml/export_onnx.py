"""
export_onnx.py

Exports the trained autoencoder to ONNX, a small portable format that runs with
onnxruntime (about 15 MB) instead of PyTorch (hundreds of MB). The deployed server
uses it to score images that users upload.

Run from the ml/ folder:   python export_onnx.py
"""

from pathlib import Path

import numpy as np
import torch

from models.conv_ae import ConvAutoencoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "conv_ae.pt"
OUT = PROJECT_ROOT / "deploy_data" / "model" / "conv_ae.onnx"


def main():
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, 64, 64)  # one 64x64 grayscale tile

    # dynamo=False uses the classic exporter, which handles this small network simply.
    torch.onnx.export(
        model,
        dummy,
        str(OUT),
        input_names=["tile"],
        output_names=["recon"],
        dynamic_axes={"tile": {0: "batch"}, "recon": {0: "batch"}},  # any number of tiles at once
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported to {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")

    # Check that ONNX gives the same output as PyTorch, if onnxruntime is installed.
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping the comparison check.")
        return

    x = torch.rand(8, 1, 64, 64)
    with torch.no_grad():
        expected = model(x).numpy()
    session = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    got = session.run(None, {"tile": x.numpy()})[0]
    print(f"Max difference between PyTorch and ONNX outputs: {np.abs(expected - got).max():.2e}")


if __name__ == "__main__":
    main()