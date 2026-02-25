from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


def build_parser():
    p = argparse.ArgumentParser(description="Export YOLO weights to ONNX and TorchScript.")
    p.add_argument("--weights", required=True, help="Path to best.pt")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--simplify", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    weights = Path(args.weights).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )
    onnx_path = Path(onnx_path)
    onnx_dst = outdir / (weights.stem + ".onnx")
    if onnx_dst.exists():
        onnx_dst.unlink()
    onnx_path.rename(onnx_dst)

    ts_path = model.export(format="torchscript", imgsz=args.imgsz)
    ts_path = Path(ts_path)
    ts_dst = outdir / (weights.stem + ".torchscript")
    if ts_dst.exists():
        ts_dst.unlink()
    ts_path.rename(ts_dst)

    print("Saved:", onnx_dst)
    print("Saved:", ts_dst)


if __name__ == "__main__":
    main()
