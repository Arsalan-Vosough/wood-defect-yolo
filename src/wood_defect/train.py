from __future__ import annotations
import argparse
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a YOLO detector for wood surface defects.")
    p.add_argument("--data", required=True, help="Path to YOLO dataset YAML")
    p.add_argument("--model", default="yolov8n.pt", help="Base pretrained weights (e.g., yolov8n.pt, yolo11n.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0", help="GPU id (e.g., 0) or 'cpu'")
    p.add_argument("--project", default="runs", help="Where Ultralytics writes runs/")
    p.add_argument("--name", default="wood_defect_exp")
    p.add_argument("--exist-ok", action="store_true", help="Allow overwrite existing run directory name")
    p.add_argument("--verbose", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
