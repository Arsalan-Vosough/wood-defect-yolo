from __future__ import annotations
from pathlib import Path
import argparse

from ultralytics import YOLO


def build_parser():
    p = argparse.ArgumentParser(description="Train baseline models (YOLOv8n and/or YOLO11n).")
    p.add_argument("--data", required=True, help="Path to split dataset data.yaml")
    p.add_argument("--project", default="runs", help="Runs directory")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--train-v8", action="store_true")
    p.add_argument("--train-y11", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    data = str(Path(args.data).expanduser().resolve())

    if not (args.train_v8 or args.train_y11):
        # default: train both
        args.train_v8 = True
        args.train_y11 = True

    if args.train_v8:
        YOLO("yolov8n.pt").train(
            data=data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=f"wood_yolov8n_baseline_e{args.epochs}",
            exist_ok=True,
            verbose=True,
        )

    if args.train_y11:
        YOLO("yolo11n.pt").train(
            data=data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=f"wood_yolo11n_baseline_e{args.epochs}",
            exist_ok=True,
            verbose=True,
        )


if __name__ == "__main__":
    main()
