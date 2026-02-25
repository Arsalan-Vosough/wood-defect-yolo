from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run predictions with a trained YOLO model.")
    p.add_argument("--weights", required=True, help="Path to trained weights .pt")
    p.add_argument("--source", required=True, help="Image file, directory, glob, video, etc.")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="predictions")
    p.add_argument("--exist-ok", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    model = YOLO(str(Path(args.weights).expanduser()))
    model.predict(
        source=args.source,
        save=True,
        conf=args.conf,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
