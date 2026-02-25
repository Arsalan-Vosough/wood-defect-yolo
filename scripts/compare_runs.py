from __future__ import annotations
import argparse
from pathlib import Path

from wood_defect.metrics import compare_runs


def build_parser():
    p = argparse.ArgumentParser(description="Compare Ultralytics runs by best mAP50-95.")
    p.add_argument("--runs-dir", default="runs/detect", help="Ultralytics detect runs directory")
    p.add_argument("--out", default="runs/wood_defects_model_comparison.csv")
    return p


def main():
    args = build_parser().parse_args()

    runs_dir = Path(args.runs_dir)
    runs = {
        "YOLOv8n (baseline)": runs_dir / "wood_yolov8n_baseline_e50",
        "YOLO11n (baseline)": runs_dir / "wood_yolo11n_baseline_e50",
        "YOLOv8n (aug mosaic rare)": runs_dir / "wood_yolov8n_augMosaicRare_e50",
        "YOLO11n (aug mosaic rare)": runs_dir / "wood_yolo11n_augMosaicRare_e50",
    }
    df = compare_runs(runs, Path(args.out))
    print("Saved comparison to:", args.out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
