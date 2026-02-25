from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO

from wood_defect.augment_mosaic import create_aug_dataset


def build_parser():
    p = argparse.ArgumentParser(description="Create mosaic-augmented dataset and train models on it.")
    p.add_argument("--baseline-split", required=True, help="Baseline split folder (contains data.yaml)")
    p.add_argument("--aug-split", required=True, help="Output augmented dataset folder")
    p.add_argument("--baseline-weights", required=True, help="Weights used to detect weak classes (best.pt)")
    p.add_argument("--map-thresh", type=float, default=0.40)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs")
    p.add_argument("--train-v8", action="store_true")
    p.add_argument("--train-y11", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    data_yaml_aug = create_aug_dataset(
        baseline_split_root=Path(args.baseline_split),
        aug_split_root=Path(args.aug_split),
        baseline_weights_for_weak=Path(args.baseline_weights),
        map_thresh=args.map_thresh,
        imgsz=args.imgsz,
        device=args.device,
    )
    data = str(Path(data_yaml_aug).resolve())

    if not (args.train_v8 or args.train_y11):
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
            name=f"wood_yolov8n_augMosaicRare_e{args.epochs}",
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
            name=f"wood_yolo11n_augMosaicRare_e{args.epochs}",
            exist_ok=True,
            verbose=True,
        )


if __name__ == "__main__":
    main()
