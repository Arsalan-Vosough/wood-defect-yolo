import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model for wood defect detection")

    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0", help="cuda device id or 'cpu'")
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="wood_defect_exp")

    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True
    )


if __name__ == "__main__":
    main()
