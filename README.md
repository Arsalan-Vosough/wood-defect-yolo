# Wood Defect Detection (YOLO)

This repository contains a reproducible YOLO training pipeline for wood surface defect detection.
Structure:
- Dataset validation + split (train/val/test)
- Baseline training (YOLOv8n, YOLO11n)
- Weak-class detection + mosaic augmentation (offline)
- Augmented training
- Run comparison (best epoch selection from results.csv)
- Export to ONNX + TorchScript

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
