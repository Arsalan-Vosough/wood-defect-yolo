from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def yolo_to_xyxy(xc, yc, w, h, W, H) -> Tuple[int, int, int, int]:
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def draw_boxes(img_bgr: np.ndarray, label_path: Path, class_names: Optional[Dict[int, str]] = None, thickness=2) -> np.ndarray:
    img = img_bgr.copy()
    H, W = img.shape[:2]
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c, xc, yc, w, h = line.split()
            c = int(float(c))
            xc, yc, w, h = map(float, [xc, yc, w, h])
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            name = str(c) if class_names is None else class_names.get(c, str(c))
            cv2.putText(img, name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img
