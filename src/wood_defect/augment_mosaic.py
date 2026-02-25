from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional

import random
import shutil
import yaml
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

from .paths import IMG_EXTS_DEFAULT


def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def to_yolo_norm(x1, y1, x2, y2, W, H):
    xc = ((x1 + x2) / 2) / W
    yc = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return xc, yc, w, h


def find_image_by_stem(img_dir: Path, stem: str, img_exts=IMG_EXTS_DEFAULT) -> Optional[Path]:
    for ext in img_exts:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    g = list(img_dir.glob(stem + ".*"))
    return g[0] if g else None


def copy_tree(src: Path, dst: Path):
    src, dst = Path(src), Path(dst)
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src)
            (dst / rel.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / rel)


def read_yolo_labels(lbl_path: Path, W: int, H: int) -> List[Tuple[int, int, int, int, int]]:
    boxes = []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
            x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
            if x2 > x1 and y2 > y1:
                boxes.append((c, x1, y1, x2, y2))
    return boxes


def make_mosaic(train_img_dir: Path, train_lbl_dir: Path, sample_stems: List[str], out_size=(1024, 1024)):
    outW, outH = out_size
    tileW, tileH = outW // 2, outH // 2
    positions = [(0, 0), (tileW, 0), (0, tileH), (tileW, tileH)]

    mosaic = np.zeros((outH, outW, 3), dtype=np.uint8)
    mosaic_lines = []

    for stem, (ox, oy) in zip(sample_stems, positions):
        img_path = find_image_by_stem(train_img_dir, stem)
        if img_path is None:
            continue
        lbl_path = train_lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        H, W = img.shape[:2]
        boxes = read_yolo_labels(lbl_path, W, H)

        img_res = cv2.resize(img, (tileW, tileH), interpolation=cv2.INTER_AREA)
        mosaic[oy:oy + tileH, ox:ox + tileW] = img_res

        sx, sy = tileW / W, tileH / H
        for (c, x1, y1, x2, y2) in boxes:
            x1r = x1 * sx + ox
            x2r = x2 * sx + ox
            y1r = y1 * sy + oy
            y2r = y2 * sy + oy

            x1r = max(0, min(outW - 1, x1r)); x2r = max(0, min(outW - 1, x2r))
            y1r = max(0, min(outH - 1, y1r)); y2r = max(0, min(outH - 1, y2r))

            if x2r > x1r and y2r > y1r:
                xc, yc, w, h = to_yolo_norm(x1r, y1r, x2r, y2r, outW, outH)
                mosaic_lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    return mosaic, mosaic_lines


def detect_weak_classes_from_val(best_weights: Path, data_yaml: Path, imgsz=1024, device="0", map_thresh=0.40):
    m = YOLO(str(best_weights))
    metrics = m.val(data=str(data_yaml), split="val", imgsz=imgsz, device=device, verbose=False)
    if not hasattr(metrics, "box") or not hasattr(metrics.box, "maps"):
        raise RuntimeError("Could not read per-class mAP from metrics.box.maps")

    maps = metrics.box.maps
    names = getattr(metrics, "names", None) or getattr(m, "names", None) or {}
    weak = [i for i, v in enumerate(maps) if (v is not None and float(v) < map_thresh)]
    weak_info = sorted([(i, names.get(i, str(i)), float(maps[i])) for i in weak], key=lambda x: x[2])
    return weak, weak_info, maps


def create_aug_dataset(
    baseline_split_root: Path,
    aug_split_root: Path,
    baseline_weights_for_weak: Path,
    map_thresh=0.40,
    mosaic_fraction=0.30,
    max_mosaics=500,
    imgsz=1024,
    device="0",
):
    baseline_split_root = Path(baseline_split_root)
    aug_split_root = Path(aug_split_root)
    data_yaml_baseline = baseline_split_root / "data.yaml"
    if not data_yaml_baseline.exists():
        raise FileNotFoundError(f"Missing baseline data.yaml: {data_yaml_baseline}")

    if aug_split_root.exists() and (aug_split_root / "data.yaml").exists():
        return aug_split_root / "data.yaml"

    weak_classes, weak_info_sorted, _ = detect_weak_classes_from_val(
        best_weights=baseline_weights_for_weak,
        data_yaml=data_yaml_baseline,
        imgsz=imgsz,
        device=device,
        map_thresh=map_thresh,
    )

    # Create structure
    for p in [
        aug_split_root / "images" / "train",
        aug_split_root / "images" / "val",
        aug_split_root / "images" / "test",
        aug_split_root / "labels" / "train",
        aug_split_root / "labels" / "val",
        aug_split_root / "labels" / "test",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    # Copy baseline split into augmented root
    copy_tree(baseline_split_root / "images", aug_split_root / "images")
    copy_tree(baseline_split_root / "labels", aug_split_root / "labels")

    train_img_dir = aug_split_root / "images" / "train"
    train_lbl_dir = aug_split_root / "labels" / "train"
    train_label_files = sorted(train_lbl_dir.glob("*.txt"))

    # Map each training image -> set of classes it contains
    img_contains_class: Dict[str, Set[int]] = {}
    for lp in train_label_files:
        stem = lp.stem
        cls_set = set()
        with open(lp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.split()
                if not parts:
                    continue
                cls_set.add(int(float(parts[0])))
        img_contains_class[stem] = cls_set

    weak_set = set(weak_classes)
    weak_stems = [s for s, cls_set in img_contains_class.items() if weak_set.intersection(cls_set)]

    if len(weak_classes) > 0 and len(weak_stems) > 0:
        num_mosaics = min(max_mosaics, max(0, int(mosaic_fraction * len(weak_stems))))
        created = 0
        for i in tqdm(range(num_mosaics), desc="Creating mosaics"):
            chosen = random.choices(weak_stems, k=4)
            mosaic_img, mosaic_lines = make_mosaic(train_img_dir, train_lbl_dir, chosen, out_size=(imgsz, imgsz))
            if len(mosaic_lines) == 0:
                continue
            out_stem = f"mosaic_weakmap_{i:05d}"
            out_img = train_img_dir / f"{out_stem}.jpg"
            out_lbl = train_lbl_dir / f"{out_stem}.txt"
            cv2.imwrite(str(out_img), mosaic_img)
            with open(out_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(mosaic_lines) + "\n")
            created += 1

    # Write YAML for augmented dataset (carry over names from baseline yaml)
    base_cfg = yaml.safe_load(open(data_yaml_baseline, "r", encoding="utf-8"))
    data_yaml_aug = aug_split_root / "data.yaml"
    aug_cfg = {
        "path": str(aug_split_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": base_cfg.get("names", []),
    }
    with open(data_yaml_aug, "w", encoding="utf-8") as f:
        yaml.safe_dump(aug_cfg, f, sort_keys=False)
    return data_yaml_aug
