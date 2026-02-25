from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Set
import random

from tqdm import tqdm

from .paths import IMG_EXTS_DEFAULT


def list_images(images_dir: Path, img_exts: Set[str] = IMG_EXTS_DEFAULT) -> List[Path]:
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts])


def list_labels(labels_dir: Path) -> List[Path]:
    return sorted([p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])


def pair_by_stem(images: List[Path], labels: List[Path]) -> Tuple[Dict[str, Path], Dict[str, Path], List[str], List[str], List[str]]:
    img_map = {p.stem: p for p in images}
    lbl_map = {p.stem: p for p in labels}
    common = sorted(set(img_map.keys()) & set(lbl_map.keys()))
    missing_lbl = sorted(set(img_map.keys()) - set(lbl_map.keys()))
    missing_img = sorted(set(lbl_map.keys()) - set(img_map.keys()))
    return img_map, lbl_map, common, missing_lbl, missing_img


def scan_yolo_labels(lbl_map: Dict[str, Path], stems: List[str]) -> Tuple[Dict[int, int], int]:
    class_counts: Dict[int, int] = {}
    bad_lines = 0

    for stem in tqdm(stems, desc="Scanning labels"):
        lp = lbl_map[stem]
        with open(lp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad_lines += 1
                    continue
                try:
                    c = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:])
                    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        bad_lines += 1
                        continue
                    class_counts[c] = class_counts.get(c, 0) + 1
                except Exception:
                    bad_lines += 1

    return class_counts, bad_lines


def split_stems(stems: List[str], train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    rng = random.Random(seed)
    stems = stems[:]
    rng.shuffle(stems)
    n = len(stems)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train = stems[:n_train]
    val = stems[n_train:n_train + n_val]
    test = stems[n_train + n_val:]
    return train, val, test
