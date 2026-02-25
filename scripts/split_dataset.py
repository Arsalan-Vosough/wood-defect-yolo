from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import yaml

from wood_defect.paths import resolve_dataset_folders
from wood_defect.dataset import list_images, list_labels, pair_by_stem, scan_yolo_labels, split_stems


def build_parser():
    p = argparse.ArgumentParser(description="Split a YOLO dataset into train/val/test folders.")
    p.add_argument("--root", required=True, help="Dataset root (where Images/Labels exist)")
    p.add_argument("--images-rel", default="dataset/baseline/Images", help="Images relative path from root")
    p.add_argument("--labels-rel", default="dataset/baseline/Labels", help="Labels relative path from root")
    p.add_argument("--out", required=True, help="Output split folder (e.g. data/baseline_split_yolo)")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--names-yaml", default="configs/classes.yaml", help="Path to classes.yaml")
    return p


def main():
    args = build_parser().parse_args()

    folders = resolve_dataset_folders(Path(args.root), args.images_rel, args.labels_rel)

    images = list_images(folders.images_dir)
    labels = list_labels(folders.labels_dir)
    img_map, lbl_map, common, missing_lbl, missing_img = pair_by_stem(images, labels)

    print("Num images:", len(images))
    print("Num labels:", len(labels))
    print("Matched pairs:", len(common))
    print("Images missing labels:", len(missing_lbl))
    print("Labels missing images:", len(missing_img))

    class_counts, bad_lines = scan_yolo_labels(lbl_map, common)
    print("Bad/invalid label lines:", bad_lines)
    for k in sorted(class_counts):
        print(f"  class {k}: {class_counts[k]}")

    out_root = Path(args.out).expanduser().resolve()

    # Create structure
    for pth in [
        out_root / "images/train", out_root / "images/val", out_root / "images/test",
        out_root / "labels/train", out_root / "labels/val", out_root / "labels/test",
    ]:
        pth.mkdir(parents=True, exist_ok=True)

    train_stems, val_stems, test_stems = split_stems(common, args.train, args.val, args.test, args.seed)
    print("Split sizes:", len(train_stems), len(val_stems), len(test_stems))

    def copy_pair(stem, split_name):
        src_img = img_map[stem]
        src_lbl = lbl_map[stem]
        shutil.copy2(src_img, out_root / "images" / split_name / src_img.name)
        shutil.copy2(src_lbl, out_root / "labels" / split_name / src_lbl.name)

    for s in train_stems:
        copy_pair(s, "train")
    for s in val_stems:
        copy_pair(s, "val")
    for s in test_stems:
        copy_pair(s, "test")

    # Load class names from configs/classes.yaml
    names_cfg = yaml.safe_load(open(args.names_yaml, "r", encoding="utf-8"))
    names = names_cfg["names"]
    # Ultralytics supports list or dict; we keep dict mapping
    data_yaml = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print("Wrote:", out_root / "data.yaml")


if __name__ == "__main__":
    main()
