from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


IMG_EXTS_DEFAULT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetFolders:
    root: Path
    images_dir: Path
    labels_dir: Path


def resolve_dataset_folders(root: Path, images_rel: str, labels_rel: str) -> DatasetFolders:
    root = Path(root).expanduser().resolve()
    images_dir = (root / images_rel).resolve()
    labels_dir = (root / labels_rel).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")
    return DatasetFolders(root=root, images_dir=images_dir, labels_dir=labels_dir)
