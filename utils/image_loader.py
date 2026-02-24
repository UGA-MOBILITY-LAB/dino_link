"""
Image loader for nuScenes / Waymo / COCO. Expects image root with subdirs or a list file.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

NUSCENES_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def default_transform(image_size: Optional[int] = 224, normalize_imagenet: bool = False):
    """If normalize_imagenet=False, output is [0,1]; DINOv2 extractor does ImageNet norm internally.
    If image_size is None: no resize, images fed at original size (variable)."""
    if image_size is not None:
        t = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    else:
        t = [transforms.ToTensor()]
    if normalize_imagenet:
        t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(t)


class ImagePathDataset(Dataset):
    """
    Dataset from a list of image paths or a root directory (recursive glob *.jpg, *.png).
    """

    def __init__(
        self,
        root_or_list,
        transform=None,
        is_list_file: bool = False,
    ):
        if transform is None:
            transform = default_transform(224)
        self.transform = transform
        if is_list_file:
            with open(root_or_list, "r") as f:
                self.paths = [p.strip() for p in f if p.strip()]
        else:
            root = Path(root_or_list)
            self.paths = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                self.paths.extend(root.rglob(ext))
            self.paths = sorted([str(p) for p in self.paths])
        assert len(self.paths) > 0, f"No images found under {root_or_list}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            # Quick validation: check if image has valid size
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError(f"Invalid image size: {img.size}")
        except Exception as e:
            # If image is corrupted, return a black image and log warning
            import warnings
            warnings.warn(f"Failed to load image {path}: {e}. Using black placeholder.")
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, path


class NuScenesMultiViewDataset(Dataset):
    """
    Each sample returns six synchronized camera images for one timestamp.
    """

    def __init__(
        self,
        data_root: str,
        transform=None,
        cameras: Sequence[str] = NUSCENES_CAMERAS,
    ):
        if transform is None:
            transform = default_transform(224)
        self.transform = transform
        self.cameras = list(cameras)
        self.samples = self._build_samples(Path(data_root))
        assert len(self.samples) > 0, f"No multi-view samples found under {data_root}"

    def _resolve_samples_root(self, root: Path) -> Path:
        # Accept:
        # - .../samples
        # - .../samples/CAM_FRONT
        # - nuScenes root containing ./samples
        if root.name.startswith("CAM_") and root.parent.name == "samples":
            return root.parent
        if (root / "samples").exists():
            return root / "samples"
        if (root / "CAM_FRONT").exists():
            return root
        return root

    def _build_samples(self, root: Path) -> List[Dict[str, str]]:
        samples_root = self._resolve_samples_root(root)
        front_dir = samples_root / "CAM_FRONT"
        if not front_dir.exists():
            raise FileNotFoundError(f"Expected CAM_FRONT directory under {samples_root}")

        front_paths: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            front_paths.extend(front_dir.glob(ext))
        front_paths = sorted(front_paths)

        sample_entries: List[Dict[str, str]] = []
        for front_path in front_paths:
            filename = front_path.name
            cam_paths: Dict[str, str] = {}
            valid = True
            for cam in self.cameras:
                p = samples_root / cam / filename
                cam_paths[cam] = str(p)
                if not p.exists():
                    valid = False
                    break
            if valid:
                sample_entries.append(cam_paths)
        return sample_entries

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        try:
            img = Image.open(path).convert("RGB")
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError(f"Invalid image size: {img.size}")
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load image {path}: {e}. Using black placeholder.")
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        cam_paths = self.samples[idx]
        imgs = [self._load_image(cam_paths[cam]) for cam in self.cameras]
        multi_view = torch.stack(imgs, dim=0)  # (6, C, H, W)
        return multi_view, cam_paths


def get_dataloader(
    data_root: str,
    dataset: str = "nuscenes",
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Optional[int] = 224,
    shuffle: bool = True,
    multi_view: bool = False,
):
    """
    Get DataLoader for nuScenes / Waymo / coco.
    If image_size is None: no resize (original resolution); batch_size is forced to 1.
    """
    if image_size is None:
        batch_size = 1
    transform = default_transform(image_size)
    root = Path(data_root)

    if dataset == "nuscenes":
        if multi_view:
            dataset_obj = NuScenesMultiViewDataset(str(root), transform=transform)
            loader = DataLoader(
                dataset_obj,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
            return loader, dataset_obj
        candidates = [
            root / "samples" / "CAM_FRONT",
            # root / "images",
            root,
        ]
    elif dataset == "waymo":
        candidates = [
            root / "images",
            root / "camera_image",
            root,
        ]
    else:
        candidates = [root / "train2017", root / "val2017", root]

    data_path = root
    for c in candidates:
        if c.exists():
            data_path = c
            break

    dataset_obj = ImagePathDataset(str(data_path), transform=transform)
    loader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset_obj
