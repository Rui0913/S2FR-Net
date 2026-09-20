"""Dataset utilities for pan-sharpening experiments."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


_MS_AND_TARGET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
_PAN_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


class PanSharpeningDataset(Dataset):
    """Load aligned PAN, multispectral, and ground-truth images."""

    def __init__(self, image_dir: str | Path) -> None:
        self.root = Path(image_dir)
        self.pan_dir = self.root / "pan"
        self.ms_dir = self.root / "ms"
        self.label_dir = self.root / "label"

        for directory in (self.pan_dir, self.ms_dir, self.label_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

        self.filenames = sorted(
            path.name for path in self.label_dir.iterdir() if path.is_file()
        )
        if not self.filenames:
            raise ValueError(f"No label images found in {self.label_dir}")

        missing = [
            name
            for name in self.filenames
            if not (self.pan_dir / name).is_file()
            or not (self.ms_dir / name).is_file()
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"Missing aligned PAN/MS files for: {preview}")

    @staticmethod
    def _read_image(
        path: Path, flags: int, expected_shape: tuple[int, ...]
    ) -> np.ndarray:
        image = cv2.imread(str(path), flags=flags)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        if image.shape != expected_shape:
            raise ValueError(
                f"Unexpected image shape for {path}: "
                f"expected {expected_shape}, got {image.shape}"
            )
        return image

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        pan = self._read_image(
            self.pan_dir / filename, cv2.IMREAD_GRAYSCALE, (256, 256)
        )
        ms = self._read_image(
            self.ms_dir / filename, cv2.IMREAD_COLOR, (64, 64, 3)
        )
        target = self._read_image(
            self.label_dir / filename, cv2.IMREAD_COLOR, (256, 256, 3)
        )

        pan = np.expand_dims(pan, axis=-1)
        return (
            _PAN_TRANSFORM(pan),
            _MS_AND_TARGET_TRANSFORM(ms),
            _MS_AND_TARGET_TRANSFORM(target),
            filename,
        )

    def __len__(self) -> int:
        return len(self.filenames)
