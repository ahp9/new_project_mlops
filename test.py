from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import typer

from datasets import load_dataset
from PIL import Image


DATASET_ID = "lukbl/LaTeX-OCR-dataset"


def pil_to_tensor_rgb(img: Image.Image, size: Tuple[int, int]) -> torch.Tensor:
    """
    Convert PIL image -> torch float tensor in [0,1], shape (3, H, W).
    size: (W, H)
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(size, resample=Image.BICUBIC)

    # PIL -> torch: (H,W,C) uint8 -> (C,H,W) float
    x = torch.from_numpy(__import__("numpy").array(img))  # lazy import numpy
    x = x.permute(2, 0, 1).contiguous().float() / 255.0
    return x


@dataclass
class SplitMeta:
    num_samples: int
    chunk_size: int
    num_chunks: int
    image_size: Tuple[int, int]  # (W,H)


class LatexOCRDataset(Dataset):
    """
    Dataset reading preprocessed chunked tensors:
      processed/
        latex_ocr/
          train_images_0.pt
          train_text_0.pt
          ...
          meta_train.json
    """

    def __init__(self, processed_dir: Path, split: str = "train") -> None:
        self.processed_dir = processed_dir
        self.split = split

        meta_path = processed_dir / f"meta_{split}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}. Run preprocessing first.")

        meta = json.loads(meta_path.read_text())
        self.meta = SplitMeta(
            num_samples=meta["num_samples"],
            chunk_size=meta["chunk_size"],
            num_chunks=meta["num_chunks"],
            image_size=tuple(meta["image_size"]),
        )

        self._cached_chunk_idx: Optional[int] = None
        self._cached_images: Optional[torch.Tensor] = None
        self._cached_text: Optional[List[str]] = None

    def __len__(self) -> int:
        return self.meta.num_samples

    def _load_chunk(self, chunk_idx: int) -> None:
        img_path = self.processed_dir / f"{self.split}_images_{chunk_idx}.pt"
        txt_path = self.processed_dir / f"{self.split}_text_{chunk_idx}.pt"

        self._cached_images = torch.load(img_path, map_location="cpu")
        self._cached_text = torch.load(txt_path)
        self._cached_chunk_idx = chunk_idx

    def __getitem__(self, index: int):
        chunk_idx = index // self.meta.chunk_size
        within = index % self.meta.chunk_size

        if self._cached_chunk_idx != chunk_idx:
            self._load_chunk(chunk_idx)

        assert self._cached_images is not None and self._cached_text is not None
        # last chunk may be shorter
        if within >= self._cached_images.shape[0]:
            raise IndexError("Index out of range inside last chunk.")

        image = self._cached_images[within]  # (3,H,W) float
        text = self._cached_text[within]  # LaTeX string
        return image, text


def download_and_preprocess(
    output_folder: Path,
    image_width: int = 192,
    image_height: int = 64,
    chunk_size: int = 2048,
) -> None:
    """
    Downloads HF dataset and writes chunked .pt files:
      train_images_k.pt, train_text_k.pt
      validation_images_k.pt, validation_text_k.pt
      plus meta_train.json, meta_validation.json
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    size = (image_width, image_height)

    # LaTeX-OCR dataset provides "train" and "validation" splits on HF. :contentReference[oaicite:1]{index=1}
    splits = ["train", "validation"]

    for split in splits:
        ds = load_dataset(DATASET_ID, split=split)  # downloads + caches via HF
        n = len(ds)

        num_chunks = (n + chunk_size - 1) // chunk_size
        meta = {
            "num_samples": n,
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "image_size": [image_width, image_height],
        }
        (output_folder / f"meta_{split}.json").write_text(json.dumps(meta, indent=2))

        images_buf: List[torch.Tensor] = []
        text_buf: List[str] = []
        chunk_idx = 0

        def flush():
            nonlocal chunk_idx, images_buf, text_buf
            if not images_buf:
                return
            imgs = torch.stack(images_buf, dim=0)  # (B,3,H,W)
            torch.save(imgs, output_folder / f"{split}_images_{chunk_idx}.pt")
            torch.save(text_buf, output_folder / f"{split}_text_{chunk_idx}.pt")
            images_buf = []
            text_buf = []
            chunk_idx += 1

        for i in range(n):
            sample = ds[i]
            img = sample["image"]  # PIL image
            txt = sample["text"]  # LaTeX string

            x = pil_to_tensor_rgb(img, size=size)
            images_buf.append(x)
            text_buf.append(txt)

            if len(images_buf) >= chunk_size:
                flush()

        flush()


def preprocess(data_path: Path, output_folder: Path) -> None:
    """
    Keep same CLI signature as your template:
      - data_path exists but is not used here (HF downloads itself)
      - output_folder is where processed tensors go
    """
    print("Downloading + preprocessing LaTeX-OCR dataset...")
    # You can choose to ignore data_path, or store HF cache under data_path if you want.
    _ = data_path
    download_and_preprocess(output_folder=output_folder)
    print("Done.")


if __name__ == "__main__":
    typer.run(preprocess)
