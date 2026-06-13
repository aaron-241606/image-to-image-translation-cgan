"""
Task-04: Paired Image Dataset for pix2pix
Prodigy Infotech Internship
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    """
    Loads paired images stored as side-by-side (src|tgt) images,
    or from separate source/ and target/ subdirectories.
    """

    def __init__(self, root_dir, transform=None, split="train"):
        self.transform = transform
        self.src_dir = os.path.join(root_dir, split, "source")
        self.tgt_dir = os.path.join(root_dir, split, "target")

        # Fallback: side-by-side images in a single folder
        self.side_by_side = False
        if not os.path.exists(self.src_dir):
            self.src_dir = os.path.join(root_dir, split)
            self.side_by_side = True

        self.files = sorted([
            f for f in os.listdir(self.src_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.src_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")

        if self.side_by_side:
            w, h = img.size
            src = img.crop((0, 0, w // 2, h))
            tgt = img.crop((w // 2, 0, w, h))
        else:
            tgt_path = os.path.join(self.tgt_dir, self.files[idx])
            tgt = Image.open(tgt_path).convert("RGB")
            src = img

        if self.transform:
            src = self.transform(src)
            tgt = self.transform(tgt)

        return src, tgt
