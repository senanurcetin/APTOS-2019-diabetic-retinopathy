"""Model, data loading, and the epoch loop.

Ported from scripts/train.py with the behaviour deliberately unchanged, so any
new result stays comparable with the six recorded single-split runs. What did
change: the constants come from config instead of being written inline in two
places, and the class-weight construction no longer breaks when a grade is
missing from a split.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from aptos.config import N_GRADES, AugmentationConfig, Config
from aptos.preprocessing import IMAGENET_MEAN, IMAGENET_STD


def set_seed(seed: int) -> None:
    """Same seed, same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------ data

class FoldDataset(Dataset):
    """Read each image from its ORIGINAL split directory.

    Under cross-validation an image's role - training or validation - changes
    from fold to fold, but its location on disk does not. So the directory comes
    from `orig_split`, never from the fold assignment.
    """

    def __init__(self, df, transform, root):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        if "orig_split" not in self.df.columns:
            raise KeyError(
                "FoldDataset needs an 'orig_split' column - use labels.pool_and_holdout()"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.root / row["orig_split"] / f"{row['id_code']}.jpg"
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(int(row["diagnosis"]))


def build_transforms(size: int, aug: AugmentationConfig | None = None):
    """Training augmentations and the plain evaluation transform.

    Vertical flip is included because a fundus photograph has no meaningful
    up/down orientation. Rotation and scale jitter cover variation in how the
    camera was aimed. Colour jitter stays mild: large shifts would fight CLAHE,
    which normalises local contrast on purpose.
    """
    aug = aug or AugmentationConfig()
    mean, std = IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()

    train = T.Compose([
        T.RandomResizedCrop(size, scale=aug.resized_crop_scale),
        T.RandomHorizontalFlip(aug.hflip_p),
        T.RandomVerticalFlip(aug.vflip_p),
        T.RandomRotation(aug.rotation_degrees),
        T.ColorJitter(brightness=aug.color_jitter_brightness,
                      contrast=aug.color_jitter_contrast),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    evaluate = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train, evaluate


def make_loader(df, transform, root, cfg: Config, *, shuffle: bool) -> DataLoader:
    workers = cfg.train.workers
    return DataLoader(
        FoldDataset(df, transform, root),
        batch_size=cfg.train.batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        # Windows workers re-import the main module by path, so keeping them
        # alive across epochs avoids paying that cost repeatedly - and avoids
        # the churn that helped exhaust the commit limit in earlier runs.
        persistent_workers=cfg.train.persistent_workers and workers > 0,
    )


# ----------------------------------------------------------------------- model

def build_model(cfg: Config, device: str):
    import timm

    num_classes = 1 if cfg.train.mode == "reg" else N_GRADES
    return timm.create_model(cfg.train.model, pretrained=True, num_classes=num_classes).to(device)


def build_criterion(cfg: Config, train_df, device: str):
    """MSE for ordinal regression, class-weighted cross-entropy for classification.

    The weight vector is built over all five grades explicitly. The original
    used `value_counts()` directly, which silently produced a shorter tensor
    when a grade was absent from the training split - and then crashed inside
    the loss.
    """
    if cfg.train.mode == "reg":
        return nn.MSELoss()

    counts = np.zeros(N_GRADES, dtype=float)
    observed = train_df["diagnosis"].value_counts()
    for grade, count in observed.items():
        counts[int(grade)] = count

    if (counts == 0).any():
        missing = [int(g) for g in np.flatnonzero(counts == 0)]
        # A present-but-unseen grade gets weight 0 rather than an infinite one.
        print(f"  warning: grade(s) {missing} absent from this training split; "
              f"their loss weight is 0")
        counts[counts == 0] = np.inf

    weights = counts.sum(where=np.isfinite(counts)) / (N_GRADES * counts)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def build_optimizer(cfg: Config, model):
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )


# ------------------------------------------------------------------ epoch loop

def run_epoch(model, loader, criterion, device, mode, optimizer=None, scaler=None):
    """One pass. `optimizer` present means training."""
    training = optimizer is not None
    model.train(training)
    total_loss, preds, trues = 0.0, [], []

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device)
            target = labels.float().unsqueeze(1) if mode == "reg" else labels

            with torch.autocast("cuda", enabled=scaler is not None):
                out = model(imgs)
                loss = criterion(out, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            raw = out.squeeze(1) if mode == "reg" else out.argmax(1)
            preds.append(raw.float().detach().cpu().numpy())
            trues.append(labels.cpu().numpy())

    return total_loss / len(loader.dataset), np.concatenate(preds), np.concatenate(trues)
