"""Phase 3: fine-tune the evidential confidence head on the recorded dataset's
ground-truth detection crops (see `common.config` / CLAUDE.md for the current
dataset -- data/raw_v2, 15,420 frames).

Quick-iteration trainer: plain Adam at a constant learning rate, saves the
LAST epoch. `scripts/generate_report_figures.py:train_and_evaluate` is the
trainer of record (adds cosine LR annealing, reports the converged mean
rather than the best epoch) -- mirror any model/loss change there too.
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.evidential_classifier import CROP_SIZE, NUM_CLASSES, EvidentialDetector
from models.evidential_head import EvidentialHead, evidential_mse_loss

CLASS_VEHICLE, CLASS_PEDESTRIAN, CLASS_BACKGROUND = 0, 1, 2
CLASS_NAMES = {CLASS_VEHICLE: "vehicle", CLASS_PEDESTRIAN: "pedestrian", CLASS_BACKGROUND: "background"}
PAD_FRAC = 0.15  # pad each labeled bbox by this fraction before cropping


def episode_split(dataset, val_frac: float, seed: int = 0):
    """Splits crops into train/val by EPISODE, never by individual crop.

    A random per-crop split leaks badly on this dataset and inflates the
    reported accuracy. Episodes are ~60 consecutive frames of the same scene;
    in the parked-ego scenarios the only thing that moves is one walker, so
    consecutive frames are near-duplicates. Splitting crops at random puts
    frame 31 of an episode in training and frame 32 in validation, and the
    model is then scored on recognising the same bus it just trained on.

    The previously reported 99.7% validation accuracy came from a per-crop
    `random_split`. Expect a grouped split to score lower -- that lower number
    is the one that means anything.
    """
    episodes = sorted({dataset.episode_of(i) for i in range(len(dataset))})
    rng = random.Random(seed)
    rng.shuffle(episodes)

    n_val = max(1, int(round(len(episodes) * val_frac)))
    val_episodes = set(episodes[:n_val])

    val_idx = [i for i in range(len(dataset)) if dataset.episode_of(i) in val_episodes]
    train_idx = [i for i in range(len(dataset)) if dataset.episode_of(i) not in val_episodes]
    # Augmentation applies ONLY to the train side -- validation must stay a
    # deterministic read of the exact recorded crop, or accuracy stops
    # measuring the model and starts measuring which random augmentation
    # landed on which held-out sample.
    return (AugmentedCrops(dataset, train_idx),
            torch.utils.data.Subset(dataset, val_idx),
            len(episodes) - n_val, n_val)


class CropDataset(Dataset):
    def __init__(self, data_dir: str, background_per_image: int = 2, seed: int = 0):
        self.data_dir = Path(data_dir)
        self.samples = []  # list of (image_path, x1, y1, x2, y2, class_id)
        rng = random.Random(seed)

        image_dir = self.data_dir / "images"
        label_dir = self.data_dir / "labels"
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = image_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                continue
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            boxes_px = []
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
                cx, cy, pw, ph = xc * w, yc * h, bw * w, bh * h
                pw *= (1 + PAD_FRAC)
                ph *= (1 + PAD_FRAC)
                x1, y1 = max(0, cx - pw / 2), max(0, cy - ph / 2)
                x2, y2 = min(w, cx + pw / 2), min(h, cy + ph / 2)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                boxes_px.append((x1, y1, x2, y2))
                self.samples.append((str(image_path), x1, y1, x2, y2, cls))

            # Background crops: random boxes that don't (much) overlap any labeled box.
            for _ in range(background_per_image):
                for _try in range(10):
                    bw_bg, bh_bg = rng.uniform(30, 100), rng.uniform(30, 100)
                    x1 = rng.uniform(0, max(1, w - bw_bg))
                    y1 = rng.uniform(0, max(1, h - bh_bg))
                    x2, y2 = x1 + bw_bg, y1 + bh_bg
                    if not any(self._iou((x1, y1, x2, y2), b) > 0.05 for b in boxes_px):
                        self.samples.append((str(image_path), x1, y1, x2, y2, CLASS_BACKGROUND))
                        break

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def episode_of(self, idx: int) -> str:
        """The episode a sample came from, parsed from its filename stem
        (`<scenario>_<weather>_ep<NNNNN>_f<NNNN>`)."""
        stem = Path(self.samples[idx][0]).stem
        return stem.rsplit("_f", 1)[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, x1, y1, x2, y2, cls = self.samples[idx]
        img = cv2.imread(path)
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(crop).permute(2, 0, 1)
        return tensor, cls


class AugmentedCrops(Dataset):
    """Train-time-only wrapper: random horizontal flip + mild brightness jitter.

    Added because the trainer had NO augmentation at all -- every one of the
    50,904 training crops was the raw recorded pixel data, once. With training
    concentrated in 161 episodes (long, slow-changing sequences -- a walker
    crossing over dozens of near-identical frames), that leaves generalisation
    on the table for free: neither a vehicle nor a pedestrian crop has an
    inherent left/right handedness, so a horizontal flip is a free additional
    view with no risk of teaching a wrong invariance. Brightness jitter is a
    coarse stand-in for the lighting variation a single-weather dataset cannot
    otherwise provide (CARLA 0.10.0's weather API does not work on this build --
    see docs/RESULTS.md).

    Deliberately NOT wrapping the base CropDataset's `__getitem__` directly:
    that would apply augmentation to validation crops too, since `episode_split`
    hands out `Subset` views over the same underlying dataset object.
    """
    FLIP_PROB = 0.5
    BRIGHTNESS_PROB = 0.5
    BRIGHTNESS_RANGE = (0.8, 1.2)

    def __init__(self, base: Dataset, indices: list[int]):
        self.base = base
        self.indices = indices
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        tensor, cls = self.base[self.indices[i]]
        if self.rng.random() < self.FLIP_PROB:
            tensor = torch.flip(tensor, dims=[2])  # width axis of (C, H, W)
        if self.rng.random() < self.BRIGHTNESS_PROB:
            factor = float(self.rng.uniform(*self.BRIGHTNESS_RANGE))
            tensor = (tensor * factor).clamp(0.0, 1.0)
        return tensor, cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--out", default="models/evidential_detector.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = CropDataset(args.data_dir)
    print(f"Loaded {len(dataset)} crops "
          f"(vehicle={sum(1 for s in dataset.samples if s[5] == CLASS_VEHICLE)}, "
          f"pedestrian={sum(1 for s in dataset.samples if s[5] == CLASS_PEDESTRIAN)}, "
          f"background={sum(1 for s in dataset.samples if s[5] == CLASS_BACKGROUND)})")

    train_set, val_set, n_train_ep, n_val_ep = episode_split(dataset, args.val_frac)
    print(f"Split by episode: {n_train_ep} train / {n_val_ep} val episodes "
          f"({len(train_set)} / {len(val_set)} crops) -- no episode appears in both.")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    model = EvidentialDetector(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for crops, labels in train_loader:
            crops, labels = crops.to(device), labels.to(device)
            target_onehot = F.one_hot(labels, NUM_CLASSES).float()

            alpha, _ = model(crops)
            loss = evidential_mse_loss(alpha, target_onehot, epoch, annealing_epochs=10)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * crops.size(0)

        train_loss = total_loss / max(len(train_set), 1)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for crops, labels in val_loader:
                crops, labels = crops.to(device), labels.to(device)
                alpha, _ = model(crops)
                pred = EvidentialHead.expected_probability(alpha).argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(total, 1)
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} val_acc={val_acc:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved model to {args.out}")

    # Sample predictions, sorted by uncertainty, to show the head behaving as
    # intended: harder/ambiguous crops (small, truncated, occluded) should
    # show up with visibly higher uncertainty than clean, unoccluded ones.
    print("\nSample predictions on validation crops (sorted by uncertainty):")
    model.eval()
    results = []
    with torch.no_grad():
        for crops, labels in val_loader:
            crops_dev = crops.to(device)
            alpha, uncertainty = model(crops_dev)
            probs = EvidentialHead.expected_probability(alpha)
            for i in range(crops.size(0)):
                results.append((uncertainty[i].item(), labels[i].item(),
                                 probs[i].cpu().numpy(), alpha[i].sum().item()))
    results.sort(key=lambda r: -r[0])
    for uncertainty, true_cls, probs, evidence_sum in results[:5]:
        print(f"  HIGH uncertainty={uncertainty:.3f} true={CLASS_NAMES[true_cls]:<10} "
              f"probs={dict(zip(CLASS_NAMES.values(), probs.round(2)))} evidence_sum={evidence_sum:.1f}")
    for uncertainty, true_cls, probs, evidence_sum in results[-5:]:
        print(f"  LOW  uncertainty={uncertainty:.3f} true={CLASS_NAMES[true_cls]:<10} "
              f"probs={dict(zip(CLASS_NAMES.values(), probs.round(2)))} evidence_sum={evidence_sum:.1f}")


if __name__ == "__main__":
    main()
