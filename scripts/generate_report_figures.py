"""Generates publication-quality figures + a metrics summary for the Phase
1-4 work, for use in the journal writeup. Everything here is measured, not
illustrative: real held-out accuracy, a real training curve, and occlusion
grid agreement computed over a sample of actual collected frames (not the
single hand-picked frame used for the earlier qualitative check).

Outputs (results/):
  figures/dataset_composition.png   -- frames per scenario x weather
  figures/training_curves.png       -- evidential head loss + val accuracy per epoch
  figures/confusion_matrix.png      -- 3-class confusion matrix on held-out crops
  figures/uncertainty_calibration.png -- accuracy vs. predicted uncertainty (selective prediction)
  figures/occlusion_grid_validation.png -- predicted vs ground-truth occlusion agreement
  metrics.json                      -- every number plotted above, machine-readable
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from common.config import load_yaml
from models.evidential_classifier import NUM_CLASSES, EvidentialDetector
from models.evidential_head import EvidentialHead, evidential_mse_loss
from models.train_evidential import (
    CLASS_BACKGROUND, CLASS_NAMES, CLASS_PEDESTRIAN, CLASS_VEHICLE, CropDataset, episode_split,
)
from perception.occlusion_grid import OCCLUDED, classify_grid
from carla_tools.occlusion_mask import OCCLUDED as GT_OCCLUDED

OUT_DIR = Path("results")
FIG_DIR = OUT_DIR / "figures"
# Default to the v2 dataset. The original hardcoded "data/raw" broke silently
# once v1 was archived: CropDataset simply found no images, the episode split
# returned zero crops, and the failure surfaced as an opaque DataLoader error
# about num_samples rather than "your dataset path is wrong".
DATA_DIR = Path(os.environ.get("DATASET_DIR", "data/raw_v2"))

# Epochs averaged for the reported accuracy. See train_and_evaluate for why the
# best epoch is not reported.
CONVERGED_EPOCHS = 5
sns.set_theme(style="whitegrid")


def fig_dataset_composition():
    counts = Counter()
    for p in (DATA_DIR / "images").glob("*.jpg"):
        # filenames: "<scenario>_<weather>_ep#####_f####.jpg"
        parts = p.stem.split("_ep")[0]
        counts[parts] += 1

    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(labels)), values, color=sns.color_palette("viridis", len(labels)))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
    ax.set_ylabel("frames")
    ax.set_title(f"Phase 2 -- pilot dataset composition ({sum(values)} frames total)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 3, str(v), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dataset_composition.png", dpi=150)
    plt.close(fig)
    return dict(zip(labels, values))


def train_and_evaluate(device: str):
    dataset = CropDataset(str(DATA_DIR))
    # Grouped by episode, not per-crop -- see `train_evidential.episode_split`.
    # A per-crop split leaked near-duplicate frames across the boundary and is
    # what produced the previously reported 99.7%.
    train_set, val_set, n_train_ep, n_val_ep = episode_split(dataset, 0.15)
    print(f"  split by episode: {n_train_ep} train / {n_val_ep} val episodes "
          f"({len(train_set)} / {len(val_set)} crops)")
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    model = EvidentialDetector(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # A constant 1e-3 LR caused visible val-accuracy oscillation on this
    # small pilot dataset (dips to ~0.67 mid-training even though loss kept
    # falling) -- cosine decay smooths that out without needing more data.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    epochs = 15
    history = {"epoch": [], "train_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    best_state = None
    for epoch in range(epochs):
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
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        print(f"  epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} val_acc={val_acc:.3f} lr={current_lr:.2e}")

    # Report/save the best checkpoint by held-out accuracy, not whichever
    # epoch happened to be last -- standard practice, and avoids the final
    # numbers depending on getting lucky with the last epoch.
    model.load_state_dict(best_state)
    acc = np.array(history["val_acc"])
    history["best_epoch"] = int(acc.argmax() + 1)
    history["final_val_acc"] = float(acc[-1])
    history["converged_val_acc"] = float(acc[-CONVERGED_EPOCHS:].mean())
    history["converged_val_std"] = float(acc[-CONVERGED_EPOCHS:].std())
    history["val_acc_std"] = float(acc.std())

    # The BEST epoch is restored for saving -- you want the better weights on
    # disk -- but it is NOT the accuracy to report.
    #
    # Under the grouped-by-episode split, validation accuracy swings hard from
    # epoch to epoch: measured 0.593 to 0.965, std 0.092, over only 8 held-out
    # episodes. Taking the maximum of 15 such draws is selecting on noise, and
    # is optimistically biased by roughly five points. The original code chose
    # best-epoch deliberately and docs/RESULTS.md defends it as a methodological
    # choice -- which it was, under the old per-crop split where consecutive
    # frames made validation nearly deterministic. It does not survive the move
    # to a grouped split.
    #
    # The converged mean over the last few epochs is the honest headline, with
    # its spread reported alongside.
    print(f"  best epoch {history['best_epoch']}: {best_val_acc:.3f} (restored for saving)")
    print(f"  REPORTED accuracy: {history['converged_val_acc']:.3f} "
          f"+/- {history['converged_val_std']:.3f} "
          f"(mean of last {CONVERGED_EPOCHS} epochs; best-of-15 would be "
          f"{best_val_acc:.3f}, which selects on noise)")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/evidential_detector.pt")

    # Full-precision pass for confusion matrix + calibration data.
    model.eval()
    all_true, all_pred, all_unc, all_correct = [], [], [], []
    with torch.no_grad():
        for crops, labels in val_loader:
            crops_dev = crops.to(device)
            alpha, uncertainty = model(crops_dev)
            probs = EvidentialHead.expected_probability(alpha)
            pred = probs.argmax(dim=-1).cpu().numpy()
            all_true.extend(labels.numpy().tolist())
            all_pred.extend(pred.tolist())
            all_unc.extend(uncertainty.cpu().numpy().flatten().tolist())
            all_correct.extend((pred == labels.numpy()).tolist())

    return history, np.array(all_true), np.array(all_pred), np.array(all_unc), np.array(all_correct)


def fig_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(history["epoch"], history["train_loss"], marker="o", color="#d03b3b")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss (evidential MSE + KL)")
    ax1.set_title("Phase 3 -- training loss")

    ax2.plot(history["epoch"], history["val_acc"], marker="o", color="#0ca30c")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("held-out validation accuracy")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Phase 3 -- validation accuracy")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def fig_confusion_matrix(y_true, y_pred):
    labels = [CLASS_VEHICLE, CLASS_PEDESTRIAN, CLASS_BACKGROUND]
    names = [CLASS_NAMES[c] for c in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names,
                cbar_kws={"label": "row-normalized"}, ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Phase 3 -- evidential classifier confusion matrix\n(held-out crops)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return cm.tolist()


def fig_uncertainty_calibration(uncertainty, correct):
    """The core evidential-learning claim: accuracy should fall as
    uncertainty rises -- i.e. the model's confidence is trustworthy, not
    just high-accuracy-on-average."""
    bins = np.linspace(0, 1, 6)
    bin_idx = np.clip(np.digitize(uncertainty, bins) - 1, 0, len(bins) - 2)
    bin_acc, bin_n, bin_centers = [], [], []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        bin_n.append(n)
        bin_acc.append(float(correct[mask].mean()) if n > 0 else np.nan)
        bin_centers.append((bins[b] + bins[b + 1]) / 2)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.bar(bin_centers, bin_acc, width=0.15, color="#3b7dd0", alpha=0.85, label="accuracy in bin")
    ax1.set_xlabel("predicted uncertainty"); ax1.set_ylabel("accuracy", color="#3b7dd0")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, bin_n, color="#d0893b", marker="o", label="# samples")
    ax2.set_ylabel("# samples in bin", color="#d0893b")
    ax1.set_title("Phase 3 -- uncertainty calibration\n(accuracy should fall as uncertainty rises)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "uncertainty_calibration.png", dpi=150)
    plt.close(fig)
    return {"bin_centers": bin_centers, "bin_accuracy": bin_acc, "bin_n": bin_n}


def fig_occlusion_grid_validation(bev_cfg, n_samples=60, seed=0):
    """Quantitative agreement between the camera(MiDaS)+radar-only occlusion
    grid and CARLA's ground-truth occlusion grid, sampled across many frames
    -- not just the one hand-picked frame from the earlier qualitative
    check. Ground truth only has VISIBLE/OCCLUDED/UNKNOWN (3 classes); our
    grid also has EMPTY, which is folded into "not occluded" for comparison
    since ground truth doesn't distinguish "camera confirms clear" from
    "radar confirms visible"."""
    import cv2
    rng = np.random.RandomState(seed)
    image_paths = sorted((DATA_DIR / "images").glob("*.jpg"))
    if len(image_paths) > n_samples:
        image_paths = [image_paths[i] for i in rng.choice(len(image_paths), n_samples, replace=False)]

    tp = fp = fn = tn = 0  # OCCLUDED as the positive class
    per_scenario = defaultdict(lambda: {"agree": 0, "total": 0})

    for img_path in image_paths:
        meta_path = DATA_DIR / "meta" / f"{img_path.stem}.npz"
        if not meta_path.exists():
            continue
        meta = np.load(meta_path, allow_pickle=True)
        gt_grid = meta["occ_grid"]
        radar_pts = meta["radar_pts"]
        scenario = str(meta["scenario_type"])

        img = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred_grid = classify_grid(rgb, radar_pts, bev_cfg)

        pred_occluded = pred_grid == OCCLUDED
        gt_occluded = gt_grid == GT_OCCLUDED

        tp += int(np.sum(pred_occluded & gt_occluded))
        fp += int(np.sum(pred_occluded & ~gt_occluded))
        fn += int(np.sum(~pred_occluded & gt_occluded))
        tn += int(np.sum(~pred_occluded & ~gt_occluded))

        agree = int(np.sum(pred_occluded == gt_occluded))
        per_scenario[scenario]["agree"] += agree
        per_scenario[scenario]["total"] += gt_grid.size

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    overall_agreement = (tp + tn) / max(tp + fp + fn + tn, 1)

    scenarios = sorted(per_scenario.keys())
    agreements = [per_scenario[s]["agree"] / max(per_scenario[s]["total"], 1) for s in scenarios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    metrics_names = ["precision", "recall", "F1", "cell agreement"]
    metrics_vals = [precision, recall, f1, overall_agreement]
    ax1.bar(metrics_names, metrics_vals, color=sns.color_palette("rocket", 4))
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"Phase 4 -- OCCLUDED-cell detection quality\n(n={len(image_paths)} sampled frames, "
                  f"{tp + fp + fn + tn} cells)")
    for i, v in enumerate(metrics_vals):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    ax2.bar([s.replace("_", "\n") for s in scenarios], agreements, color=sns.color_palette("mako", len(scenarios)))
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("cell-level agreement with ground truth")
    ax2.set_title("Phase 4 -- agreement by scenario")
    for i, v in enumerate(agreements):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "occlusion_grid_validation.png", dpi=150)
    plt.close(fig)

    return {
        "n_frames_sampled": len(image_paths),
        "precision_occluded": precision,
        "recall_occluded": recall,
        "f1_occluded": f1,
        "overall_cell_agreement": overall_agreement,
        "per_scenario_agreement": dict(zip(scenarios, agreements)),
    }


def main():
    if not (DATA_DIR / "images").is_dir():
        raise SystemExit(
            f"No images under {DATA_DIR}. Set DATASET_DIR to a collected dataset, "
            f"e.g. DATASET_DIR=data/raw_v2")
    n_img = len(list((DATA_DIR / "images").glob("*.jpg")))
    print(f"Dataset: {DATA_DIR} ({n_img} frames)")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bev_cfg = load_yaml("bev.yaml")
    metrics = {}

    print("Dataset composition...")
    metrics["dataset_composition"] = fig_dataset_composition()

    print("Training evidential head (with history logging) + evaluating...")
    history, y_true, y_pred, uncertainty, correct = train_and_evaluate(device)
    metrics["training_history"] = history
    metrics["best_val_accuracy"] = max(history["val_acc"])
    metrics["best_epoch"] = history["best_epoch"]
    metrics["reported_val_accuracy"] = history["converged_val_acc"]
    metrics["reported_val_std"] = history["converged_val_std"]

    print("Training curves...")
    fig_training_curves(history)

    print("Confusion matrix...")
    metrics["confusion_matrix"] = fig_confusion_matrix(y_true, y_pred)

    print("Uncertainty calibration...")
    metrics["uncertainty_calibration"] = fig_uncertainty_calibration(uncertainty, correct)

    print("Occlusion grid validation (this runs MiDaS over ~60 frames, may take a minute)...")
    metrics["occlusion_grid_validation"] = fig_occlusion_grid_validation(bev_cfg, n_samples=200)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Figures in {FIG_DIR}/, metrics in {OUT_DIR / 'metrics.json'}")
    print(f"Validation accuracy: {metrics['reported_val_accuracy']:.3f} "
          f"+/- {metrics['reported_val_std']:.3f}  "
          f"(best epoch was {metrics['best_val_accuracy']:.3f} -- not reported, see above)")
    ov = metrics["occlusion_grid_validation"]
    print(f"Occlusion detection: precision={ov['precision_occluded']:.2f} recall={ov['recall_occluded']:.2f} "
          f"f1={ov['f1_occluded']:.2f} cell_agreement={ov['overall_cell_agreement']:.2f}")


if __name__ == "__main__":
    main()
