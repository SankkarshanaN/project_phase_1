"""Phase 4 verification: run the camera+radar-only occlusion_grid classifier
on a captured frame and compare it against the ground-truth occ_grid saved
during Phase 2 collection (from CARLA's own depth buffer -- see
carla_tools.occlusion_mask). Saves a side-by-side visualization: RGB frame,
predicted grid, ground-truth grid.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carla_tools.occlusion_mask import labels_to_rgb as gt_labels_to_rgb
from common.config import load_yaml
from perception.occlusion_grid import classify_grid, labels_to_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-name", required=True,
                         help="Basename (no extension) shared by images/, labels/, meta/")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    bev_cfg = load_yaml("bev.yaml")

    img_bgr = cv2.imread(str(data_dir / "images" / f"{args.frame_name}.jpg"))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    meta = np.load(data_dir / "meta" / f"{args.frame_name}.npz", allow_pickle=True)
    radar_pts = meta["radar_pts"]
    gt_occ_grid = meta["occ_grid"]

    pred_grid = classify_grid(rgb, radar_pts, bev_cfg)

    pred_vis = labels_to_rgb(pred_grid)
    gt_vis = gt_labels_to_rgb(gt_occ_grid)

    cell_px = 20
    pred_big = cv2.resize(pred_vis, (pred_vis.shape[1] * cell_px, pred_vis.shape[0] * cell_px),
                           interpolation=cv2.INTER_NEAREST)
    gt_big = cv2.resize(gt_vis, (gt_vis.shape[1] * cell_px, gt_vis.shape[0] * cell_px),
                         interpolation=cv2.INTER_NEAREST)

    rgb_resized = cv2.resize(img_bgr, (pred_big.shape[1], pred_big.shape[0]))
    combined = np.hstack([rgb_resized, cv2.cvtColor(pred_big, cv2.COLOR_RGB2BGR),
                           cv2.cvtColor(gt_big, cv2.COLOR_RGB2BGR)])

    out_path = args.out or f"{args.frame_name}_occlusion_check.jpg"
    cv2.imwrite(out_path, combined)
    print(f"Saved {out_path} (left: RGB, middle: predicted grid [camera+radar only], "
          f"right: ground truth [CARLA depth buffer])")

    print("\nPredicted grid label counts:", {int(k): int(v) for k, v in zip(*np.unique(pred_grid, return_counts=True))})
    print("Ground-truth grid label counts:", {int(k): int(v) for k, v in zip(*np.unique(gt_occ_grid, return_counts=True))})


if __name__ == "__main__":
    main()
