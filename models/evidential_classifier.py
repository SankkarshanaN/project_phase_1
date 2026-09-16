"""Detection-confidence classifier: a small CNN crop encoder feeding the
EvidentialHead, trained on the bounding-box crops from the Phase 2 dataset.

Design note: rather than performing surgery on YOLOv8's internal multi-scale
DFL detection head (which is tightly coupled to ultralytics' anchor-free loss
and decode logic), the evidential head is attached as a separate classifier
over each detected crop -- YOLOv8n proposes WHERE an object is, this module
scores WHAT it is and HOW CONFIDENT that call is, per the spec's "camera
confidence score" + "uncertainty flag" outputs. This keeps YOLO's proven
detection quality intact and is directly trainable on our own ground-truth
boxes without depending on YOLO's own (unfine-tuned) box quality.

Classes: 0=vehicle, 1=pedestrian, 2=background (matches
carla_tools.data_collector's CLASS_VEHICLE/CLASS_PEDESTRIAN, plus a
background class sampled from crops that don't overlap any labeled box).
"""
import torch
import torch.nn as nn

from models.evidential_head import EvidentialHead

CROP_SIZE = 64
NUM_CLASSES = 3  # vehicle, pedestrian, background


class CropEncoder(nn.Module):
    def __init__(self, out_features: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),   # 64 -> 32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # 32 -> 16
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 16 -> 8
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x).flatten(1)
        return self.fc(feat)


class EvidentialDetector(nn.Module):
    def __init__(self, feature_dim: int = 128, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = CropEncoder(feature_dim)
        self.head = EvidentialHead(feature_dim, num_classes)

    def forward(self, crops: torch.Tensor):
        features = self.encoder(crops)
        alpha, uncertainty = self.head(features)
        return alpha, uncertainty
