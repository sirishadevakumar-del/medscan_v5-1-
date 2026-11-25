"""Image quality assessment: blur (Laplacian variance), contrast (intensity std), artefacts (edge density).

Thresholds differ by modality and by acquisition resolution, so they are calibrated per task from the
training set (see `calibrate`). The chest X-ray values from the paper are kept as a fallback default.
"""
from dataclasses import dataclass, asdict

import numpy as np
import cv2

SIZE = 224  # fixed analysis size so metrics are comparable across images


@dataclass
class Thresholds:
    blur_min: float = 100.0
    contrast_min: float = 30.0
    artifact_max: float = 0.15

    def to_dict(self):
        return asdict(self)


def metrics(img):
    g = np.asarray(img.convert("L").resize((SIZE, SIZE)))
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    contrast = float(g.std())
    edge = float((cv2.Canny(g, 50, 150) > 0).mean())
    return {"blur": blur, "contrast": contrast, "edge_density": edge}


def assess(img, th: Thresholds):
    m = metrics(img)
    bp = float(np.clip(1 - m["blur"] / max(th.blur_min, 1e-6), 0, 1))
    cp = float(np.clip(1 - m["contrast"] / max(th.contrast_min, 1e-6), 0, 1))
    ap = float(np.clip((m["edge_density"] - th.artifact_max) / max(th.artifact_max, 1e-6), 0, 1))
    score = 1 - (bp + cp + ap) / 3
    flags = []
    if bp > 0:
        flags.append("blurry")
    if cp > 0:
        flags.append("low contrast")
    if ap > 0:
        flags.append("possible artefacts")
    return {**m, "blur_ok": bp == 0, "contrast_ok": cp == 0, "artifact_ok": ap == 0,
            "passed": not flags, "score": score, "flags": flags}


def calibrate(images, low_pct=5, high_pct=95):
    """Data-driven thresholds: flag the worst ~5% of the task's own training distribution."""
    ms = [metrics(i) for i in images]
    return Thresholds(
        blur_min=float(np.percentile([m["blur"] for m in ms], low_pct)),
        contrast_min=float(np.percentile([m["contrast"] for m in ms], low_pct)),
        artifact_max=float(np.percentile([m["edge_density"] for m in ms], high_pct)),
    )
