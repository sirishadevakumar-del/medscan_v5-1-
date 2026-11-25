"""Checks the multi-task / multi-label inference path and the page, with random weights (no data, CPU).
    python tests/test_multitask.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medscan.config import Config
from medscan.models import Ensemble, save_checkpoint
from medscan.pipeline import Analyzer
from medscan.quality import Thresholds

tmp = Path(tempfile.mkdtemp())
cfg = Config(); cfg.pretrained = False; cfg.image_size = 64; cfg.mc_samples = 5
meta = {
    "xr_pneumonia": {"n": 2, "multilabel": False, "classes": ["normal", "pneumonia"], "modality": "xray"},
    "xr_chest14": {"n": 4, "multilabel": True, "classes": ["atelectasis", "effusion", "mass", "edema"], "modality": "xray"},
    "ct_cancer": {"n": 3, "multilabel": False, "classes": ["normal", "adeno", "squamous"], "modality": "ct"},
    "_modality": {"n": 2, "multilabel": False, "classes": ["ct", "xray"]},
}
torch.manual_seed(0)
ens = Ensemble(meta, cfg).eval()
th = {k: Thresholds(0, 0, 1).to_dict() for k in meta if k != "_modality"}
save_checkpoint(tmp / "best.pt", ens, cfg, meta, th, ["ct", "xray"])
Image.fromarray((np.random.RandomState(0).rand(80, 80) * 255).astype("uint8")).save(tmp / "a.png")

an = Analyzer(str(tmp / "best.pt"), device="cpu", audit_path=str(tmp / "audit.jsonl"))

# every task, including the multi-label one, must run end to end
for t in ("xr_pneumonia", "xr_chest14", "ct_cancer"):
    res, ov, _ = an.analyze(str(tmp / "a.png"), t, log=False)
    assert 0 <= res["confidence"] <= 1, (t, res["confidence"])
    print(t, "->", res["prediction"], f"{res['confidence']:.2f}", res["tier"], "multilabel:", res["multilabel"])

# automatic mode runs all tasks of the detected image type
results = an.analyze_all(str(tmp / "a.png"), log=False)
mod = results[0][0]["modality"]
expected = sum(1 for v in meta.values() if v.get("modality") == mod)
assert len(results) == expected, (len(results), expected)
print("analyze_all:", mod, "->", [r[0]["task"] for r in results])
# force the X-ray type: both X-ray tasks (single-label and multi-label) must run and be returned
an.route = lambda x: ("xray", np.array([0.1, 0.9]))
results = an.analyze_all(str(tmp / "a.png"), log=False)
assert [r[0]["task"] for r in results] == ["xr_pneumonia", "xr_chest14"], [r[0]["task"] for r in results]
print("forced xray:", [r[0]["task"] for r in results])
print("MULTITASK TEST PASSED")
