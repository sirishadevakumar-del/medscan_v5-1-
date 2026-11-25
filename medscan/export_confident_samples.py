"""Save the images the trained model recognises best (correct AND high confidence) for each task.

Run from the project folder (the one containing train.py), with the environment active:
    python export_confident_samples.py                  # from the TEST set (honest: unseen images)
    python export_confident_samples.py --split train    # from the TRAINING set (images the model has seen)
Images go to  samples_confident/<task>/<class>_<index>_conf<percent>.png
"""
import argparse
import json
import re
from pathlib import Path

import torch

from medscan.config import load_tasks
from medscan.data import build_task, make_loader
from medscan.models import load_checkpoint
from medscan.uncertainty import mc_predict

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/exp1/best.pt")
ap.add_argument("--tasks", default="configs/tasks.json")
ap.add_argument("--split", default="test", choices=["test", "val", "train"])
ap.add_argument("--per-class", type=int, default=2, help="images to save per class")
ap.add_argument("--scan", type=int, default=4000, help="max images to scan per task")
ap.add_argument("--out", default="samples_confident")
a = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
ens, cfg, ck = load_checkpoint(a.ckpt, device)
cfg.num_workers = 0

for spec in load_tasks(a.tasks):
    name = spec["name"]
    if name not in ck["task_meta"] or ck["task_meta"][name]["multilabel"]:
        continue
    b = build_task(spec, cfg)
    classes = b["classes"]
    ds = b[a.split]
    ds.transform = b["test"].transform          # no random augmentation, even for the training split
    ld = make_loader(ds, 64, False, 0)
    best = {k: [] for k in range(len(classes))}     # class -> [(confidence, index)]
    seen = 0
    for x, y in ld:
        with torch.no_grad():
            r = mc_predict(ens, x.to(device), name, cfg.mc_samples, cfg.confidence_mode)
        pred, conf = r["pred"].cpu(), r["confidence"].cpu()
        for j in range(len(y)):
            if int(pred[j]) == int(y[j]):
                best[int(y[j])].append((float(conf[j]), seen + j))
        seen += len(y)
        if seen >= a.scan:
            break
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for k, items in best.items():
        for c, idx in sorted(items, reverse=True)[: a.per_class]:
            label = re.sub(r"[^A-Za-z0-9]+", "_", classes[k]).strip("_")
            ds.raw(idx).save(out / f"{label}_{idx}_conf{int(c * 100)}.png")
            saved += 1
    print(f"{name}: scanned {seen} {a.split} images, saved {saved} to {out}")
