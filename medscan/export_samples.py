"""Save a few test images (one per class) from each MedMNIST task as PNG files for trying the demo.

Run from the project folder (the one containing train.py):  python export_samples.py
Images go to  samples/<task>/<class>_<index>.png
"""
import json
import re
from pathlib import Path

import medmnist
from medmnist import INFO

tasks = [t for t in json.load(open("configs/tasks.json"))
         if t.get("enabled", True) and t["source"] == "medmnist"]

for t in tasks:
    info = INFO[t["flag"]]
    cls = getattr(medmnist, info["python_class"])
    ds = cls(split="test", download=False, root="data/medmnist", size=128)
    labels = {int(k): v for k, v in info["label"].items()}
    out = Path("samples") / t["name"]
    out.mkdir(parents=True, exist_ok=True)
    seen = set()
    for i in range(len(ds)):
        img, y = ds[i]
        k = int(y.reshape(-1)[0])
        if k in seen:
            continue
        seen.add(k)
        name = re.sub(r"[^A-Za-z0-9]+", "_", labels[k]).strip("_")
        img.save(out / f"{name}_{i}.png")
        if len(seen) == len(labels):
            break
    print(f"{t['name']}: saved {len(seen)} images to {out}")
