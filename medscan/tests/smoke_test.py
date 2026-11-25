"""End-to-end smoke test on tiny synthetic data (CPU, no downloads).  python tests/smoke_test.py"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def make_task(root, n_classes, per_class, rng):
    for split, k in (("train", per_class), ("val", per_class // 2), ("test", per_class // 2)):
        for c in range(n_classes):
            d = root / split / f"class{c}"; d.mkdir(parents=True, exist_ok=True)
            for i in range(k):
                a = rng.rand(80, 80) * 60 + 40
                yy, xx = np.ogrid[:80, :80]
                cx, cy = 20 + 20 * c, 40
                a += 150 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 120.0)
                Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(d / f"{i}.png")


def run(*args):
    print("$", " ".join(map(str, args)))
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.RandomState(0)
    make_task(tmp / "xr", 2, 24, rng); make_task(tmp / "ct", 3, 24, rng)
    tasks = [{"name": "xr_demo", "modality": "xray", "source": "folder", "path": str(tmp / "xr")},
             {"name": "ct_demo", "modality": "ct", "source": "folder", "path": str(tmp / "ct"), "hflip": False}]
    (tmp / "tasks.json").write_text(json.dumps(tasks))
    out = tmp / "run"
    run("train.py", "--tasks", tmp / "tasks.json", "--out", out, "--epochs", 3, "--batch-size", 8,
        "--image-size", 64, "--no-pretrained", "--freeze-bn", "--dropout-after-bn", "--num-workers", 0, "--lr", 3e-4)
    run("evaluate.py", "--ckpt", out / "best.pt", "--tasks", tmp / "tasks.json", "--num-workers", 0, "--max-cam", 6)
    img = next((tmp / "xr" / "test").rglob("*.png"))
    run("infer.py", "--ckpt", out / "best.pt", "--image", img, "--patient-id", "p1", "--out", tmp / "ov.png",
        "--audit", tmp / "audit.jsonl")
    run("infer.py", "--ckpt", out / "best.pt", "--image", img, "--task", "xr_demo", "--patient-id", "p1",
        "--out", tmp / "ov2.png", "--audit", tmp / "audit.jsonl")
    print("\nSMOKE TEST PASSED; artefacts in", out)
