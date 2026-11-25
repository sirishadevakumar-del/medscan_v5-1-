"""Prepare a downloaded dataset that is already split into train / test folders (optionally also validation).

Expected input: one folder per class inside each split folder, e.g.
    <src>/Training/<class>/*.jpg      <src>/Testing/<class>/*.jpg
Output (what configs/tasks.json expects):  <dst>/{train,val,test}/<class>/*

Examples
  Brain tumour MRI (Kaggle "Brain Tumor MRI Dataset"):
    python scripts/prepare_presplit.py --src raw_brain_mri --dst data/brain_mri
  Chest CT (Kaggle "Chest CT-Scan images Dataset", folders train / valid / test):
    python scripts/prepare_presplit.py --src raw_chest_ct --dst data/chest_ct --train-name train --val-name valid --test-name test

If no validation folder is given, 15% of the training images become the validation set. These datasets usually
have no patient IDs, so images of one patient can land in different splits and results may be optimistic.
"""
import argparse
import random
import shutil
from pathlib import Path

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_dir(root, name):
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    raise SystemExit(f"Could not find a '{name}' folder inside {root}. Check --src and the folder names.")


def class_dirs(d):
    return sorted(p for p in d.iterdir() if p.is_dir())


def images(d):
    return sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in EXTS)


def copy_into(files, dst, split, cls, counts):
    (dst / split / cls).mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / split / cls / f.name)
        counts[split] = counts.get(split, 0) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--train-name", default="Training"); ap.add_argument("--test-name", default="Testing")
    ap.add_argument("--val-name", default=None, help="folder name of an existing validation split")
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    src, dst = Path(a.src), Path(a.dst)
    train_dir, test_dir = find_dir(src, a.train_name), find_dir(src, a.test_name)
    val_dir = find_dir(src, a.val_name) if a.val_name else None
    counts = {}
    for cdir in class_dirs(train_dir):
        files = images(cdir)
        if val_dir is None:
            rnd.shuffle(files)
            nval = int(len(files) * a.val_frac)
            copy_into(files[:nval], dst, "val", cdir.name, counts)
            files = files[nval:]
        copy_into(files, dst, "train", cdir.name, counts)
    if val_dir is not None:
        for cdir in class_dirs(val_dir):
            copy_into(images(cdir), dst, "val", cdir.name, counts)
    for cdir in class_dirs(test_dir):
        copy_into(images(cdir), dst, "test", cdir.name, counts)
    print("done:", counts, "->", dst)


if __name__ == "__main__":
    main()
