"""Split a flat <src>/<class>/images tree into <dst>/{train,val,test}/<class>/ (70/15/15).

IMPORTANT: if one patient has several images, pass --patient-regex so splitting is done per patient
(otherwise images of the same patient leak across splits and inflate results).
  python scripts/prepare_folder_dataset.py --src raw/mri --dst data/brain_mri --patient-regex "(patient\\d+)"
COVID-19 Radiography Database (classes have an images and a masks folder; keep only images):
  python scripts/prepare_folder_dataset.py --src COVID-19_Radiography_Dataset --dst data/covid_xray --only-subfolder images
"""
import argparse
import random
import re
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--patient-regex", help="regex whose first group is the patient id (taken from the file path)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-subfolder", help="keep only files inside a folder with this name (e.g. 'images' to skip 'masks')")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    src, dst = Path(a.src), Path(a.dst)
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm")
    files = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts
             and len(p.relative_to(src).parts) >= 2                      # needs a class folder
             and (not a.only_subfolder or a.only_subfolder in p.relative_to(src).parts[:-1])]
    classes = sorted({p.relative_to(src).parts[0] for p in files})
    groups = {}
    for p in files:
        cls = p.relative_to(src).parts[0]
        key = (re.search(a.patient_regex, str(p)).group(1) if a.patient_regex and re.search(a.patient_regex, str(p)) else str(p))
        groups.setdefault((cls, key), []).append(p)
    for cls in classes:
        keys = [k for k in groups if k[0] == cls]
        rnd.shuffle(keys)
        n = len(keys); cut1, cut2 = int(.7 * n), int(.85 * n)
        for i, k in enumerate(keys):
            split = "train" if i < cut1 else "val" if i < cut2 else "test"
            (dst / split / cls).mkdir(parents=True, exist_ok=True)
            for p in groups[k]:
                shutil.copy2(p, dst / split / cls / p.name)
    print(f"done: {dst} ({len(classes)} classes)")


if __name__ == "__main__":
    main()
