"""Unified data layer: MedMNIST benchmarks and folder datasets (PNG/JPG/DICOM/NIfTI)."""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

IMN_MEAN = (0.485, 0.456, 0.406)
IMN_STD = (0.229, 0.224, 0.225)
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom", ".nii", ".nii.gz")


def _to_uint8(a, lo=1, hi=99):
    a = a.astype(np.float32)
    l, h = np.percentile(a, [lo, hi])
    if h <= l:
        h = l + 1
    return (np.clip((a - l) / (h - l), 0, 1) * 255).astype(np.uint8)


def _load_dicom(p):
    import pydicom
    ds = pydicom.dcmread(p)
    a = ds.pixel_array.astype(np.float32)
    if a.ndim == 3 and a.shape[-1] not in (3, 4):   # multi-frame -> middle frame
        a = a[a.shape[0] // 2]
    a = a * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    wc, ww = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        wc = float(wc[0] if hasattr(wc, "__len__") else wc)
        ww = float(ww[0] if hasattr(ww, "__len__") else ww)
        a = np.clip((a - (wc - ww / 2)) / max(ww, 1), 0, 1) * 255
        a = a.astype(np.uint8)
    else:
        a = _to_uint8(a)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = 255 - a
    return Image.fromarray(a)


def _load_nifti(p):
    import nibabel as nib
    v = np.asarray(nib.load(p).dataobj)
    while v.ndim > 3:
        v = v[..., 0]
    s = v[..., v.shape[-1] // 2] if v.ndim == 3 else v   # middle slice (2D analysis of a volume)
    return Image.fromarray(_to_uint8(s))


def load_any_image(path):
    p = str(path)
    low = p.lower()
    if low.endswith((".dcm", ".dicom")):
        return _load_dicom(p)
    if low.endswith((".nii", ".nii.gz")):
        return _load_nifti(p)
    im = Image.open(p)
    im.load()
    return im


def to_rgb(im):
    if im.mode in ("I;16", "I", "F"):
        im = Image.fromarray(_to_uint8(np.asarray(im)))
    return im.convert("RGB")


def build_transforms(size, train, hflip=True):
    t = [to_rgb, T.Resize((size, size))]
    if train:
        t += [T.RandomRotation(15), T.RandomAffine(0, translate=(0.1, 0.1))]
        if hflip:
            t.append(T.RandomHorizontalFlip())
        t.append(T.ColorJitter(brightness=0.2, contrast=0.2))
    t += [T.ToTensor(), T.Normalize(IMN_MEAN, IMN_STD)]
    return T.Compose(t)


class TaskDataset(Dataset):
    def __init__(self, base, targets, transform, multilabel):
        self.base, self.targets, self.transform, self.multilabel = base, np.asarray(targets), transform, multilabel

    def __len__(self):
        return len(self.base)

    def raw(self, i):
        return to_rgb(self.base[i][0])

    def __getitem__(self, i):
        x = self.transform(self.base[i][0])
        dtype = torch.float32 if self.multilabel else torch.long
        return x, torch.as_tensor(self.targets[i], dtype=dtype)


def _medmnist_bundle(spec, cfg, tfs):
    import medmnist
    from medmnist import INFO
    info = INFO[spec["flag"]]
    cls = getattr(medmnist, info["python_class"])
    multilabel = "multi-label" in info["task"]
    classes = [info["label"][str(i)] for i in range(len(info["label"]))]
    root = Path(cfg.data_root) / "medmnist"
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "val", "test"):
        base = cls(split=split, transform=None, download=True, root=str(root), size=cfg.medmnist_size)
        y = np.asarray(base.labels)
        y = y if multilabel else y.reshape(-1)
        out[split] = TaskDataset(base, y, tfs[split], multilabel)
    return out, classes, multilabel


def _folder_bundle(spec, cfg, tfs):
    root = Path(spec["path"])
    out, classes = {}, None
    for split in ("train", "val", "test"):
        d = root / split
        if not d.is_dir():
            raise FileNotFoundError(
                f"{d} not found. Expected {root}/{{train,val,test}}/<class_name>/images. "
                f"Use scripts/prepare_folder_dataset.py to create it (patient-level split recommended).")
        base = ImageFolder(str(d), loader=load_any_image, is_valid_file=lambda p: p.lower().endswith(EXTS))
        if classes is None:
            classes = base.classes
        elif base.classes != classes:
            raise ValueError(f"Class folders differ between splits in {root}")
        out[split] = TaskDataset(base, base.targets, tfs[split], False)
    return out, classes, False


def build_task(spec, cfg):
    hflip = spec.get("hflip", True)
    tfs = {"train": build_transforms(cfg.image_size, True, hflip),
           "val": build_transforms(cfg.image_size, False),
           "test": build_transforms(cfg.image_size, False)}
    if spec["source"] == "medmnist":
        sets, classes, multilabel = _medmnist_bundle(spec, cfg, tfs)
    elif spec["source"] == "folder":
        sets, classes, multilabel = _folder_bundle(spec, cfg, tfs)
    else:
        raise ValueError(f"Unknown source {spec['source']}")
    return dict(spec=spec, classes=classes, multilabel=multilabel, **sets)


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=shuffle and len(ds) > batch_size,
                      persistent_workers=num_workers > 0)
