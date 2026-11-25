"""CT / MRI volumes: loading (NIfTI, DICOM series, multi-frame DICOM, zip), orientation, slices, masks.

Storage convention: `arr` has shape (X, Y, Z) with voxel spacing `spacing` = (sx, sy, sz) in mm.
NIfTI is reordered to the closest canonical (RAS+) orientation. DICOM keeps its patient (LPS) orientation.
Everything is kept at the original resolution; only the 3D rendering uses a down-sampled copy.
"""
import io
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MAX_SLICES_DEFAULT = 24


@dataclass
class Volume:
    arr: np.ndarray                       # (X, Y, Z) float32, original intensities (HU for CT when rescale tags exist)
    spacing: tuple                        # mm (sx, sy, sz)
    orientation: str                      # "RAS" (NIfTI canonical) or "LPS" (DICOM native)
    source: str                           # nifti | dicom_series | dicom_multiframe
    dicom_modality: str = None
    study_date: str = None
    meta: dict = field(default_factory=dict)
    lo: float = 0.0                       # display / model window computed over the WHOLE volume
    hi: float = 1.0

    def __post_init__(self):
        a = self.arr
        self.lo, self.hi = [float(v) for v in np.percentile(a[:, :, ::max(1, a.shape[2] // 64)], [1, 99.5])]
        if self.hi <= self.lo:
            self.hi = self.lo + 1.0

    @property
    def shape(self):
        return self.arr.shape

    @property
    def n_slices(self):
        return self.arr.shape[2]

    def to_u8(self, a):
        return (np.clip((a - self.lo) / (self.hi - self.lo), 0, 1) * 255).astype(np.uint8)

    def _plane(self, a, plane, i):
        if plane == "axial":
            s, ph, pw = a[:, :, i].T, self.spacing[1], self.spacing[0]
            if self.orientation == "RAS":
                s = s[::-1]                                  # Y grows anterior -> flip so anterior is at the top
        elif plane == "coronal":
            s, ph, pw = a[:, i, :].T[::-1], self.spacing[2], self.spacing[0]
        elif plane == "sagittal":
            s, ph, pw = a[i, :, :].T[::-1], self.spacing[2], self.spacing[1]
        else:
            raise ValueError(plane)
        return s, ph, pw

    def view(self, plane, i, a=None, physical_aspect=True, nearest=False):
        """2D view of `a` (default: this volume) in the given plane, upright, with physical pixel aspect.
        Pass another array of the same grid (mask, attention) to get an exactly matching view."""
        s, ph, pw = self._plane(self.arr if a is None else a, plane, i)
        s = np.ascontiguousarray(s, dtype=np.float32)
        if physical_aspect and abs(ph - pw) > 1e-3:
            h, w = s.shape
            size = (w, max(1, int(round(h * ph / pw)))) if ph > pw else (max(1, int(round(w * pw / ph))), h)
            s = cv2.resize(s, size, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
        return s

    def slice2d(self, plane, i, physical_aspect=True):
        """uint8 2D image of one plane. plane: axial | coronal | sagittal. Superior is up; for axial, anterior is up."""
        return self.to_u8(self.view(plane, i, physical_aspect=physical_aspect))

    def axial_pil(self, k):
        """Axial slice as a PIL image for the 2D model: whole-volume window (consistent between slices)."""
        s = self.arr[:, :, k].T
        if self.orientation == "RAS":
            s = s[::-1]
        return Image.fromarray(self.to_u8(s)).convert("RGB")

    def plane_count(self, plane):
        return {"axial": self.arr.shape[2], "coronal": self.arr.shape[1], "sagittal": self.arr.shape[0]}[plane]

    def downsampled(self, max_dim=56):
        """Small copy for interactive 3D rendering (spacing is updated so proportions stay physical)."""
        f = [max(1, int(np.ceil(n / max_dim))) for n in self.arr.shape]
        a = self.arr[::f[0], ::f[1], ::f[2]]
        return a, tuple(s * k for s, k in zip(self.spacing, f)), f

    def describe(self):
        return {"shape_xyz": list(map(int, self.shape)), "spacing_mm": [round(float(s), 3) for s in self.spacing],
                "orientation": self.orientation, "source": self.source, "dicom_modality": self.dicom_modality,
                "study_date": self.study_date, "intensity_window": [round(self.lo, 1), round(self.hi, 1)]}


# ----------------------------------------------------------------------------------------------- NIfTI
def load_nifti(path):
    import nibabel as nib
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)                       # RAS+, keeps voxel sizes
    a = np.asarray(img.dataobj, dtype=np.float32)
    while a.ndim > 3:                                         # 4D (time/b-values): first volume only
        a = a[..., 0]
    if a.ndim != 3:
        raise ValueError("NIfTI file is not a 3D volume.")
    sp = tuple(float(z) for z in img.header.get_zooms()[:3])
    return Volume(np.nan_to_num(a), sp, "RAS", "nifti", meta={"note": "4D files use the first volume"})


# ----------------------------------------------------------------------------------------------- DICOM
def _slice_pixels(ds):
    a = ds.pixel_array.astype(np.float32)
    a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a
    return a


def _first(v, default=None):
    try:
        return float(v[0] if hasattr(v, "__len__") else v)
    except Exception:
        return default


def load_dicom_series(paths):
    """Order slices by their position along the slice normal (not by file name), take spacing from the tags."""
    import pydicom
    dss = []
    for p in paths:
        try:
            ds = pydicom.dcmread(str(p), force=False)
            if hasattr(ds, "PixelData"):
                dss.append(ds)
        except Exception:
            continue
    if not dss:
        raise ValueError("No readable DICOM images found.")
    # keep the largest series (a folder can hold localizers or several series)
    groups = {}
    for ds in dss:
        groups.setdefault(str(getattr(ds, "SeriesInstanceUID", "")), []).append(ds)
    dss = max(groups.values(), key=len)
    ds0 = dss[0]
    if len(dss) == 1 and int(getattr(ds0, "NumberOfFrames", 1) or 1) > 1:
        a = _slice_pixels(ds0)                                # (frames, rows, cols)
        arr = np.transpose(a, (2, 1, 0))
        ps = [float(x) for x in getattr(ds0, "PixelSpacing", [1, 1])]
        sz = _first(getattr(ds0, "SpacingBetweenSlices", None), None) or _first(getattr(ds0, "SliceThickness", None), 1.0)
        return Volume(arr, (ps[1], ps[0], sz), "LPS", "dicom_multiframe", str(getattr(ds0, "Modality", "")) or None,
                      str(getattr(ds0, "StudyDate", "")) or None)
    iop = getattr(ds0, "ImageOrientationPatient", None)
    if iop is not None and all(hasattr(d, "ImagePositionPatient") for d in dss):
        r, c = np.array(iop[:3], float), np.array(iop[3:], float)
        n = np.cross(r, c)
        pos = [float(np.dot(n, np.array(d.ImagePositionPatient, float))) for d in dss]
        order = np.argsort(pos)
        dss = [dss[i] for i in order]
        pos = np.array(pos)[order]
        dz = float(np.median(np.diff(pos))) if len(pos) > 1 else None
    else:
        dss.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0) or 0))
        dz = None
    ps = [float(x) for x in getattr(ds0, "PixelSpacing", [1.0, 1.0])]
    sz = abs(dz) if dz else (_first(getattr(ds0, "SpacingBetweenSlices", None)) or _first(getattr(ds0, "SliceThickness", None), 1.0))
    shape = _slice_pixels(dss[0]).shape
    dss = [d for d in dss if _slice_pixels(d).shape == shape]   # drop slices with a different matrix
    arr = np.stack([_slice_pixels(d) for d in dss], axis=-1)    # (rows, cols, Z)
    arr = np.transpose(arr, (1, 0, 2))                          # (X=cols, Y=rows, Z)
    return Volume(np.nan_to_num(arr), (ps[1], ps[0], float(sz)), "LPS", "dicom_series",
                  str(getattr(ds0, "Modality", "")) or None, str(getattr(ds0, "StudyDate", "")) or None,
                  {"series_description": str(getattr(ds0, "SeriesDescription", "")), "n_files_used": len(dss)})


# ----------------------------------------------------------------------------------------------- uploads
def _is_nifti(n):
    return n.lower().endswith((".nii", ".nii.gz"))


def _is_dicom_like(n):
    return n.lower().endswith((".dcm", ".dicom")) or "." not in os.path.basename(n)


def dicom_tag(path):
    """Modality tag and study date of a single DICOM file (or (None, None))."""
    try:
        import pydicom
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        return str(getattr(ds, "Modality", "")) or None, str(getattr(ds, "StudyDate", "")) or None
    except Exception:
        return None, None


def load_study(paths):
    """paths: local files (one image, one NIfTI, one multi-frame DICOM, a DICOM series, or a .zip of DICOMs/NIfTI).
    Returns {"kind": "image"|"volume", "image": PIL|None, "volume": Volume|None, "dicom_modality", "date", "is_nifti"}."""
    paths = [Path(p) for p in paths]
    tmpdir = None
    if len(paths) == 1 and paths[0].suffix.lower() == ".zip":
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(paths[0]) as z:
            for m in z.namelist():                              # extract by index to avoid path traversal
                if m.endswith("/"):
                    continue
                with z.open(m) as src, open(Path(tmpdir) / f"{abs(hash(m))}_{Path(m).name}", "wb") as dst:
                    dst.write(src.read())
        paths = sorted(Path(tmpdir).iterdir())
    out = {"kind": None, "image": None, "volume": None, "dicom_modality": None, "date": None, "is_nifti": False}
    nifti = [p for p in paths if _is_nifti(p.name)]
    if nifti:
        v = load_nifti(nifti[0])
        out.update(kind="volume", volume=v, is_nifti=True)
        if v.n_slices < 2:
            raise ValueError("The NIfTI file has a single slice; upload it as an image instead.")
        return out
    dcm = [p for p in paths if _is_dicom_like(p.name)]
    if dcm:
        tag, date = dicom_tag(dcm[0])
        out.update(dicom_modality=tag, date=date)
        multiframe = False
        if len(dcm) == 1:
            import pydicom
            ds = pydicom.dcmread(str(dcm[0]), stop_before_pixels=True)
            multiframe = int(getattr(ds, "NumberOfFrames", 1) or 1) > 1
        if len(dcm) > 1 or multiframe:
            v = load_dicom_series(dcm)
            out.update(kind="volume", volume=v, dicom_modality=v.dicom_modality or tag, date=v.study_date or date)
        else:                                                   # one single-frame file = a 2D image (e.g. X-ray)
            from .data import load_any_image
            out.update(kind="image", image=load_any_image(str(dcm[0])))
        return out
    from .data import load_any_image
    out.update(kind="image", image=load_any_image(str(paths[0])))
    return out


# ----------------------------------------------------------------------------------------------- slices & masks
def select_slices(vol, n_max=MAX_SLICES_DEFAULT, min_foreground=0.02):
    """Evenly spaced axial slices across the part of the volume that contains tissue (skips empty / air-only
    slices), at most n_max. Keeps memory and run time bounded for large volumes."""
    a = vol.arr
    thr = vol.lo + 0.15 * (vol.hi - vol.lo)
    fg = np.array([(a[::2, ::2, k] > thr).mean() for k in range(a.shape[2])])
    ok = np.where(fg >= min_foreground)[0]
    if len(ok) == 0:
        ok = np.arange(a.shape[2])
    idx = np.unique(np.linspace(ok[0], ok[-1], min(n_max, len(ok))).round().astype(int))
    return [int(i) for i in idx]


def load_mask(path, vol):
    """External segmentation mask (NIfTI). Must have exactly the same grid as the volume; it is never resampled,
    because a mis-registered overlay would be misleading. Returns a bool array (X,Y,Z) or raises ValueError."""
    if not _is_nifti(str(path)):
        raise ValueError("Segmentation masks must be NIfTI files (.nii / .nii.gz).")
    m = load_nifti(path)
    if vol.source != "nifti":
        raise ValueError("Mask upload is supported for NIfTI volumes (the mask must share the volume's grid).")
    if m.shape != vol.shape:
        raise ValueError(f"Mask grid {tuple(m.shape)} does not match the volume {tuple(vol.shape)}; "
                         "resample it to the volume first (e.g. in 3D Slicer).")
    if not np.allclose(m.spacing, vol.spacing, atol=1e-3):
        raise ValueError("Mask voxel spacing differs from the volume's spacing.")
    labels = np.unique(m.arr)
    if len(labels) > 64:
        raise ValueError("This file looks like an intensity image, not a label mask.")
    return m.arr > 0


def mask_stats(mask, spacing):
    """Measured size of a validated mask: volume (mL), largest axial area (mm2), per-slice area curve."""
    vox = float(np.prod(spacing))
    per_slice = mask.sum(axis=(0, 1)) * spacing[0] * spacing[1]
    return {"voxels": int(mask.sum()), "volume_ml": float(mask.sum() * vox / 1000.0),
            "max_axial_area_mm2": float(per_slice.max()) if mask.size else 0.0,
            "area_per_slice_mm2": per_slice.astype(float).tolist()}


def attention_volume(vol, cams, factors=(1, 1, 1)):
    """Build a (down-sampled) volume of Grad-CAM attention from the per-slice maps of the analysed slices.
    cams: {z_index: cam_xy (X,Y) in 0..1}. Slices between analysed ones are linearly interpolated; slices outside the
    analysed range are 0. This is a visual aid, NOT a segmentation."""
    fx, fy, fz = factors
    zs = sorted(cams)
    X, Y, Z = vol.shape
    out = np.zeros((len(range(0, X, fx)), len(range(0, Y, fy)), len(range(0, Z, fz))), np.float32)
    small = {z: np.asarray(cams[z], np.float32)[::fx, ::fy] for z in zs}
    for j, z in enumerate(range(0, Z, fz)):
        if z < zs[0] or z > zs[-1]:
            continue
        hi = next(i for i, k in enumerate(zs) if k >= z)
        lo = max(hi - 1, 0) if zs[hi] > z else hi
        if zs[hi] == zs[lo]:
            out[:, :, j] = small[zs[hi]]
        else:
            t = (z - zs[lo]) / (zs[hi] - zs[lo])
            out[:, :, j] = (1 - t) * small[zs[lo]] + t * small[zs[hi]]
    return out


def cam_to_grid(cam, vol):
    """Map a model-space heat-map (square, as fed to the network) back onto the slice grid (X, Y)."""
    X, Y = vol.shape[:2]
    c = cv2.resize(cam.astype(np.float32), (X, Y))      # (Y, X) in display orientation
    if vol.orientation == "RAS":
        c = c[::-1]
    return c.T                                           # (X, Y)
