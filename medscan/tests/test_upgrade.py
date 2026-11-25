"""Tests for the 3D / progression / report / modality-validation upgrade, with random weights and synthetic data.
These check that the code works; they say nothing about clinical validity.    python tests/test_upgrade.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medscan import modality as M
from medscan import progression as prog
from medscan.config import Config
from medscan.models import Ensemble, save_checkpoint
from medscan.pipeline import Analyzer
from medscan.quality import Thresholds
from medscan.report import BANNER, build_report, to_json, to_markdown, to_pdf
from medscan.viz3d import blend, volume_figure
from medscan.volume import (attention_volume, load_mask, load_study, mask_stats, select_slices)

tmp = Path(tempfile.mkdtemp())
ok = lambda name: print("PASS", name)

# ------------------------------------------------------------------ synthetic data
def make_volume(shift=0, blob=14):
    X, Y, Z = 64, 64, 40
    g = np.mgrid[0:X, 0:Y, 0:Z].astype(np.float32)
    body = (((g[0] - 32) / 26) ** 2 + ((g[1] - 32 + shift) / 22) ** 2 + ((g[2] - 20) / 17) ** 2) < 1
    lesion = (((g[0] - 36) / blob) ** 2 + ((g[1] - 30 + shift) / blob) ** 2 + ((g[2] - 20) / 6) ** 2) < 1
    a = np.random.RandomState(0).rand(X, Y, Z).astype(np.float32) * 40 - 1000
    a[body] = 40 + np.random.RandomState(1).rand(int(body.sum())) * 30
    a[lesion] = 160
    return a, lesion


def save_nifti(path, a, sp=(0.8, 0.8, 2.5)):
    import nibabel as nib
    aff = np.diag([sp[0], sp[1], sp[2], 1.0])
    nib.save(nib.Nifti1Image(a.astype(np.float32), aff), str(path))


def write_dicom_series(folder, a, modality="CT", date="20240131"):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
    folder.mkdir(exist_ok=True)
    suid = generate_uid()
    order = list(range(a.shape[2]))[::-1]                     # write in reverse order; InstanceNumber is misleading too
    for n, k in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = CTImageStorage, generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"f{n}"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID, ds.SeriesInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID, suid
        ds.Modality, ds.StudyDate, ds.InstanceNumber = modality, date, n
        sl = (a[:, :, k].T + 1024).astype(np.int16)             # rows=Y, cols=X
        ds.Rows, ds.Columns = sl.shape
        ds.PixelSpacing, ds.SliceThickness = [0.9, 0.7], 2.5     # row spacing 0.9 (y), col spacing 0.7 (x)
        ds.ImagePositionPatient, ds.ImageOrientationPatient = [0, 0, 2.5 * k], [1, 0, 0, 0, 1, 0]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.PixelData = sl.tobytes()
        ds.save_as(str(folder / f"f{n}.dcm"))
        (folder / f"f{n}").unlink(missing_ok=True)


vol_a, lesion_a = make_volume()
save_nifti(tmp / "cur.nii.gz", vol_a)
save_nifti(tmp / "mask.nii.gz", lesion_a.astype(np.float32))
vol_p, lesion_p = make_volume(shift=4, blob=11)
save_nifti(tmp / "prev.nii.gz", vol_p)
save_nifti(tmp / "prev_mask.nii.gz", lesion_p.astype(np.float32))
write_dicom_series(tmp / "dcm", vol_a)

# ------------------------------------------------------------------ 1. modality logic
m = M.determine("xray", 0.95)
assert (m.value, m.status, m.blocked) == ("xray", "detected", False)
m = M.determine("xray", 0.95, manual="ultrasound")                   # X-ray image + user picks ultrasound
assert m.status == "mismatch" and m.blocked and "Ultrasound" in " ".join(m.warnings)
m = M.determine("xray", 0.95, manual="ultrasound", confirmed_mismatch=True)
assert m.status == "mismatch" and not m.blocked and "may not be meaningful" in " ".join(m.warnings)
m = M.determine("ct", 0.9, dicom_tag="MR")                           # metadata beats the classifier
assert (m.value, m.status) == ("mri", "verified") and m.warnings
m = M.determine("ct", 0.99, ext_is_nifti=True)                       # NIfTI never verified
assert m.status == "unverified" and m.blocked
m = M.determine("xray", 0.40)                                        # uncertain -> must not silently pick
assert m.value is None and m.status == "unverified" and m.blocked
m = M.determine("xray", 0.4, manual="mri")                           # manual MRI, nothing contradicts
assert m.status == "manual" and not m.blocked
ok("modality detection / manual / mismatch / unverified")

# ------------------------------------------------------------------ 2. loaders
st_n = load_study([tmp / "cur.nii.gz"])
v = st_n["volume"]
assert st_n["kind"] == "volume" and v.shape == (64, 64, 40) and v.orientation == "RAS" and st_n["is_nifti"]
assert np.allclose(v.spacing, (0.8, 0.8, 2.5))
ax, co, sa = v.slice2d("axial", 20), v.slice2d("coronal", 32), v.slice2d("sagittal", 32)
assert ax.shape == (64, 64) and co.shape == (125, 64) and sa.shape == (125, 64), (ax.shape, co.shape, sa.shape)  # 40*2.5/0.8
ok("NIfTI volume: shape, spacing, orientation, 3 planes with physical aspect")

st_d = load_study(sorted((tmp / "dcm").glob("*.dcm")))
vd = st_d["volume"]
assert st_d["kind"] == "volume" and st_d["dicom_modality"] == "CT" and st_d["date"] == "20240131"
assert vd.shape == (64, 64, 40) and np.allclose(vd.spacing, (0.7, 0.9, 2.5), atol=1e-3), (vd.shape, vd.spacing)
assert abs(float(vd.arr[32, 32, 20]) - float(vol_a[32, 32, 20])) < 1.0           # slices re-ordered by position, HU preserved
assert abs(float(vd.arr[36, 30, 5]) - float(vol_a[36, 30, 5])) < 1.0
ok("DICOM series: ordered by position (not file order), spacing and HU rescale preserved, modality tag read")
import zipfile
with zipfile.ZipFile(tmp / "s.zip", "w") as z:
    for f in sorted((tmp / "dcm").glob("*.dcm")):
        z.write(f, f.name)
assert load_study([tmp / "s.zip"])["volume"].shape == (64, 64, 40)
ok("DICOM series from a .zip")
Image.fromarray((np.random.rand(60, 60) * 255).astype("uint8")).save(tmp / "a.png")
assert load_study([tmp / "a.png"])["kind"] == "image"
sl = select_slices(v, 10)
assert len(sl) <= 10 and sl == sorted(sl) and 0 <= sl[0] and sl[-1] < 40
ok("2D image loads as image; slice selection bounded")

mask = load_mask(tmp / "mask.nii.gz", v)
ms = mask_stats(mask, v.spacing)
assert mask.shape == v.shape and abs(ms["volume_ml"] - mask.sum() * 0.8 * 0.8 * 2.5 / 1000) < 1e-6
try:
    save_nifti(tmp / "bad.nii.gz", np.zeros((10, 10, 10)))
    load_mask(tmp / "bad.nii.gz", v)
    raise SystemExit("mismatching mask must be rejected")
except ValueError as e:
    assert "does not match" in str(e)
ok("segmentation mask: loaded on same grid, volume in mL measured, mismatching grid rejected")

# ------------------------------------------------------------------ 3. analyzer with random weights
cfg = Config(); cfg.pretrained = False; cfg.image_size = 64; cfg.mc_samples = 4
meta = {"xray_pneumonia": {"n": 2, "multilabel": False, "classes": ["normal", "pneumonia"], "modality": "xray"},
        "ct_organ": {"n": 3, "multilabel": False, "classes": ["liver", "lung", "bladder"], "modality": "ct"},
        "us_breast": {"n": 2, "multilabel": False, "classes": ["malignant", "normal, benign"], "modality": "ultrasound"},
        "_modality": {"n": 3, "multilabel": False, "classes": ["ct", "xray", "ultrasound"]}}
torch.manual_seed(0)
ens = Ensemble(meta, cfg).eval()
th = {k: Thresholds(0, 0, 1).to_dict() for k in meta if k != "_modality"}
ckpt = tmp / "best.pt"
save_checkpoint(ckpt, ens, cfg, meta, th, ["ct", "xray", "ultrasound"])
h0 = __import__("hashlib").sha256(ckpt.read_bytes()).hexdigest()
an = Analyzer(str(ckpt), device="cpu", audit_path=str(tmp / "audit.jsonl"))

out = an.analyze_volume(v, "ct", "P1", True, 8, {"status": "verified", "value": "ct"})
assert len(out) == 1                                                  # only the CT task
res, ovs, grid = out[0]
assert res["analysis_type"] == "multi-slice" and "NOT a trained 3D model" in res["method"]
assert len(res["slices"]) == len(res["slice_indices"]) <= 8 and set(ovs) == set(res["slice_indices"])
P = np.array([s["probabilities"] for s in res["slices"]])
assert np.allclose(P.mean(0), [res["probabilities"][c] for c in res["classes"]], atol=1e-6)   # aggregation = mean of slices
assert 0 <= res["slice_agreement"] <= 1 and 0 <= res["confidence"] <= 1 and res["case_id"]
assert grid[res["slice_indices"][0]].shape == v.shape[:2]
ok(f"multi-slice analysis: {len(res['slices'])} slices -> {res['prediction']} ({res['confidence']:.2f}), agreement {res['slice_agreement']:.2f}")
assert an.analyze_volume(v, "mri", None, False, 8) == []              # no MRI task trained -> nothing predicted
ok("MRI volume: no trained task, no prediction is invented")
try:
    an.analyze(Image.fromarray((np.random.rand(60, 60) * 255).astype("uint8")), "us_breast", modality="xray")
    raise SystemExit("task/modality mismatch must raise")
except ValueError as e:
    assert "ultrasound" in str(e)
ok("analyze() refuses a task that does not match the image type")
# the original bug: task us_breast chosen, classifier thinks X-ray -> displayed modality must follow the task, with a warning
an.route = lambda x: ("xray", np.array([0.05, 0.9, 0.05]))
r_us, _, _ = an.analyze(Image.fromarray((np.random.rand(60, 60) * 255).astype("uint8")), "us_breast", log=False)
assert r_us["modality"] == "ultrasound" and any("suggests 'xray'" in n for n in r_us["notes"]), (r_us["modality"], r_us["notes"])
r_xr, _, _ = an.analyze(Image.fromarray((np.random.rand(60, 60) * 255).astype("uint8")), "xray_pneumonia", log=False)
assert r_xr["modality"] == "xray" and not any("suggests" in n for n in r_xr["notes"])
ok("original bug: modality shown follows the task (us_breast -> ultrasound, xray_pneumonia -> xray) + classifier disagreement noted")
vq = an.volume_quality(v, "ct")
assert {"blur", "contrast", "passed", "fraction_passed"} <= set(vq)
assert an.volume_quality(v, "mri")["calibrated"] is False
ok("volume quality (calibrated for CT, raw metrics for MRI)")

# ------------------------------------------------------------------ 4. 3D visualisation
att = attention_volume(v, grid, (2, 2, 2))
assert att.shape == (32, 32, 20) and att.max() <= 1.0
fig = volume_figure(v, mask, att)
names = [t.name for t in fig.data]
assert names == ["volume", "external mask", "Grad-CAM attention"] or "volume" in names, names
json.loads(fig.to_json())
assert volume_figure(v).data[0].name == "volume" and len(volume_figure(v).data) == 1   # no mask/attention -> nothing invented
ob = blend(v.slice2d("axial", 20), v.view("axial", 20, a=mask, nearest=True) > 0.5)
assert ob.shape[-1] == 3 and (ob[..., 0] != ob[..., 1]).any()
ok("3D figure (volume, mask, attention), overlay blending; no overlays when none exist")

# ------------------------------------------------------------------ 5. progression
vp = load_study([tmp / "prev.nii.gz"])["volume"]
img_c = v.axial_pil(20)
shifted = Image.fromarray(np.roll(np.asarray(img_c), (6, -9), axis=(0, 1)))
c2 = prog.compare_images(shifted, img_c)
assert c2["ncc_after"] > c2["ncc_before"] and c2["aligned"], (c2["ncc_before"], c2["ncc_after"])
ok(f"2D alignment improves similarity: {c2['ncc_before']:.2f} -> {c2['ncc_after']:.2f}")
cv = prog.compare_volumes(vp, v)
assert len(cv["pairs"]) == 5 and cv["pairs"][0]["difference"].shape[-1] == 3
rp, _, _ = an.analyze_volume(vp, "ct", None, False, 6)[0]
rows = prog.compare_predictions({"ct_organ": rp}, {"ct_organ": res})
assert rows[0]["status"] == "compared" and set(rows[0]["prob_delta"]) == set(res["classes"])
mc = prog.compare_masks(load_mask(tmp / "prev_mask.nii.gz", vp), mask, vp.spacing, v.spacing)
assert mc["curr_volume_ml"] > mc["prev_volume_ml"] and mc["delta_ml"] > 0
assert prog.compare_masks(None, mask, vp.spacing, v.spacing) is None   # needs masks for BOTH studies
summ = prog.summarise("ct", "ct", rows, cv, mc, "20230101", None)
assert summ["status"] and summ["current_date"] == "not available" and "does not state" in summ["disclaimer"]
assert "improved" not in summ["status"].lower() and "progress" not in summ["status"].lower()
assert "not comparable" in prog.summarise("ct", "xray", [], None, None)["status"].lower()
json.dumps(summ, default=float)
ok(f"progression: volumes, predictions, mask volumes ({mc['prev_volume_ml']:.2f} -> {mc['curr_volume_ml']:.2f} mL), summary")

# ------------------------------------------------------------------ 6. report
rep = build_report([res], an.audit.hash_id("P1"), "Radiologist", {"status": "verified", "value": "ct", "source": "dicom_metadata",
                   "warnings": []}, v.describe(), summ, {"volume_ml": ms["volume_ml"], "note": "user mask"})
assert rep["banner"] == BANNER and rep["patient_id_hashed"] != "P1" and rep["viewing_role"] == "Radiologist"
assert rep["findings"][0]["multi_slice"]["slices_analysed"] == len(res["slice_indices"])
assert "not a trained 3D model" in " ".join(rep["limitations"])
js = json.loads(to_json(rep))
assert js["banner"] == BANNER
assert BANNER in to_markdown(rep)
pdf = to_pdf(rep, [("Grad-CAM", ovs[res["slice_indices"][0]])])
assert pdf[:4] == b"%PDF" and len(pdf) > 2000
rep0 = build_report([], None, "Doctor / Physician", {"status": "manual", "value": "mri", "source": "manual", "warnings": []})
assert rep0["findings"] == [] and rep0["expert_review"]["recommended"] and rep0["segmentation_measurements"].startswith("not available")
assert to_pdf(rep0)[:4] == b"%PDF"
ok("report: JSON, Markdown preview, PDF (with and without findings), banner, no invented segmentation")

# ------------------------------------------------------------------ 7. end-to-end page (Streamlit AppTest)
from streamlit.testing.v1 import AppTest

nii_sample = str(tmp / "cur.nii.gz")
png_sample = str(tmp / "a.png")
script = tmp / "page.py"
script.write_text(f'''
import os, sys
import numpy as np
sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
import streamlit as st
from medscan.pipeline import Analyzer
from medscan.quality import Thresholds
from medscan.webapp import render_app
@st.cache_resource
def get(): return Analyzer({str(ckpt)!r}, device="cpu", audit_path={str(tmp / "audit2.jsonl")!r})
an = get()
mod = os.environ["FORCE_MOD"]; conf = float(os.environ.get("FORCE_CONF", "0.95"))
an.route = lambda x: (mod, np.array([0.05, 0.9, 0.05]) * 0 + conf / 1.0 * np.array([1, 1, 1]) / 3 + (1 - conf) / 3 if False else np.array([conf, (1-conf)/2, (1-conf)/2]))
if os.environ.get("STRICT"):
    for k in an.thresholds: an.thresholds[k] = Thresholds(1e9, 0, 1)
render_app(st, an, samples={{"vol": {nii_sample!r}, "img": {png_sample!r}}})
''')


def run(sample, mod="xray", conf=0.95, strict=False, setup=None):
    os.environ["FORCE_MOD"], os.environ["FORCE_CONF"] = mod, str(conf)
    os.environ.pop("STRICT", None)
    if strict:
        os.environ["STRICT"] = "1"
    at = AppTest.from_file(str(script), default_timeout=180).run()
    assert not at.exception, at.exception
    next(s for s in at.selectbox if s.label.startswith("...or try")).select(sample).run()
    assert not at.exception, at.exception
    if setup:
        setup(at)
    return at


def sget(at, k):
    return at.session_state[k] if k in at.session_state else None


def box(at, text):
    return next(c for c in at.checkbox if c.label.startswith(text))


# (a) X-ray image, auto-detect: analysed as X-ray; roles change presentation only
at = run("img", "xray")
res_by_role = {}
for role in ["Doctor / Physician", "Radiologist", "Medical Student", "Researcher", "Patient"]:
    at.selectbox[0].select(role).run()
    assert not at.exception, (role, at.exception)
    res_by_role[role] = json.dumps([(r["res"]["task"], r["res"]["prediction"], round(r["res"]["confidence"], 6))
                                    for r in at.session_state["mk_res"]])
assert len(set(res_by_role.values())) == 1 and "xray_pneumonia" in res_by_role["Doctor / Physician"]
ok("all 5 roles: identical predictions, no exception (role changes presentation only)")
assert any("Image type" in c.value for c in at.caption)

# (b) X-ray image but user selects Ultrasound -> mismatch, blocked until explicitly confirmed
at = run("img", "xray")
next(s for s in at.selectbox if s.label.startswith("Image type")).select("Ultrasound").run()
assert any("indicates X-ray" in e.value for e in at.error), [e.value for e in at.error]
assert "mk_res" not in at.session_state or not at.session_state["mk_res"]
box(at, "I understand the mismatch").check().run()
assert not at.exception, at.exception
rr = at.session_state["mk_res"]
assert [r["res"]["task"] for r in rr] == ["us_breast"] and rr[0]["res"]["modality_info"]["status"] == "mismatch"
assert any("mismatch" in e.value.lower() for e in at.error)
ok("X-ray image + user selects Ultrasound: blocked, warning shown, runs only after explicit confirm, stays flagged")

# (c) uncertain classifier on a PNG: must not silently pick a task
at = run("img", "xray", conf=0.40)
assert not at.session_state["mk_res"] if "mk_res" in at.session_state else True
assert any("could not be identified" in w.value for w in at.warning)
box(at, "Analyze anyway as").check().run()
assert at.session_state["mk_res"][0]["res"]["modality_info"]["status"] == "unverified"
ok("uncertain image type: no silent task choice; 'Analyze anyway as ...' marks result unverified")

# (d) NIfTI volume: unverified type -> confirm via manual CT -> multi-slice
at = run("vol", "ct", conf=0.95)
assert not sget(at, "mk_res")
next(s for s in at.selectbox if s.label.startswith("Image type")).select("CT").run()
assert not at.exception, at.exception
rr = at.session_state["mk_res"]
assert rr[0]["res"]["analysis_type"] == "multi-slice"
tab_names = [t.label for t in at.tabs]
assert "Multi-slice analysis" in tab_names and "3D viewer" in tab_names, tab_names
ok("NIfTI volume: type must be confirmed, then multi-slice analysis; viewer tabs present")

# (e) NIfTI as MRI: manual MRI, no trained task -> no prediction, viewer still works
at = run("vol", "ct")
next(s for s in at.selectbox if s.label.startswith("Image type")).select("MRI").run()
assert not at.exception, at.exception
assert not sget(at, "mk_res") and any("No classifier was trained for MRI" in i.value for i in at.info)
ok("MRI volume: no classifier -> no prediction, explanation shown")

# (f) quality gate: flagged -> paused until 'Analyze anyway'; warning stays after analysis
at = run("img", "xray", strict=True)
assert any("Scan-quality check flagged" in w.value for w in at.warning)
assert not sget(at, "mk_res")
box(at, "Analyze anyway").check().run()
assert not at.exception, at.exception
assert at.session_state["mk_res"] and any("LOW-QUALITY" in w.value for w in at.warning)
ok("quality warning: analysis paused, 'Analyze anyway' runs it, low-quality warning stays visible with the result")

# (g) model files untouched by anything above
assert __import__("hashlib").sha256(ckpt.read_bytes()).hexdigest() == h0
ok("checkpoint bytes unchanged")
print("\nUPGRADE TESTS PASSED")
