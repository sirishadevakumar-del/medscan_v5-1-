"""Viewer + progression + report tabs end to end, with a patched file uploader (AppTest cannot upload files)."""
import os, sys, tempfile
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("tu", Path(__file__).with_name("test_upgrade.py"))
# reuse fixtures (volumes, checkpoint, analyzer) by running the data-building part of test_upgrade
import runpy
ns = runpy.run_path(str(Path(__file__).with_name("test_upgrade.py")), run_name="fixtures")
tmp, ckpt = ns["tmp"], ns["ckpt"]
from streamlit.testing.v1 import AppTest

Image.fromarray((np.random.RandomState(3).rand(60, 60) * 255).astype("uint8")).save(tmp / "prev.png")
script = tmp / "page2.py"
script.write_text(f'''
import os, sys
import numpy as np
sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
import streamlit as st
from medscan.pipeline import Analyzer
from medscan.webapp import render_app
class F:
    def __init__(s, p): s.name = os.path.basename(p); s._b = open(p, "rb").read(); s.size = len(s._b)
    def getvalue(s): return s._b
FILES = {{"mask_": os.environ.get("MASK"), "prev_": os.environ.get("PREV"), "pmask_": os.environ.get("PMASK")}}
_orig = st.file_uploader
def fu(label, **k):
    for pre, path in FILES.items():
        if k.get("key", "").startswith(pre):
            return ([F(path)] if k.get("accept_multiple_files") else F(path)) if path else ([] if k.get("accept_multiple_files") else None)
    return [] if k.get("accept_multiple_files") else None
st.file_uploader = fu
@st.cache_resource
def get(): return Analyzer({str(ckpt)!r}, device="cpu", audit_path={str(tmp / "audit3.jsonl")!r})
an = get()
an.route = lambda x: (os.environ["FORCE_MOD"], np.array([0.9, 0.05, 0.05]) if os.environ["FORCE_MOD"] == "ct" else np.array([0.05, 0.9, 0.05]))
render_app(st, an, samples={{"vol": {str(tmp / "cur.nii.gz")!r}, "img": {str(tmp / "a.png")!r}}})
''')

def go(sample, mod, **env):
    os.environ.update(FORCE_MOD=mod, **{k: v for k, v in env.items()})
    for k in ("MASK", "PREV", "PMASK"):
        if k not in env:
            os.environ.pop(k, None)
    at = AppTest.from_file(str(script), default_timeout=240).run()
    next(s for s in at.selectbox if s.label.startswith("...or try")).select(sample).run()
    return at

# volume: CT confirmed manually, with mask, previous volume + previous mask, 3D rendering on
at = go("vol", "ct", MASK=str(tmp / "mask.nii.gz"), PREV=str(tmp / "prev.nii.gz"), PMASK=str(tmp / "prev_mask.nii.gz"))
next(s for s in at.selectbox if s.label.startswith("Image type")).select("CT").run()
assert not at.exception, at.exception
assert any("Mask loaded" in s.value for s in at.success)
next(c for c in at.checkbox if c.label.startswith("Render interactive 3D")).check().run()
assert not at.exception, at.exception
radio = next(r for r in at.radio if r.label == "View")
assert "External mask" in radio.options and "Both overlays" in radio.options, radio.options
radio.set_value("Both overlays").run()
assert not at.exception, at.exception
assert any("Comparison status" in w.value for w in list(at.warning) + list(at.success)), [w.value for w in at.warning]
txt = " ".join(m.value for m in at.markdown)
assert "Validated segmentation measurements" in txt and "Image differences" in txt and "Prediction differences" in txt
assert any("AI-GENERATED DRAFT REPORT" in e.value for e in at.error)
print("PASS volume: viewer (mask + attention overlays, 3D render), progression (images, predictions, masks), report")
for role in ["Radiologist", "Researcher", "Medical Student", "Patient", "Doctor / Physician"]:
    at.selectbox[0].select(role).run()
    assert not at.exception, (role, at.exception)
print("PASS volume pages under all roles")

# 2D: previous PNG
at = go("img", "xray", PREV=str(tmp / "prev.png"))
assert not at.exception, at.exception
assert any("Comparison status" in w.value for w in list(at.warning) + list(at.success))
print("PASS 2D progression with previous image")

# previous of a different kind -> not comparable
at = go("img", "xray", PREV=str(tmp / "prev.nii.gz"))
assert not at.exception, at.exception
assert any("Not comparable" in e.value for e in at.error)
print("PASS image vs volume: not comparable")
print("PAGE TESTS PASSED")
