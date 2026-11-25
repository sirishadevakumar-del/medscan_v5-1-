"""Clinician-facing demo.  streamlit run app.py -- --ckpt runs/exp1/best.pt"""
import argparse
import json
import sys
from pathlib import Path

import streamlit as st

from medscan.pipeline import Analyzer
from medscan.webapp import render_app

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/trial2/best.pt")
args, _ = p.parse_known_args(sys.argv[1:])

st.set_page_config(page_title="MedScan decision support", layout="wide")
st.title("Adaptive AI medical scan analysis: decision support")
st.caption("Research prototype. Not a medical device; the clinician makes the final decision.")


@st.cache_resource
def get_analyzer(path):
    return Analyzer(path)


_m = Path(args.ckpt).parent / "eval" / "results.json"
metrics = json.loads(_m.read_text()) if _m.exists() else None
render_app(st, get_analyzer(args.ckpt), metrics=metrics)
