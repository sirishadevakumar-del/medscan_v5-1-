"""Analyse one scan.  python infer.py --ckpt runs/exp1/best.pt --image scan.png [--task xray_pneumonia]"""
import argparse
import json

from PIL import Image

from medscan.pipeline import Analyzer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--image", required=True)
    ap.add_argument("--task"); ap.add_argument("--patient-id"); ap.add_argument("--out", default="result_overlay.png")
    ap.add_argument("--audit", default="runs/audit_log.jsonl")
    a = ap.parse_args()
    an = Analyzer(a.ckpt, audit_path=a.audit)
    res, ov, _ = an.analyze(a.image, a.task, a.patient_id)
    Image.fromarray(ov).save(a.out)
    print(json.dumps(res, indent=2, default=float))
    print(f"\nGrad-CAM overlay saved to {a.out}")


if __name__ == "__main__":
    main()
