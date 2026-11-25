"""Privacy-preserving audit log, clinician feedback, and longitudinal comparison.

No image data or raw patient identifiers are stored: patient ids are salted SHA-256 hashes.
"""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path


class AuditLog:
    def __init__(self, path="runs/audit_log.jsonl", salt=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.salt = salt or os.environ.get("MEDSCAN_SALT", "change-me")

    def hash_id(self, patient_id):
        return hashlib.sha256((self.salt + str(patient_id)).encode()).hexdigest()[:16] if patient_id else None

    def _read(self):
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def log(self, **rec):
        rec = {"case_id": rec.pop("case_id", None) or uuid.uuid4().hex[:12], "time": time.time(), "type": "prediction", **rec}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec["case_id"]

    def feedback(self, case_id, agrees, final_label=None, comment=""):
        with open(self.path, "a") as f:
            f.write(json.dumps({"type": "feedback", "case_id": case_id, "time": time.time(),
                                "agrees": bool(agrees), "final_label": final_label, "comment": comment}) + "\n")

    def prior(self, patient_hash, task):
        rows = [r for r in self._read() if r.get("type") == "prediction" and r.get("patient") == patient_hash and r.get("task") == task]
        return rows[-1] if rows else None

    def compare_with_prior(self, patient_hash, task, probs):
        """Longitudinal change: difference in class probabilities vs. the patient's previous study."""
        if not patient_hash:
            return None
        p = self.prior(patient_hash, task)
        if not p:
            return None
        delta = [round(a - b, 4) for a, b in zip(probs, p["probs"])]
        return {"prior_case": p["case_id"], "prior_time": p["time"], "prob_change": delta,
                "prior_pred": p["pred_label"]}
