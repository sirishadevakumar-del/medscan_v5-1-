"""Record and verify SHA-256 hashes of trained weights, so you can prove the upgrade did not touch them.

  python scripts/verify_models.py record runs          # BEFORE upgrading: writes model_hashes.json
  python scripts/verify_models.py check               # AFTER upgrading: compares every file with the record
"""
import hashlib
import json
import sys
from pathlib import Path

RECORD = Path("model_hashes.json")


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "check"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "record":
        root = Path(sys.argv[2] if len(sys.argv) > 2 else "runs")
        files = sorted(root.rglob("*.pt")) + sorted(root.rglob("*.pth"))
        rec = {str(p): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}
        RECORD.write_text(json.dumps(rec, indent=2))
        for k, v in rec.items():
            print(f"{v['sha256'][:16]}...  {v['bytes']:>12,d} bytes  {k}")
        print(f"recorded {len(rec)} file(s) in {RECORD}")
        return
    rec = json.loads(RECORD.read_text())
    bad = 0
    for k, v in rec.items():
        p = Path(k)
        ok = p.exists() and sha256(p) == v["sha256"]
        bad += not ok
        print(("UNCHANGED " if ok else "CHANGED/MISSING ") + k)
    print("ALL MODEL FILES UNCHANGED" if not bad else f"{bad} FILE(S) DIFFER")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
