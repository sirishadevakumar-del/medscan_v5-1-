"""Previous-vs-current comparison. Three kinds of difference are kept strictly apart:
  1. image differences   (alignment + difference map; says nothing about disease),
  2. prediction differences (model output changed; NOT evidence of progression or improvement),
  3. validated segmentation measurements (only from external masks supplied by the user).
Alignment is a 2D affine registration (ECC) per slice, not a deformable 3D registration.
"""
import cv2
import numpy as np

from .volume import select_slices, mask_stats

DISCLAIMER = ("This comparison lists differences. It does not state that a condition has improved or progressed: a change in "
              "model prediction or confidence is not a clinical measurement, and image differences can come from "
              "positioning, scanner settings or contrast rather than disease.")


def _gray(pil, size=256):
    g = np.asarray(pil.convert("L"), dtype=np.float32) / 255.0
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def _ncc(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def compare_images(prev_pil, curr_pil, size=256):
    """Register the previous image to the current one (affine ECC) and compute a difference map."""
    p, c = _gray(prev_pil, size), _gray(curr_pil, size)
    before = _ncc(p, c)
    warp = np.eye(2, 3, dtype=np.float32)
    ok = True
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6)
        _, warp = cv2.findTransformECC(cv2.GaussianBlur(c, (5, 5), 0), cv2.GaussianBlur(p, (5, 5), 0), warp,
                                       cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error:
        ok, warp = False, np.eye(2, 3, dtype=np.float32)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    aligned = cv2.warpAffine(p, warp, (size, size), flags=flags, borderValue=0)
    valid = cv2.warpAffine(np.ones_like(p), warp, (size, size), flags=flags, borderValue=0) > 0.99
    after = _ncc(aligned, c, valid)
    use_aligned = ok and after >= before
    ref = aligned if use_aligned else p
    diff = np.abs(c - ref)
    if use_aligned:
        diff = diff * valid
    heat = cv2.cvtColor(cv2.applyColorMap((np.clip(diff * 2, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
                        cv2.COLOR_BGR2RGB)
    status = "ok" if (use_aligned and after >= 0.5) else "poor"
    return {"ncc_before": before, "ncc_after": after if use_aligned else before, "aligned": bool(use_aligned),
            "mean_abs_diff": float(diff[valid].mean()) if use_aligned and valid.any() else float(diff.mean()),
            "alignment_status": status,
            "previous": (p * 255).astype(np.uint8), "previous_aligned": (ref * 255).astype(np.uint8),
            "current": (c * 255).astype(np.uint8), "difference": heat}


def compare_volumes(prev_vol, curr_vol, n_pairs=5):
    """Compare corresponding axial slices, matched by relative position inside the tissue range of each volume."""
    pi, ci = select_slices(prev_vol, 256), select_slices(curr_vol, 256)
    pairs = []
    for t in np.linspace(0.1, 0.9, n_pairs):
        a = pi[int(round(t * (len(pi) - 1)))]
        b = ci[int(round(t * (len(ci) - 1)))]
        r = compare_images(prev_vol.axial_pil(a), curr_vol.axial_pil(b))
        r.update(prev_z=int(a), curr_z=int(b), fraction=float(t))
        pairs.append(r)
    notes = []
    if prev_vol.orientation != curr_vol.orientation:
        notes.append("The two volumes use different orientation conventions; slice matching is approximate.")
    if np.abs(np.array(prev_vol.spacing) - np.array(curr_vol.spacing)).max() > 0.5:
        notes.append("Voxel spacing differs between studies; slice matching by relative position is approximate.")
    return {"pairs": pairs, "mean_ncc_after": float(np.mean([p["ncc_after"] for p in pairs])),
            "alignment_status": "ok" if all(p["alignment_status"] == "ok" for p in pairs) else "poor", "notes": notes,
            "method": "slice matching by relative position + 2D affine (ECC) registration per slice"}


def compare_predictions(prev, curr):
    """prev / curr: dict task -> result (single study). Only tasks present in both with identical classes are compared."""
    rows = []
    for task, c in curr.items():
        p = prev.get(task)
        if p is None:
            rows.append({"task": task, "status": "not available in previous study"})
            continue
        if list(p["probabilities"]) != list(c["probabilities"]):
            rows.append({"task": task, "status": "class lists differ: not comparable"})
            continue
        cl = list(c["probabilities"])
        pp, cc = [p["probabilities"][k] for k in cl], [c["probabilities"][k] for k in cl]
        rows.append({"task": task, "status": "compared", "classes": cl, "prev_probs": pp, "curr_probs": cc,
                     "prev_prediction": p["prediction"], "curr_prediction": c["prediction"],
                     "prev_confidence": p["confidence"], "curr_confidence": c["confidence"],
                     "prediction_changed": p["prediction"] != c["prediction"],
                     "prob_delta": {k: round(b - a, 4) for k, a, b in zip(cl, pp, cc)}})
    return rows


def compare_masks(prev_mask, curr_mask, prev_spacing, curr_spacing):
    """Measured size of two external masks. Only reported when both exist."""
    if prev_mask is None or curr_mask is None:
        return None
    a, b = mask_stats(prev_mask, prev_spacing), mask_stats(curr_mask, curr_spacing)
    dv = b["volume_ml"] - a["volume_ml"]
    return {"prev_volume_ml": a["volume_ml"], "curr_volume_ml": b["volume_ml"], "delta_ml": dv,
            "percent_change": (100.0 * dv / a["volume_ml"]) if a["volume_ml"] > 0 else None,
            "prev_curve": a["area_per_slice_mm2"], "curr_curve": b["area_per_slice_mm2"],
            "note": "Measured from masks supplied by the user. This system did not create or validate them."}


def fmt_date(d):
    if not d:
        return None
    d = str(d)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 and d.isdigit() else d


def summarise(prev_mod, curr_mod, pred_rows, image_cmp, mask_cmp, prev_date=None, curr_date=None, extra_notes=()):
    """Structured comparison summary (JSON-serialisable)."""
    notes = list(extra_notes)
    if prev_mod != curr_mod:
        status = f"Not comparable: previous study is {prev_mod}, current is {curr_mod}."
    elif image_cmp and image_cmp["alignment_status"] != "ok":
        status = "Limited: the images could not be aligned reliably, so the difference maps are unreliable."
    else:
        status = "Comparable"
    return {"status": status, "previous_date": fmt_date(prev_date) or "not available",
            "current_date": fmt_date(curr_date) or "not available", "modality": curr_mod,
            "predictions": pred_rows,
            "image": None if not image_cmp else {k: image_cmp[k] for k in image_cmp
                                                   if k in ("mean_ncc_after", "ncc_before", "ncc_after", "mean_abs_diff",
                                                            "alignment_status", "method", "notes")},
            "segmentation": None if not mask_cmp else {k: v for k, v in mask_cmp.items() if not k.endswith("curve")},
            "segmentation_available": mask_cmp is not None, "notes": notes, "disclaimer": DISCLAIMER}
