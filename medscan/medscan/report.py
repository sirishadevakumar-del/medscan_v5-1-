"""Structured draft report built ONLY from verified pipeline outputs (no free-text generation, no invented findings).
Outputs: a dict (-> JSON), Markdown for the on-screen preview, and a PDF. Every output carries the draft banner."""
import io
import json
import time
from xml.sax.saxutils import escape

import numpy as np

from .modality import label as mod_label
from .view import info

BANNER = "AI-GENERATED DRAFT REPORT - REQUIRES CLINICIAN REVIEW"
LIMITATIONS = [
    "Decision support only; it is not a diagnosis and not a medical device.",
    "The model was trained on small public benchmark images (128 px) and has not been externally or clinically validated.",
    "Confidence is a model-internal score (probability x Monte Carlo stability), not a measured diagnostic accuracy.",
    "Grad-CAM shows where the model looked; it does not prove that the highlighted region is abnormal.",
    "Only the conditions the model was trained on can be recognised; other conditions may be missed or mislabelled.",
]
VOLUME_LIMITATION = ("Volume results come from the existing 2D model applied to selected axial slices and aggregated; "
                     "it is not a trained 3D model and does not use 3D context.")


def _pred(p):
    return ", ".join(p) if isinstance(p, list) else str(p)


def _finding(res):
    area, what, is_dx = info(res["task"])
    q = res["quality"]
    f = {"task": res["task"], "task_description": f"{area}: {what}", "is_diagnostic_task": bool(is_dx),
         "prediction": _pred(res["prediction"]), "confidence_percent": round(100 * res["confidence"], 1),
         "confidence_category": res["tier"],
         "probabilities": {k: round(v, 4) for k, v in res["probabilities"].items()},
         "uncertainty_std": {k: round(v, 4) for k, v in res["uncertainty_std"].items()},
         "scan_quality": {"passed": bool(q["passed"]), "flags": q["flags"],
                          "blur": round(q["blur"], 1), "contrast": round(q["contrast"], 1),
                          "edge_density": round(q["edge_density"], 3)},
         "gradcam_explanation": (f"Grad-CAM heat-map for '{res['gradcam_target']}' highlights the image regions that raised "
                                 "the model's score for that class the most. It shows where the model looked, not proof of "
                                 "disease; its reliability was not validated for this task.")}
    if res.get("analysis_type") == "multi-slice":
        f["multi_slice"] = {"method": res["method"], "slices_analysed": len(res["slice_indices"]),
                            "slices_in_volume": res["n_slices_total"], "slice_indices": res["slice_indices"],
                            "slice_agreement_percent": round(100 * res["slice_agreement"], 1),
                            "fraction_slices_quality_ok": round(res["quality"].get("fraction_passed", float("nan")), 3)}
    return f


def build_report(results, patient_hash=None, role="", mod_info=None, volume=None, comparison=None,
                 mask_measure=None, extra_reasons=()):
    """results: list of result dicts (2D or multi-slice). All content is copied from these outputs."""
    mod = (mod_info or {}).get("value") or (results[0]["modality"] if results else None)
    status = (mod_info or {}).get("status", "not recorded")
    reasons = list(extra_reasons)
    if status in ("unverified", "mismatch"):
        reasons.append(f"Image type is {status}.")
    if mod_info and mod_info.get("warnings"):
        reasons += mod_info["warnings"]
    findings = []
    for r in results:
        findings.append(_finding(r))
        if r["tier"] != "high":
            reasons.append(f"{r['task']}: {r['tier']} confidence.")
        if not r["quality"]["passed"]:
            reasons.append(f"{r['task']}: scan-quality check flagged ({', '.join(r['quality']['flags']) or 'low score'}).")
        if r.get("slice_agreement", 1) < 0.7:
            reasons.append(f"{r['task']}: slices disagree (agreement {r['slice_agreement']:.0%}).")
    if not results:
        reasons.append("No trained analysis exists for this image type; no AI prediction was made.")
    lim = list(LIMITATIONS) + ([VOLUME_LIMITATION] if any(r.get("analysis_type") == "multi-slice" for r in results) else [])
    rep = {"banner": BANNER, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "patient_id_hashed": patient_hash or "not provided", "viewing_role": role,
           "modality": mod_label(mod) if mod else "unknown", "modality_verification": status,
           "modality_source": (mod_info or {}).get("source", "not recorded"),
           "modality_warnings": (mod_info or {}).get("warnings", []),
           "scan": volume or "single 2D image", "findings": findings,
           "segmentation_measurements": mask_measure or "not available (no externally supplied segmentation mask)",
           "comparison": comparison or "not available (no previous study supplied)",
           "limitations": lim,
           "expert_review": {"recommended": bool(reasons), "reasons": sorted(set(reasons)),
                             "statement": ("Expert review is recommended." if reasons else
                                           "No automatic flags were raised; the usual clinician review of AI output still applies.")}}
    return rep


def to_json(rep):
    return json.dumps(rep, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


def to_markdown(rep):
    L = [f"### :red[{rep['banner']}]", f"**Patient (hashed):** {rep['patient_id_hashed']}  \n**Viewed as:** {rep['viewing_role']}  \n"
         f"**Generated:** {rep['generated_at']}", "", f"**Modality:** {rep['modality']} ({rep['modality_verification']}, "
         f"source: {rep['modality_source']})"]
    for w in rep["modality_warnings"]:
        L.append(f"- Warning: {w}")
    L.append("")
    if not rep["findings"]:
        L.append("**AI prediction:** none (no trained analysis for this image type).")
    for f in rep["findings"]:
        L += [f"#### {f['task_description']}", f"- **Prediction:** {f['prediction']}",
              f"- **Confidence:** {f['confidence_percent']}% ({f['confidence_category']})",
              f"- **Scan quality:** {'passed' if f['scan_quality']['passed'] else 'flagged: ' + ', '.join(f['scan_quality']['flags'])}",
              f"- **Grad-CAM:** {f['gradcam_explanation']}"]
        if "multi_slice" in f:
            m = f["multi_slice"]
            L.append(f"- **Multi-slice:** {m['slices_analysed']} of {m['slices_in_volume']} axial slices, agreement "
                     f"{m['slice_agreement_percent']}%. {m['method']}")
    seg = rep["segmentation_measurements"]
    L += ["", "**Segmentation measurements:** " + (seg if isinstance(seg, str) else
          f"external mask volume {seg['volume_ml']:.2f} mL (user-supplied, not validated by this system)")]
    c = rep["comparison"]
    L.append("**Previous vs current:** " + (c if isinstance(c, str) else c["status"] +
             f" (previous {c['previous_date']}, current {c['current_date']}). " + c["disclaimer"]))
    L += ["", "**Uncertainty and limitations:**"] + [f"- {x}" for x in rep["limitations"]]
    ev = rep["expert_review"]
    L += ["", f"**Expert review:** {ev['statement']}"] + [f"- {r}" for r in ev["reasons"]]
    return "\n".join(L)


def to_pdf(rep, images=None):
    """images: optional list of (caption, HxWx3 uint8) e.g. Grad-CAM overlays. Returns PDF bytes."""
    from PIL import Image
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

    ss = getSampleStyleSheet()
    body, h2 = ss["BodyText"], ParagraphStyle("h2", parent=ss["Heading3"], spaceBefore=8)
    ban = ParagraphStyle("ban", parent=ss["Heading2"], textColor=colors.white, backColor=colors.HexColor("#b00020"),
                         alignment=1, borderPadding=6, spaceAfter=10)
    P = lambda t, s=body: Paragraph(escape(str(t)).replace("\n", "<br/>"), s)
    story = [Paragraph(BANNER, ban),
             P(f"Patient (hashed): {rep['patient_id_hashed']}   |   Viewed as: {rep['viewing_role']}   |   {rep['generated_at']}"),
             P(f"Scan modality: {rep['modality']}  -  verification: {rep['modality_verification']} ({rep['modality_source']})")]
    for w in rep["modality_warnings"]:
        story.append(P("Warning: " + w))
    if isinstance(rep["scan"], dict):
        story.append(P("Volume: " + json.dumps(rep["scan"])))
    if not rep["findings"]:
        story.append(P("AI prediction: none (no trained analysis for this image type)."))
    for f in rep["findings"]:
        story.append(P(f["task_description"], h2))
        rows = [["Prediction", f["prediction"]], ["Confidence", f"{f['confidence_percent']}% ({f['confidence_category']})"],
                ["Scan quality", "passed" if f["scan_quality"]["passed"] else "flagged: " + ", ".join(f["scan_quality"]["flags"])]]
        if "multi_slice" in f:
            m = f["multi_slice"]
            rows.append(["Multi-slice", f"{m['slices_analysed']} of {m['slices_in_volume']} slices; agreement {m['slice_agreement_percent']}%"])
        t = Table([[P(a), P(b)] for a, b in rows], colWidths=[35 * mm, 135 * mm])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story += [t, P("Grad-CAM: " + f["gradcam_explanation"])]
        if "multi_slice" in f:
            story.append(P("Method: " + f["multi_slice"]["method"]))
    for cap, arr in (images or [])[:6]:
        b = io.BytesIO()
        Image.fromarray(arr).save(b, "PNG")
        b.seek(0)
        story += [Spacer(1, 4), RLImage(b, width=60 * mm, height=60 * mm), P(cap)]
    seg = rep["segmentation_measurements"]
    story.append(P("Segmentation measurements", h2))
    story.append(P(seg if isinstance(seg, str) else f"External mask volume {seg['volume_ml']:.2f} mL. {seg.get('note', '')}"))
    c = rep["comparison"]
    story.append(P("Previous vs current comparison", h2))
    story.append(P(c if isinstance(c, str) else f"{c['status']}  (previous {c['previous_date']}, current {c['current_date']})\n{c['disclaimer']}"))
    story.append(P("Uncertainty and limitations", h2))
    story += [P("- " + x) for x in rep["limitations"]]
    ev = rep["expert_review"]
    story.append(P("Recommendation for expert review", h2))
    story += [P(ev["statement"])] + [P("- " + r) for r in ev["reasons"]]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.HexColor("#b00020"))
        canvas.drawCentredString(A4[0] / 2, 10 * mm, f"{BANNER}   -   page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                      bottomMargin=18 * mm, title="MedScan draft report").build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
