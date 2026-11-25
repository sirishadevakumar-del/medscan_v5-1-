"""Role-aware result display, shared by the local app and the public demo.

`st` is the streamlit module, passed in so this file does not import it.
"""

ROLES = ["Doctor / Physician", "Radiologist", "Medical Student", "Researcher", "Patient"]
CLINICAL_ROLES = [r for r in ROLES if r != "Patient"]

# task -> (image type shown to people, what the model decides, is the answer a diagnosis?)
TASKS = {
    "xray_pneumonia": ("chest X-ray", "whether signs of pneumonia are present", True),
    "ct_organ": ("CT scan", "which organ is shown", False),
    "us_breast": ("breast ultrasound", "whether the finding looks malignant", True),
    "derm_lesion": ("skin (dermoscopy) image", "the type of skin lesion", True),
    "fundus_dr": ("retina (fundus) image", "the grade of diabetic retinopathy", True),
    "histo_colon": ("colon tissue (histopathology) slide", "the tissue type, including cancer tissue", True),
    "mri_brain_tumor": ("brain MRI", "the type of brain tumour, or none", True),
    "xray_chest14": ("chest X-ray", "which of 14 common chest findings are present", True),
    "xray_covid": ("chest X-ray", "COVID-19, pneumonia, lung opacity or normal lungs", True),
    "ct_chest_cancer": ("chest CT scan", "the type of lung cancer, or normal", True),
}

CLINICAL_TIER = {"high": "High-confidence finding.",
                 "medium": "Medium confidence: clinician review recommended.",
                 "low": "Low confidence: do not rely on this prediction; refer for expert review."}
PATIENT_TIER = {"high": "The AI is fairly sure about this result. Please still talk to your doctor.",
                "medium": "The AI is somewhat unsure. A doctor should review this image.",
                "low": "The AI is not sure. Please do not rely on this result and ask your doctor."}


def info(task):
    return TASKS.get(task, ("medical image", "the image class", True))


def task_label(task):
    return task if task.startswith("(") else f"{info(task)[0]} ({task})"


def _pred(res):
    p = res["prediction"]
    return ", ".join(p) if isinstance(p, list) else str(p)


def result_title(res):
    area, what, _ = info(res["task"])
    return f"{area.capitalize()}: {what}"


def render_summary(st, results, role):
    """One line per analysis that was run for the detected image type."""
    st.subheader("Summary")
    for res, _, _ in results:
        area, what, _ = info(res["task"])
        pred = _pred(res)
        if role == "Patient":
            st.markdown(f"- {what.capitalize()}: **{pred}** (AI certainty: {res['tier']})")
        else:
            st.markdown(f"- {what.capitalize()}: **{pred}** ({res['confidence']:.0%} confidence, {res['tier']} tier)")


def modality_status_text(mi, modality):
    """Plain description of how the image type was established (detected / manual / verified / unverified)."""
    from .modality import label
    if not mi:
        return f"{label(modality)} (not recorded)"
    st_, src = mi.get("status"), mi.get("source")
    if st_ == "verified":
        return f"{label(modality)}: VERIFIED from DICOM metadata"
    if st_ == "detected":
        return f"{label(modality)}: detected by the image classifier ({mi.get('classifier_confidence', 0):.0%} certain)"
    if st_ == "manual":
        return f"{label(modality)}: selected manually (not independently verified)"
    if st_ == "mismatch":
        return f"{label(modality)}: selected manually, CONFLICTS with the detected type"
    return f"{label(modality)}: UNVERIFIED"


def banners(st, res):
    """Warnings that stay visible next to every result: modality problems and low scan quality."""
    mi = res.get("modality_info") or {}
    if mi.get("status") == "mismatch":
        st.error("Modality/task mismatch was confirmed by the user. This prediction may not be meaningful.")
    elif mi.get("status") in ("unverified", "manual"):
        st.warning("Image type not independently verified: the prediction may not be meaningful if the type is wrong.")
    for w in mi.get("warnings", []):
        st.caption("Image type: " + w)
    if not res["quality"]["passed"]:
        st.warning("Based on a LOW-QUALITY scan (" + ", ".join(res["quality"]["flags"]) + "): interpret with extra caution.")


def render_result(st, res, ov, role, metrics=None):
    area, what, is_dx = info(res["task"])
    pred, tier = _pred(res), res["tier"]
    patient = role == "Patient"
    st.divider()
    st.subheader(result_title(res))
    if res.get("analysis_type") == "multi-slice":
        st.caption(f"Volume-level result: {res['method']}")
    c1, c2 = st.columns(2)
    c1.caption("Prediction")
    c1.markdown(f"## {pred}")
    c2.caption("How sure is the AI?" if patient else "Confidence")
    c2.markdown(f"## {tier.capitalize()}" if patient else f"## {res['confidence']:.1%}  ({tier})")
    {"high": st.success, "medium": st.warning, "low": st.error}[tier]((PATIENT_TIER if patient else CLINICAL_TIER)[tier])
    banners(st, res)
    if res.get("multilabel"):
        top = sorted(res["probabilities"].items(), key=lambda kv: -kv[1])[:3]
        st.caption("Most likely findings: " + (", ".join(k for k, _ in top) if patient
                                               else ", ".join(f"{k} {v:.0%}" for k, v in top)))
    if tier == "low":
        st.caption("No reliable finding: the image may show a condition this system was not trained on.")
    if patient:
        _patient(st, res, ov, area, what, is_dx, pred)
    else:
        _clinical(st, res, ov, area, what, role, metrics)


def _patient(st, res, ov, area, what, is_dx, pred):
    st.write(f"The AI looked at your **{area}** and tried to decide **{what}**. Its answer: **{pred}**.")
    if not is_dx:
        st.info("This task recognises what is shown in the image. It is not a diagnosis.")
    if res["notes"]:
        st.warning("The picture quality may be too low for a reliable result. A clearer image could help.")
    with st.expander("Show where the AI looked"):
        st.image(ov, caption="Warmer colours: areas that influenced the AI most", width=400)
    st.caption("This is a research prototype and cannot replace a doctor.")


def _facts(st, res, area, what):
    from .modality import label
    q = res["quality"]
    st.markdown(f"**Modality:** {modality_status_text(res.get('modality_info'), res['modality'])}  \n"
                f"**Task:** {area}: {what} (`{res['task']}`)  \n"
                f"**Scan quality:** {'passed' if q['passed'] else 'FLAGGED: ' + ', '.join(q['flags'])} "
                f"(score {q['score']:.2f}, blur {q['blur']:.0f}, contrast {q['contrast']:.0f}, "
                f"edge density {q['edge_density']:.2f})")


def _prior(st, res):
    lo = res.get("longitudinal")
    if lo:
        st.write("Change vs previous study in the audit log (probability delta; not a measure of disease progression):",
                 lo["prob_change"])


def _clinical(st, res, ov, area, what, role, metrics):
    _facts(st, res, area, what)
    for n in res["notes"]:
        st.warning(n)
    if role == "Doctor / Physician":                       # concise
        _prior(st, res)
        with st.expander("Heat-map and probabilities"):
            st.image(ov, caption=f"Grad-CAM for '{res['gradcam_target']}'", width=420)
            st.bar_chart(res["probabilities"])
        return
    a, b = st.columns(2)
    a.image(ov, caption=f"Grad-CAM for '{res['gradcam_target']}'", width=480)
    b.bar_chart(res["probabilities"])
    _prior(st, res)
    with st.expander("Uncertainty per class (standard deviation over Monte Carlo passes)", expanded=role == "Researcher"):
        st.write({k: round(v, 4) for k, v in res["uncertainty_std"].items()})
        st.write({"MC stability (1 - sigma/0.5)": round(res.get("mc_confidence", float("nan")), 4),
                  "combined confidence": round(res["confidence"], 4)})
    if role == "Radiologist":
        with st.expander("Technical details", expanded=True):
            st.write({"quality metrics": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["quality"].items()
                                          if k not in ("flags",)},
                      "image-type probabilities": {k: round(v, 3) for k, v in res.get("modality_probs", {}).items()}})
    if role == "Researcher":
        with st.expander("Model performance (held-out test set, from evaluate.py)", expanded=True):
            if metrics:
                st.json(metrics.get(res["task"], metrics))
            else:
                st.caption("No evaluation results found next to the checkpoint (run evaluate.py to create runs/<exp>/eval/results.json).")
        with st.expander("Technical image-analysis information"):
            st.write({"modality probabilities": res.get("modality_probs", {}), "quality": res["quality"]})
    if role == "Medical Student":
        with st.expander("What is the confidence score?"):
            st.write("It is the probability of the predicted class multiplied by how stable that prediction is across "
                     "20 random dropout passes of two networks (Monte Carlo Dropout). Above 0.85 is high, "
                     "0.70 to 0.85 medium, below 0.70 low. It is not the chance of being correct.")
        with st.expander("What is Grad-CAM?"):
            st.write("The heat-map shows which image regions increased the score of the predicted class the most. "
                     "Red/yellow = influential. It shows where the model looked, not proof that it looked at the disease.")
        with st.expander("Why does the AI predict this?"):
            top = sorted(res["probabilities"].items(), key=lambda kv: -kv[1])[:2]
            st.write("The two highest model probabilities were " + " and ".join(f"'{k}' ({v:.0%})" for k, v in top) +
                     ". A prediction is only a statistical pattern match against the training images.")
        with st.expander("Why can the AI be wrong?"):
            st.write("It was trained on small public benchmark images (128 pixels). Images from other scanners, "
                     "other patients or unsupported image types can give unreliable results.")
