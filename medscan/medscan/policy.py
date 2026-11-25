"""Adaptive output presentation: what the system shows depends on confidence and scan quality."""


def decide(confidence, quality_passed, quality_flags, conf_high=0.85, conf_low=0.70):
    tier = "high" if confidence > conf_high else ("medium" if confidence >= conf_low else "low")
    if not quality_passed and tier == "high":
        tier = "medium"     # poor image quality caps the confidence tier
    action = {
        "high": "Show prediction with heat-map as decision support.",
        "medium": "Show prediction with an uncertainty warning; clinician review recommended.",
        "low": "Do not rely on this prediction; refer for expert review.",
    }[tier]
    notes = []
    if not quality_passed:
        notes.append("Scan quality issue(s): " + ", ".join(quality_flags) + ". Consider re-acquisition.")
    return {"tier": tier, "action": action, "notes": notes}
