"""Image-type (modality) determination and task/modality validation.

Priority of evidence:
  1. manual    : the user chose the image type explicitly.
  2. DICOM tag : (0008,0060) Modality written by the scanner  -> "verified" (but only as reliable as the file).
  3. classifier: the trained routing head, only trusted when its certainty is >= MIN_CONF -> "detected".
  4. otherwise "unverified": the image type is uncertain and the user is asked to confirm it.
File extensions never identify a modality. NIfTI carries no reliable modality, so it is never "verified".
"""
from dataclasses import dataclass, field

MIN_CONF = 0.70

# DICOM Modality tag -> our modality names
DICOM_MAP = {"CR": "xray", "DX": "xray", "DR": "xray", "RF": "xray", "MG": "xray",
             "CT": "ct", "MR": "mri", "US": "ultrasound", "OP": "fundus", "OPT": "fundus",
             "SM": "histopathology", "XC": "dermoscopy"}

LABELS = {"xray": "X-ray", "ct": "CT", "mri": "MRI", "ultrasound": "Ultrasound", "dermoscopy": "Dermoscopy (skin)",
          "fundus": "Fundus (retina)", "histopathology": "Histopathology"}
ALL_MODALITIES = list(LABELS)


def label(m):
    return LABELS.get(m, str(m))


@dataclass
class Modality:
    value: str                      # modality used for analysis
    status: str                     # verified | detected | manual | unverified | mismatch
    source: str                     # dicom_metadata | classifier | manual
    classifier_guess: str = None
    classifier_confidence: float = 0.0
    dicom_modality: str = None
    warnings: list = field(default_factory=list)
    blocked: bool = False           # incompatible: analysis must not run without explicit confirmation

    def to_dict(self):
        return dict(self.__dict__)


def determine(classifier_guess, classifier_conf, dicom_tag=None, manual=None, ext_is_nifti=False,
              confirmed_mismatch=False, known=None):
    """Combine every source of evidence into one Modality verdict. Never silently picks an arbitrary type.
    known: image types the classifier was trained on. It can only contradict a manual choice inside that set (it has never
    seen e.g. MRI, so its guess says nothing about whether an image is MRI)."""
    meta = DICOM_MAP.get((dicom_tag or "").upper()) if dicom_tag else None
    ref, ref_src = (meta, "dicom_metadata") if meta else (None, None)
    w = []
    if dicom_tag and not meta:
        w.append(f"DICOM modality tag '{dicom_tag}' is not one of the supported image types.")
    if manual:                                              # the user chose
        m = Modality(manual, "manual", "manual", classifier_guess, classifier_conf, dicom_tag, w)
        trusted = classifier_guess if (classifier_conf >= MIN_CONF and (known is None or manual in known)) else None
        if known is not None and manual not in known and not ref:
            m.warnings.append(f"The image classifier was not trained on {label(manual)}, so it cannot verify this choice.")
        other = ref or trusted
        if other and other != manual:
            src = "the file's DICOM metadata" if ref else "the image classifier"
            m.status = "mismatch"
            m.warnings.append(f"You selected {label(manual)}, but {src} indicates {label(other)}. "
                              "Analysis for the selected type is blocked unless you explicitly confirm.")
            m.blocked = not confirmed_mismatch
            if confirmed_mismatch:
                m.warnings.append("Mismatch confirmed by the user: the result may not be meaningful.")
        return m
    if ref:
        m = Modality(ref, "verified", ref_src, classifier_guess, classifier_conf, dicom_tag, w)
        if classifier_conf >= MIN_CONF and classifier_guess != ref and classifier_guess in ALL_MODALITIES:
            m.warnings.append(f"The image classifier suggests {label(classifier_guess)}, but the file's DICOM "
                              f"metadata says {label(ref)}. The metadata is used.")
        return m
    if ext_is_nifti:
        w.append("NIfTI files carry no reliable image-type information: please confirm the image type.")
        return Modality(classifier_guess if classifier_conf >= MIN_CONF else None, "unverified", "classifier",
                        classifier_guess, classifier_conf, dicom_tag, w, blocked=True)
    if classifier_conf >= MIN_CONF:
        return Modality(classifier_guess, "detected", "classifier", classifier_guess, classifier_conf, dicom_tag, w)
    w.append("The image type could not be identified with confidence. Please choose it manually.")
    return Modality(None, "unverified", "classifier", classifier_guess, classifier_conf, dicom_tag, w, blocked=True)


def tasks_for(trained_tasks, modality):
    """Tasks (name -> meta) trained for this modality. Empty for e.g. MRI when no MRI task was trained."""
    return [t for t, v in trained_tasks.items() if v["modality"] == modality]


def validate_task(task_meta, modality):
    """Strict task/modality check used before any analysis."""
    return task_meta["modality"] == modality
