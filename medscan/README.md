# MedScan — adaptive, uncertainty-aware, explainable analysis for all diagnostic image types

Generalises the chest-X-ray pipeline of the STCR-2026 paper to X-ray, CT, MRI, ultrasound, fundus,
dermoscopy and histopathology. One shared DenseNet121 + EfficientNet-B0 ensemble with a classifier head per
task and an auxiliary modality-routing head.

## Pipeline (paper section → code)

| Paper | Here |
|---|---|
| III-B quality assessment (blur / contrast / artefacts) | `medscan/quality.py` — thresholds **calibrated per task** from training data |
| III-C DenseNet121 + EfficientNet-B0, MC-Dropout head (Dropout-512-BN-ReLU-Dropout) | `medscan/models.py` |
| III-D loss / Adam / LR halving / early stop / grad-clip 1.0 | `train.py` (CE for multi-class, BCE for multi-label) |
| III-E Monte Carlo Dropout, confidence tiers 0.85 / 0.70 | `medscan/uncertainty.py`, `medscan/policy.py` |
| III-F Grad-CAM | `medscan/gradcam.py` (+ quantitative deletion test) |
| Calibration (ref. [13]) | temperature scaling on the validation set + ECE / reliability diagram |
| Audit, clinician feedback, longitudinal comparison | `medscan/audit.py`, `app.py` |
| Tables II–IV, Figs 1,3,4,5 | `evaluate.py` → `report.md`, `results.json`, PNGs |

## Quick start

```bash
pip install -r requirements.txt
python tests/smoke_test.py                       # 1–2 min, CPU, synthetic data: verifies everything runs

python train.py --out runs/exp1                  # downloads MedMNIST (6 tasks) and trains
python evaluate.py --ckpt runs/exp1/best.pt      # writes runs/exp1/eval/report.md + figures
python infer.py --ckpt runs/exp1/best.pt --image scan.png          # auto-detects modality
streamlit run app.py -- --ckpt runs/exp1/best.pt                    # demo UI with feedback logging
```

For an 8 GB GPU the defaults (batch 32, AMP, 224 px) fit. First run: try `--epochs 15` to check timing
(6 tasks × 100 epochs is long). Use `--image-size 128` to speed up.

## Using your own data (any modality, incl. MRI and your CheXpert subset)

1. Arrange `root/{train,val,test}/<class_name>/*.png|jpg|dcm|nii.gz`
   (`scripts/prepare_folder_dataset.py` does a 70/15/15 split, **by patient** if you give `--patient-regex`).
2. Add an entry to `configs/tasks.json` with `"source": "folder"`, a unique `name`, a `modality`, `"path"`.
   Set `"hflip": false` where left/right matters.
3. Re-run `train.py`. No code changes are needed.

DICOM is windowed automatically. For training, a NIfTI volume contributes its middle slice; the **app** analyses many slices (see below).

## The app

`streamlit run app.py -- --ckpt runs/exp1/best.pt`

No new training is needed: the app only uses your existing `best.pt`.

| Section (in page order) | What it does |
|---|---|
| Who is viewing? | Doctor / Physician (concise), Radiologist (technical + Grad-CAM + overlays), Medical Student (explanations), Researcher (probabilities, uncertainty, test-set metrics), Patient (plain language). The role changes only the presentation, never the prediction. |
| Medical scan upload | PNG/JPG, DICOM (single file, a series of files, or one .zip), NIfTI volume. |
| Image type | Auto-detect (DICOM metadata first, else the image classifier when it is >= 70 % certain). Otherwise you are asked to choose. Choosing a type that conflicts with the evidence shows a warning and **blocks** analysis until you explicitly confirm; the result stays marked "mismatch". NIfTI is never treated as verified. |
| Scan quality | Blur, brightness, contrast and a status. If flagged, analysis waits for **Analyze anyway** (this does not override an image-type problem). The low-quality warning stays visible next to the result. |
| AI findings | Prediction, confidence and tier, quality status, modality-verification status, Grad-CAM, probability graph, uncertainty. |
| Multi-slice analysis (CT/MRI volumes) | Up to 24-48 evenly spaced axial slices are each classified by the existing 2D ensemble; the slice probabilities are averaged into a volume result with a slice-agreement score. Per-slice table and plot, slice inspector with Grad-CAM. **This is not a 3D model.** |
| 3D viewer (volumes) | Axial / coronal / sagittal sliders with the original voxel spacing, overlay switch, interactive Plotly 3D rendering (rotate/zoom/pan). Overlays: an external NIfTI mask you upload, and/or the Grad-CAM attention region (where the model looked, **not a segmentation**). |
| Progression | Upload the previous scan: side-by-side images, 2D affine alignment, difference map, prediction/confidence comparison graphs, and mask volumes (mL) only if you supply masks for both studies. The summary never says a condition improved or progressed. |
| Report | Structured draft report built only from model outputs; preview, **PDF** and **JSON** download; every page says "AI-GENERATED DRAFT REPORT - REQUIRES CLINICIAN REVIEW". |
| Feedback & audit | Clinician feedback and the patient's audit history. |

Limits you should know about:
- MRI has no trained classifier: MRI volumes can be viewed, quality-checked and reported, but no prediction is made.
  The image classifier has never seen MRI, so for a PNG/JPG MRI without DICOM metadata the type must be chosen by you.
- Segmentation measurements exist only for masks you upload (same grid as the volume; they are never resampled).
- Registration is 2D per slice, not deformable 3D registration. Slice pairs are matched by relative position.
- Volume analysis feeds axial slices to a model trained on 2D benchmark images (OrganAMNIST for CT); it is not a trained
  3D network and not validated on full CT/MRI studies.

Verify that the upgrade left your trained weights untouched:

```
python scripts/verify_models.py record runs     # once, from your own copy
python scripts/verify_models.py check           # any time later
```

Tests (random weights, synthetic data; they check the code, not clinical validity):
`python tests/test_multitask.py`, `python tests/test_upgrade.py`, `python tests/test_pages.py`.

## Which diseases can it recognise?

Only the ones in the datasets it was trained on. "Any disease" is not possible; more datasets mean more coverage.

| Image type | Task (name in configs/tasks.json) | Data | Setup |
|---|---|---|---|
| Chest X-ray | `xray_pneumonia` (normal / pneumonia) | MedMNIST | automatic download |
| Chest X-ray | `xray_chest14` (14 common findings, multi-label) | MedMNIST ChestMNIST (NIH ChestX-ray14) | automatic download |
| Chest X-ray | `xray_covid` (COVID-19, lung opacity, viral pneumonia, normal) | Kaggle "COVID-19 Radiography Database" | manual |
| CT | `ct_organ` (which organ: not a disease) | MedMNIST | automatic download |
| CT | `ct_chest_cancer` (lung cancer types, normal) | Kaggle "Chest CT-Scan images Dataset" | manual |
| MRI | `mri_brain_tumor` (glioma, meningioma, pituitary, no tumour) | Kaggle "Brain Tumor MRI Dataset" | manual |
| Ultrasound, skin, eye, tissue | `us_breast`, `derm_lesion`, `fundus_dr`, `histo_colon` | MedMNIST | automatic download |

Manual datasets: download from Kaggle (free account), unzip, then

```
python scripts/prepare_folder_dataset.py --src <unzipped COVID-19_Radiography_Dataset> --dst data/covid_xray --only-subfolder images
python scripts/prepare_presplit.py --src <unzipped chest CT> --dst data/chest_ct --train-name train --val-name valid --test-name test
python scripts/prepare_presplit.py --src <unzipped brain MRI> --dst data/brain_mri
```

then set `"enabled": true` for those tasks in `configs/tasks.json` and train all tasks together:

`python train.py --out runs/exp2 --epochs 30 --num-workers 0 --image-size 128 --max-batches 300 --freeze-bn --dropout-after-bn`

Check each dataset's licence and terms before using it.

## Things worth knowing

- **Confidence.** The paper's `1 − σ/σmax` ignores the predicted probability (a model that always outputs 0.5
  would get confidence 1). Default here is `p(pred) × (1 − σ/σmax)`; set `confidence_mode: "paper"` in the config to
  reproduce the original definition. σ includes disagreement between the two ensemble members.
- **MC Dropout speed.** Dropout exists only in the heads, so the backbones run once and T=20 head passes are sampled.
- **Ensembling.** Member *probabilities* are averaged (paper Eq. 1 averages outputs; its text also says "concatenated").
- **Quality thresholds** are the 5th/95th percentile of each task's training images, so ~95 % pass on in-distribution
  data by construction. Hand-set thresholds in `quality_thresholds.json` if you want clinically defined ones.
- **MedMNIST** images are low-resolution (≤224 px) benchmark data. Good for developing and comparing, not a substitute
  for full-resolution clinical datasets.
- **Not a medical device.** Needs external validation, patient-level splits and clinical review before any clinical claim.
- Set `MEDSCAN_SALT` for the audit-log patient-id hashing.
