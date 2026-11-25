"""Evaluate a trained checkpoint on every task's test split.  python evaluate.py --ckpt runs/exp1/best.pt"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from medscan.config import Config, load_tasks
from medscan.data import build_task, make_loader
from medscan.gradcam import faithfulness, gradcam, overlay
from medscan.metrics import classification_metrics, ece_score, risk_coverage, tier_table
from medscan.models import count_params, load_checkpoint
from medscan.plots import confidence_hist, confusion_plot, gallery, reliability_plot, roc_plot
from medscan.quality import Thresholds, assess
from medscan.uncertainty import mc_predict


def latency_ms(fn, x, n=15):
    fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out")
    ap.add_argument("--tasks", help="override tasks file"); ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-cam", type=int, default=100, help="images for Grad-CAM faithfulness test")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ens, cfg, ck = load_checkpoint(a.ckpt, device)
    cfg.num_workers = a.num_workers
    if a.tasks:
        cfg.tasks_file = a.tasks
    out = Path(a.out or Path(a.ckpt).parent / "eval"); out.mkdir(parents=True, exist_ok=True)
    specs = [s for s in load_tasks(cfg.tasks_file) if s["name"] in ck["task_meta"]]
    results, report = {}, ["# Evaluation report\n"]

    for s in specs:
        n = s["name"]
        meta = ck["task_meta"][n]
        classes, ml = meta["classes"], meta["multilabel"]
        b = build_task(s, cfg)
        test = b["test"]
        ld = make_loader(test, 64, False, cfg.num_workers)
        print(f"== {n} ({meta['modality']}), {len(test)} test images")
        member_p = [[] for _ in ens.members]; ens_p, mc_p, conf, ys = [], [], [], []
        with torch.no_grad():
            for x, y in ld:
                x = x.to(device)
                zs = ens.member_logits(x, n)
                for k, z in enumerate(zs):
                    member_p[k].append(ens.to_probs(z, n).cpu())
                ens_p.append(torch.stack([ens.to_probs(z, n) for z in zs]).mean(0).cpu())
                r = mc_predict(ens, x, n, cfg.mc_samples, cfg.confidence_mode)
                mc_p.append(r["probs"].cpu()); conf.append(r["confidence"].cpu()); ys.append(y)
        y = torch.cat(ys).numpy()
        P = {"DenseNet121": torch.cat(member_p[0]).numpy(), "EfficientNet-B0": torch.cat(member_p[1]).numpy(),
             "Ensemble": torch.cat(ens_p).numpy()}
        mcP, conf = torch.cat(mc_p).numpy(), torch.cat(conf).numpy()

        x1 = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
        lat = {"DenseNet121": latency_ms(lambda z: ens.members[0](z, n), x1),
               "EfficientNet-B0": latency_ms(lambda z: ens.members[1](z, n), x1),
               "Ensemble": latency_ms(lambda z: ens(z, n), x1)}
        params = {"DenseNet121": count_params(ens.members[0]) / 1e6, "EfficientNet-B0": count_params(ens.members[1]) / 1e6}
        params["Ensemble"] = sum(params.values())

        comp = {k: classification_metrics(y, v, ml) for k, v in P.items()}
        main_m = classification_metrics(y, mcP, ml)              # MC-Dropout mean prediction (headline)
        r = {"n_test": len(test), "classes": classes, "headline_mc_ensemble": main_m,
             "comparison": comp, "latency_ms": lat, "params_M": params}

        if not ml:
            pred = mcP.argmax(1); correct = pred == y
            r["tiers"] = tier_table(conf, correct, cfg.conf_high, cfg.conf_low)
            r["risk_coverage"] = risk_coverage(conf, correct)
            r["mean_confidence"] = float(conf.mean())
            r["ece_confidence"] = ece_score(conf, correct)
            confusion_plot(main_m["confusion_matrix"], classes, out / f"{n}_confusion.png", n)
            roc_plot(y, mcP, classes, out / f"{n}_roc.png", n)
            confidence_hist(conf, correct, out / f"{n}_confidence.png", cfg.conf_high, cfg.conf_low, n)
            reliability_plot(conf, correct, out / f"{n}_reliability.png", title=n)

        # quality pass rates (threshold calibrated on train, so ~95% expected on in-distribution data)
        th = Thresholds(**ck["thresholds"][n])
        qs = [assess(test.raw(i), th) for i in np.random.RandomState(0).choice(len(test), min(500, len(test)), replace=False)]
        r["quality_pass_rate"] = {k: float(np.mean([q[k] for q in qs])) for k in ("blur_ok", "contrast_ok", "artifact_ok", "passed")}
        # accuracy on quality-passing vs failing images (does the quality gate flag harder images?)
        if not ml:
            idx = np.random.RandomState(0).choice(len(test), min(500, len(test)), replace=False)
            ok = np.array([q["passed"] for q in qs]); c = (mcP.argmax(1) == y)[idx]
            r["accuracy_quality_pass"] = float(c[ok].mean()) if ok.any() else None
            r["accuracy_quality_fail"] = float(c[~ok].mean()) if (~ok).any() else None

        # Grad-CAM gallery + quantitative faithfulness (deletion test)
        items, drops_cam, drops_rand = [], [], []
        pick = np.random.RandomState(1).choice(len(test), min(a.max_cam, len(test)), replace=False)
        for j, i in enumerate(pick):
            x, _ = test[int(i)]
            x = x.unsqueeze(0).to(device)
            cam, tgt = gradcam(ens, x, n)
            dc, dr = faithfulness(ens, x, n, cam, tgt)
            drops_cam.append(dc); drops_rand.append(dr)
            if j < 8:
                rgb = np.asarray(test.raw(int(i)).resize((cfg.image_size,) * 2))
                items.append((overlay(rgb, cam), f"pred {classes[tgt][:16]} | true {str(y[int(i)])[:16] if ml else classes[int(y[int(i)])][:16]}"))
        gallery(items, out / f"{n}_gradcam.png")
        r["gradcam_faithfulness"] = {"mean_prob_drop_top20pct_cam": float(np.mean(drops_cam)),
                                     "mean_prob_drop_random20pct": float(np.mean(drops_rand)),
                                     "cam_beats_random_fraction": float(np.mean(np.array(drops_cam) > np.array(drops_rand)))}
        results[n] = r

        report.append(f"\n## {n}  ({meta['modality']}; {len(test)} test images)\n")
        report.append("| Model | Params (M) | Acc | Sens(macro) | Spec(macro) | AUC | Latency (ms) |\n|---|---|---|---|---|---|---|")
        for k, m in comp.items():
            report.append(f"| {k} | {params[k]:.2f} | {m['accuracy']:.4f} | {m['sensitivity_macro']:.4f} | "
                          f"{m.get('specificity_macro', float('nan')):.4f} | {m['auc_macro']:.4f} | {lat[k]:.1f} |")
        if not ml:
            report.append(f"\nMC-Dropout ensemble: acc {main_m['accuracy']:.4f}, AUC {main_m['auc_macro']:.4f}, "
                          f"ECE {main_m['ece']:.4f}, mean confidence {r['mean_confidence']:.3f}")
            report.append("\n| Confidence tier | Fraction of cases | Accuracy |\n|---|---|---|")
            for t, v in r["tiers"].items():
                acc = "n/a" if v["accuracy"] is None else f"{v['accuracy']:.4f}"
                report.append(f"| {t} | {v['fraction']:.3f} | {acc} |")
        report.append(f"\nQuality pass rates: {json.dumps({k: round(v, 3) for k, v in r['quality_pass_rate'].items()})}")
        report.append(f"\nGrad-CAM deletion test: {json.dumps({k: round(v, 3) for k, v in r['gradcam_faithfulness'].items()})}")

    json.dump(results, open(out / "results.json", "w"), indent=2)
    (out / "report.md").write_text("\n".join(report))
    print(f"wrote {out/'report.md'}")


if __name__ == "__main__":
    main()
