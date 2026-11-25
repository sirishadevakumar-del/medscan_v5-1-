"""Train the multi-modality ensemble.  python train.py --out runs/exp1"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from medscan.config import Config, load_tasks
from medscan.data import build_task, make_loader
from medscan.metrics import classification_metrics
from medscan.models import Ensemble, save_checkpoint
from medscan.plots import training_curves
from medscan.quality import calibrate
from medscan.uncertainty import fit_temperature


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def freeze_backbone_bn(ens):
    for m in ens.members:
        for mod in m.features.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.eval()


def task_loss(logits, y, multilabel, weight):
    logits = logits.float()
    if multilabel:
        return F.binary_cross_entropy_with_logits(logits, y)
    return F.cross_entropy(logits, y, weight=weight)


@torch.no_grad()
def collect_logits(ens, loader, task, device, max_batches=None):
    ens.eval()
    zs, ys = [[] for _ in ens.members], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device)
        for k, z in enumerate(ens.member_logits(x, task)):
            zs[k].append(z.float().cpu())
        ys.append(y)
    return torch.stack([torch.cat(z) for z in zs]), torch.cat(ys)       # [M,N,C], [N]


def validate(ens, loaders, device, max_batches=None):
    aucs, accs, per = [], [], {}
    for task, ld in loaders.items():
        z, y = collect_logits(ens, ld, task, device, max_batches)
        probs = torch.stack([ens.to_probs(m, task) for m in z]).mean(0).numpy()
        m = classification_metrics(y.numpy(), probs, ens.task_meta[task]["multilabel"])
        per[task] = m
        aucs.append(m["auc_macro"]); accs.append(m["accuracy"])
    auc = float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float(np.mean(accs))
    return auc, float(np.mean(accs)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--out"); ap.add_argument("--tasks")
    ap.add_argument("--epochs", type=int); ap.add_argument("--batch-size", type=int)
    ap.add_argument("--image-size", type=int); ap.add_argument("--lr", type=float)
    ap.add_argument("--num-workers", type=int); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--max-batches", type=int, help="debug: cap batches per task per epoch")
    ap.add_argument("--freeze-bn", action="store_true", help="keep backbone BatchNorm in eval mode while training")
    ap.add_argument("--dropout-after-bn", action="store_true", help="head: Linear-BN-ReLU-Dropout-Linear")
    a = ap.parse_args()

    cfg = Config.load(a.config) if a.config else Config()
    for k, v in {"out_dir": a.out, "tasks_file": a.tasks, "epochs": a.epochs, "batch_size": a.batch_size,
                 "image_size": a.image_size, "lr": a.lr, "num_workers": a.num_workers}.items():
        if v is not None:
            setattr(cfg, k, v)
    if a.no_pretrained:
        cfg.pretrained = False
    if a.freeze_bn:
        cfg.freeze_backbone_bn = True
    if a.dropout_after_bn:
        cfg.dropout_after_bn = True
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = cfg.amp and device == "cuda"
    print(f"device={device} amp={amp}")

    specs = load_tasks(cfg.tasks_file)
    bundles = {s["name"]: build_task(s, cfg) for s in specs}
    modalities = sorted({s["modality"] for s in specs})
    mod_id = {m: i for i, m in enumerate(modalities)}
    task_meta = {n: {"n": len(b["classes"]), "multilabel": b["multilabel"], "classes": b["classes"],
                     "modality": b["spec"]["modality"]} for n, b in bundles.items()}
    task_meta["_modality"] = {"n": len(modalities), "multilabel": False, "classes": modalities}
    json.dump(task_meta, open(out / "task_meta.json", "w"), indent=2)

    # per-task quality thresholds calibrated on a sample of training images
    thresholds = {}
    for n, b in bundles.items():
        rng = np.random.RandomState(0)
        idx = rng.choice(len(b["train"]), min(300, len(b["train"])), replace=False)
        thresholds[n] = calibrate([b["train"].raw(int(i)) for i in idx]).to_dict()
    json.dump(thresholds, open(out / "quality_thresholds.json", "w"), indent=2)

    train_ld = {n: make_loader(b["train"], cfg.batch_size, True, cfg.num_workers) for n, b in bundles.items()}
    val_ld = {n: make_loader(b["val"], cfg.batch_size * 2, False, cfg.num_workers) for n, b in bundles.items()}

    weights = {}
    for n, b in bundles.items():
        if cfg.class_weighted and not b["multilabel"]:
            c = np.bincount(b["train"].targets, minlength=len(b["classes"])).astype(float)
            w = np.clip(c.sum() / (len(c) * np.maximum(c, 1)), 0.2, 10.0)
            weights[n] = torch.tensor(w, dtype=torch.float32, device=device)
        else:
            weights[n] = None

    ens = Ensemble(task_meta, cfg).to(device)
    opt = torch.optim.Adam(ens.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, stale, history = -1.0, 0, []
    for epoch in range(1, cfg.epochs + 1):
        ens.train()
        if cfg.freeze_backbone_bn:
            freeze_backbone_bn(ens)
        t0 = time.time()
        plan = [n for n, ld in train_ld.items() for _ in range(min(len(ld), a.max_batches or len(ld)))]
        random.shuffle(plan)
        its = {n: iter(ld) for n, ld in train_ld.items()}
        run, cnt = 0.0, 0
        for n in plan:
            x, y = next(its[n])
            x, y = x.to(device), y.to(device)
            mid = torch.full((x.size(0),), mod_id[bundles[n]["spec"]["modality"]], device=device)
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = 0.0
                for m in ens.members:
                    pooled = m.pool(m.feature_map(x))
                    l = task_loss(m.heads[n](pooled), y, bundles[n]["multilabel"], weights[n])
                    l = l + cfg.modality_loss_weight * F.cross_entropy(m.heads["_modality"](pooled).float(), mid)
                    loss = loss + l / len(ens.members)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ens.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            run += float(loss.detach()); cnt += 1
        auc, acc, per = validate(ens, val_ld, device, a.max_batches)
        sched.step(auc)
        h = {"epoch": epoch, "train_loss": run / max(cnt, 1), "val_auc": auc, "val_acc": acc,
             "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0}
        history.append(h)
        print(f"epoch {epoch:3d} loss {h['train_loss']:.4f} val_auc {auc:.4f} val_acc {acc:.4f} "
              f"lr {h['lr']:.1e} ({h['sec']:.0f}s)  " + " ".join(f"{k}:{v['accuracy']:.2f}" for k, v in per.items()))
        if auc > best:
            best, stale = auc, 0
            torch.save(ens.state_dict(), out / "_best_state.pt")
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                print("early stopping"); break

    ens.load_state_dict(torch.load(out / "_best_state.pt", map_location=device))
    # temperature scaling on the validation set (confidence calibration)
    for n in bundles:
        z, y = collect_logits(ens, val_ld[n], n, "cpu" if False else device, a.max_batches)
        ens.temperatures[n] = fit_temperature(z, y, bundles[n]["multilabel"])
    print("temperatures:", {k: round(v, 3) for k, v in ens.temperatures.items()})
    save_checkpoint(out / "best.pt", ens, cfg, task_meta, thresholds, modalities)
    (out / "_best_state.pt").unlink(missing_ok=True)
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    training_curves(history, out / "training_curves.png")
    print(f"saved {out/'best.pt'}")


if __name__ == "__main__":
    main()
