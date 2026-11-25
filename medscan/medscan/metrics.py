import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def ece_score(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def _ovr_auc(y, probs):
    aucs = []
    for k in range(probs.shape[1]):
        pos = (y == k)
        if pos.any() and (~pos).any():
            aucs.append(roc_auc_score(pos, probs[:, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


def classification_metrics(y, probs, multilabel=False):
    y, probs = np.asarray(y), np.asarray(probs)
    if multilabel:
        pred = (probs >= 0.5).astype(int)
        aucs = [roc_auc_score(y[:, k], probs[:, k]) for k in range(y.shape[1]) if len(np.unique(y[:, k])) == 2]
        return {"auc_macro": float(np.mean(aucs)) if aucs else float("nan"),
                "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
                "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
                "sensitivity_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
                "exact_match": float((pred == y).all(1).mean()),
                "accuracy": float((pred == y).mean())}
    C = probs.shape[1]
    pred = probs.argmax(1)
    cm = np.zeros((C, C), dtype=int)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    present = cm.sum(1) > 0
    sens = tp / np.maximum(tp + fn, 1)
    spec = tn / np.maximum(tn + fp, 1)
    prec = tp / np.maximum(tp + fp, 1)
    f1 = 2 * prec * sens / np.maximum(prec + sens, 1e-12)
    conf = probs.max(1)
    return {"accuracy": float((pred == y).mean()),
            "sensitivity_macro": float(sens[present].mean()),     # = balanced accuracy
            "specificity_macro": float(spec[present].mean()),
            "precision_macro": float(prec[present].mean()),
            "f1_macro": float(f1[present].mean()),
            "auc_macro": _ovr_auc(y, probs),
            "ece": ece_score(conf, pred == y),
            "per_class_sensitivity": sens.tolist(),
            "confusion_matrix": cm.tolist()}


def tier_table(conf, correct, hi=0.85, lo=0.70):
    conf, correct = np.asarray(conf), np.asarray(correct)
    out = {}
    for name, m in {"high": conf > hi, "medium": (conf >= lo) & (conf <= hi), "low": conf < lo}.items():
        out[name] = {"fraction": float(m.mean()), "accuracy": float(correct[m].mean()) if m.any() else None}
    return out


def risk_coverage(conf, correct, points=10):
    order = np.argsort(-np.asarray(conf))
    c = np.asarray(correct)[order]
    n = len(c)
    res = []
    for f in np.linspace(0.1, 1.0, points):
        k = max(1, int(n * f))
        res.append({"coverage": float(k / n), "accuracy": float(c[:k].mean())})
    return res
