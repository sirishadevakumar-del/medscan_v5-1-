import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def confusion_plot(cm, classes, path, title="Confusion matrix"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(classes) + 3),) * 2)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), [c[:14] for c in classes], rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), [c[:14] for c in classes])
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def roc_plot(y, probs, classes, path, title="ROC"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for k in range(min(probs.shape[1], 10)):
        pos = (y == k)
        if pos.any() and (~pos).any():
            fpr, tpr, _ = roc_curve(pos, probs[:, k])
            ax.plot(fpr, tpr, label=classes[k][:18])
    ax.plot([0, 1], [0, 1], "r--", label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title(title)
    ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def confidence_hist(conf, correct, path, hi=0.85, lo=0.70, title="Confidence distribution"):
    conf, correct = np.asarray(conf), np.asarray(correct).astype(bool)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bins = np.linspace(0, 1, 26)
    ax.hist(conf[correct], bins, alpha=0.7, label="correct", color="tab:green")
    ax.hist(conf[~correct], bins, alpha=0.7, label="incorrect", color="tab:red")
    ax.axvline(hi, ls="--", c="k"); ax.axvline(lo, ls=":", c="k")
    ax.set_xlabel("confidence"); ax.set_ylabel("count"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def reliability_plot(conf, correct, path, bins=10, title="Reliability diagram"):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--"); ax.plot(xs, ys, "o-")
    ax.set_xlabel("mean confidence"); ax.set_ylabel("accuracy"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def training_curves(history, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ep = [h["epoch"] for h in history]
    axs[0].plot(ep, [h["train_loss"] for h in history]); axs[0].set_title("Training loss"); axs[0].set_xlabel("epoch")
    axs[1].plot(ep, [h["val_auc"] for h in history], label="val macro-AUC")
    axs[1].plot(ep, [h["val_acc"] for h in history], label="val accuracy"); axs[1].legend()
    axs[1].set_title("Validation"); axs[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def gallery(items, path, cols=4):
    """items: list of (overlay_rgb, caption)."""
    rows = int(np.ceil(len(items) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (img, cap) in zip(axs.ravel(), items):
        ax.imshow(img); ax.set_title(cap, fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
