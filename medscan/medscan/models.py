"""DenseNet121 + EfficientNet-B0 ensemble with per-task heads (MC-Dropout classifier head as in the paper)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def make_head(in_f, n_out, hidden, p, after_bn=False):
    if after_bn:   # FC -> BN -> ReLU -> Dropout -> output (dropout no longer feeds BatchNorm statistics)
        return nn.Sequential(nn.Linear(in_f, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                             nn.Dropout(p), nn.Linear(hidden, n_out))
    # Dropout(p) -> FC(512) -> BN -> ReLU -> Dropout(p) -> output   (paper, Sec. III-C)
    return nn.Sequential(nn.Dropout(p), nn.Linear(in_f, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                         nn.Dropout(p), nn.Linear(hidden, n_out))


class Member(nn.Module):
    def __init__(self, arch, task_meta, cfg):
        super().__init__()
        self.arch = arch
        tv = torchvision.models
        try:
            if arch == "densenet121":
                m = tv.densenet121(weights=tv.DenseNet121_Weights.DEFAULT if cfg.pretrained else None)
            else:
                m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.DEFAULT if cfg.pretrained else None)
        except Exception as e:  # offline etc.
            print(f"[warn] could not load pretrained {arch} ({e}); using random init")
            m = tv.densenet121() if arch == "densenet121" else tv.efficientnet_b0()
        if arch == "densenet121":
            self.features, feat = m.features, m.classifier.in_features
        else:
            self.features, feat = m.features, m.classifier[1].in_features
        self.heads = nn.ModuleDict({k: make_head(feat, v["n"], cfg.hidden, cfg.dropout, getattr(cfg, "dropout_after_bn", False)) for k, v in task_meta.items()})

    def feature_map(self, x):
        f = self.features(x)
        return F.relu(f) if self.arch == "densenet121" else f   # out-of-place ReLU keeps Grad-CAM hooks valid

    @staticmethod
    def pool(fmap):
        return F.adaptive_avg_pool2d(fmap, 1).flatten(1)

    def head_forward(self, fmap, task):
        return self.heads[task](self.pool(fmap))

    def forward(self, x, task):
        return self.head_forward(self.feature_map(x), task)


class Ensemble(nn.Module):
    ARCHS = ("densenet121", "efficientnet_b0")

    def __init__(self, task_meta, cfg):
        super().__init__()
        self.task_meta = task_meta
        self.members = nn.ModuleList([Member(a, task_meta, cfg) for a in self.ARCHS])
        self.temperatures = {}

    def to_probs(self, logits, task):
        t = self.temperatures.get(task, 1.0)
        z = logits / t
        return torch.sigmoid(z) if self.task_meta[task]["multilabel"] else torch.softmax(z, -1)

    def member_logits(self, x, task):
        return [m(x, task) for m in self.members]

    def forward(self, x, task):
        """Ensemble probabilities (mean of member probabilities), dropout off, temperature applied."""
        return torch.stack([self.to_probs(z, task) for z in self.member_logits(x, task)]).mean(0)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def save_checkpoint(path, ens, cfg, task_meta, thresholds, modalities):
    torch.save({"state": ens.state_dict(), "task_meta": task_meta, "cfg": cfg.__dict__,
                "temperatures": ens.temperatures, "thresholds": thresholds, "modalities": modalities}, path)


def load_checkpoint(path, device="cpu"):
    from .config import Config
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    cfg.pretrained = False
    ens = Ensemble(ck["task_meta"], cfg)
    ens.load_state_dict(ck["state"])
    ens.temperatures = ck.get("temperatures", {})
    return ens.to(device).eval(), cfg, ck
