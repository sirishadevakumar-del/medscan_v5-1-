import json
from dataclasses import dataclass, asdict, fields


@dataclass
class Config:
    # data
    tasks_file: str = "configs/tasks.json"
    data_root: str = "data"
    medmnist_size: int = 128          # 28 / 64 / 128 / 224; images are resized to image_size anyway
    image_size: int = 224
    num_workers: int = 4
    # model
    pretrained: bool = True           # ImageNet weights (downloads on first use)
    hidden: int = 512
    dropout: float = 0.5
    freeze_backbone_bn: bool = False    # keep backbone BatchNorm in eval mode (stable shared stats across tasks)
    dropout_after_bn: bool = False      # head = Linear-BN-ReLU-Dropout-Linear (avoids dropout/BN variance shift)
    # optimisation (paper: Adam, halve LR after 5 stale epochs, stop after 15, clip 1.0)
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4                  # paper used 1e-3; 1e-4 is safer for fine-tuning pretrained nets
    weight_decay: float = 1e-5
    lr_patience: int = 5
    lr_factor: float = 0.5
    early_stop_patience: int = 15
    grad_clip: float = 1.0
    class_weighted: bool = True
    modality_loss_weight: float = 0.2   # auxiliary modality-routing head
    amp: bool = True
    seed: int = 42
    out_dir: str = "runs/default"
    # uncertainty / decision policy (paper: >0.85 high, 0.70-0.85 medium, <0.70 expert review)
    mc_samples: int = 20
    conf_high: float = 0.85
    conf_low: float = 0.70
    confidence_mode: str = "combined"   # "combined" = p(pred) * (1 - std/0.5); "paper" = 1 - std/0.5

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        names = {x.name for x in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def load_tasks(path):
    with open(path) as f:
        return [t for t in json.load(f) if t.get("enabled", True)]
