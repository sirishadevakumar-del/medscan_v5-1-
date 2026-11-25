"""Monte-Carlo Dropout uncertainty + temperature scaling.

Dropout lives only in the classifier heads, so the backbone is run once and T stochastic head passes are
drawn from the pooled features. This is mathematically identical to T full forward passes, but much faster.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_MAX = 0.5  # max std of a [0,1]-bounded probability


def set_mc_mode(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()   # dropout on, BatchNorm stays in eval mode


@torch.no_grad()
def mc_predict(ens, x, task, T=20, mode="combined"):
    multilabel = ens.task_meta[task]["multilabel"]
    ens.eval()
    pooled = [m.pool(m.feature_map(x)) for m in ens.members]
    set_mc_mode(ens)
    samples = []
    for m, p in zip(ens.members, pooled):
        for _ in range(T):
            samples.append(ens.to_probs(m.heads[task](p), task))
    ens.eval()
    S = torch.stack(samples)                       # [members*T, B, C]
    mean, std = S.mean(0), S.std(0, unbiased=False)
    ent = -(mean.clamp_min(1e-8).log() * mean).sum(1) if not multilabel else None
    if multilabel:
        # Confidence of the call about the most suspicious finding: if its probability is >= 0.5 the call is
        # "present" (confidence p), otherwise the call is "no finding" (confidence 1-p); times MC stability.
        top = mean.argmax(1, keepdim=True)
        pt, st_ = mean.gather(1, top)[:, 0], std.gather(1, top)[:, 0]
        mc_conf = 1 - st_ / SIGMA_MAX
        p_hat = torch.where(pt >= 0.5, pt, 1 - pt)
        conf = p_hat * mc_conf if mode == "combined" else mc_conf
        pred = (mean >= 0.5).long()
    else:
        pred = mean.argmax(1)
        p_hat = mean.gather(1, pred[:, None])[:, 0]
        s_hat = std.gather(1, pred[:, None])[:, 0]
        mc_conf = 1 - s_hat / SIGMA_MAX                      # paper Eq. (5)
        conf = p_hat * mc_conf if mode == "combined" else mc_conf
    return {"probs": mean, "std": std, "pred": pred, "confidence": conf, "mc_confidence": mc_conf, "entropy": ent}


def fit_temperature(member_logits, y, multilabel):
    """member_logits: [M,N,C] tensor, y: labels. Returns scalar T minimising ensemble NLL (Guo et al. 2017)."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

    def nll():
        opt.zero_grad()
        z = member_logits / logT.exp().clamp(0.05, 10)
        if multilabel:
            p = torch.sigmoid(z).mean(0).clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(p, y.float())
        else:
            p = torch.softmax(z, -1).mean(0)
            loss = F.nll_loss(p.clamp_min(1e-8).log(), y)
        loss.backward()
        return loss

    opt.step(nll)
    return float(logT.exp().clamp(0.05, 10))
