"""Grad-CAM (paper Eqs. 6-7) computed per ensemble member on its last feature map, then averaged."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def gradcam(ens, x, task, target=None):
    """x: [1,3,H,W]. Returns (cam[H,W] in 0..1 as np.ndarray, target_class)."""
    ens.eval()
    H, W = x.shape[-2:]
    cams = []
    with torch.enable_grad():
        if target is None:
            with torch.no_grad():
                target = int(ens(x, task).argmax(1))
        for m in ens.members:
            fmap = m.feature_map(x.clone().requires_grad_(False))
            fmap.retain_grad()
            score = m.head_forward(fmap, task)[0, target]
            m.zero_grad(set_to_none=True)
            score.backward()
            w = fmap.grad.mean(dim=(2, 3), keepdim=True)            # alpha_k^c = GAP of gradients
            cam = F.relu((w * fmap).sum(1, keepdim=True))           # L^c = ReLU(sum_k alpha_k A^k)
            cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)[0, 0].detach()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cams.append(cam)
    cam = torch.stack(cams).mean(0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy(), target


def overlay(rgb_uint8, cam, alpha=0.5):
    """rgb_uint8: HxWx3 uint8 (any size); cam in 0..1. Returns HxWx3 uint8 blended heatmap."""
    h, w = rgb_uint8.shape[:2]
    c = cv2.resize(cam.astype(np.float32), (w, h))
    heat = cv2.cvtColor(cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * rgb_uint8).astype(np.uint8)


@torch.no_grad()
def faithfulness(ens, x, task, cam, target, top_frac=0.2, seed=0):
    """Deletion test (no annotations needed): drop in target probability when the top-CAM pixels are
    removed vs. when the same number of random pixels are removed. Larger CAM drop = more faithful."""
    p0 = float(ens(x, task)[0, target])
    cam_t = torch.as_tensor(cam, device=x.device)
    k = int(cam_t.numel() * top_frac)
    thr = torch.topk(cam_t.flatten(), k).values.min()
    m_cam = (cam_t >= thr).float()
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(cam_t.numel(), generator=g)[:k]
    m_rand = torch.zeros(cam_t.numel()); m_rand[idx] = 1
    m_rand = m_rand.view_as(cam_t).to(x.device)
    p_cam = float(ens(x * (1 - m_cam), task)[0, target])
    p_rand = float(ens(x * (1 - m_rand), task)[0, target])
    return p0 - p_cam, p0 - p_rand
