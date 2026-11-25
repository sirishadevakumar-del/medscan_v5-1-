"""Viewers: slice views with overlays (2D) and interactive 3D volume rendering (Plotly).

Overlays are only drawn from (a) a user-supplied external segmentation mask or (b) the Grad-CAM attention region,
which is NOT a segmentation. Nothing is invented: without either, only the plain volume is shown.
"""
import numpy as np

MASK_RGB = (255, 60, 60)
ATT_THRESHOLD = 0.6


def blend(gray_u8, mask=None, heat=None, alpha=0.45, heat_thr=ATT_THRESHOLD):
    """gray_u8: HxW uint8. mask: HxW bool (external segmentation, red). heat: HxW 0..1 (Grad-CAM attention region, orange
    outline-free fill above heat_thr). Returns HxWx3 uint8."""
    rgb = np.repeat(gray_u8[..., None], 3, axis=2).astype(np.float32)
    if heat is not None:
        m = heat >= heat_thr
        col = np.array([255, 170, 0], np.float32)
        rgb[m] = (1 - alpha) * rgb[m] + alpha * col
    if mask is not None:
        m = mask > 0.5
        rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(MASK_RGB, np.float32)
    return rgb.clip(0, 255).astype(np.uint8)


def volume_figure(vol, mask=None, attention=None, max_dim=56, att_thr=ATT_THRESHOLD, show_mask=True, show_att=True):
    """Interactive 3D rendering (rotate / zoom / pan with the mouse). Works on a down-sampled copy for speed."""
    import plotly.graph_objects as go
    a, sp, f = vol.downsampled(max_dim)
    X, Y, Z = a.shape
    xs, ys, zs = (np.arange(n) * s for n, s in zip((X, Y, Z), sp))
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    v = np.clip((a - vol.lo) / (vol.hi - vol.lo), 0, 1)
    fig = go.Figure()
    fig.add_trace(go.Volume(x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=v.ravel(), isomin=0.15, isomax=1.0,
                            opacity=0.12, surface_count=14, colorscale="Gray", showscale=False, name="volume",
                            caps=dict(x_show=False, y_show=False, z_show=False)))
    if show_mask and mask is not None:
        m = mask[::f[0], ::f[1], ::f[2]].astype(np.float32)
        if m.max() > 0:
            fig.add_trace(go.Isosurface(x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=m.ravel(), isomin=0.5, isomax=1.0,
                                        surface_count=1, opacity=0.55, colorscale=[[0, "red"], [1, "red"]], showscale=False,
                                        caps=dict(x_show=False, y_show=False, z_show=False), name="external mask"))
    if show_att and attention is not None and attention.max() > att_thr:
        fig.add_trace(go.Isosurface(x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=attention.ravel(), isomin=att_thr,
                                    isomax=1.0, surface_count=1, opacity=0.4, colorscale=[[0, "orange"], [1, "orange"]],
                                    showscale=False, caps=dict(x_show=False, y_show=False, z_show=False),
                                    name="Grad-CAM attention"))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)"),
                      margin=dict(l=0, r=0, t=0, b=0), height=520, template="plotly_dark")
    return fig


def probability_comparison_figure(classes, prev, curr, title="Model probabilities: previous vs current"):
    import plotly.graph_objects as go
    fig = go.Figure([go.Bar(name="previous", x=classes, y=prev), go.Bar(name="current", x=classes, y=curr)])
    fig.update_layout(barmode="group", title=title, yaxis_title="probability", template="plotly_dark", height=340,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def area_curve_figure(prev_curve, curr_curve, title="External-mask area per axial slice"):
    import plotly.graph_objects as go
    fig = go.Figure()
    for name, c in (("previous", prev_curve), ("current", curr_curve)):
        if c is not None:
            fig.add_trace(go.Scatter(y=c, mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="axial slice index", yaxis_title="area (mm²)", template="plotly_dark",
                      height=340, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def slice_curve_figure(slices, classes, top):
    """Per-slice probability of the volume's top class and confidence across the analysed slices."""
    import plotly.graph_objects as go
    z = [s["z"] for s in slices]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=[s["probabilities"][top] for s in slices], mode="lines+markers",
                             name=f"P({classes[top]})"))
    fig.add_trace(go.Scatter(x=z, y=[s["confidence"] for s in slices], mode="lines+markers", name="confidence"))
    fig.update_layout(xaxis_title="axial slice index", yaxis_range=[0, 1], template="plotly_dark", height=320,
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig
