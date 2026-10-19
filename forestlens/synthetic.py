"""Matplotlib forest plots with known ground truth. Used for unit tests only (never for the headline eval)."""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def make_forest(n=8, seed=0, log=True, dpi=110, printed_override=None):
    """Returns (RGB array, truth list of (est, lo, hi))."""
    rng = np.random.default_rng(seed)
    est = np.exp(rng.normal(0.0, 0.35, n))
    se = rng.uniform(0.08, 0.35, n)
    lo, hi = est * np.exp(-1.96 * se), est * np.exp(1.96 * se)
    truth = [(round(e, 2), round(l, 2), round(h, 2)) for e, l, h in zip(est, lo, hi)]
    fig = plt.figure(figsize=(9, 0.5 * n + 1.6), dpi=dpi)
    ax = fig.add_axes([0.35, 0.14, 0.38, 0.8])
    y = np.arange(n)[::-1]
    for (e, l, h), yy in zip(truth, y):
        ax.plot([l, h], [yy, yy], color="k", lw=1.2)
        ax.plot([e], [yy], "s", color="k", ms=7)
    ax.axvline(1.0, color="k", lw=0.8)
    if log:
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        ax.set_xlim(0.2, 5)
    ax.set_yticks([]); ax.minorticks_off()
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    for i, yy in enumerate(y):
        fig.text(0.02, 0.14 + 0.8 * (yy + 0.5) / n, f"Study {chr(65 + i)} 20{10 + i}", va="center", fontsize=9)
        e, l, h = (printed_override or {}).get(i, truth[i])
        fig.text(0.78, 0.14 + 0.8 * (yy + 0.5) / n, f"{e:.2f} [{l:.2f}, {h:.2f}]", va="center", fontsize=9)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")), truth
