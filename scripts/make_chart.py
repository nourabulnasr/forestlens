"""Render docs/results.png from eval_results.json (every run, in order) + data/results_test.json. No numbers typed by hand."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
hist = json.loads((ROOT / "eval_results.json").read_text(encoding="utf-8"))
last = hist[-1]
INK, MUTE, ACC, GOOD = "#222222", "#9a9a9a", "#c8553d", "#3d7a5a"

fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.1, 1.5, 1.2]})
fig.patch.set_facecolor("white")

# (1) claim A across runs
ax = axs[0]
labels = [h["run"].split("_")[0] for h in hist]
vals = [h["A_row_agreement"] for h in hist]
ax.bar(labels, vals, color=[GOOD if h["pass_A"] else ACC for h in hist], width=0.55)
ax.axhline(last["thresholds"]["A_row_agreement_min"], color=INK, ls="--", lw=1)
ax.text(len(hist) - 0.5, last["thresholds"]["A_row_agreement_min"] + 0.02, "threshold", ha="right", fontsize=8, color=INK)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, color=INK)
ax.set_ylim(0, 1.1); ax.set_title(f"A. Geometry vs printed text\nrows agreeing within {last['thresholds']['tol']:.0%} (n={last['rows_supported']})", fontsize=10, loc="left")

# (2) claim B by corruption type
ax = axs[1]
types = list(last["B_by_type"])
rec = [last["B_by_type"][t]["recall"] for t in types]
cols = [GOOD if t != "digit_transpose" else MUTE for t in types]
ax.barh(types[::-1], rec[::-1], color=cols[::-1], height=0.6)
ax.axvline(last["thresholds"]["B_large_corruption_recall_min"], color=INK, ls="--", lw=1)
for i, v in enumerate(rec[::-1]):
    ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9, color=INK)
ax.set_xlim(0, 1.15)
ax.set_title("B. Seeded printed-text corruption caught\n(grey = small typo, not part of the claim)", fontsize=10, loc="left")

# (3) coverage
ax = axs[2]
n_sup, n_all = last["figures_supported"], last["figures_total"]
ax.bar(["supported", "unsupported"], [n_sup, n_all - n_sup], color=[GOOD, MUTE], width=0.55)
for i, v in enumerate([n_sup, n_all - n_sup]):
    ax.text(i, v + 1, str(v), ha="center", fontsize=10, color=INK)
ax.set_title(f"Coverage: {n_sup}/{n_all} held-out figures\nwere calibratable", fontsize=10, loc="left")

for a in axs:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.tick_params(labelsize=8)
fig.tight_layout()
(ROOT / "docs").mkdir(exist_ok=True)
fig.savefig(ROOT / "docs" / "results.png", dpi=130)
print("wrote docs/results.png")
