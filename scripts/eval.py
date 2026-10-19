"""EVAL GATE for ForestLens.

CLAIM_UNDER_TEST (stated exactly as the README states it):
  (A) On real CC-BY forest-plot figures with a printed 'est [lo, hi]' column, the pixel-geometry reading
      agrees with the printed text within 5% on est, lo AND hi for at least 60% of rows
      (rows from figures the pipeline can calibrate; unsupported-figure coverage is reported separately).
  (B) When the printed text of a row that agreed is corrupted with a realistic error of >=10% (decimal
      slip, +/-10% estimate, upper bound +25%, swapped bounds), the cross-check flags it for >=90% of such corruptions.
Thresholds below were fixed BEFORE the first run on the held-out split and are not edited afterwards.

Reads data/results_test.json (held-out figures, produced by scripts/run_corpus.py test).
Appends one record per invocation to eval_results.json (never overwrites; first record is ground truth).
"""
import copy, datetime, json, pathlib, random, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from forestlens import core

THRESH = {"tol": 0.05, "A_row_agreement_min": 0.60, "B_large_corruption_recall_min": 0.90}
SEED = 20261019
tag = sys.argv[1] if len(sys.argv) > 1 else "run"
note = sys.argv[2] if len(sys.argv) > 2 else ""

res = json.loads((ROOT / "data" / "results_test.json").read_text(encoding="utf-8"))
sup = [f for f in res if f["supported"]]
rows = [r for f in sup for r in f["rows"]]
n_ok = sum(not r["flag"] for r in rows)
A = n_ok / len(rows) if rows else 0.0
fig_rates = [sum(not r["flag"] for r in f["rows"]) / len(f["rows"]) for f in sup if f["rows"]]
fail_reasons = {}
for r in rows:
    if r["flag"]:
        fail_reasons[r["reason"].split(":")[0] if r["reason"].startswith("no") else r["reason"]] = \
            fail_reasons.get(r["reason"].split(":")[0] if r["reason"].startswith("no") else r["reason"], 0) + 1


def as_row(d):
    r = core.RowResult(d["raw"], d["p_est"], d["p_lo"], d["p_hi"], d["g_est"], d["g_lo"], d["g_hi"])
    return r


def corrupt(kind, e, lo, hi, rng):
    if kind == "decimal_slip":
        return (e * 10, lo, hi) if rng.random() < 0.5 else (e / 10, lo / 10, hi / 10)
    if kind == "est_plus10":
        return (e * 1.10, lo, hi)
    if kind == "est_minus10":
        return (e * 0.90, lo, hi)
    if kind == "hi_plus25":
        return (e, lo, hi * 1.25)
    if kind == "swap_bounds":
        return (e, hi, lo)
    if kind == "digit_transpose":  # e.g. 1.23 -> 1.32: a small, realistic typo (reported separately)
        s = f"{e:.2f}"
        d = list(s.replace(".", ""))
        for i in range(len(d) - 1):
            if d[i] != d[i + 1]:
                d[i], d[i + 1] = d[i + 1], d[i]
                break
        t = float("".join(d[:1]) + "." + "".join(d[1:]))
        return (t, lo, hi)
    raise ValueError(kind)


rng = random.Random(SEED)
agree = [r for r in rows if not r["flag"]]
kinds = ["decimal_slip", "est_plus10", "est_minus10", "hi_plus25", "swap_bounds", "digit_transpose"]
corr = {}
for k in kinds:
    hit = tot = 0
    for d in agree:
        e, lo, hi = corrupt(k, d["p_est"], d["p_lo"], d["p_hi"], rng)
        r = as_row(d)
        r.p_est, r.p_lo, r.p_hi = e, lo, hi
        r = core.compare(r, THRESH["tol"])
        tot += 1; hit += r.flag
    corr[k] = {"n": tot, "flagged": hit, "recall": hit / tot if tot else None}
large = [k for k in kinds if k != "digit_transpose"]
B_n = sum(corr[k]["n"] for k in large); B_hit = sum(corr[k]["flagged"] for k in large)
B = B_hit / B_n if B_n else 0.0
clean_flag_rate = 1 - A
prev = 0.10  # precision if 10% of printed rows carried one of the large corruptions
prec = (B * prev) / (B * prev + clean_flag_rate * (1 - prev)) if (B * prev + clean_flag_rate * (1 - prev)) else None

record = {
    "run": tag, "note": note, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "thresholds": THRESH, "seed": SEED,
    "figures_total": len(res), "figures_supported": len(sup),
    "coverage_figures": len(sup) / len(res) if res else 0.0,
    "unsupported_reasons": {},
    "rows_supported": len(rows), "rows_agree": n_ok,
    "A_row_agreement": A, "A_figure_macro_mean": sum(fig_rates) / len(fig_rates) if fig_rates else 0.0,
    "flagged_row_reasons": fail_reasons,
    "B_large_corruption_recall": B, "B_by_type": corr,
    "clean_row_flag_rate": clean_flag_rate, "precision_at_10pct_corruption_prevalence": prec,
    "pass_A": A >= THRESH["A_row_agreement_min"], "pass_B": B >= THRESH["B_large_corruption_recall_min"],
}
for f in res:
    if not f["supported"]:
        key = f["why"].split(":")[0]
        record["unsupported_reasons"][key] = record["unsupported_reasons"].get(key, 0) + 1

path = ROOT / "eval_results.json"
hist = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
hist.append(record)  # append-only: the first record is ground truth
path.write_text(json.dumps(hist, indent=1), encoding="utf-8")
print(json.dumps(record, indent=1))
