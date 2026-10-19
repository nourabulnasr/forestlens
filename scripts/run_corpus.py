"""Run the pipeline over the corpus manifest. Writes data/results_<split>.json (committed, read by eval + chart).
split=dev  -> first 15 manifest figures, used to debug the pipeline
split=test -> the remaining held-out figures, used by the eval"""
import json, pathlib, sys, time, traceback
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from forestlens import core

import pathlib as _p
core.CACHE_DIR = ROOT / 'data' / 'ocr_cache'  # gitignored; makes re-runs after a pipeline change fast
DEV_N = 15
split = sys.argv[1] if len(sys.argv) > 1 else "test"
man = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
sel = man[:DEV_N] if split == "dev" else man[DEV_N:]
part = ROOT / "data" / f"results_{split}.partial.json"  # resumable: OCR is slow, so a run may be split into chunks
out = json.loads(part.read_text(encoding="utf-8")) if part.exists() else []
done = {e["id"] for e in out}
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9  # seconds before stopping cleanly for a later resume
t_start = time.time()
for i, m in enumerate(sel):
    if m["id"] in done:
        continue
    if time.time() - t_start > BUDGET:
        print("budget reached; resume by re-running the same command"); sys.exit(3)
    t = time.time()
    try:
        r = core.analyse_path(ROOT / "data" / "corpus" / m["file"])
        e = {"id": m["id"], "supported": r.supported, "why": r.why_unsupported,
             "axis": r.axis and {"kind": r.axis.kind, "ticks": r.axis.n_ticks, "inliers": r.axis.n_inliers},
             "rows": [core.row_dict(x) for x in r.rows]}
    except Exception as ex:  # a crash is a result too: record it, never skip silently
        e = {"id": m["id"], "supported": False, "why": f"crash: {ex!r}", "axis": None, "rows": []}
        traceback.print_exc()
    out.append(e)
    part.write_text(json.dumps(out), encoding="utf-8")
    n = len(e["rows"]); ok = sum(not x["flag"] for x in e["rows"])
    print(f"[{i+1}/{len(sel)}] {m['id']:<40} supported={e['supported']!s:<5} rows={n:<3} ok={ok:<3} {time.time()-t:.1f}s {e['why'][:60]}", flush=True)
(ROOT / "data" / f"results_{split}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
