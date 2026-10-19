"""CLI: python -m forestlens figure.png [--csv out.csv] [--overlay out.png]"""
import argparse, csv, sys
import cv2
from . import core


def main(argv=None):
    ap = argparse.ArgumentParser(prog="forestlens", description="Digitise a forest plot and cross-check it against its printed text column.")
    ap.add_argument("image")
    ap.add_argument("--csv", help="write per-row results to this CSV")
    ap.add_argument("--overlay", help="write an annotated PNG (green = agrees with printed text, red = flagged)")
    a = ap.parse_args(argv)
    img = core.load_image(a.image)
    res = core.analyse(img)
    if not res.supported:
        print(f"UNSUPPORTED: {res.why_unsupported}")
        return 2
    ax = res.axis
    print(f"axis: {ax.kind}, {ax.n_inliers}/{ax.n_ticks} ticks used")
    print(f"{'printed':<26}{'geometry (est [lo, hi])':<28}{'max err':>8}  verdict")
    for r in res.rows:
        g = "-" if r.g_est is None else f"{r.g_est:.2f} [{r.g_lo:.2f}, {r.g_hi:.2f}]"
        errs = [e for e in (r.err_est, r.err_lo, r.err_hi) if e is not None]
        me = f"{100 * max(errs):.1f}%" if errs else "-"
        print(f"{r.raw:<26}{g:<28}{me:>8}  {'FLAG ' + r.reason if r.flag else 'ok'}")
    n_flag = sum(r.flag for r in res.rows)
    print(f"{len(res.rows)} rows, {n_flag} flagged (tolerance {100 * core.TOL:.0f}%)")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            rows = [core.row_dict(r) for r in res.rows]
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if a.overlay:
        cv2.imwrite(a.overlay, cv2.cvtColor(core.draw_overlay(img, res), cv2.COLOR_RGB2BGR))
    return 1 if n_flag else 0


if __name__ == "__main__":
    sys.exit(main())
