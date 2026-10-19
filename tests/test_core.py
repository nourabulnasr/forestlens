import pytest
from forestlens import core
from forestlens.synthetic import make_forest


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clean_synthetic_recovers_printed_values(seed):
    img, truth = make_forest(seed=seed)
    res = core.analyse(img)
    assert res.supported and res.axis.kind == "log"
    assert len(res.rows) == len(truth)
    assert not any(r.flag for r in res.rows)
    for r in res.rows:
        assert max(r.err_est, r.err_lo, r.err_hi) < 0.03


def test_valid_but_wrong_printed_value_flagged():
    img, truth = make_forest(seed=0)
    e, lo, hi = truth[3]
    wrong = (round(e * 1.2, 2), lo, round(max(hi, e * 1.2 + 0.01), 2))
    img2, _ = make_forest(seed=0, printed_override={3: wrong})
    res = core.analyse(img2)
    flagged = [i for i, r in enumerate(res.rows) if r.flag]
    assert flagged == [3]


def test_parser_variants():
    mk = lambda t: [core.Box(t, 0, 0, 10, 10)]
    for t in ["0.85 [0.70, 1.03]", "0.85 (0.70-1.03)", "0.85 (0.70 to 1.03)", "0,85 [0,70; 1,03]"]:
        rows = core.parse_printed(mk(t))
        assert len(rows) == 1 and (rows[0].est, rows[0].lo, rows[0].hi) == (0.85, 0.70, 1.03), t
    assert core.parse_printed(mk("Total 12 345")) == []
    assert core.parse_printed(mk("2.00 [3.00, 4.00]")) == []  # est outside its own CI: rejected


def test_axis_fit_ignores_a_misread_tick():
    mk = lambda t, x: core.Box(t, x - 5, 0, x + 5, 10)
    import math
    ticks = [mk(t, 100 + 80 * math.log(float(t))) for t in ("0.25", "0.5", "1", "2", "4")]
    ticks.append(mk("7", 150))  # OCR junk
    ax = core.fit_axis(ticks)
    assert ax.kind == "log" and ax.n_inliers == 5
    assert ax.to_value(100 + 80 * math.log(3.0)) == pytest.approx(3.0, rel=1e-3)
