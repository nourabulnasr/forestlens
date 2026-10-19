"""ForestLens core: read a forest plot twice (pixel geometry vs printed "est [lo, hi]" text) and compare.

Pipeline
  1. OCR (RapidOCR, ONNX, CPU) -> text boxes
  2. Printed-text parser: regex for "0.85 [0.70, 1.03]" style cells -> one printed row per text line
  3. Axis calibration: numeric tick labels under the plot -> RANSAC fit of pixel-x vs value
     (log-linear for ratio axes, linear otherwise; the better-fitting model wins)
  4. Classical CV geometry per row: longest horizontal run = CI whisker (or diamond width),
     marker = blob that survives a 4x4 morphological opening, estimate = its widest column
  5. Cross-check: geometry-derived value vs printed value; any |rel. diff| > TOL flags the row
"""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
from PIL import Image

TOL = 0.05  # relative disagreement above which a row is flagged (fixed before the full-corpus eval)

NUM = r"-?\d{1,3}(?:[.,]\d{1,3})?"
SEP = r"(?:\s*[,;]\s*|\s*(?:to|TO)\s*|\s*[-–—−]\s*|\s+)"
ROW_RE = re.compile(rf"({NUM})\s*[\[\(\{{]\s*({NUM}){SEP}({NUM})\s*[\]\)\}}]")
TICK_RE = re.compile(r"^-?\d{1,3}(?:[.,]\d{1,2})?$")


# ----------------------------------------------------------------------------- OCR
@dataclass
class Box:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self): return (self.x0 + self.x1) / 2

    @property
    def cy(self): return (self.y0 + self.y1) / 2

    @property
    def h(self): return self.y1 - self.y0


_ENGINE = None


def load_image(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


CACHE_DIR = None  # set to a pathlib.Path to memoise OCR on disk (keyed by image content); off by default


def ocr(img: np.ndarray) -> list[Box]:
    global _ENGINE
    key = None
    if CACHE_DIR is not None:
        import hashlib, json
        key = CACHE_DIR / (hashlib.md5(img.tobytes() + str(img.shape).encode()).hexdigest() + ".json")
        if key.exists():
            return [Box(*b) for b in json.loads(key.read_text(encoding="utf-8"))]
    out = _ocr_uncached(img)
    if key is not None:
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_text(json.dumps([[b.text, b.x0, b.y0, b.x1, b.y1] for b in out]), encoding="utf-8")
    return out


def _ocr_uncached(img: np.ndarray) -> list[Box]:
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        # cap threads: default (-1 = all cores) thrashes badly when the machine is busy
        _ENGINE = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=1)
    res, _ = _ENGINE(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    boxes = []
    for pts, text, _score in res or []:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        boxes.append(Box(text, min(xs), min(ys), max(xs), max(ys)))
    return boxes


def ocr_strip(img: np.ndarray, y_from: float, scale: float = 2.5) -> list[Box]:
    """Second OCR pass on the strip under the last data row, upscaled: the default pass drops
    short tick labels such as '1', '2', '4'. Boxes are mapped back to full-image coordinates."""
    y0 = max(0, int(y_from))
    strip = img[y0:]
    if strip.shape[0] < 8:
        return []
    big = cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return [Box(b.text, (b.x0 - 20) / scale, (b.y0 - 20) / scale + y0, (b.x1 - 20) / scale, (b.y1 - 20) / scale + y0)
            for b in ocr(big)]


def _num(s: str) -> float:
    return float(s.replace(",", "."))


# ----------------------------------------------------------------------------- printed text
@dataclass
class PrintedRow:
    est: float
    lo: float
    hi: float
    cy: float
    x0: float
    x1: float
    raw: str
    ulp: tuple = (0.0, 0.0, 0.0)  # half a unit in the last printed decimal, per value


def group_lines(boxes: list[Box]) -> list[list[Box]]:
    lines: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        for ln in lines:
            if abs(np.mean([x.cy for x in ln]) - b.cy) < max(5.0, 0.55 * b.h):
                ln.append(b)
                break
        else:
            lines.append([b])
    return [sorted(ln, key=lambda b: b.cx) for ln in lines]


def parse_printed(boxes: list[Box]) -> list[PrintedRow]:
    rows = []
    for ln in group_lines(boxes):
        s, spans = "", []
        for b in ln:  # concatenate the boxes of one text line, remembering char offsets
            spans.append((len(s), len(s) + len(b.text), b))
            s += b.text + " "
        for m in ROW_RE.finditer(s):
            est, lo, hi = (_num(m.group(i)) for i in (1, 2, 3))
            if not (0 < lo <= est <= hi):  # ratio-scale rows only; also rejects OCR junk
                continue
            hit = [b for a, z, b in spans if a < m.end() and z > m.start()]
            ulp = tuple(0.5 * 10 ** -(len(re.split(r"[.,]", m.group(i))[1]) if re.search(r"[.,]", m.group(i)) else 0)
                        for i in (1, 2, 3))
            rows.append(PrintedRow(est, lo, hi, float(np.mean([b.cy for b in hit])),
                                   min(b.x0 for b in hit), max(b.x1 for b in hit), m.group(0), ulp))
    return rows


# ----------------------------------------------------------------------------- calibration
@dataclass
class Axis:
    kind: str  # "log" or "linear"
    a: float   # x = a + b * f(v)
    b: float
    n_ticks: int
    n_inliers: int
    ticks: list = field(default_factory=list)  # (x_pixel, value) inliers

    def to_value(self, x: float) -> float:
        u = (x - self.a) / self.b
        return math.exp(u) if self.kind == "log" else u

    def to_x(self, v: float) -> float:
        return self.a + self.b * (math.log(v) if self.kind == "log" else v)


def find_tick_lines(boxes: list[Box], below_y: float) -> list[list[Box]]:
    """Candidate tick rows: lines of >=3 numeric-only labels below the last data row, nearest first.
    Numeric table rows ('Total events 716 557') qualify too, so the caller keeps the first that
    actually fits a consistent axis."""
    cands = [b for b in boxes if TICK_RE.match(b.text.strip())]
    out = []
    for ln in group_lines(cands):
        uniq = []
        for b in ln:  # two OCR passes can read the same label twice
            if not any(b.text == u.text and abs(b.cx - u.cx) < 6 for u in uniq):
                uniq.append(b)
        dy = np.mean([b.cy for b in uniq]) - below_y
        if len(uniq) >= 3 and dy > -2:
            out.append((dy, uniq))
    return [ln for _, ln in sorted(out, key=lambda t: t[0])]


def snap_ticks(mask: np.ndarray, ticks: list[Box]) -> list[Box]:
    """OCR box centres are off by several px on small fonts. Find the axis line above the tick labels
    and snap each label to the nearest drawn tick mark (a short vertical stroke crossing the line)."""
    if len(ticks) < 3:
        return ticks
    xs = sorted(t.cx for t in ticks)
    span = xs[-1] - xs[0]
    top = int(np.mean([t.y0 for t in ticks]))
    H, W = mask.shape
    y_ax = None
    for y in range(min(H - 1, top + 2), max(0, top - 40), -1):
        a, b = int(max(0, xs[0] - 10)), int(min(W, xs[-1] + 10))
        runs = _runs(mask[y, a:b], gap=1)
        if runs and max(e - s for s, e in runs) >= 0.7 * span:
            y_ax = y
            break
    if y_ax is None:
        return ticks
    marks = []
    for x in range(int(max(0, xs[0] - 25)), int(min(W, xs[-1] + 25))):
        below = mask[y_ax + 2:y_ax + 5, x].all() if y_ax + 5 < H else False
        above = mask[y_ax - 4:y_ax - 1, x].all() if y_ax - 4 >= 0 else False
        if below or above:
            marks.append(x)
    groups, cur = [], []
    for x in marks:  # merge adjacent columns of one tick stroke
        if cur and x - cur[-1] > 1:
            groups.append(cur); cur = []
        cur.append(x)
    if cur:
        groups.append(cur)
    centres = [float(np.mean(g)) for g in groups if len(g) <= 4]
    if len(centres) < 3:
        return ticks
    gap = np.median(np.diff(xs)) if len(xs) > 1 else 30
    out = []
    for t in ticks:
        c = min(centres, key=lambda c: abs(c - t.cx))
        near = abs(c - t.cx) <= min(15.0, 0.45 * gap)
        out.append(Box(t.text, c, t.y0, c, t.y1) if near else t)
    return out


def fit_axis(ticks: list[Box], px_tol: float = 4.0) -> Axis | None:
    pts = []
    for t in ticks:  # the two OCR passes can both read the same label: keep one per (value, ~position)
        v = _num(t.text.strip())
        if not any(v == pv and abs(t.cx - px) < 6 for px, pv in pts):
            pts.append((t.cx, v))
    best = None
    for kind in ("log", "linear"):
        usable = [(x, v) for x, v in pts if (v > 0 or kind == "linear")]
        f = (lambda v: math.log(v)) if kind == "log" else (lambda v: v)
        for (x1, v1), (x2, v2) in itertools.combinations(usable, 2):
            if f(v1) == f(v2):
                continue
            b = (x2 - x1) / (f(v2) - f(v1))
            a = x1 - b * f(v1)
            if b <= 0:
                continue
            inl = [(x, v) for x, v in usable if abs(a + b * f(v) - x) <= px_tol]
            if len(inl) < 3:
                continue
            X = np.array([x for x, _ in inl]); F = np.array([f(v) for _, v in inl])
            b2, a2 = np.polyfit(F, X, 1)
            rms = float(np.sqrt(np.mean((a2 + b2 * F - X) ** 2)))
            score = (len(inl), -rms, kind == "log")
            if best is None or score > best[0]:
                best = (score, Axis(kind, float(a2), float(b2), len(pts), len(inl), inl))
    return best[1] if best else None


# ----------------------------------------------------------------------------- geometry
def _runs(row: np.ndarray, gap: int = 2):
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return []
    out, s, p = [], xs[0], xs[0]
    for x in xs[1:]:
        if x - p > gap + 1:
            out.append((s, p)); s = x
        p = x
    out.append((s, p))
    return out


def dark_mask(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (g < 190).astype(np.uint8)


@dataclass
class Geometry:
    est_x: float | None
    lo_x: float | None
    hi_x: float | None
    y: float | None
    note: str = ""


def measure_row(mask, opened, y_c, half, xlo, xhi) -> Geometry:
    H = mask.shape[0]
    y0, y1 = max(0, int(y_c - half)), min(H, int(y_c + half) + 1)
    xlo, xhi = int(max(0, xlo)), int(min(mask.shape[1], xhi))
    if xhi - xlo < 10 or y1 <= y0:
        return Geometry(None, None, None, None, "empty-band")
    # longest horizontal run in the band = whisker (or diamond mid-row); skip near-full-width rules
    best = None
    for y in range(y0, y1):
        for s, e in _runs(mask[y, xlo:xhi]):
            L = e - s + 1
            if L >= 0.92 * (xhi - xlo):
                continue
            if best is None or L > best[0]:
                best = (L, y, s + xlo, e + xlo)
    if best is None or best[0] < 4:
        return Geometry(None, None, None, None, "no-whisker")
    _, yw, xs, xe = best
    # marker: blobs surviving a morphological opening, inside the whisker x-extent, in the band.
    # Try 4x4 first (rejects text strokes and rules); small low-weight markers need 3x3 / 2x2.
    for k, op in opened:
        sub = op[y0:y1, xs:xe + 1]
        n, lab, stats, cent = cv2.connectedComponentsWithStats(sub, connectivity=8)
        band_h = y1 - y0
        # a vertical null/reference line wider than k px survives the opening: it spans the band and is thin
        cands = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= k * k
                 and not (stats[i, cv2.CC_STAT_HEIGHT] >= 0.9 * band_h
                          and stats[i, cv2.CC_STAT_WIDTH] <= 0.5 * stats[i, cv2.CC_STAT_HEIGHT])]
        if not cands:
            continue
        i = min(cands, key=lambda c: (abs(cent[c][1] + y0 - yw) > 0.35 * half, -stats[c, cv2.CC_STAT_AREA]))
        cols = (lab == i).sum(axis=0)
        if cols.max() <= 2 and k == 2:  # a 2px-thick whisker alone is not a marker
            continue
        peak = np.flatnonzero(cols >= 0.85 * cols.max())
        est_x = xs + float(peak.mean())
        # a marker wider than the whisker (CI narrower than the square) widens the extent
        left = xs + stats[i, cv2.CC_STAT_LEFT]; right = left + stats[i, cv2.CC_STAT_WIDTH] - 1
        return Geometry(est_x, float(min(xs, left)), float(max(xe, right)), float(yw), "")
    # last resort for markers too small to survive any opening: widest vertical stroke on the raw mask,
    # ignoring columns that are a long vertical rule (reference line) rather than a marker
    thick = np.zeros(xe - xs + 1, int)
    for j, x in enumerate(range(xs, xe + 1)):
        up = dn = 0
        while yw - up - 1 >= y0 and mask[yw - up - 1, x]:
            up += 1
        while yw + dn + 1 < y1 and mask[yw + dn + 1, x]:
            dn += 1
        thick[j] = up + dn + 1
    ok = (thick >= 3) & (thick < 0.7 * (y1 - y0))
    if ok.any():
        peak = np.flatnonzero(ok & (thick >= 0.85 * thick[ok].max()))
        return Geometry(xs + float(peak.mean()), float(xs), float(xe), float(yw), "")
    return Geometry(None, xs, xe, yw, "no-marker")


# ----------------------------------------------------------------------------- pipeline
@dataclass
class RowResult:
    raw: str
    p_est: float; p_lo: float; p_hi: float
    g_est: float | None = None; g_lo: float | None = None; g_hi: float | None = None
    err_est: float | None = None; err_lo: float | None = None; err_hi: float | None = None
    flag: bool = True
    reason: str = ""
    y: float = 0.0
    ulp: tuple = (0.0, 0.0, 0.0)
    gx: tuple = ()  # (est_x, lo_x, hi_x, y) pixels, for the overlay


@dataclass
class FigureResult:
    supported: bool
    why_unsupported: str = ""
    axis: Axis | None = None
    rows: list[RowResult] = field(default_factory=list)


def rel(g, p):
    return None if g is None else abs(g - p) / p


def compare(r: RowResult, tol: float = TOL) -> RowResult:
    r.err_est, r.err_lo, r.err_hi = rel(r.g_est, r.p_est), rel(r.g_lo, r.p_lo), rel(r.g_hi, r.p_hi)
    errs = [r.err_est, r.err_lo, r.err_hi]
    if any(e is None for e in errs):
        r.flag, r.reason = True, r.reason or "no-geometry"
    else:
        # a printed "0.03" is only known to +/-0.005: never demand more precision than the text carries
        tols = [max(tol, u / v) for u, v in zip(r.ulp, (r.p_est, r.p_lo, r.p_hi))]
        bad = [n for n, e, t in zip(("est", "lo", "hi"), errs, tols) if e > t]
        r.flag, r.reason = bool(bad), ("mismatch:" + "+".join(bad)) if bad else "ok"
    return r


def analyse(img: np.ndarray, boxes: list[Box] | None = None) -> FigureResult:
    boxes = boxes if boxes is not None else ocr(img)
    printed = parse_printed(boxes)
    if len(printed) < 2:
        return FigureResult(False, "fewer than 2 printed 'est [lo, hi]' rows")
    first, last = min(p.cy for p in printed), max(p.cy for p in printed)
    mask = dark_mask(img)
    # one axis per tick row: stacked panels (A/B/C) each carry their own scale and x-offset
    axes: list[tuple[float, Axis]] = []
    for ticks in find_tick_lines(boxes + ocr_strip(img, last - 4), below_y=first):
        ax_ = fit_axis(snap_ticks(mask, ticks))
        cy = float(np.mean([t.cy for t in ticks]))
        if ax_ is not None and not any(abs(cy - y) < 15 for y, _ in axes):
            axes.append((cy, ax_))
    if not axes:
        return FigureResult(False, "no calibratable tick row (need >=3 consistent numeric ticks)")
    axes.sort(key=lambda t: t[0])

    opened = [(k, cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))) for k in (4, 3, 2)]
    printed.sort(key=lambda p: p.cy)
    ys = [p.cy for p in printed]

    def panel_axis(cy):  # nearest axis below the row (its own panel's), else the last one
        below = [a_ for y, a_ in axes if y >= cy - 2]
        return below[0] if below else axes[-1][1]

    def plot_range(axis, rows):
        tx = [x for x, _ in axis.ticks]
        pad = 0.08 * (max(tx) - min(tx))
        xlo, xhi, mid = min(tx) - pad, max(tx) + pad, (min(tx) + max(tx)) / 2
        for p in rows:  # keep the printed-text column out of the plot band
            if p.x0 > mid:
                xhi = min(xhi, p.x0 - 2)
            else:
                xlo = max(xlo, p.x1 + 2)
        return xlo, xhi

    res = FigureResult(True, axis=panel_axis(first))
    ranges = {}
    for k, p in enumerate(printed):
        axis = panel_axis(p.cy)
        if id(axis) not in ranges:
            ranges[id(axis)] = plot_range(axis, [q for q in printed if panel_axis(q.cy) is axis])
        xlo, xhi = ranges[id(axis)]
        gaps = [abs(ys[k] - ys[j]) for j in (k - 1, k + 1) if 0 <= j < len(ys)]
        half = 0.45 * min(gaps) if gaps else 12.0
        g = measure_row(mask, opened, p.cy, half, xlo, xhi)
        r = RowResult(p.raw, p.est, p.lo, p.hi, y=p.cy, reason=g.note, ulp=p.ulp)
        if g.est_x is not None:
            r.g_est, r.g_lo, r.g_hi = (axis.to_value(x) for x in (g.est_x, g.lo_x, g.hi_x))
            r.gx = (g.est_x, g.lo_x, g.hi_x, g.y)
        res.rows.append(compare(r))
    return res


def analyse_path(path) -> FigureResult:
    return analyse(load_image(path))


def row_dict(r: RowResult) -> dict:
    d = asdict(r)
    d.pop("gx", None)
    return d


def draw_overlay(img: np.ndarray, res: FigureResult) -> np.ndarray:
    """Green = geometry agrees with the printed text, red = flagged. Circle = estimate, bars = CI ends."""
    out = img.copy()
    for r in res.rows:
        if not r.gx:
            continue
        ex, lx, hx, y = r.gx
        col = (220, 40, 40) if r.flag else (30, 160, 60)
        cv2.line(out, (int(lx), int(y) - 7), (int(lx), int(y) + 7), col, 2)
        cv2.line(out, (int(hx), int(y) - 7), (int(hx), int(y) + 7), col, 2)
        cv2.circle(out, (int(ex), int(y)), 5, col, 2)
    return out
