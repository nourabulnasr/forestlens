"""Generate demo/synthetic_clean.png and demo/synthetic_corrupted.png (same figure; row 4's printed estimate has a slip)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from PIL import Image
from forestlens.synthetic import make_forest

out = pathlib.Path(__file__).resolve().parent.parent / "demo"
img, truth = make_forest(seed=0)
Image.fromarray(img).save(out / "synthetic_clean.png")
e, lo, hi = truth[3]
bad = (round(e * 1.2, 2), lo, round(max(hi, e * 1.2 + 0.01), 2))
img2, _ = make_forest(seed=0, printed_override={3: bad})
Image.fromarray(img2).save(out / "synthetic_corrupted.png")
print("printed row 4:", truth[3], "->", bad)
