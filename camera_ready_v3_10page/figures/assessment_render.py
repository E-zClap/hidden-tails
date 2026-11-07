"""Copy compiled vector PDFs, check size, and render color/grayscale QA PNGs."""
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parent
(ROOT / "previews").mkdir(parents=True, exist_ok=True)
NAMES = ("fig02_learning", "fig03_assessment", "supp_sensitivity", "supp_coverage")
for name in NAMES:
    pdf = ROOT / (name + ".pdf")
    pdf.write_bytes((ROOT / "build" / (name + ".pdf")).read_bytes())
    with fitz.open(pdf) as doc:
        assert len(doc) == 1
        page = doc[0]
        assert abs(page.rect.width - 396) < 0.001, (name, page.rect)
        page.get_pixmap(matrix=fitz.Matrix(3, 3)).save(ROOT / "previews" / (name + ".png"))
        page.get_pixmap(matrix=fitz.Matrix(96 / 72, 96 / 72), colorspace=fitz.csGRAY).save(
            ROOT / "previews" / (name + "_grayscale.png")
        )
        print(f"{name}: 1 page, {page.rect.width:.6f} x {page.rect.height:.6f} PDF pt")
