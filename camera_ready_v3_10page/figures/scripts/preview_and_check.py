"""Render the actual vector exports and record dimensions/text-bound checks."""
from pathlib import Path
import json
import fitz

FIG = Path(__file__).resolve().parents[1]
PREVIEW = FIG / "previews"
PREVIEW.mkdir(exist_ok=True)
NAMES = ["fig01_concept", "fig02_learning", "fig03_assessment", "fig04_revelation",
         "supp_frontier", "supp_diagnostics", "supp_sensitivity", "supp_coverage", "supp_revelation"]
report = []
overview = fitz.open()
for name in NAMES:
    doc = fitz.open(FIG / f"{name}.pdf")
    assert len(doc) == 1
    page = doc[0]
    assert abs(page.rect.width - 396) < 0.1, (name, page.rect)
    spans = [s for b in page.get_text("dict")["blocks"] if "lines" in b
             for line in b["lines"] for s in line["spans"]]
    outside = [s["text"] for s in spans if s["bbox"][0] < -0.4 or s["bbox"][1] < -0.4
               or s["bbox"][2] > page.rect.width + 0.4 or s["bbox"][3] > page.rect.height + 0.4]
    assert not outside, (name, outside)
    assert len(page.get_images()) == 0, (name, "raster image embedded")
    page.get_pixmap(dpi=160, alpha=False).save(PREVIEW / f"{name}.png")
    page.get_pixmap(dpi=96, colorspace=fitz.csGRAY, alpha=False).save(PREVIEW / f"{name}_grayscale.png")
    report.append({"name": name, "width_pdf_bp": page.rect.width,
                   "height_pdf_bp": page.rect.height,
                   "smallest_font_including_math_scripts": min(s["size"] for s in spans),
                   "text_spans_outside_page": outside, "embedded_raster_images": 0})
    if name.startswith("fig0"):
        overview.insert_pdf(doc)
    doc.close()
overview.save(FIG / "figures_overview.pdf")
(FIG / "final_qa.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
