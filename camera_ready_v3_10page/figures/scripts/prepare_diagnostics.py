"""Export recorded simulation diagnostics without regenerating experiments."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(parents=True, exist_ok=True)


def write(name, fields, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


rows = list(csv.DictReader((ROOT / "experiments/results/experiment4_lower_bound_scaling.csv").open()))
points = [{"x": float(r["x_rate"]), "y": float(r["excess_cvar"])} for r in rows]
write("diagnostics_learning.csv", ["x", "y"], points)
xmean = sum(r["x"] for r in points) / len(points)
ymean = sum(r["y"] for r in points) / len(points)
slope = sum((r["x"] - xmean) * (r["y"] - ymean) for r in points) / sum((r["x"] - xmean) ** 2 for r in points)
intercept = ymean - slope * xmean
xs = [min(r["x"] for r in points), max(r["x"] for r in points)]
write("diagnostics_fit.csv", ["x", "y"], [{"x": x, "y": intercept + slope * x} for x in xs])

rows = list(csv.DictReader((ROOT / "experiments/results/experiment5_sensitivity.csv").open()))
for key in sorted({r["sweep"] for r in rows}):
    write(f"diagnostics_{key}.csv", ["x", "width"], [{"x": r["x"], "width": r["width"]} for r in rows if r["sweep"] == key])
print(f"Diagnostics: {len(points)} recorded simulation points; OLS slope={slope:.8f}, intercept={intercept:.8f}")
