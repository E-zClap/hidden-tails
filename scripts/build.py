"""Build the supplied manuscript and vector figures on Windows, macOS or Linux."""
from pathlib import Path
import argparse
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "camera_ready_v3_10page"
FIGURES = (
    "fig01_concept", "fig02_learning", "fig03_assessment", "fig04_revelation",
    "supp_diagnostics", "supp_sensitivity", "supp_coverage", "supp_frontier",
    "supp_revelation",
)


def run(command, logfile):
    result = subprocess.run(command, cwd=PAPER, capture_output=True)
    logfile.write_bytes(result.stdout + result.stderr)
    if result.returncode:
        print((result.stdout + result.stderr).decode("utf-8", errors="replace")[-8000:])
        raise SystemExit(f"Build failed. See {logfile.relative_to(ROOT)}")


def latex(source, output, logfile):
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         "-file-line-error", f"-output-directory={output}", source], logfile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true", help="Rebuild all nine vector figures")
    parser.add_argument("--paper", action="store_true", help="Compile the complete manuscript")
    args = parser.parse_args()
    if not (args.figures or args.paper):
        parser.error("select --figures, --paper, or both")
    for tool in (["pdflatex", "bibtex"] if args.paper else ["pdflatex"]):
        if not shutil.which(tool):
            raise SystemExit(f"Missing {tool}: install MiKTeX or TeX Live; see README.md")
    if args.figures:
        output = PAPER / "figures/build"
        output.mkdir(parents=True, exist_ok=True)
        for name in FIGURES:
            for pass_id in (1, 2):
                latex(f"figures/{name}.tex", "figures/build", output / f"{name}_pass{pass_id}.txt")
            shutil.copy2(output / f"{name}.pdf", PAPER / "figures" / f"{name}.pdf")
            print(f"Built figures/{name}.pdf", flush=True)
    if args.paper:
        output = PAPER / "build"
        output.mkdir(parents=True, exist_ok=True)
        for pass_id in (1, 2, 3, 4):
            latex("main.tex", "build", output / f"latex_pass{pass_id}.txt")
            if pass_id == 1:
                run(["bibtex", "build/main"], output / "bibtex_output.txt")
        log = (output / "main.log").read_text(encoding="utf-8", errors="replace")
        if any(message in log for message in (
            "There were undefined references", "There were undefined citations",
            "Label(s) may have changed", "Citation `", "multiply defined",
        )):
            raise SystemExit("Unresolved references in build/main.log")
        shutil.copy2(output / "main.pdf", PAPER / "main.pdf")
        print("Built camera_ready_v3_10page/main.pdf", flush=True)


if __name__ == "__main__":
    main()
