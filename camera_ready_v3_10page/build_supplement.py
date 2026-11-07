"""Build the standalone supplement using the current main-paper labels and citations."""
from pathlib import Path
import argparse
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"


def run(command, log_name):
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    log = BUILD / log_name
    log.write_bytes(result.stdout + result.stderr)
    if result.returncode:
        print((result.stdout + result.stderr).decode("utf-8", errors="replace")[-6000:])
        raise SystemExit(f"Build failed; see {log}")


def compile_document(name):
    for pass_id in range(1, 5):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-file-line-error", "-output-directory=build", f"{name}.tex"],
            f"{name}_supplement_build_pass{pass_id}.txt")
        if pass_id == 1:
            run(["bibtex", f"build/{name}"], f"{name}_supplement_bibtex.txt")
    log = (BUILD / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    for message in ("There were undefined references", "There were undefined citations",
                    "Label(s) may have changed", "multiply defined", "Overfull"):
        if message in log:
            raise SystemExit(f"Resolve '{message}' in build/{name}.log before delivery")
    shutil.copy2(BUILD / f"{name}.pdf", ROOT / f"{name}.pdf")


def prepare_reference_inputs():
    source = (ROOT / "main.tex").read_text(encoding="utf-8")
    preamble, separator, _ = source.partition(r"\begin{document}")
    if not separator or preamble.count(r"\usepackage{hyperref}") != 1:
        raise ValueError("Cannot identify the current main preamble/hyperref load")
    preamble = preamble.replace(r"\usepackage{hyperref}",
                                "\\usepackage{xr-hyper}\n\\usepackage{hyperref}")
    (BUILD / "supplement_preamble.tex").write_text(
        "% Generated from main.tex; edit main.tex rather than this file.\n" + preamble,
        encoding="utf-8")
    aux = (BUILD / "main.aux").read_text(encoding="utf-8")
    end = re.search(r"\\newlabel\{main-text-end\}\{\{[^}]*\}\{(\d+)\}", aux)
    if end is None:
        raise ValueError("Compile the main paper before preparing supplement references")
    main_pages = int(end.group(1))
    labels = []
    for line in aux.splitlines():
        match = re.match(r"\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}", line)
        if match and int(match.group(2)) <= main_pages:
            labels.append(line)
    keys = list(dict.fromkeys(re.findall(r"\\bibcite\{([^}]+)\}", aux)))
    if not labels or not keys:
        raise ValueError("Missing main-text labels or bibliography entries")
    (BUILD / "supplement_main_refs.aux").write_text(
        "\\relax\n" + "\n".join(labels) + "\n", encoding="utf-8")
    (BUILD / "supplement_citations.tex").write_text(
        "% Retain the main paper's bibliography numbering.\n\\nocite{" +
        ",".join(keys) + "}\n", encoding="utf-8")
    print(f"Imported {len(labels)} main-text labels and {len(keys)} citation keys.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-existing-main", action="store_true",
                        help="Use a freshly compiled build/main.aux (for the combined build)")
    args = parser.parse_args()
    for tool in ("pdflatex", "bibtex"):
        if not shutil.which(tool):
            raise SystemExit(f"Missing {tool}; install MiKTeX or TeX Live")
    BUILD.mkdir(exist_ok=True)
    if not args.use_existing_main:
        compile_document("main")
    prepare_reference_inputs()
    compile_document("supplement")
    print(f"Built {ROOT / 'supplement.pdf'}")


if __name__ == "__main__":
    main()
