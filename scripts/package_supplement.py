"""Package exactly the verified release files into a portable supplement ZIP."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "RELEASE_MANIFEST.json"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def source_path(root, name):
    """Reject nonportable archive names, private assets, and linked source files."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\0" in name:
        raise ValueError(f"Unsafe release path: {name!r}")
    parts = PurePosixPath(name).parts
    if (PurePosixPath(name).is_absolute() or ".." in parts
            or "/".join(parts) != name):
        raise ValueError(f"Unsafe release path: {name!r}")
    lower_parts = tuple(part.casefold() for part in parts)
    if (any(part in {".git", ".venv", "venv", "__pycache__", "prepared", "before_update"}
            or part.startswith(".venv-") for part in lower_parts)
            or lower_parts[-1] == ".env" or lower_parts[-1].startswith(".env.")
            or name.casefold().endswith((".parquet", ".pkl", ".joblib", ".npy", ".npz", ".csv.gz", ".csv.zip"))):
        raise ValueError(f"Private/runtime asset is not a supplement member: {name}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Linked release source is not permitted: {name}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Release path escapes the source root: {name}")
    if not path.is_file():
        raise ValueError(f"Missing release file: {name}")
    return path


def build_archive(root, output=None):
    """Validate sources, create and verify a temporary ZIP, then replace output."""
    root = Path(root).resolve()
    manifest_path = source_path(root, MANIFEST)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes, object_pairs_hook=unique_object)
    if not isinstance(manifest, dict) or manifest.get("algorithm") != "SHA-256":
        raise ValueError("Expected a SHA-256 release manifest")
    release = manifest.get("release")
    if not isinstance(release, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release):
        raise ValueError("Missing or invalid release identifier")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Release manifest must contain a nonempty files mapping")
    sources, hashes = {}, {}
    seen = {MANIFEST.casefold()}
    for name, expected in files.items():
        if not isinstance(name, str) or name.casefold() in seen:
            raise ValueError(f"Duplicate or reserved release member: {name!r}")
        seen.add(name.casefold())
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid SHA-256 for release member: {name}")
        path = source_path(root, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Release checksum mismatch: {name}; use an unmodified release checkout")
        sources[name] = path
        hashes[name] = expected
    hashes[MANIFEST] = hashlib.sha256(manifest_bytes).hexdigest()
    output = (Path(output).expanduser() if output is not None
              else root / "dist" / f"hidden_tails_supplement_v{release}.zip").resolve()
    if output in {manifest_path, *(p.resolve() for p in sources.values())}:
        raise ValueError("Output would overwrite a release source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(sources):
                archive.write(sources[name], arcname=name)
            archive.writestr(MANIFEST, manifest_bytes)
        with zipfile.ZipFile(temporary) as archive:
            if len(archive.namelist()) != len(hashes) or set(archive.namelist()) != set(hashes):
                raise ValueError("Archive member set differs from the release allowlist")
            if archive.testzip() is not None:
                raise ValueError("Archive integrity check failed")
            for name, expected in hashes.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError(f"Archive checksum mismatch: {name}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output, len(hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="Output ZIP (default: dist/hidden_tails_supplement_vVERSION.zip)")
    args = parser.parse_args()
    try:
        output, count = build_archive(ROOT, args.output)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Packaging failed: {error}\n")
    print(f"Packaged {count} verified release files: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
