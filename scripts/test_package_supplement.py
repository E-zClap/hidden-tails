"""Focused archive checks; uses only temporary fixtures and the standard library."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


SCRIPT = Path(__file__).with_name("package_supplement.py")
package = None
if SCRIPT.is_file():
    spec = importlib.util.spec_from_file_location("package_supplement", SCRIPT)
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)


class SupplementPackageTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(package, "The manifest-based supplement packager is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "scripts/example.py").write_bytes(b"print('reproduce')\n")
        self.manifest = {
            "release": "1.2.3", "algorithm": "SHA-256",
            "files": {"scripts/example.py": hashlib.sha256(
                (self.root / "scripts/example.py").read_bytes()).hexdigest()},
        }
        self.write_manifest()

    def write_manifest(self):
        (self.root / "RELEASE_MANIFEST.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_roundtrip_excludes_unlisted_files_and_preserves_bytes(self):
        (self.root / "borrower_records.csv").write_bytes(b"private fixture\n")
        output, count = package.build_archive(self.root)
        self.assertEqual(output.parent, self.root / "dist")
        self.assertEqual(count, 2)
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(set(archive.namelist()), {"scripts/example.py", "RELEASE_MANIFEST.json"})
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (self.root / name).read_bytes())

    def test_changed_input_preserves_existing_output(self):
        output = self.root / "existing.zip"
        output.write_bytes(b"previous delivery")
        (self.root / "scripts/example.py").write_bytes(b"changed\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            package.build_archive(self.root, output)
        self.assertEqual(output.read_bytes(), b"previous delivery")

    def test_rejects_unsafe_or_private_member_paths(self):
        bad_names = ["../outside.txt", "/absolute.txt", "C:/data.txt", "a\\b.txt",
                     "a/../b.txt", "a//b.txt", "./b.txt", ".git/config", ".GIT/config",
                     ".venv/site.py", "artifacts/prepared/cohort.parquet",
                     "review/before_update/main.tex", "RELEASE_MANIFEST.json"]
        digest = next(iter(self.manifest["files"].values()))
        for name in bad_names:
            with self.subTest(name=name):
                self.manifest["files"] = {name: digest}
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "Unsafe|Private|reserved"):
                    package.build_archive(self.root)

    def test_rejects_source_overwrite(self):
        for name in ["RELEASE_MANIFEST.json", "scripts/example.py"]:
            with self.subTest(name=name):
                previous = (self.root / name).read_bytes()
                with self.assertRaisesRegex(ValueError, "source"):
                    package.build_archive(self.root, self.root / name)
                self.assertEqual((self.root / name).read_bytes(), previous)

    def test_rejects_duplicate_manifest_keys(self):
        (self.root / "RELEASE_MANIFEST.json").write_text(
            '{"release":"1.2.3","algorithm":"SHA-256","files":{},"files":{}}',
            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            package.build_archive(self.root)

    def test_rejects_missing_input(self):
        (self.root / "scripts/example.py").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            package.build_archive(self.root)


if __name__ == "__main__":
    unittest.main()
