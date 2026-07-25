"""Portability proofs for the versioned implementation hash.

Scheme repo-relative-path-and-content-v1 must be deterministic across checkout
directories and OS path-separator styles, while staying sensitive to renames,
moves, and single-byte content changes.
"""

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from poker44.utils.model_manifest import (
    IMPLEMENTATION_SHA256_SCHEME,
    _sha256_for_files,
    build_local_model_manifest,
)

_FIXTURE = {
    "neurons/miner.py": b"print('serve')\n",
    "poker44/miner_model/features.py": b"FEATURES = 35\n",
    "poker44/miner_model/feature_core/base30.py": b"def extract():\n    return []\n",
    "poker44/miner_model/model_meta.json": b'{"feature_count": 35}\n',
}


def _make_tree(root: Path, files=_FIXTURE) -> list:
    paths = []
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        paths.append(p)
    return paths


class ImplementationHashPortabilityTests(unittest.TestCase):
    def test_1_same_tree_different_absolute_dirs_same_hash(self):
        # second checkout path contains spaces AND non-ASCII characters
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            ra = Path(a) / "checkout-one"
            rb = Path(b) / "a dir with spaces" / "wörk spáce ünïcode-δ" / "checkout-two"
            pa, pb = _make_tree(ra), _make_tree(rb)
            self.assertEqual(_sha256_for_files(pa, repo_root=ra),
                             _sha256_for_files(pb, repo_root=rb))

    def test_2_single_byte_change_changes_hash(self):
        with TemporaryDirectory() as a:
            root = Path(a)
            paths = _make_tree(root)
            before = _sha256_for_files(paths, repo_root=root)
            target = root / "poker44/miner_model/features.py"
            target.write_bytes(target.read_bytes()[:-1] + b"6")   # 35 -> 36 last byte
            self.assertNotEqual(before, _sha256_for_files(paths, repo_root=root))

    def test_3_rename_changes_hash(self):
        with TemporaryDirectory() as a:
            root = Path(a)
            paths = _make_tree(root)
            before = _sha256_for_files(paths, repo_root=root)
            old = root / "poker44/miner_model/features.py"
            new = root / "poker44/miner_model/features_renamed.py"
            old.rename(new)
            renamed = [new if p == old else p for p in paths]
            self.assertNotEqual(before, _sha256_for_files(renamed, repo_root=root))

    def test_4_moving_parent_directory_does_not_change_hash(self):
        with TemporaryDirectory() as a:
            root = Path(a) / "original"
            paths = _make_tree(root)
            before = _sha256_for_files(paths, repo_root=root)
            moved = Path(a) / "relocated elsewhere"
            shutil.move(str(root), str(moved))
            moved_paths = [moved / p.relative_to(root) for p in paths]
            self.assertEqual(before, _sha256_for_files(moved_paths, repo_root=moved))

    def test_5_input_ordering_is_irrelevant(self):
        with TemporaryDirectory() as a:
            root = Path(a)
            paths = _make_tree(root)
            self.assertEqual(_sha256_for_files(paths, repo_root=root),
                             _sha256_for_files(list(reversed(paths)), repo_root=root))

    def test_6_file_outside_repo_root_rejected(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            root = Path(a)
            paths = _make_tree(root)
            outsider = Path(b) / "outside.py"
            outsider.write_bytes(b"x = 1\n")
            with self.assertRaises(ValueError):
                _sha256_for_files(paths + [outsider], repo_root=root)

    def test_7_missing_file_rejected(self):
        with TemporaryDirectory() as a:
            root = Path(a)
            paths = _make_tree(root)
            with self.assertRaises(FileNotFoundError):
                _sha256_for_files(paths + [root / "poker44/nope.py"], repo_root=root)

    def test_8_windows_style_separators_normalize(self):
        with TemporaryDirectory() as a:
            root = Path(a)
            _make_tree(root)
            posix = list(_FIXTURE)
            windows = [rel.replace("/", "\\") for rel in _FIXTURE]
            self.assertEqual(_sha256_for_files(posix, repo_root=root),
                             _sha256_for_files(windows, repo_root=root))

    def test_9_manifest_reports_scheme_and_relative_files(self):
        with TemporaryDirectory() as a:
            root = Path(a)
            paths = _make_tree(root)
            manifest = build_local_model_manifest(
                repo_root=root, implementation_files=paths,
                defaults={"model_name": "test", "model_version": "v0"},
            )
            self.assertEqual(manifest["implementation_sha256_scheme"],
                             IMPLEMENTATION_SHA256_SCHEME)
            self.assertEqual(manifest["implementation_files"], sorted(_FIXTURE))
            self.assertEqual(len(manifest["implementation_sha256"]), 64)

    def test_10_scheme_is_domain_separated_from_plain_concat(self):
        # a trivially concatenated digest of the same bytes must not collide
        import hashlib
        with TemporaryDirectory() as a:
            root = Path(a)
            paths = _make_tree(root)
            naive = hashlib.sha256(b"".join(_FIXTURE[k] for k in sorted(_FIXTURE))).hexdigest()
            self.assertNotEqual(naive, _sha256_for_files(paths, repo_root=root))


if __name__ == "__main__":
    unittest.main()
