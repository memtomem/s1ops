"""Release guard: tag refs must match the version, and the sdist must hold exactly the tracked include list."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from scripts.release_check import CheckError, check_sdist_manifest, check_version
from s1ops import __version__

EXPECTED = {"PKG-INFO", ".gitignore", "README.md", "src/s1/__init__.py", "src/s1ops/__init__.py"}


def _sdist(tmp_path: Path, names: list[str], *, symlink: str | None = None, prefix: str | None = None) -> Path:
    path = tmp_path / "s1ops.tar.gz"
    root = prefix if prefix is not None else f"s1ops-{__version__}/"
    with tarfile.open(path, "w:gz") as tar:
        dir_info = tarfile.TarInfo(root + "src")
        dir_info.type = tarfile.DIRTYPE
        tar.addfile(dir_info)
        for name in names:
            info = tarfile.TarInfo(root + name)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
        if symlink is not None:
            link = tarfile.TarInfo(root + symlink)
            link.type = tarfile.SYMTYPE
            link.linkname = "/Users/someone/private"
            tar.addfile(link)
    return path


def test_sdist_manifest_accepts_exact_inventory(tmp_path: Path) -> None:
    check_sdist_manifest(_sdist(tmp_path, sorted(EXPECTED)), __version__, EXPECTED)


@pytest.mark.parametrize(
    ("names", "extra", "message"),
    [
        (sorted(EXPECTED | {"CLAUDE.md"}), {}, "unexpected files ['CLAUDE.md']"),
        (sorted(EXPECTED | {"src/../CLAUDE.md"}), {}, "unexpected files ['src/../CLAUDE.md']"),
        (sorted(EXPECTED - {"src/s1/__init__.py"}), {}, "missing files ['src/s1/__init__.py']"),
        (sorted(EXPECTED), {"symlink": "docs/private-link"}, "docs/private-link is not a regular file"),
        (sorted(EXPECTED), {"prefix": "other-0.0.1/"}, "member outside"),
    ],
)
def test_sdist_manifest_rejects(tmp_path: Path, names: list[str], extra: dict, message: str) -> None:
    with pytest.raises(CheckError, match="^s1ops.tar.gz: ") as exc:
        check_sdist_manifest(_sdist(tmp_path, names, **extra), __version__, EXPECTED)
    assert message in str(exc.value)


@pytest.mark.parametrize(
    "ref",
    ["refs/heads/main", "refs/tags/v0.0.0", f"refs/tags/{__version__}", f"refs/tags/v{__version__}-rc1"],
)
def test_version_guard_rejects_ref(ref: str) -> None:
    pytest.importorskip("tomllib")
    with pytest.raises(CheckError):
        check_version(ref)


def test_version_guard_accepts_matching_tag() -> None:
    pytest.importorskip("tomllib")
    assert check_version(f"refs/tags/v{__version__}") == __version__
