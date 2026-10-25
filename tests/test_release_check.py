"""Release guard: tag refs must match the version, and the sdist must hold exactly the tracked include list."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.release_check import CheckError, check_no_repo_identity, check_sdist_manifest, check_version
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
    with pytest.raises(CheckError):
        check_version(ref)


def test_version_guard_accepts_matching_tag() -> None:
    assert check_version(f"refs/tags/v{__version__}") == __version__


REPO = "example-org/example-repo"


def _dists(tmp_path: Path, wheel_files: dict[str, str], sdist_files: dict[str, str]) -> tuple[Path, Path]:
    wheel = tmp_path / f"s1ops-{__version__}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        for name, text in wheel_files.items():
            zf.writestr(name, text)
    sdist = tmp_path / f"s1ops-{__version__}.tar.gz"
    with tarfile.open(sdist, "w:gz") as tar:
        for name, text in sdist_files.items():
            info = tarfile.TarInfo(f"s1ops-{__version__}/{name}")
            info.size = len(text.encode())
            tar.addfile(info, io.BytesIO(text.encode()))
    return wheel, sdist


_CLEAN_WHEEL = {f"s1ops-{__version__}.dist-info/METADATA": "Name: s1ops\n", "s1ops/cli.py": "x = 1\n"}
_CLEAN_SDIST = {"PKG-INFO": "Name: s1ops\n", "scripts/release_check.py": "x = 1\n"}


def test_artifacts_without_repo_identity_pass(tmp_path: Path) -> None:
    check_no_repo_identity(*_dists(tmp_path, _CLEAN_WHEEL, _CLEAN_SDIST), REPO)


@pytest.mark.parametrize(
    ("archive", "name"),
    [
        ("wheel", f"s1ops-{__version__}.dist-info/METADATA"),
        ("wheel", "s1ops/cli.py"),
        ("sdist", "PKG-INFO"),
        ("sdist", "scripts/release_check.py"),
    ],
)
def test_repo_identity_in_any_file_fails(tmp_path: Path, archive: str, name: str) -> None:
    wheel_files, sdist_files = dict(_CLEAN_WHEEL), dict(_CLEAN_SDIST)
    leak = "url = 'https://github.com/Example-Org/example-repo'\n"
    (wheel_files if archive == "wheel" else sdist_files)[name] = leak
    with pytest.raises(CheckError, match="name the private repository") as exc:
        check_no_repo_identity(*_dists(tmp_path, wheel_files, sdist_files), REPO)
    assert name in str(exc.value)


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com-alias:example-org/example-repo.git",
        "https://github.com/example-org/example-repo",
        "https://github.com/example-org/example-repo.git",
    ],
)
def test_repo_identity_from_origin_url(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    import scripts.release_check as release_check

    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.setattr(release_check, "_run", lambda cmd, cwd: url)
    assert release_check.repo_identity() == REPO


def test_repo_identity_prefers_actions_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.release_check import repo_identity

    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    assert repo_identity() == REPO
