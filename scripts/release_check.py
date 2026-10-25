"""Release checks run by CI: version agreement, sdist manifest, installed-artifact smoke.

Usage:
    python scripts/release_check.py version [--ref refs/tags/vX.Y.Z]
    python scripts/release_check.py dist DIST_DIR

Standard library only, so it runs before the project's dependencies are installed.
Needs Python 3.11+ (``tomllib``) and, for ``dist``, ``uv`` on PATH.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAG_REF = re.compile(r"refs/tags/v(\d+\.\d+\.\d+)")
# Hatch adds these to every sdist on top of the configured include list.
HATCH_FORCED = {".gitignore", "PKG-INFO"}
# Package data the wheel and sdist must carry; the web workbench fails without index.html.
PACKAGE_DATA = ("s1ops/web/static/index.html", "s1ops/py.typed")


class CheckError(Exception):
    pass


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _module_version() -> str:
    tree = ast.parse((ROOT / "src/s1ops/__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            return node.value.value
    raise CheckError("src/s1ops/__init__.py defines no literal __version__")


def check_version(ref: str | None) -> str:
    """Return the project version after checking pyproject, ``__version__`` and the tag agree."""
    version = _pyproject()["project"]["version"]
    module = _module_version()
    if module != version:
        raise CheckError(f"pyproject.toml version {version} != s1ops.__version__ {module}")
    if ref is not None:
        match = TAG_REF.fullmatch(ref)
        if match is None:
            raise CheckError(f"release must be dispatched on a tag ref refs/tags/vX.Y.Z, got {ref!r}")
        if match.group(1) != version:
            raise CheckError(f"tag {ref} does not match project version {version}")
    return version


def check_sdist_manifest(sdist: Path, version: str) -> None:
    """Every sdist member must come from the pyproject include list or Hatch's forced files."""
    build = _pyproject().get("tool", {}).get("hatch", {}).get("build", {})
    include = build.get("targets", {}).get("sdist", {}).get("include")
    if not include:
        raise CheckError("pyproject.toml has no [tool.hatch.build.targets.sdist] include list")
    dirs = tuple(f"{entry}/" for entry in include if (ROOT / entry).is_dir())
    files = {entry for entry in include if not (ROOT / entry).is_dir()} | HATCH_FORCED
    prefix = f"s1ops-{version}/"
    with tarfile.open(sdist) as tar:
        members = [m.name for m in tar.getmembers() if m.isfile()]
    names = set()
    for name in members:
        if not name.startswith(prefix):
            raise CheckError(f"{sdist.name}: member outside {prefix}: {name}")
        names.add(name[len(prefix):])
    unexpected = sorted(n for n in names if n not in files and not n.startswith(dirs))
    if unexpected:
        raise CheckError(f"{sdist.name}: unexpected files {unexpected}")
    missing = sorted((files | {f"src/{p}" for p in PACKAGE_DATA}) - names)
    if missing:
        raise CheckError(f"{sdist.name}: missing files {missing}")


def _run(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise CheckError(f"{' '.join(cmd)} exited {result.returncode}:\n{result.stderr.strip()}")
    return result.stdout.strip()


_PROBE = """
import importlib.metadata, importlib.resources, s1ops
print(importlib.metadata.version("s1ops"))
print(s1ops.__file__)
for path in {data!r}:
    print(importlib.resources.files("s1ops").joinpath(path.removeprefix("s1ops/")).is_file())
"""


def check_install(artifact: Path, version: str) -> None:
    """Install one built artifact into a fresh venv outside the checkout and probe it."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        venv = work / "venv"
        _run(["uv", "venv", "--quiet", "--python", sys.executable, str(venv)], work)
        python = venv / "bin" / "python"
        _run(["uv", "pip", "install", "--quiet", "--python", str(python), str(artifact)], work)
        cli = _run([str(venv / "bin" / "s1"), "--version"], work)
        if cli != f"s1 {version}":
            raise CheckError(f"{artifact.name}: `s1 --version` printed {cli!r}")
        meta, location, *present = _run(
            [str(python), "-c", _PROBE.format(data=PACKAGE_DATA)], work
        ).splitlines()
        if meta != version:
            raise CheckError(f"{artifact.name}: installed metadata version {meta}")
        if not Path(location).resolve().is_relative_to(venv.resolve()):
            raise CheckError(f"{artifact.name}: s1ops imported from {location}, not the venv")
        absent = [p for p, ok in zip(PACKAGE_DATA, present) if ok != "True"]
        if absent:
            raise CheckError(f"{artifact.name}: installed package lacks {absent}")


def check_dist(dist: Path) -> str:
    version = check_version(None)
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    expected = ([f"s1ops-{version}-py3-none-any.whl"], [f"s1ops-{version}.tar.gz"])
    if ([w.name for w in wheels], [s.name for s in sdists]) != expected:
        raise CheckError(f"{dist}: expected {expected}, found {[p.name for p in wheels + sdists]}")
    check_sdist_manifest(sdists[0], version)
    for artifact in (wheels[0], sdists[0]):
        check_install(artifact.resolve(), version)
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    version_cmd = sub.add_parser("version", help="check that the version declarations agree")
    version_cmd.add_argument("--ref", help="git ref the release was dispatched on")
    dist_cmd = sub.add_parser("dist", help="check built distributions")
    dist_cmd.add_argument("dist_dir", type=Path, help="directory holding the wheel and sdist")
    args = parser.parse_args(argv)
    try:
        if args.command == "version":
            version = check_version(args.ref)
        else:
            version = check_dist(args.dist_dir)
    except CheckError as exc:
        print(f"release_check: {exc}", file=sys.stderr)
        return 1
    print(f"release_check: {args.command} ok ({version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
