"""Release checks run by CI: version agreement, sdist manifest, installed-artifact smoke.

Usage:
    python scripts/release_check.py version [--ref refs/tags/vX.Y.Z]
    python scripts/release_check.py dist DIST_DIR

Standard library only, so it runs before the project's dependencies are installed.
Needs ``git`` and ``uv`` on PATH for ``dist``.
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
# Package data the installed wheel must carry; the web workbench fails without index.html.
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


def expected_sdist_files() -> set[str]:
    """Tracked files under the pyproject sdist include list, plus Hatch's forced files."""
    build = _pyproject().get("tool", {}).get("hatch", {}).get("build", {})
    include = build.get("targets", {}).get("sdist", {}).get("include")
    if not include:
        raise CheckError("pyproject.toml has no [tool.hatch.build.targets.sdist] include list")
    listed = _run(["git", "ls-files", "-z", "--", *include], ROOT)
    return set(filter(None, listed.split("\0"))) | HATCH_FORCED


def check_sdist_manifest(sdist: Path, version: str, expected: set[str]) -> None:
    """The sdist must hold exactly the expected files, as regular files under one prefix.

    Names are compared verbatim, so ``src/../CLAUDE.md`` is unexpected rather than
    normalised into the ``src`` tree.
    """
    prefix = f"s1ops-{version}/"
    names = set()
    with tarfile.open(sdist) as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise CheckError(f"{sdist.name}: {member.name} is not a regular file")
            if not member.name.startswith(prefix):
                raise CheckError(f"{sdist.name}: member outside {prefix}: {member.name}")
            names.add(member.name[len(prefix):])
    unexpected = sorted(names - expected)
    if unexpected:
        raise CheckError(f"{sdist.name}: unexpected files {unexpected}")
    missing = sorted(expected - names)
    if missing:
        raise CheckError(f"{sdist.name}: missing files {missing}")


def _run(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise CheckError(f"{' '.join(cmd)} exited {result.returncode}:\n{result.stderr.strip()}")
    return result.stdout.strip()


_PROBE = """
import importlib.metadata, importlib.resources, s1, s1ops
assert s1.__version__ == s1ops.__version__ and s1.DecisionGate is s1ops.DecisionGate
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
        # The README quickstart starts from `s1 demo`, so it must work from the installed package.
        demo = work / "demo"
        _run([str(venv / "bin" / "s1"), "demo", str(demo)], work)
        _run([str(venv / "bin" / "s1"), "lint", "check", str(demo / "questions.json")], work)


def check_dist(dist: Path) -> str:
    version = check_version(None)
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    expected = ([f"s1ops-{version}-py3-none-any.whl"], [f"s1ops-{version}.tar.gz"])
    if ([w.name for w in wheels], [s.name for s in sdists]) != expected:
        raise CheckError(f"{dist}: expected {expected}, found {[p.name for p in wheels + sdists]}")
    check_sdist_manifest(sdists[0], version, expected_sdist_files())
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
