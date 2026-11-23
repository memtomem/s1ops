"""The repository is public: no tracked file names a local home directory, a private repository or an
operator account.

The test reads `git ls-files` of the checkout this file is in (a clone or a worktree, whose `.git` is
a file) and skips in an unpacked sdist, which ships `tests/` but has no `.git`. The names are written
in pieces so this file does not match itself.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_WORD = r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])"
# Matched case-insensitively against every line of every tracked file. A home directory is a path
# prefix; the names are whole words, so an identifier that merely contains one does not match.
PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        re.escape("/Us" "ers/"),
        _WORD.format("pd" "studio"),
        _WORD.format("pandas" "-studio"),
        _WORD.format("agent" "-harness"),
        _WORD.format("decision" "-plane"),
        _WORD.format("jev" "[-_]toolgate"),
        _WORD.format("jev" "-utils"),
        _WORD.format("jev" "-system1"),
        _WORD.format("ts" "data"),
        _WORD.format(re.escape("github.com" "-memtomem")),
        _WORD.format(re.escape("github.com" "-pandas")),
    )
)

# (tracked path, exact text, quotes included): a match inside that text on that file is allowed.
ALLOWED = {
    ("tests/test_release_check.py", '"/Us' 'ers/someone/private"'),  # a synthetic symlink target
}


def _tracked_files() -> list[str]:
    if not (ROOT / ".git").exists():
        pytest.skip("not a checkout (an unpacked sdist)")
    # In a checkout, any failure to list it is a failure, never a skip or an empty list that passes.
    top = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    assert Path(top.stdout.strip()).resolve() == ROOT, f"git names another top level: {top.stdout.strip()}"
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True)
    files = [name for name in listed.stdout.decode("utf-8").split("\0") if name]
    assert files, "git ls-files listed nothing"
    return files


def _hits(path: str, text: str) -> list[str]:
    allowed = [s for p, s in ALLOWED if p == path]
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        covered: set[int] = set()
        for s in allowed:
            start = line.find(s)
            while start != -1:
                covered.update(range(start, start + len(s)))
                start = line.find(s, start + 1)
        for pattern in PATTERNS:
            for m in pattern.finditer(line):
                if not set(range(m.start(), m.end())) <= covered:
                    found.append(f"{path}:{number}: {m.group(0)!r}")
    return found


def test_no_tracked_file_names_a_private_path_repository_or_account():
    found = []
    for name in _tracked_files():
        found.extend(f"path {hit}" for hit in _hits(name, name))
        path = ROOT / name
        if path.is_symlink():
            # The link's own target text is what git stores; never follow it.
            found.extend(_hits(name, os.readlink(path)))
        elif path.is_file():
            found.extend(_hits(name, path.read_bytes().decode("utf-8", errors="replace")))
        else:
            found.append(f"{name}: tracked but not in the working tree, so it was not read")
    assert not found, "\n".join(found)


def test_an_allowed_text_covers_only_itself_in_its_own_file():
    home = "/Us" "ers/"
    fixture = f'"{home}someone/private"'
    assert _hits("tests/test_release_check.py", f"x = {fixture}") == []
    assert _hits("README.md", f"x = {fixture}") != []
    # A longer path sharing the fixture's prefix is not the fixture.
    assert _hits("tests/test_release_check.py", f'x = "{home}someone/private/more"') != []
    assert _hits("tests/test_release_check.py", f'{fixture} and "{home}real"') != []


def test_names_match_as_words_only():
    account = "ts" "data"
    assert _hits("README.md", f"chart{account.title()} = 1") == []
    assert _hits("README.md", f"active account {account}") != []
    alias = "github.com" "-memtomem"
    assert _hits("README.md", f"git@{alias}:o/r.git") != []
    assert _hits("README.md", f"git@{alias}x:o/r.git") == []
    consumer = "decision" "-plane"
    assert _hits("README.md", f"~/.{consumer}/config") != []
    assert _hits("README.md", f"the {consumer}-hook") != []


def test_a_tracked_path_is_checked_as_well_as_its_contents():
    consumer = "decision" "-plane"
    assert _hits(f"docs/{consumer}.md", f"docs/{consumer}.md") == [f"docs/{consumer}.md:1: {consumer!r}"]
