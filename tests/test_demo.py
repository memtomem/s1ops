"""`s1 demo`: writes the seeded generator's workspace, and only into a new or empty directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.demo import generate_workspace

# Wall-clock fields: fits.json `fitted_at` (also copied into each calibrated record) and
# policy.json `created_at`. Everything else in the workspace is a function of the seed.
_CLOCK_KEYS = {"fitted_at", "created_at"}


def _strip_clock(value):
    if isinstance(value, dict):
        return {k: _strip_clock(v) for k, v in value.items() if k not in _CLOCK_KEYS}
    if isinstance(value, list):
        return [_strip_clock(v) for v in value]
    return value


def _load(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines()]
    return json.loads(text)


def _workspace(root: Path) -> dict[str, object]:
    return {p.name: _strip_clock(_load(p)) for p in sorted(root.iterdir())}


def test_demo_matches_generator(tmp_path: Path, capsys) -> None:
    out = tmp_path / "nested" / "demo"
    assert cli_main(["demo", str(out)]) == 0
    stdout, stderr = capsys.readouterr()
    paths = json.loads(stdout)
    assert sorted(Path(p).name for p in paths.values()) == sorted(p.name for p in out.iterdir())
    assert len(paths) == 11
    assert stderr == f"next: s1 web --workspace {out}\n"

    generate_workspace(tmp_path / "direct")
    assert _workspace(out) == _workspace(tmp_path / "direct")


def test_demo_seed_changes_the_data(tmp_path: Path, capsys) -> None:
    assert cli_main(["demo", str(tmp_path / "a")]) == 0
    assert cli_main(["demo", str(tmp_path / "b"), "--seed", "7"]) == 0
    capsys.readouterr()
    p_a = [r["p"] for r in _load(tmp_path / "a" / "records.jsonl")]
    p_b = [r["p"] for r in _load(tmp_path / "b" / "records.jsonl")]
    assert len(p_a) == len(p_b) and p_a != p_b


def test_demo_next_step_quotes_the_path(tmp_path: Path, capsys) -> None:
    out = tmp_path / "my demo"
    assert cli_main(["demo", str(out)]) == 0
    assert capsys.readouterr().err == f"next: s1 web --workspace '{out}'\n"


def test_demo_accepts_existing_empty_directory(tmp_path: Path, capsys) -> None:
    out = tmp_path / "empty"
    out.mkdir()
    assert cli_main(["demo", str(out)]) == 0
    capsys.readouterr()
    assert (out / "policy.json").is_file()


@pytest.mark.parametrize("kind", ["file", "non-empty directory"])
def test_demo_refuses_without_writing(tmp_path: Path, capsys, kind: str) -> None:
    if kind == "file":
        target = tmp_path / "taken"
        target.write_text("keep\n")
        before = {target: "keep\n"}
        expected = "not a directory"
    else:
        target = tmp_path / "ws"
        target.mkdir()
        (target / "policy.json").write_text("mine\n")
        before = {target / "policy.json": "mine\n"}
        expected = "not empty"
    assert cli_main(["demo", str(target)]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("s1: ") and expected in err and err.count("\n") == 1
    assert {p: p.read_text() for p in before} == before
    if kind != "file":
        assert sorted(p.name for p in target.iterdir()) == ["policy.json"]
