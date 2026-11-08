"""`source_sha256` is the digest of the bytes a fit or policy was made from, even from a FIFO."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from s1ops.cli import main as cli_main
from s1ops.contracts.records import RecordError, read_records, read_records_hashed
from s1ops.demo import generate_workspace


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("ws"))


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _through_fifo(tmp_path: Path, source: str, run):
    """Serve the bytes once, then an empty stream to any later open, as `<(cat f)` does: a second
    read gets the empty digest and fails the assertion, rather than blocking the test forever."""
    fifo = tmp_path / "records.fifo"
    os.mkfifo(fifo)
    done = threading.Event()
    data = Path(source).read_bytes()  # here, so a missing source fails rather than leaving a reader blocked

    def serve() -> None:
        fifo.write_bytes(data)
        while not done.is_set():
            try:
                os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            except OSError:  # no reader yet
                done.wait(0.01)

    writer = threading.Thread(target=serve, daemon=True)  # a command that never opens it cannot hold the process
    writer.start()
    try:
        return run(fifo)
    finally:
        done.set()


def _without_wall_clock(doc: dict, *listed: str) -> dict:
    for key in listed:
        for row in doc.get(key, []):
            row.pop("fitted_at", None)
            row.pop("created_at", None)
    doc.pop("created_at", None)
    return doc


def test_fit_records_the_digest_of_a_fifo_and_otherwise_the_same_file(ws, tmp_path, capsys) -> None:
    source = ws["records_train"]
    regular, piped = tmp_path / "regular.json", tmp_path / "piped.json"
    assert cli_main(["fit", source, "-o", str(regular)]) == 0
    assert _through_fifo(tmp_path, source, lambda fifo: cli_main(["fit", str(fifo), "-o", str(piped)])) == 0
    regular_doc, piped_doc = json.loads(regular.read_text()), json.loads(piped.read_text())
    assert {fit["source_sha256"] for fit in piped_doc["fits"]} == {_digest(source)}
    assert {fit["source_sha256"] for fit in regular_doc["fits"]} == {_digest(source)}
    assert _without_wall_clock(piped_doc, "fits") == _without_wall_clock(regular_doc, "fits")


def test_policy_tune_records_the_digest_of_a_fifo_and_otherwise_the_same_file(ws, tmp_path, capsys) -> None:
    source = ws["records"]
    regular, piped = tmp_path / "regular.json", tmp_path / "piped.json"
    args = ["--error-budget", "0.1", "-o"]
    assert cli_main(["policy", "tune", source, *args, str(regular)]) == 0
    assert _through_fifo(tmp_path, source, lambda fifo: cli_main(["policy", "tune", str(fifo), *args, str(piped)])) == 0
    regular_doc, piped_doc = json.loads(regular.read_text()), json.loads(piped.read_text())
    assert {row["source_sha256"] for row in piped_doc["policies"]} == {_digest(source)}
    assert _without_wall_clock(piped_doc, "policies") == _without_wall_clock(regular_doc, "policies")


def test_policy_tune_use_calib_records_the_digest(ws, tmp_path, capsys) -> None:
    out = tmp_path / "policy.json"
    assert cli_main(["policy", "tune", ws["records"], "--use-calib", "--error-budget", "0.1", "-o", str(out)]) == 0
    assert {row["source_sha256"] for row in json.loads(out.read_text())["policies"]} == {_digest(ws["records"])}
    assert all(row["calibrated"] for row in json.loads(out.read_text())["policies"])


_ROW = {"schema": "s1-record/1", "item_id": "i", "question_id": "q-1", "primitive": "noul", "model_requested": "m", "model_served": "m", "p": 0.5}
_LINE = json.dumps(_ROW).encode()


@pytest.mark.parametrize(
    "data, lines",
    [
        pytest.param(_LINE + b"\n" + json.dumps({**_ROW, "item_id": "j"}).encode() + b"\n", [1, 2], id="plain"),
        pytest.param(_LINE + b"\r\n", [1], id="crlf"),
        pytest.param(_LINE + b"\r" + json.dumps({**_ROW, "item_id": "j"}).encode(), [1, 2], id="old-mac-newline"),
        pytest.param(b"\n\n" + _LINE + b"\n\n", [3], id="blank-lines"),
        pytest.param(json.dumps({**_ROW, "note": "a\u2028b"}, ensure_ascii=False).encode() + b"\n", [1], id="raw-line-separator-in-a-string"),
        pytest.param(b"", [], id="empty"),
    ],
)
def test_the_hashed_reader_parses_as_read_records_does(tmp_path, data, lines) -> None:
    path = tmp_path / "r.jsonl"
    path.write_bytes(data)
    records, digest = read_records_hashed(path)
    assert [r.raw for r in records] == [r.raw for r in read_records(path)]
    assert [r.line for r in records] == lines == [r.line for r in read_records(path)]
    assert digest == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(_LINE + b"\n" + b'{"item_id": "\xff"}\n', id="not-utf8"),
        pytest.param(_LINE + b"\n{broken\n", id="not-json"),
        pytest.param(b"\xef\xbb\xbf" + _LINE + b"\n", id="bom"),
    ],
)
def test_the_hashed_reader_refuses_as_read_records_does(tmp_path, data) -> None:
    path = tmp_path / "r.jsonl"
    path.write_bytes(data)
    with pytest.raises(RecordError) as plain:
        read_records(path)
    with pytest.raises(RecordError) as hashed:
        read_records_hashed(path)
    assert str(hashed.value) == str(plain.value)
