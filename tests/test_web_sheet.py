"""The blind review sheet in the workbench: `read_sheet` / `SheetForm.fill`, `GET /api/sheet` and its
loopback rule, `s1 web --sheet`, and the Review view's functions.

The sheet is the one workspace file that holds state text, and `/api/sheet` the one response that
carries it, so this file also holds the review to what it may not do: read the key, answer a remote
request, or say anything a refusal quoted from the sheet."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import random
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from s1ops.cli import main as cli_main
from s1ops.contracts.records import parse_record
from s1ops.demo import generate_workspace
from s1ops.improve.labelaudit import (
    SheetFormatError,
    SheetInputError,
    build_sheet,
    parse_answers,
    read_sheet,
    score_sheet,
    sheet_options,
)
from s1ops.web.app import SHEET_REFUSED, create_app, sheet_request_allowed
from tests.test_web_render import _all_functions, _node

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the page's functions")
LOOPBACK = {"client": ("127.0.0.1", 50000), "base_url": "http://127.0.0.1:8765"}


def _quiet_cli(argv: list[str]) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return cli_main(argv)


@pytest.fixture(scope="module")
def made(tmp_path_factory) -> dict:
    """README's walkthrough: a demo workspace, its fit, and a sheet of 6 flags and 6 controls."""
    root = tmp_path_factory.mktemp("sheet")
    ws = generate_workspace(root / "w")
    sheet, key = root / "blind_sheet.md", root / "blind_key.json"
    assert _quiet_cli(["labelaudit", "sheet", ws["records"], "--states", ws["states_local"], "--questions", ws["questions"],
                       "--fit", ws["fits"], "-o", str(sheet), "-k", str(key), "--top", "6", "--controls", "6"]) == 0
    return {"ws": ws, "root": root, "sheet": sheet, "key": key, "form": sheet.read_text(encoding="utf-8"),
            "key_doc": json.loads(key.read_text(encoding="utf-8"))}


def _korean(form: str, header: bool) -> str:
    """The sheet as s1ops printed it in Korean: 0.6.0 (no question block) or the unreleased commits
    before e072369 (`질문 ID:`)."""
    head, body = form.split("\n## 1\n", 1)
    title, _, instruction = head.split("\n")[:3]
    assert title == "# Label blind review"
    options = instruction[len("Choose one answer for each item: "): -len(". Write it after `Answer:`.")]
    block = head.split("\n", 4)[4].replace("Question ID:", "질문 ID:") if header else ""
    lines = ["# 라벨 블라인드 재검수", "", f"각 항목의 올바른 답을 하나 고르세요: {options}. `답:` 뒤에 적어 주세요.", ""]
    return "\n".join(lines) + ("\n" + block if block else "") + "\n## 1\n" + body.replace("\nAnswer: ", "\n답: ")


def _answered(form: str, answers: dict[int, str]) -> str:
    out = form
    for n, a in answers.items():
        out = re.sub(rf"(^## {n}\n.*?^Answer: )$", lambda m: m.group(1) + a, out, count=1, flags=re.M | re.S)
    return out


@pytest.fixture(scope="module")
def forms(made) -> dict[str, str]:
    crlf = made["root"] / "crlf.md"
    crlf.write_bytes(made["form"].replace("\n", "\r\n").encode("utf-8"))
    return {
        "fresh": made["form"],
        "v060": _korean(made["form"], header=False),
        "ko-header": _korean(made["form"], header=True),
        "answered": _answered(made["form"], {2: "True", 5: "False"}),
        "crlf": crlf.read_text(encoding="utf-8"),
    }


# --- read_sheet and fill -----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["fresh", "v060", "ko-header", "answered", "crlf"])
def test_filling_nothing_gives_the_sheet_back(forms, name) -> None:
    sheet = read_sheet(forms[name])
    assert sheet.fill({}) == forms[name]
    assert len(sheet.pieces) == len(sheet.items) + 1 == 13
    assert all(p.endswith(("Answer: ", "답: ")) for p in sheet.pieces[:-1])


@pytest.mark.parametrize("name", ["fresh", "v060", "ko-header", "answered", "crlf"])
def test_every_answer_map_reads_back_as_written(forms, made, name) -> None:
    sheet = read_sheet(forms[name])
    own = {it.n: it.answer for it in sheet.items}
    rng = random.Random(name)
    maps = [{n: rng.choice([None, *sheet.options]) for n in rng.sample(range(1, 13), rng.randint(0, 12))} for _ in range(40)]
    maps.append({n: sheet.options[n % 2] for n in range(1, 13)})
    for a in maps:
        merged = {**own, **a}
        out = sheet.fill(a)
        again = read_sheet(out)
        assert {it.n: it.answer for it in again.items} == merged
        assert again.pieces == sheet.pieces
        if None not in merged.values():
            assert parse_answers(out, sheet.options) == {str(n): v for n, v in merged.items()}
            score_sheet(out, made["key_doc"])


def test_a_partly_answered_sheet_fails_estimate_on_the_first_empty_answer(made) -> None:
    out = read_sheet(made["form"]).fill({1: "True"})
    with pytest.raises(ValueError, match=r"^item 2: answer '' is not one of \['False', 'True'\]$"):
        score_sheet(out, made["key_doc"])


def test_an_answer_goes_right_after_the_marker(made) -> None:
    out = read_sheet(made["form"]).fill({1: "True"})
    assert "\n## 1\n" in out and re.search(r"^## 1\n.*?^Answer: True$", out, re.M | re.S)
    assert out.count("\nAnswer: True\n") == 1


def test_the_question_and_header_as_each_version_wrote_them(forms, made) -> None:
    fresh, old, ko = read_sheet(forms["fresh"]), read_sheet(forms["v060"]), read_sheet(forms["ko-header"])
    assert fresh.question_id == ko.question_id == made["key_doc"]["question_id"]
    assert fresh.header.startswith("Question ID:\n\n> q-") and not fresh.header.endswith("\n")
    assert (old.question_id, old.header) == (None, "")
    assert fresh.options == old.options == ("True", "False")
    assert [it.state for it in fresh.items] == [it.state for it in old.items]
    assert all(it.state and not it.state.startswith("> ") for it in fresh.items)


def test_a_states_own_blank_lines_and_a_reviewers_note_are_kept() -> None:
    form = "# Label blind review\n\nChoose one answer for each item: a, b. Write it after `Answer:`.\n\n## 1\n\n> \n> hello\n>\nnote\n\nAnswer: \n"
    assert read_sheet(form).items[0].state == "\nhello\n\nnote"


def test_an_option_with_a_dot_is_read_whole() -> None:
    form = "# Label blind review\n\nChoose one answer for each item: v1.0, v2. Write it after `Answer:`.\n\n## 1\n\n> s\n\nAnswer: v1.0\n"
    sheet = read_sheet(form)
    assert sheet.options == ("v1.0", "v2") and sheet.items[0].answer == "v1.0"


def _swap(form: str, a: str, b: str) -> str:
    return form.replace(a, "\0").replace(b, a).replace("\0", b)


REFUSALS = [
    ("no-title", lambda f: f.split("\n", 1)[1], r"^the first line is not the title `# Label blind review`$"),
    ("note-after-title", lambda f: f.replace("\n\nChoose", "\n\nA note.\n\nChoose", 1), r"^the line after the title is not the instruction line"),
    ("empty-option", lambda f: f.replace(": True, False.", ": True, , False.", 1), r"^the instruction line's options cannot be read"),
    ("repeated-option", lambda f: f.replace(": True, False.", ": True, True.", 1), r"^the instruction line's options cannot be read"),
    ("spaced-option", lambda f: f.replace(": True, False.", ": True,  False.", 1), r"^the instruction line's options cannot be read"),
    ("no-item", lambda f: f.split("\n## 1\n")[0] + "\n", r"^the sheet has no item$"),
    ("unicode-heading", lambda f: f.replace("\n## 3\n", "\n## ٣\n", 1), r"^the heading of item 3 is not `## 3`"),
    ("unicode-heading-at-end", lambda f: f + "\n## ٢", r"^line \d+ is an item heading that opens no item"),
    ("heading-at-end", lambda f: f + "\n## 13", r"^line \d+ is an item heading that opens no item"),
    ("leading-zero", lambda f: f.replace("\n## 2\n", "\n## 02\n", 1), r"^the heading of item 2 is not `## 2`"),
    ("out-of-order", lambda f: _swap(f, "\n## 2\n", "\n## 3\n"), r"^the heading of item 2 is not `## 2`"),
    ("two-markers", lambda f: f.replace("\nAnswer: \n", "\nAnswer: \nAnswer: \n", 1), r"^item 1: expected one line starting with Answer:$"),
    ("no-space", lambda f: f.replace("\nAnswer: \n", "\nAnswer:True\n", 1), r"^item 1: the web reads an answer only as"),
    ("trailing-space", lambda f: f.replace("\nAnswer: \n", "\nAnswer: True \n", 1), r"^item 1: the web reads an answer only as"),
    ("next-line", lambda f: f.replace("\nAnswer: \n", "\nAnswer: \nTrue\n", 1), r"^item 1: the web reads an answer only as"),
    ("not-an-option", lambda f: f.replace("\nAnswer: \n", "\nAnswer: maybe\n", 1), r"^item 1: the web reads an answer only as"),
    ("text-below", lambda f: f.replace("\nAnswer: \n", "\nAnswer: True\nsee notes\n", 1), r"^item 1: the web reads an answer only as"),
]


@pytest.mark.parametrize("name, edit, message", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_a_sheet_not_in_its_writers_form_is_refused_without_quoting_it(made, name, edit, message) -> None:
    form = edit(made["form"])
    with pytest.raises(SheetFormatError, match=message) as caught:
        read_sheet(form)
    text = str(caught.value)
    # Nothing of the sheet's own text: no state, no reviewer answer, no option.
    for leaked in ("Customer message", "wire transfer", "maybe", "see notes", "True", "False", "٣", "٢"):
        assert leaked not in text.replace("Answer: <option>", "")


@pytest.mark.parametrize("key", [True, 1.0, 0, 13, "1"])
def test_fill_refuses_an_item_the_sheet_does_not_have(made, key) -> None:
    with pytest.raises(ValueError, match="the sheet has no item"):
        read_sheet(made["form"]).fill({key: "True"})


@pytest.mark.parametrize("value", ["maybe", "true", 1, True])
def test_fill_refuses_an_answer_that_is_not_an_option(made, value) -> None:
    with pytest.raises(ValueError, match="item 1: the answer is not one of the sheet's options"):
        read_sheet(made["form"]).fill({1: value})


# --- options a sheet cannot carry --------------------------------------------------------------


def _choice_records(tmp_path: Path) -> tuple[list, Path, Path]:
    rows, states = [], []
    for i in range(12):
        p = 0.2 + 0.05 * i
        rows.append({"schema": "s1-record/1", "primitive": "choice", "question_id": "q-0000000000000c0a", "dataset": "d",
                     "item_id": f"i{i}", "model_requested": "m", "model_served": "m", "probabilities": {"No, escalate": p, "Yes": 1 - p},
                     "choice": "No, escalate" if p >= 0.5 else "Yes", "confidence": max(p, 1 - p),
                     "label": "Yes" if i % 3 else "No, escalate"})
        states.append({"dataset": "d", "item_id": f"i{i}", "state": f"state {i}"})
    rec, st = tmp_path / "records.jsonl", tmp_path / "states.jsonl"
    rec.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    st.write_text("".join(json.dumps(r) + "\n" for r in states), encoding="utf-8")
    return [parse_record(r) for r in rows], rec, st


def test_an_option_joined_like_the_instruction_line_is_refused_on_a_sheet_only(tmp_path, capsys) -> None:
    records, rec, st = _choice_records(tmp_path)
    # `patch` and `pull` check their options through `sheet_options`, which still takes it.
    assert sheet_options(records) == ["No, escalate", "Yes"]
    with pytest.raises(SheetInputError, match=r"^option 'No, escalate' cannot be put on a sheet: its instruction line joins the options with ', '$"):
        build_sheet(records, {("d", f"i{i}"): "s" for i in range(12)}, 2, 2)
    capsys.readouterr()
    assert cli_main(["labelaudit", "sheet", str(rec), "--states", str(st), "-o", str(tmp_path / "s.md"), "-k", str(tmp_path / "k.json"),
                     "--top", "2", "--controls", "2"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "cannot be put on a sheet" in err
    assert not (tmp_path / "s.md").exists()


# --- the route ---------------------------------------------------------------------------------


def _request(client: tuple[str, int] | None, headers: list[tuple[str, str]]) -> Request:
    scope = {"type": "http", "method": "GET", "path": "/api/sheet", "query_string": b"", "client": client,
             "headers": [(k.lower().encode("latin-1"), v.encode("utf-8")) for k, v in headers]}
    return Request(scope)


ALLOWED = [
    (("127.0.0.1", 1), [("Host", "127.0.0.1:8765")]),
    (("::1", 1), [("Host", "localhost:8765")]),
    (("127.0.0.1", 1), [("Host", "LOCALHOST:8765")]),
    (("127.0.0.1", 1), [("Host", "localhost")]),
    (("127.0.0.1", 1), [("Host", "127.0.0.1:65535"), ("Accept", "*/*")]),
]
REFUSED = [
    ("client-testclient", ("testclient", 1), [("Host", "127.0.0.1:8765")]),
    ("client-remote", ("10.0.0.1", 1), [("Host", "127.0.0.1:8765")]),
    ("client-127-2", ("127.0.0.2", 1), [("Host", "127.0.0.1:8765")]),
    ("no-client", None, [("Host", "127.0.0.1:8765")]),
    ("no-host", ("127.0.0.1", 1), []),
    ("two-hosts", ("127.0.0.1", 1), [("Host", "127.0.0.1:8765"), ("Host", "127.0.0.1:8765")]),
    ("rebound-name", ("127.0.0.1", 1), [("Host", "evil.example:8765")]),
    ("port-0", ("127.0.0.1", 1), [("Host", "127.0.0.1:0")]),
    ("port-65536", ("127.0.0.1", 1), [("Host", "127.0.0.1:65536")]),
    ("port-plus", ("127.0.0.1", 1), [("Host", "127.0.0.1:+80")]),
    ("port-unicode", ("127.0.0.1", 1), [("Host", "localhost:١٢٣٤")]),
    ("ipv6-authority", ("::1", 1), [("Host", "[::1]:8765")]),
    ("userinfo", ("127.0.0.1", 1), [("Host", "user@127.0.0.1:8765")]),
    ("trailing-dot", ("127.0.0.1", 1), [("Host", "localhost.:8765")]),
    *[(f"proxy-{h.lower()}", ("127.0.0.1", 1), [("Host", "127.0.0.1:8765"), (h, "203.0.113.9")])
      for h in ("Forwarded", "Via", "X-Real-IP", "X-Forwarded-For", "X-Forwarded-Proto", "X-Forwarded-Host")],
]


@pytest.mark.parametrize("client, headers", ALLOWED)
def test_a_loopback_request_may_read_the_sheet(client, headers) -> None:
    assert sheet_request_allowed(_request(client, headers)) is True


@pytest.mark.parametrize("name, client, headers", REFUSED, ids=[r[0] for r in REFUSED])
def test_any_other_request_may_not(name, client, headers) -> None:
    assert sheet_request_allowed(_request(client, headers)) is False


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(set(), *(_keys(v) for v in value))
    return set()


def test_the_route_serves_the_sheet_and_nothing_of_the_key(made) -> None:
    app = create_app(workspace_dir=made["ws"]["records"].rsplit("/", 1)[0], sheet_path=made["sheet"])
    r = TestClient(app, **LOOPBACK).get("/api/sheet")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    body = r.json()["sheet"]
    sheet = read_sheet(made["form"])
    assert body == {
        "file_name": "blind_sheet.md",
        "sheet_sha256": hashlib.sha256(made["sheet"].read_bytes()).hexdigest(),
        "question_id": made["key_doc"]["question_id"],
        "header": sheet.header,
        "options": ["True", "False"],
        "items": [{"n": it.n, "state": it.state, "answer": None} for it in sheet.items],
        "pieces": list(sheet.pieces),
    }
    assert not _keys(r.json()) & {"label", "predicted", "score", "arm", "item_id", "dataset", "seed"}
    assert str(made["root"]) not in r.text


def test_the_route_refuses_the_test_clients_own_request_without_saying_what_it_saw(made) -> None:
    for app in (create_app(workspace_dir=made["root"], sheet_path=made["sheet"]), create_app(workspace_dir=made["root"])):
        r = TestClient(app).get("/api/sheet")
        assert (r.status_code, r.json()["detail"], r.headers["cache-control"]) == (403, SHEET_REFUSED, "no-store")
        assert "testserver" not in r.text and "testclient" not in r.text
    # Every other route is open to it, as before.
    assert TestClient(create_app(workspace_dir=made["root"], sheet_path=made["sheet"])).get("/api/overview").status_code == 200


def test_no_sheet_given_is_null_and_none_is_looked_for(made) -> None:
    # README's walkthrough leaves blind_sheet.md in the workspace; only `sheet_path` serves one.
    ws = made["root"] / "with-sheet"
    ws.mkdir()
    shutil.copy(made["sheet"], ws / "blind_sheet.md")
    assert TestClient(create_app(workspace_dir=ws), **LOOPBACK).get("/api/sheet").json() == {"sheet": None}


def _sheet_400(made, path: Path) -> str:
    r = TestClient(create_app(workspace_dir=made["root"], sheet_path=path), **LOOPBACK).get("/api/sheet")
    assert r.status_code == 400, r.text
    return r.json()["detail"]


def test_a_sheet_that_cannot_be_read_is_a_400_naming_it_once(made, tmp_path) -> None:
    missing = tmp_path / "gone.md"
    assert _sheet_400(made, missing) == f"{missing}: not found"
    folder = tmp_path / "folder.md"
    folder.mkdir()
    assert _sheet_400(made, folder).startswith(f"{folder}: cannot read: ")
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"\xff# Label blind review\n")
    assert _sheet_400(made, bad) == f"{bad}: not valid UTF-8: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"
    edited = tmp_path / "edited.md"
    edited.write_text(made["form"].replace("\nAnswer: \n", "\nAnswer: maybe\n", 1), encoding="utf-8")
    detail = _sheet_400(made, edited)
    assert detail.startswith(f"{edited}: item 1: the web reads an answer only as") and detail.count(str(edited)) == 1


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_a_fifo_sheet_is_a_400_not_a_hang(made, tmp_path) -> None:
    fifo = tmp_path / "sheet.fifo"
    os.mkfifo(fifo)
    assert _sheet_400(made, fifo) == f"{fifo}: not a regular file"


def test_a_crlf_sheet_reads_as_the_cli_reads_it(made, tmp_path) -> None:
    crlf = tmp_path / "crlf.md"
    raw = made["form"].replace("\n", "\r\n").encode("utf-8")
    crlf.write_bytes(raw)
    body = TestClient(create_app(workspace_dir=made["root"], sheet_path=crlf), **LOOPBACK).get("/api/sheet").json()["sheet"]
    assert body["sheet_sha256"] == hashlib.sha256(raw).hexdigest()
    assert body["pieces"] == list(read_sheet(made["form"]).pieces)


def test_a_byte_order_mark_an_editor_added_is_read_past(made, tmp_path, capsys) -> None:
    bom = tmp_path / "bom.md"
    raw = b"\xef\xbb\xbf" + made["form"].replace("\n", "\r\n").encode("utf-8")
    bom.write_bytes(raw)
    body = TestClient(create_app(workspace_dir=made["root"], sheet_path=bom), **LOOPBACK).get("/api/sheet").json()["sheet"]
    assert body["sheet_sha256"] == hashlib.sha256(raw).hexdigest()
    assert body["pieces"] == list(read_sheet(made["form"]).pieces) and len(body["items"]) == 12
    # The download is the sheet without the mark, and `estimate` scores it as it scores the file.
    done = tmp_path / "done.md"
    done.write_text(read_sheet(made["form"]).fill({n: "True" for n in range(1, 13)}), encoding="utf-8")
    capsys.readouterr()
    assert cli_main(["labelaudit", "estimate", str(done), "--key", str(made["key"])]) == 0


def test_each_request_reads_the_file_once_and_again(made, tmp_path) -> None:
    path = tmp_path / "sheet.md"
    path.write_text(made["form"], encoding="utf-8")
    client = TestClient(create_app(workspace_dir=made["root"], sheet_path=path), **LOOPBACK)
    first = client.get("/api/sheet").json()["sheet"]
    path.write_text(_answered(made["form"], {1: "False"}), encoding="utf-8")
    second = client.get("/api/sheet").json()["sheet"]
    assert first["sheet_sha256"] != second["sheet_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (first["items"][0]["answer"], second["items"][0]["answer"]) == (None, "False")


# --- s1 web --sheet ----------------------------------------------------------------------------


@pytest.fixture
def served(monkeypatch) -> list:
    import uvicorn

    calls: list = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: calls.append((app, host, port)))
    return calls


@pytest.mark.parametrize("host", ["0.0.0.0", "::1", "127.0.0.2", "192.168.1.5", "LOCALHOST"])
def test_web_refuses_a_sheet_on_any_other_address(made, served, capsys, host) -> None:
    capsys.readouterr()
    assert cli_main(["web", "--workspace", str(made["root"]), "--sheet", str(made["sheet"]), "--host", host]) == 1
    assert capsys.readouterr().err == f"s1: --sheet serves state text to this computer only: --host must be 127.0.0.1 or localhost, not {host!r}\n"
    assert served == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_web_refuses_a_sheet_it_could_read_once(made, served, capsys, tmp_path) -> None:
    fifo = tmp_path / "sheet.fifo"
    os.mkfifo(fifo)
    capsys.readouterr()
    assert cli_main(["web", "--workspace", str(made["root"]), "--sheet", str(fifo)]) == 1
    assert "review sheet is not a regular file" in capsys.readouterr().err
    assert served == []


def test_web_names_a_missing_sheet(made, served, capsys, tmp_path) -> None:
    capsys.readouterr()
    assert cli_main(["web", "--workspace", str(made["root"]), "--sheet", str(tmp_path / "none.md")]) == 1
    assert capsys.readouterr().err.startswith(f"s1: review sheet not found: {tmp_path / 'none.md'} (run `s1 labelaudit sheet` first)")
    assert served == []


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_web_serves_the_sheet_it_was_given(made, served, host) -> None:
    assert cli_main(["web", "--workspace", str(made["root"]), "--sheet", str(made["sheet"]), "--host", host, "--port", "8799"]) == 0
    app, bound, port = served[0]
    assert (bound, port) == (host, 8799)
    assert TestClient(app, **LOOPBACK).get("/api/sheet").json()["sheet"]["file_name"] == "blind_sheet.md"
    # Without --sheet, nothing is served there.
    assert cli_main(["web", "--workspace", str(made["root"])]) == 0
    assert TestClient(served[1][0], **LOOPBACK).get("/api/sheet").json() == {"sheet": None}


# --- the page ----------------------------------------------------------------------------------


def _served(form: str, name: str = "blind_sheet.md") -> dict:
    sheet = read_sheet(form)
    return {"file_name": name, "sheet_sha256": hashlib.sha256(form.encode()).hexdigest(), "question_id": sheet.question_id,
            "header": sheet.header, "options": list(sheet.options),
            "items": [{"n": it.n, "state": it.state, "answer": it.answer} for it in sheet.items], "pieces": list(sheet.pieces)}


def _js(expr: str, prelude: str = "") -> object:
    return _node(f"{prelude}\n{_all_functions()}\nprocess.stdout.write(JSON.stringify({expr}));")


@needs_node
@pytest.mark.parametrize("name", ["fresh", "v060", "answered", "crlf"])
def test_the_pages_fill_is_the_librarys(forms, made, name, tmp_path, capsys) -> None:
    sheet = read_sheet(forms[name])
    rng = random.Random(name)
    maps = [{n: rng.choice([None, *sheet.options]) for n in rng.sample(range(1, 13), rng.randint(0, 12))} for _ in range(12)]
    maps.append({n: sheet.options[n % 2] for n in range(1, 13)})
    got = _js(f"{json.dumps([{str(k): v for k, v in a.items()} for a in maps])}.map(a => fillSheet(sheet, a))",
              prelude=f"const sheet = {json.dumps(_served(forms[name]))};")
    assert got == [sheet.fill(a) for a in maps]
    done = tmp_path / "done.md"
    done.write_text(got[-1], encoding="utf-8")
    capsys.readouterr()
    assert cli_main(["labelaudit", "estimate", str(done), "--key", str(made["key"])]) == 0


@needs_node
def test_one_rule_decides_the_answer_shown_counted_searched_and_downloaded(forms) -> None:
    # Item 2 is answered in the file; the page cleared it and answered item 3.
    sheet = _served(forms["answered"])
    answers = {"2": None, "3": "False"}
    out = _js(
        "({two: reviewAnswer(sheet, answers, 2), three: reviewAnswer(sheet, answers, 3), five: reviewAnswer(sheet, answers, 5),"
        " count: reviewAnsweredCount(sheet, answers), next: reviewNextUnanswered(sheet, answers, 1),"
        " html: renderReview(sheet, answers, 2), file: fillSheet(sheet, answers)})",
        prelude=f"const sheet = {json.dumps(sheet)}; const answers = {json.dumps(answers)};",
    )
    assert (out["two"], out["three"], out["five"], out["count"], out["next"]) == (None, "False", "False", 2, 2)
    assert "Not answered yet" in out["html"] and "2 of 12 answered" in out["html"]
    assert "10 of 12 items are not answered yet" in out["html"]
    kept = read_sheet(out["file"])
    assert [it.answer for it in kept.items][:5] == [None, None, "False", None, "False"]


@needs_node
def test_next_unanswered_wraps_and_stops_when_none_is_left(forms) -> None:
    sheet = _served(forms["fresh"])
    all_but = {str(n): "True" for n in range(1, 13) if n != 4}
    out = _js(
        "[reviewNextUnanswered(sheet, {}, 12), reviewNextUnanswered(sheet, allBut, 9), reviewNextUnanswered(sheet, allBut, 4),"
        " reviewNextUnanswered(sheet, every, 1)]",
        prelude=f"const sheet = {json.dumps(sheet)}; const allBut = {json.dumps(all_but)};"
                f" const every = {json.dumps({str(n): 'True' for n in range(1, 13)})};",
    )
    assert out == [1, 4, None, None]


@needs_node
def test_a_stored_draft_keeps_only_what_can_answer_this_sheet(forms) -> None:
    sheet = _served(forms["fresh"])
    raws = ["null", "[]", '"x"', "{", json.dumps({"1": "maybe", "99": "True", "01": "True", " 2": "True", "1.0": "True",
                                                  "__proto__": "True", "2": None, "3": "True", "4": 1})]
    out = _js(f"{json.dumps(raws)}.map(r => reviewDraft(sheet, r))", prelude=f"const sheet = {json.dumps(sheet)};")
    assert out == [{}, {}, {}, {}, {"2": None, "3": "True"}]


@needs_node
def test_the_draft_survives_storage_that_throws_or_is_absent(forms) -> None:
    sheet = _served(forms["fresh"])
    throws = "const localStorage = {getItem() { throw new Error('denied'); }, setItem() { throw new Error('full'); }, removeItem() { throw new Error('x'); }};"
    assert _js("[readReviewDraft(sheet), (writeReviewDraft(sheet, {'1': 'True'}), 'ok')]", prelude=f"const sheet = {json.dumps(sheet)}; {throws}") == [{}, "ok"]
    # No localStorage at all (a ReferenceError).
    assert _js("readReviewDraft(sheet)", prelude=f"const sheet = {json.dumps(sheet)};") == {}


@needs_node
def test_a_draft_holds_only_what_differs_from_the_file(forms) -> None:
    sheet = _served(forms["answered"])
    store = "const log = []; const localStorage = {setItem(k, v) { log.push(['set', k, v]); }, removeItem(k) { log.push(['remove', k]); }};"
    out = _js(
        "(() => { let a = reviewWithAnswer(sheet, {}, 2, null); writeReviewDraft(sheet, a); const cleared = a;"
        " a = reviewWithAnswer(sheet, a, 2, 'True'); writeReviewDraft(sheet, a);"
        " return {cleared, back: a, log}; })()",
        prelude=f"const sheet = {json.dumps(sheet)}; {store}",
    )
    key = "s1ops:v1:sheet:" + sheet["sheet_sha256"]
    assert out == {"cleared": {"2": None}, "back": {}, "log": [["set", key, '{"2":null}'], ["remove", key]]}
    # A draft restored from storage with entries equal to the file (item 1 is unanswered there, item 2
    # answered True) is written back without them.
    restored = _js("(writeReviewDraft(sheet, {'1': null, '2': 'True', '3': 'False'}), writeReviewDraft(sheet, {'1': null}), log)",
                   prelude=f"const sheet = {json.dumps(sheet)}; {store}")
    assert restored == [["set", key, '{"3":"False"}'], ["remove", key]]


@needs_node
def test_the_keys(forms) -> None:
    sheet = _served(forms["fresh"])
    out = _js("['j', 'k', 'n', 'u', '1', '2', '3', '9', 'x', 'J'].map(k => reviewKeyAction(k, sheet))", prelude=f"const sheet = {json.dumps(sheet)};")
    assert out == [{"act": "next"}, {"act": "prev"}, {"act": "next-unanswered"}, {"act": "clear"},
                   {"act": "answer", "value": "True"}, {"act": "answer", "value": "False"}, None, None, None, None]


def _shell_run(steps: str, payload: dict) -> dict:
    """The shell around the review view: state, history, the view's request, a stub DOM (the focus
    and the live-DOM lookups are the browser check's)."""
    js = f"""
let STATE = null;
let viewSeq = {{}}, loadedSig = {{}}, viewData = {{}}, generation = 0, TSTATE = {{}}, policySeq = 0;
let GROUPS = [], groupsState = "ok", groupsError = "";
const els = {{}};
const el = id => (els[id] ??= {{id, innerHTML: "", textContent: "", value: "", hidden: false, disabled: false, attrs: {{}}, classList: {{toggle() {{}}}},
  setAttribute(k, v) {{ this.attrs[k] = v; }}, removeAttribute(k) {{ delete this.attrs[k]; }}, querySelector: () => null,
  querySelectorAll: () => [], focus() {{ out.focused = id; }}}});
const document = {{getElementById: el, querySelectorAll: () => []}};
const location = {{hash: "", pathname: "/", search: ""}};
const history = {{replaceState(s, t, url) {{ out.replaced.push(url); location.hash = url; }}, pushState(s, t, url) {{ out.pushed.push(url); location.hash = url; }}}};
const stored = {{}};
const localStorage = {{getItem: k => (k in stored ? stored[k] : null), setItem(k, v) {{ out.writes += 1; stored[k] = v; }}, removeItem(k) {{ out.removes += 1; delete stored[k]; }}}};
const pending = [];
const fetchJson = url => new Promise((resolve, reject) => pending.push({{url, resolve, reject}}));
const tick = () => new Promise(r => setTimeout(r, 0));
const payload = {json.dumps(payload)};
const out = {{replaced: [], pushed: [], writes: 0, removes: 0}};
const shown = () => (el("review-container").innerHTML.match(/Item (\\d+) of/) || [])[1] || null;
{_all_functions()}
STATE = {{...defaultState(), view: "review"}};
(async () => {{
{steps}
process.stdout.write(JSON.stringify(out));
}})();
"""
    return _node(js)


@needs_node
def test_moving_between_items_rerenders_without_a_request(forms) -> None:
    steps = """
  STATE = {...STATE, item: 5};
  runView("review");
  out.url = pending[0].url;
  pending.shift().resolve(payload);
  await tick(); await tick();
  out.first = shown();
  out.kept = Object.keys(viewData.review).sort();
  out.sigs = viewSig("review", {...STATE, item: 1}) === viewSig("review", {...STATE, item: 9});
  reviewAct("next", undefined, {});
  out.next = [shown(), STATE.item];
  reviewAct("goto", "12", {act: "review-goto", n: "12"});
  reviewAct("next", undefined, {});
  out.end = [shown(), STATE.item];
  // A cached visit to a link naming item 3, then a digit: it answers the item shown.
  STATE = {...STATE, view: "lint"};
  STATE = {...STATE, view: "review", item: 3};
  runView("review");
  out.cached = shown();
  reviewAct("answer", "False", {radio: true});
  out.answered = viewData.review.answers;
  out.requests = pending.length;
"""
    out = _shell_run(steps, {"sheet": _served(forms["fresh"])})
    assert out["url"] == "/api/sheet" and out["first"] == "5" and out["kept"] == ["answers", "sheet"]
    assert out["sigs"] is True
    assert out["next"] == ["6", 6] and out["end"] == ["12", 12]
    assert out["cached"] == "3" and out["answered"] == {"3": "False"}
    assert out["requests"] == 0 and out["writes"] == 1 and out["pushed"] == []
    assert out["replaced"][-1].endswith("item=12") and out["focused"] == "review-item-heading"


@needs_node
def test_an_item_past_the_end_shows_the_first_and_is_rewritten_only_on_the_review(forms) -> None:
    steps = """
  STATE = {...STATE, item: 99};
  runView("review");
  pending.shift().resolve(payload);
  await tick(); await tick();
  out.shown = shown();
  out.state = STATE.item;
  out.hash = location.hash;
  // Answered while another view is shown, the same request writes no hash.
  generation += 1;
  loadedSig = {};
  STATE = {...STATE, item: 50};
  runView("review");
  STATE = {...STATE, view: "lint"};
  pending.shift().resolve(payload);
  await tick(); await tick();
  out.away = [STATE.item, out.replaced.length];
"""
    out = _shell_run(steps, {"sheet": _served(forms["fresh"])})
    assert (out["shown"], out["state"], out["hash"]) == ("1", 1, "#/review")
    assert out["away"] == [50, 1]


@needs_node
def test_a_stale_answer_renders_and_keeps_nothing(forms) -> None:
    steps = """
  runView("review");
  generation += 1;
  pending.shift().resolve(payload);
  await tick(); await tick();
  out.html = el("review-container").innerHTML.includes("Item 1 of");
  out.kept = "review" in viewData;
"""
    assert _shell_run(steps, {"sheet": _served(forms["fresh"])}) == {
        "replaced": [], "pushed": [], "writes": 0, "removes": 0, "html": False, "kept": False}


@needs_node
def test_keys_act_only_on_the_review_and_never_in_a_text_field(forms) -> None:
    steps = """
  runView("review");
  pending.shift().resolve(payload);
  await tick(); await tick();
  const press = (key, extra = {}) => { let prevented = false; reviewKey({key, target: {tagName: "BODY"}, preventDefault() { prevented = true; }, ...extra}); return prevented; };
  out.ctrl = press("1", {ctrlKey: true});
  out.text = press("1", {target: {tagName: "INPUT", type: "text"}});
  out.select = press("1", {target: {tagName: "SELECT"}});
  out.past = press("3");
  out.radio = press("2", {target: {tagName: "INPUT", type: "radio"}});
  out.after = viewData.review.answers;
  STATE = {...STATE, view: "lint"};
  out.away = press("1");
  out.final = viewData.review.answers;
"""
    out = _shell_run(steps, {"sheet": _served(forms["fresh"])})
    assert (out["ctrl"], out["text"], out["select"], out["past"], out["radio"], out["away"]) == (False, False, False, False, True, False)
    assert out["after"] == out["final"] == {"1": "False"} and out["writes"] == 1


@needs_node
def test_no_sheet_and_no_context(forms) -> None:
    out = _shell_run('runView("review"); pending.shift().resolve({sheet: null}); await tick(); await tick();'
                     ' out.html = el("review-container").innerHTML; out.kept = "review" in viewData;', {})
    assert "s1 web --sheet PATH" in out["html"] and out["kept"] is False
    # The node harnesses call a loader with no context and no shell state.
    js = (f"const written = {{}}; const document = {{getElementById: () => written}};"
          f" const fetchJson = async () => ({json.dumps({'sheet': _served(forms['fresh'])})});\n{_all_functions()}\n"
          "loadReview().then(() => process.stdout.write(JSON.stringify(written.innerHTML)));")
    assert "Item 1 of 12" in _node(js)


@needs_node
def test_a_403_says_the_sheet_is_for_this_computer_only() -> None:
    js = ("const box = {innerHTML: ''}; const document = {getElementById: () => box};\n" + _all_functions() +
          "\nshowError('review-container', Object.assign(new Error('the review sheet is served to this computer only'), {status: 403}));"
          "\nprocess.stdout.write(JSON.stringify(box.innerHTML));")
    html = _node(js)
    assert "served to this computer only (a loopback address and Host, no proxy headers)" in html
    assert "fix the file named above" not in html


def test_only_the_review_requests_the_sheet_and_it_requests_nothing_else() -> None:
    # The overview's checks and every other view: no request names /api/sheet; the review's loader
    # makes one request, so it never prefetches the labels or predictions of /api/labelaudit.
    script = (Path(__file__).parent.parent / "src" / "s1ops" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert re.findall(r'["`]/api/sheet[^"`]*["`]', script) == ['"/api/sheet"']
    loader = script.split("    async function loadReview")[1].split("\n    }\n", 1)[0]
    assert loader.count("fetchJson(") == 1 and 'fetchJson("/api/sheet")' in loader


@needs_node
def test_a_cached_visit_past_the_end_shows_the_first_item_and_rewrites_the_link(forms) -> None:
    steps = """
  runView("review");
  pending.shift().resolve(payload);
  await tick(); await tick();
  STATE = {...STATE, view: "lint"};
  STATE = {...STATE, view: "review", item: 40};
  runView("review");
  out.shown = shown();
  out.state = STATE.item;
  out.requests = pending.length;
"""
    out = _shell_run(steps, {"sheet": _served(forms["fresh"])})
    assert (out["shown"], out["state"], out["requests"], out["replaced"]) == ("1", 1, 0, ["#/review"])


@needs_node
def test_a_sheet_the_page_cannot_render_is_not_kept(forms) -> None:
    steps = """
  runView("review");
  pending.shift().resolve({sheet: {...payload.sheet, items: 5}});
  await tick(); await tick();
  out.error = el("review-container").innerHTML.includes("ERROR</span>");
  out.kept = "review" in viewData;
"""
    out = _shell_run(steps, {"sheet": _served(forms["fresh"])})
    assert (out["error"], out["kept"]) == (True, False)


@needs_node
def test_a_radio_names_its_option_by_position_so_any_option_text_survives() -> None:
    # The HTML parser turns a NUL in an attribute into U+FFFD (measured in Chrome), and `read_sheet`
    # accepts such an option; the radio carries its index, never the text.
    form = ("# Label blind review\n\nChoose one answer for each item: a\u0000b, c. Write it after `Answer:`.\n\n"
            "## 1\n\n> \n> first line blank\n\nAnswer: \n")
    sheet = _served(form)
    assert sheet["options"] == ["a\u0000b", "c"] and sheet["items"][0]["state"] == "\nfirst line blank"
    steps = """
  runView("review");
  pending.shift().resolve(payload);
  await tick(); await tick();
  const html = el("review-container").innerHTML;
  out.attrs = (html.match(/<input [^>]*>/g) || []).map(t => t.includes("\\u0000") || t.includes("a\\u0000b"));
  out.indexed = html.includes('data-i="0" value="0"') && html.includes('data-i="1" value="1"');
  out.pre = html.includes('<pre class="review-state">\\n\\nfirst line blank</pre>');
  reviewAct("answer", reviewOption("0"), {act: "review-answer", i: "0"});
  out.answer = viewData.review.answers["1"];
  out.file = fillSheet(viewData.review.sheet, viewData.review.answers);
  out.bad = [reviewOption("2"), reviewOption("x"), reviewOption("-1")];
"""
    out = _shell_run(steps, {"sheet": sheet})
    assert out["attrs"] == [False, False] and out["indexed"] is True and out["pre"] is True
    assert out["answer"] == "a\u0000b"
    assert read_sheet(out["file"]).items[0].answer == "a\u0000b"
    assert out["bad"] == [None, None, None]
