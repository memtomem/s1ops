"""A reader that leaves early is not the command's failure, and never means exit 120.

`console_main` owns this. `main` is a library call, so it leaves the process's stdout alone.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import s1ops
from s1ops.cli import console_main
from s1ops.cli import main as cli_main
from s1ops.demo import generate_workspace

_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict[str, str]:
    return generate_workspace(tmp_path_factory.mktemp("ws"))


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUNBUFFERED", "S1_DEBUG")}
    return {**env, "PYTHONPATH": str(Path(s1ops.__file__).parents[1]), **extra}


def _closed_pipe(argv: list[str], *, setup: str = "pass", **env: str) -> subprocess.CompletedProcess:
    """The read end is closed before the child starts, so its first write finds no reader."""
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        return subprocess.run(
            [sys.executable, "-c", f"import sys; {setup}; from s1ops.cli import console_main; console_main()", *argv],
            stdout=write_end, stderr=subprocess.PIPE, text=True, env=_env(**env), timeout=120,
        )
    finally:
        os.close(write_end)


def test_both_commands_run_the_console_entry() -> None:
    scripts = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts == {"s1": "s1ops.cli:console_main", "s1ops": "s1ops.cli:console_main"}


def test_a_finished_command_keeps_exit_0(ws) -> None:
    done = _closed_pipe(["audit", ws["records"]])
    assert (done.returncode, done.stderr) == (0, "")


def test_lint_keeps_its_failure_exit(tmp_path) -> None:
    questions = tmp_path / "q.json"
    questions.write_text("{}", encoding="utf-8")
    done = _closed_pipe(["lint", "check", str(questions)])
    assert (done.returncode, done.stderr) == (1, "")


def test_drift_keeps_exit_4(ws) -> None:
    done = _closed_pipe(["drift", ws["records"], ws["records_canary"], "--fail-on-critical"])
    assert (done.returncode, done.stderr) == (4, "")


@pytest.mark.parametrize("argv", [["fit", "-h"], ["--version"]], ids=["help", "version"])
def test_argparse_exits_keep_exit_0(argv) -> None:
    done = _closed_pipe(argv)
    assert (done.returncode, done.stderr) == (0, "")


@pytest.fixture(scope="module")
def ranked_past_the_buffer(ws, tmp_path_factory) -> list[str]:
    """A `labelaudit rank` whose output is larger than the child's stream buffer (8 KiB through 3.13,
    128 KiB on 3.14), so its `print` writes, and meets a closed pipe, before the command returns: the
    demo records with one ranked item's id made longer than twice that buffer."""
    size = int(subprocess.run([sys.executable, "-c", "import io; print(io.DEFAULT_BUFFER_SIZE)"],
                              capture_output=True, text=True, check=True, env=_env(), timeout=120).stdout)
    lines = Path(ws["records"]).read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert first["label"] is not None and first["model_served"] is not None
    first["item_id"] = "long-" + "x" * (2 * size)
    records = tmp_path_factory.mktemp("rank") / "records.jsonl"
    records.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n", encoding="utf-8")
    argv = ["labelaudit", "rank", str(records), "--top", "100000"]
    done = subprocess.run([sys.executable, "-c", "from s1ops.cli import console_main; console_main()", *argv],
                          capture_output=True, env=_env(), timeout=120)
    assert done.returncode == 0 and len(done.stdout) > 2 * size
    return argv


def test_a_pipe_that_breaks_mid_print_is_one_line_and_exit_1(ranked_past_the_buffer) -> None:
    # Larger than the stream's buffer, so `print` itself writes and meets the closed pipe.
    done = _closed_pipe(ranked_past_the_buffer)
    assert done.returncode == 1
    assert done.stderr.strip().splitlines() == ["s1: [Errno 32] Broken pipe"]


def test_debug_mid_print_is_a_traceback_and_exit_1_not_120(ranked_past_the_buffer) -> None:
    done = _closed_pipe(ranked_past_the_buffer, S1_DEBUG="1")
    assert done.returncode == 1
    assert "BrokenPipeError" in done.stderr and "Exception ignored" not in done.stderr


def test_debug_on_a_line_buffered_stdout_is_a_traceback_and_exit_1_not_120() -> None:
    # A write fails with its text still buffered, so without a cleanup the shutdown flush fails too.
    done = _closed_pipe(["lint", "rules"], setup="sys.stdout.reconfigure(line_buffering=True)", S1_DEBUG="1")
    assert done.returncode == 1
    assert "BrokenPipeError" in done.stderr and "Exception ignored" not in done.stderr


def test_stdout_closed_after_startup_never_reaches_120(ws) -> None:
    script = "import os; from s1ops.cli import console_main; os.close(1); console_main()"
    done = subprocess.run([sys.executable, "-c", script, "audit", ws["records"]], capture_output=True, text=True, env=_env(), timeout=120)
    assert (done.returncode, done.stderr) == (1, "s1: [Errno 9] Bad file descriptor\n")


_FLUSH_FAILS = (
    "import io, sys\n"
    "class Stream(io.StringIO):\n"
    "    def flush(self):\n"
    "        raise {error}\n"
    "sys.stdout = Stream()\n"
    "from s1ops.cli import console_main\n"
    "console_main()\n"
)


@pytest.mark.parametrize(
    "error, code, stderr",
    [
        pytest.param("BrokenPipeError(32, 'Broken pipe')", 0, "", id="reader-left"),
        pytest.param("OSError(28, 'No space left on device')", 1, "s1: [Errno 28] No space left on device\n", id="disk-full"),
    ],
)
def test_a_stream_without_a_descriptor_survives_shutdown(error, code, stderr) -> None:
    script = _FLUSH_FAILS.format(error=error)
    done = subprocess.run([sys.executable, "-c", script, "--version"], capture_output=True, text=True, env=_env(), timeout=120)
    assert (done.returncode, done.stderr) == (code, stderr)


@pytest.mark.parametrize("dev_mode", ["", "1"], ids=["normal", "dev-mode"])
def test_a_replaced_buffered_stream_leaves_no_finalizer_diagnostic(dev_mode) -> None:
    done = _closed_pipe(
        ["--version"],
        setup="import io; sys.stdout = io.TextIOWrapper(io.BufferedWriter(io.FileIO(1, 'w', closefd=False)))",
        PYTHONDEVMODE=dev_mode,
    )
    assert (done.returncode, done.stderr) == (0, "")


def test_cleanup_needs_no_free_descriptor() -> None:
    script = (
        "import os, resource\n"
        "from s1ops.cli import console_main\n"
        "hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE, (64 if hard == resource.RLIM_INFINITY else min(64, hard), hard))\n"
        "held = []\n"
        "try:\n"
        "    while True:\n"
        "        held.append(os.open(os.devnull, os.O_RDONLY))\n"
        "except OSError:\n"
        "    pass\n"
        "console_main()\n"
    )
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        done = subprocess.run([sys.executable, "-c", script, "--version"], stdout=write_end, stderr=subprocess.PIPE, text=True, env=_env(), timeout=120)
    finally:
        os.close(write_end)
    assert (done.returncode, done.stderr) == (0, "")


class _Full(io.StringIO):
    """A stream on a full disk: nothing can be flushed, and it has no descriptor."""

    def flush(self) -> None:
        raise OSError(28, "No space left on device")


def _console(argv: list[str], monkeypatch) -> int:
    monkeypatch.setattr("sys.argv", ["s1", *argv])
    with pytest.raises(SystemExit) as exited:
        console_main()
    return exited.value.code


def test_another_write_failure_on_success_is_exit_1(ws, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdout", _Full())
    assert _console(["audit", ws["records"]], monkeypatch) == 1
    assert capsys.readouterr().err.strip().splitlines() == ["s1: [Errno 28] No space left on device"]


@pytest.mark.parametrize("argv", [["fit", "-h"], ["--version"]], ids=["help", "version"])
def test_a_write_failure_on_argparse_output_is_exit_1(monkeypatch, capsys, argv) -> None:
    monkeypatch.setattr("sys.stdout", _Full())
    assert _console(argv, monkeypatch) == 1
    assert capsys.readouterr().err.strip().splitlines() == ["s1: [Errno 28] No space left on device"]


def test_a_write_failure_is_reported_once(ws, monkeypatch, capsys) -> None:
    class Failing(_Full):
        def write(self, text: str) -> int:
            raise OSError(28, "No space left on device")

    monkeypatch.setattr("sys.stdout", Failing())
    assert _console(["audit", ws["records"]], monkeypatch) == 1
    assert capsys.readouterr().err.strip().splitlines() == ["s1: [Errno 28] No space left on device"]


def test_a_refusal_keeps_its_own_line_over_a_write_failure(tmp_path, monkeypatch, capsys) -> None:
    missing = tmp_path / "missing.jsonl"
    monkeypatch.setattr("sys.stdout", _Full())
    assert _console(["audit", str(missing)], monkeypatch) == 1
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1 and str(missing) in lines[0]


def test_a_usage_error_keeps_exit_1(monkeypatch, capsys) -> None:
    assert _console(["fit"], monkeypatch) == 1
    assert "s1: the following arguments are required" in capsys.readouterr().err


def test_a_closed_stdout_descriptor_is_not_an_error(ws, monkeypatch) -> None:
    monkeypatch.setattr("sys.stdout", None)
    assert _console(["audit", ws["records"]], monkeypatch) == 0


def test_a_normal_run_prints_and_exits_0(ws, monkeypatch, capsys) -> None:
    assert _console(["audit", ws["records"]], monkeypatch) == 0
    assert capsys.readouterr().out.startswith("{")


def test_main_as_a_library_leaves_stdout_alone(ws, monkeypatch) -> None:
    """An embedding process keeps its stream: no flush, no touching its descriptor, and (Python 3.14's
    argparse colors help by default, deciding by asking `sys.stdout`) no question about it to choose
    a color."""

    class Watching(io.StringIO):
        flushes = 0

        def flush(self) -> None:
            Watching.flushes += 1

        def fileno(self) -> int:
            raise AssertionError("main() must not reach for the descriptor")

        def isatty(self) -> bool:
            raise AssertionError("main() must not ask whether the stream is a terminal")

    # An environment in which 3.14 would ask: these settle the color without looking at the stream.
    monkeypatch.setenv("TERM", "xterm-256color")
    for name in ("NO_COLOR", "FORCE_COLOR", "PYTHON_COLORS"):
        monkeypatch.delenv(name, raising=False)
    stream = Watching()
    monkeypatch.setattr("sys.stdout", stream)
    assert cli_main(["audit", ws["records"]]) == 0
    assert stream.getvalue().startswith("{")
    assert Watching.flushes == 0


def test_main_as_a_library_does_not_repoint_a_descriptor_on_a_broken_pipe(ws) -> None:
    script = (
        "import os, sys\n"
        "from s1ops.cli import main\n"
        "r, w = os.pipe(); os.close(r)\n"
        "host = sys.stdout\n"
        "sys.stdout = os.fdopen(w, 'w', buffering=1)\n"
        "code = main(['labelaudit', 'rank', sys.argv[1], '--top', '100000'])\n"
        "broken, sys.stdout = sys.stdout, host\n"
        "print('host still prints', code)\n"
        "try:\n"
        "    broken.close()\n"
        "except OSError:\n"
        "    pass\n"
    )
    done = subprocess.run([sys.executable, "-c", script, ws["records"]], capture_output=True, text=True, env=_env(), timeout=120)
    assert done.stdout.strip() == "host still prints 1"
