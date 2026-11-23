# Contributing to s1ops

Thank you for your interest in contributing to s1ops!

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development Setup

```bash
# Install (requires Python 3.11+ and uv; Node.js too, or the web page's renderer tests are skipped)
uv sync --all-extras

# Run tests: fully offline, no API keys (a few backend tests talk to an HTTP server on 127.0.0.1)
uv run pytest

# A seeded demo workspace to try the CLI and the web workbench on
uv run s1 demo examples/data
uv run s1 web --workspace examples/data --host 127.0.0.1 --port 8765
```

No linter or type checker is configured. CI runs the locked suite on Python 3.11–3.14 and
`scripts/release_check.py` on the built wheel and sdist.

## Project Structure

- `src/s1ops/` — the package; `src/s1/` only re-exports it so `import s1` works.
- `src/s1ops/contracts/` — the data contracts every other module reads (questions, records, policies).
- `src/s1ops/cli.py` — every `s1` subcommand, in one argparse file.
- `src/s1ops/web/` — the local web workbench (`s1 web`); the page is `web/static/index.html`.
- `tests/` — the suite; fixtures build their workspace with `s1ops.demo.generate_workspace` into `tmp_path`.
- `docs/` — design notes (in Korean).

## Pull Request Guidelines

1. Create a feature branch from `main`.
2. Keep changes focused: one feature or fix per pull request.
3. Add tests for new behavior. A new CLI argument needs `help=`; a user-facing failure is one
   `s1: <message>` line with the exit code listed in README's *CLI Exit Codes*.
4. Make sure `uv run pytest` passes.
5. Write a commit message that says why, in the Conventional Commits form the history uses
   (`feat(web): ...`, `fix(registry): ...`).
6. Sign the CLA on your first pull request (see below).

Features marked **beta** in README's *Stability* section can still change shape; if your change
touches one, say so in the pull request.

## Contributor License Agreement (CLA)

Before we can merge your first pull request, you need to sign the
[Contributor License Agreement](CLA.md). You sign by replying on your pull request with:

> I have read the CLA Document and I hereby sign the CLA

You only need to sign once per GitHub account.

The CLA is adapted from the Apache Software Foundation Individual Contributor License Agreement
with one additional section covering future licensing rights. This preserves DAPADA Inc.'s
ability to adopt different license terms for the Work in the future (for example, a
dual-licensing arrangement) without needing to re-collect consent from every contributor. The
CLA does not change the current license of the Work, which remains Apache License 2.0.

For questions about the CLA, contact contact@dapada.co.kr.

## Reporting Issues

Open an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version, s1ops version)

Never paste item `state` text, API keys or a server's raw responses into an issue, and check any
record or metadata you attach for text copied from an item: s1ops identifies items without their
state text, and a bug report should too. A failing record can be described by its
`(dataset, item_id)` and `state_sha256`.

Security issues go to [SECURITY.md](SECURITY.md), not to a public issue.
