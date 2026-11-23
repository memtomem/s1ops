# Security Policy

## Reporting Vulnerabilities

Please report security issues by email to **contact@dapada.co.kr**. Do NOT open public issues for
vulnerabilities. We aim to acknowledge security reports within 2 business days.

## Supported Versions

s1ops is beta (`0.x`); only the latest published minor receives security fixes.

| Version | Supported |
|---------|-----------|
| 0.14.x  | Yes       |
| < 0.14  | No        |

## Security Measures

### Item state text (`safe-v1`)

- The records, ledger entries, probe output and exports s1ops writes identify an item by
  `(dataset, item_id)` (records and exports also by `state_sha256`); s1ops adds no state text of
  its own to them. What a backend answers with, such as the model name it reports, is written as
  the backend sent it, so a server that echoes its input there puts that text in the record.
- Commands that copy input rows (`s1 apply`, `s1 labelaudit patch`) drop the row's top-level
  `state` key only: text you put under another key, or nested in metadata, is copied as it is. The
  ledger refuses a top-level `state` metadata key.
- A local states file is read by two commands only: `s1 probe` sends each state to the selected
  backend, and `s1 labelaudit sheet` writes the reviewed items' state text into the review sheet,
  which is meant for a human reviewer.
- Backend errors carry an error code, never the server's text (a server may echo the state).

### Web workbench (`s1 web`)

- Binds to `127.0.0.1` by default; it has no authentication, so do not expose it on another
  interface.
- `GET /api/sheet` (under `s1 web --sheet`), the only response that carries state text, answers
  only a loopback client that names `127.0.0.1` or `localhost` as its `Host` and sends no proxy
  header (`Forwarded`, `Via`, `X-Real-IP`, `X-Forwarded-*`); `--sheet` refuses any other `--host`.
- Every server-derived string is escaped before it reaches the page's markup.

### Network

- s1ops makes no network call except a live `s1 probe` (`--backend systemone` / `openai`), through
  that backend's SDK, to the endpoint configured by `--base-url`, else `TYPESAFE_BASE_URL` /
  `OPENAI_BASE_URL`, else the SDK's own hosted default. The SDK's HTTP client may follow a redirect
  the server answers with. `s1 labelaudit pull` reads a Langfuse response you fetched yourself.
- Spend on a live backend is reserved before each request and settled after it, under a file lock;
  a measured cost above its reservation halts the run.

### Data handling

- No pickle, no YAML, no `eval`/`exec`, no subprocess.
- The CLI's error line escapes every non-printable character, so a path or id in it cannot move the
  terminal's cursor or reorder its text.

## Best Practices

- Keep states files, review sheets and their secret keys out of version control and out of issues.
- `s1 probe` takes no API key option: the live backends read the SDKs' environment variables
  (`TYPESAFE_API_KEY`, `OPENAI_API_KEY`). Keep them out of committed files.
- Review a sheet with `s1 web --sheet` on the machine that holds it; do not put the workbench
  behind a proxy.
