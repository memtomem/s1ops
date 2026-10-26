# `s1ops` 후속 확장 계획서 (Phase 4 & Ecosystem Integration Plan)

> **현재 상태 요약 (`v0.2.0`, Commit `2fa8405`)**
> - **완료된 범위 (Phase 1 ~ Phase 3)**:
>   - 벤더 중립 계약(`questions.py`, `records.py`, `schemas.py`) 및 예산 안전 원장(`ledger/journal.py`)
>   - 정적 문항 린터(`JL000`–`JL060`) 및 Noul/Choice Twin · Inverse 프로브 일관성 감사(`lint/`)
>   - 보정 전/후 평가(`eval/metrics.py`, `audit.py`), 다중 백엔드 Shadow 비교(`compare.py`), 루브릭 심사자 합의도(`judge.py`)
>   - 사후 보정 컴파일러(`improve/calibrate.py`), 임계값/에스컬레이션 밴드 파레토 튜너(`policy.py`), 슬라이스 분석(`slices.py`), 블라인드 라벨 감사(`labelaudit.py`)
>   - 분포/모델 드리프트 모니터(`monitor/drift.py`), 예산 안전 프로브 러너(`monitor/runner.py`), 런타임 이진 게이트(`runtime/gate.py`)
>   - `safe-v1` Langfuse / Arize Phoenix 익스포터(`bridges/`), 6-탭 로컬 웹 워크벤치(`web/`), 통합 CLI(`cli.py`), 19개 오프라인 테스트 전원 통과

---

## 남은 후속 작업 (4대 트랙)

### Track 1. GitHub 원격 저장소(`memtomem/s1ops`) 연결 및 배포 파이프라인 구성

- **목표**: 로컬 Git 저장소(`/Users/pdstudio/Work/s1ops`)를 `memtomem/s1ops` 원격 저장소와 연결하고 CI 테스트 워크플로우를 구성합니다.
- **세부 작업**:
  - [x] 비공개 저장소 `memtomem/s1ops` 생성 및 `main` Push (2026-10-25). `origin`은 SSH alias `git@github.com-memtomem:memtomem/s1ops.git`; `gh` 활성 계정이 `tsdata`이므로 `gh` 호출은 `GH_TOKEN=$(gh auth token --user memtomem)`로 실행
  - [x] PyPI·TestPyPI 이름 확보 (2026-10-25): 코드 없는 `s1ops 0.0.1`을 올리고 두 곳 모두 yank. pip는 yank된 버전을 건너뛰지만 `uv`는 고정하지 않은 `s1ops`에도 0.0.1을 고르므로, 첫 실제 릴리스 전까지 uv 사용자는 빈 패키지를 받음
  - [x] `.github/workflows/ci.yml` 추가 (2026-10-25, PR #2 run 40169216043 녹색: 3.11–3.13 각 92 passed, `package` job `release_check: dist ok`):
    - Python 3.11–3.13 매트릭스 (`eval/judge.py`가 쓰는 `cohen_kappa_score(replace_undefined_by=...)`가 scikit-learn 1.9부터 있고, 1.9는 Python 3.11 이상 필요), `uv sync --all-extras --locked` (lock이 낡으면 실패)
    - `uv run pytest -v` (100% 오프라인 무의존성 검증)
    - `examples/data/`는 gitignore된 생성물이므로 `uv run s1 demo examples/data`로 먼저 만든 뒤 `uv run s1 lint check examples/data/questions.json` 스모크 체크 (데모 문항에 의도된 경고가 있어 `--strict` 없이 실행)
    - `package` job: `uv build` → `twine check --strict` (publish action과 같은 twine 7.0.0) → `scripts/release_check.py dist`가 sdist 목록이 `[tool.hatch.build.targets.sdist] include` 아래의 `git ls-files` 목록(+ Hatch가 강제로 넣는 `.gitignore`·`PKG-INFO`)과 정확히 같은지(일반 파일만, 링크 불가) 확인하고, wheel·sdist를 각각 checkout 밖의 새 venv에 설치해 버전·`import s1` 별칭·`web/static/index.html`을 확인
  - [x] `.github/workflows/release.yml` 추가: 토큰 대신 GitHub Actions trusted publishing
    - 수동 `workflow_dispatch`만 받음 (입력 `index`: `testpypi` | `pypi`). dispatch 자체가 공개 시점을 정하는 관문
    - `guard`: ref가 `refs/tags/vX.Y.Z`이고 X.Y.Z가 `pyproject.toml` version·`s1ops.__version__`과 같아야 함
    - `ci`: 같은 커밋에서 `ci.yml` 전체를 재사용(`workflow_call`), publish job은 checkout 없이 그 run의 `dist` artifact만 업로드. `id-token: write`는 publish job에만 부여
    - `attestations: false`: 저장소가 비공개인 동안 끔. 서명 단계가 업로드 전에 저장소·워크플로·ref·커밋·run URI를 공개 Rekor 로그에 영구 기록하기 때문 (업로드가 실패해도 남음). 저장소 공개 시 제거
  - [ ] 소유자 1회 설정:
    - GitHub environment `testpypi`, `pypi` 생성
    - TestPyPI·PyPI의 **기존** `s1ops` 프로젝트(yank된 0.0.1이 있으므로 pending publisher가 아님) → Manage → Publishing에서 GitHub publisher 추가: owner `memtomem`, repository `s1ops`, workflow `release.yml`, environment `testpypi` / `pypi`
  - 릴리스 절차: `vX.Y.Z` 태그 push → `GH_TOKEN=$(gh auth token --user memtomem) gh workflow run release.yml --ref vX.Y.Z -f index=testpypi` (`--ref`는 태그 이름도 받음). TestPyPI로 먼저 확인한 뒤 PyPI. 실패한 run을 publish job만 재실행하면 이전에 검증된 artifact를 다시 올림 (현재 태그를 새로 검증하지 않음)
  - [x] 첫 실제 릴리스 준비 (2026-10-25 소유자 결정: 패키지만 먼저 공개, 저장소는 비공개 유지. 버전은 0.2.0 대신 0.0.2로 결정. `v0.2.0` 태그로 한 TestPyPI 시도는 publisher 불일치로 업로드 전에 실패해 아무것도 공개되지 않음):
    - 데모 생성기를 패키지로 옮겨 `s1 demo OUT_DIR [--seed N]` 추가 (새 디렉터리나 빈 디렉터리에만 씀, `--force` 없음). `release_check.py`가 wheel·sdist 설치본에서 `s1 demo` + lint까지 실행
    - README를 PyPI용으로 재작성: `pip install s1ops` → `s1 demo` → 파이프라인. 저장소·`docs/` 링크, Mermaid, `$...$` 수식 제거 (PyPI는 렌더링하지 않음)
    - `[project.urls]` 제거, `release_check.py`가 METADATA·PKG-INFO에 `memtomem/s1ops`가 없는지 검사
  - [ ] 저장소 공개 시: `release.yml`의 `attestations: false`와 `release_check.py`의 `PRIVATE_REPO` 검사 제거, `[project.urls]` 복원, `docs/*.md`의 로컬 경로 정리

---

### Track 2. 라이브 HTTP 백엔드 어댑터 (`HttpSystemOneBackend` & `OpenAIDecisionsBackend`)

- **목표**: `src/s1ops/monitor/runner.py`의 `Backend` 프로토콜을 구현하는 실제 HTTP 클라이언트 어댑터를 추가하여, `s1 probe`가 모의 백엔드(`DeterministicMockBackend`)뿐 아니라 라이브 API 엔드포인트도 예산 상한(`cap_usd`) 하에서 안전하게 호출할 수 있도록 확장합니다.
- **대상 파일**: `src/s1ops/monitor/backends.py` (신규), `src/s1ops/monitor/runner.py`, `src/s1ops/cli.py` (`s1 probe --backend mock|systemone`)
- **세부 설계**:
  - [x] **`HttpSystemOneBackend`** (2026-10-25, 브랜치 `feat/live-systemone-backend`):
    - 직접 HTTP 구현 대신 `typesafe` extra의 `typesafe-sdk`를 감쌈. SDK 재시도는 끄고(`RetryPolicy(max_retries=0)`) 재시도는 `run_batch`가 수행해 시도마다 원장 예약을 따로 잡음
    - 요청: 문항 1개/요청, `questions={"q": question.wire_dict()}` (SDK가 raw dict를 받으므로 `question_id`와 같은 wire 형태가 그대로 감)
    - 응답 매핑: `model_served = response.model`. 응답에 USD가 없고 토큰 수만 있으므로 비용 = `usage.input_tokens × --usd-per-input-token` (출력 토큰 무료, 기본 단가 없음)
    - SDK는 `usage={}`, 빈 `answers`, 범위 밖 확률을 통과시키므로 백엔드가 답 타입·선택지·레벨·score 범위를 확인하고 만든 레코드를 `parse_record`로 검증
    - safe-v1: `BackendError` 메시지는 오류 코드 + 고정 문구만 (SDK 예외 문자열은 서버 본문을 최대 200자 포함하고, `request_id`도 서버 헤더). SDK 로그(DEBUG 요청 본문, INFO request ID, WARNING 미지의 answer type)도 서버가 되돌려준 state를 담을 수 있음: `s1` 프로세스에선 핸들러가 없어 출력되지 않지만(실측), 라이브러리 사용자가 `logging.basicConfig()` 등으로 핸들러를 붙이면 출력됨 → `infer` 호출 동안 `typesafe_sdk` 로거의 레코드를 버리는 ContextVar 필터(호출 밖 SDK 로그는 통과)
  - [x] **재시도 및 원장 정산 안전성** (원 설계의 "최종 실패 시 `settle(0.0)`"은 과금됐을 수 있는 요청을 0으로 기록해 cap을 무력화하므로 바꿈):
    - 응답 usage가 유효하면 답이 무효여도 `measured`; 요청이 확실히 안 나갔으면 `unsent`; 그 외(타임아웃·연결 오류·HTTP 오류 응답·예약~정산 사이 인터럽트)는 `unknown`(예약액 과금)
    - 타임아웃·연결 오류·408·429·5xx는 `--max-retries`까지 재시도, `Retry-After`/`retry-after-ms` 우선, 60초 초과 요청이면 재시도 중단. 최종 실패는 `model_served: null` + `error_code` 레코드
    - `run_batch`의 `min(측정, 예약)` 클램프 제거. `Ledger.settle`은 초과를 기록만 하고 다음 `reserve`에서 멈추므로, `run_batch`가 정산 직후 `ledger.overrun`을 확인해 그 레코드를 내보낸 뒤 `LedgerUnusable`
    - 레코드는 `on_record`로 즉시 기록·flush: 중간 정지 시 이미 비용을 낸 응답이 남음. `-o`가 원장·lock·입력과 같은 파일(심볼릭·하드링크·대소문자 무시 FS)이면 inode 비교로 거부
  - [x] **`OpenAIDecisionsBackend`** (2026-10-25, 브랜치 `feat/openai-decisions-backend`, `s1 probe --backend openai`):
    - 공식 SDK `openai`(extra `openai>=3.32,<4`: `decisions.create`는 3.26.0부터, `prompt_cache_options`는 3.32.0부터, 응답 필드는 3.26.0 이후 동일)를 SDK 재시도 없이 감쌈. SDK가 응답을 기본으로 검증하지 않으므로 `with_raw_response`로 원시 JSON을 받아 직접 검증
    - served model: API reference와 SDK 타입 모두 `Decision.model: str "The model used."` — 원래 막혀 있던 조건은 문서로 해결. 예시 값이 별칭 `gpt-6-luna`라 실제로 스냅샷 이름을 주는지는 미확인(아래)
    - prompt caching은 기본 켜짐(캐시 쓰기 과금, 적중 할인) → `prompt_cache_options={"mode": "explicit"}`에 breakpoint 없음으로 끔(문서: "the request does not use prompt caching"). `input_tokens × --usd-per-input-token`이 측정 비용, `cache_write_tokens > 0`이면 단가를 모르므로 `unknown`. state가 서버 캐시에 30분 남지 않음
    - 응답: predicate `probability` → `p`; choice·score `probabilities` 배열은 항목 수 = 선택지/레벨 수이고 값 집합이 일치해야 dict로 바꿈(중복이 dict에서 유효한 분포로 접히지 않게); `refusal` → `error_code: "refusal"` 레코드(측정 정산, 재시도 없음)
    - HTTP 오류의 usage는 SDK의 `e.body`(내부 `error` 객체만)가 아니라 전체 응답 JSON에서; 읽기 실패(HTML·깨진 UTF-8·미판독)는 usage 없음. `timeout=None`을 넘기면 SDK가 타임아웃을 끄므로 생략
    - `to_openai_decision`: score(`levels`) 추가, noul `criteria`·맨 객체 instructions(문자열·input part 배열은 통과)·개수 범위 밖은 `ValueError`(이전엔 noul criteria를 조용히 버려 같은 `question_id`로 다른 질문을 보냄). `parse_wire_question`: OpenAI score `levels` 지원, 중복 choice `value`는 `ValueError`(이전엔 덮어씀 — lint `JL000`로도 보고됨). `run_batch`는 백엔드의 `check_question`으로 첫 예약 전에 거부
    - 로그 필터를 전송 계층까지 확대(두 백엔드 공통): `httpx2`(INFO reason phrase), `httpcore2.*`(DEBUG 응답 헤더), `openai._base_client` 등 자식 로거. 테스트가 실제 요청 중 로그를 남긴 로거가 모두 목록에 있는지 확인(라이브러리 업그레이드 시 실패로 드러남)
  - [x] 실측 (2026-10-25, 합성 state, 질문 1개 `"Does the customer report a damaged item?"`, 요청 7건):

    | state | 글자 | `token_bound` 추정 | `/v1/systemone` (jev-1.13.0) | `/v1/decisions` (gpt-6-luna) |
    |---|---|---|---|---|
    | 짧은 영문 | 68 | 71 | 287 | 169 |
    | 긴 영문(×12) | 803 | 440 | 430 | 312 |
    | 짧은 한글 | 36 | 98 | 304 | 179 |

    - 청구 − 추정(관측): systemone +216 / −10 / +206, openai +98 / −128 / +81(표의 행 순서). 짧은 state에서만 모자라고 긴 state에선 남음 — `token_bound`의 글자당 상한(영문 0.5, 한글 2.0)이 실측(영문 ≈0.2, 한글 ≈0.65–0.8)보다 보수적이기 때문
    - 영문 735자 증가에 두 API 모두 +143토큰(≈0.195토큰/자) → 나머지는 요청당 고정분: systemone ≈274, openai ≈156토큰. 이 고정분에서 질문 추정(38)을 빼면 state가 비었을 때의 부족분 외삽치 ≈236 / ≈118(관측치가 아니라 3개 표본·한 질문에서의 외삽)
    - → `reserve_overhead_tokens` 백엔드 기본값: `HttpSystemOneBackend` 300, `OpenAIDecisionsBackend` 200(외삽치에 여유), mock 0. `run_batch(reserve_overhead_tokens=None)`과 CLI 기본값이 이를 씀. 예약 단가(`--worst-usd-per-token` 기본 2e-6)가 실제 단가(openai $1e-7)보다 훨씬 크면 이 값과 무관하게 초과가 없고, 예약 단가를 실제 단가에 가깝게 잡을 때 짧은 state의 매 요청 초과(`LedgerUnusable`)를 막음
    - served model: `/v1/systemone`은 별칭을 실제 버전으로 풀어 돌려줌(`jev-latest` → `jev-1.14.0`) → 무통보 교체 감지 동작. `/v1/decisions`는 별칭 `gpt-6-luna`를 그대로 돌려줌 → 별칭 뒤 스냅샷 교체는 `model_served`로 보이지 않고 분포 drift로만 드러남

---

### Track 3. `DecisionGate`의 `choice` / `score` 런타임 평가 확장

- **목표**: 현재 `evaluate_noul()`만 지원하는 `src/s1ops/runtime/gate.py`의 `DecisionGate`에 다지선다(`choice`) 및 순서형 루브릭(`score`) 런타임 보정·게이팅 메서드를 추가합니다.
- **대상 파일**:
  - `src/s1ops/runtime/gate.py`, `src/s1ops/runtime/routing.py` (신규)
  - `src/s1ops/cli.py` (`s1 gate`에서 `--probs '{"A":0.7,"B":0.3}'` 지원)
  - `src/s1ops/contracts/records.py` (`probability_total`), `src/s1ops/monitor/drift.py`, `src/s1ops/bridges/langfuse.py`, `src/s1ops/bridges/openinference.py`
  - `tests/test_gate_routing.py` (신규)
- **세부 설계**:
  - [x] **`DecisionGate.evaluate_choice(...)` / `evaluate_score(...)`** (2026-10-26, 브랜치 `feat/gate-choice-score`):
    - 입력: `question_id`, `model_served`, `probs_raw` (choice는 선택지→확률, score는 레벨(정수 또는 `"0"` 같은 10진 문자열)→확률). `classes` 인자는 두지 않음: choice는 호출자 매핑 순서, score는 오름차순
    - 검증: 유한한 `[0,1]` 숫자(bool 불가), 2개 이상, 합 `1 ± 0.02`, 같은 레벨로 접히는 score 키(`0`과 `"0"`, `"0"`과 `"00"`) 거부 → `ValueError`
    - 판정: `s1 policy tune`이 잰 값과 같은 top-label 확률이 `threshold` 이상이면 `SELECT:<label>`, 미만·`threshold: null`·**최상위 동률**이면 `ESCALATE`. 튜너는 동률을 그룹 첫 레코드의 선택지 순서로 깨는데 이 순서는 파일 순서와 `s1 apply`의 `sort_keys`로 바뀌어 런타임에 재현할 수 없음(소유자 결정: 동률은 escalate. 최상위 동률이면 확률 ≤ 0.5라 threshold ≤ 0.5인 정책에서만 차이)
    - 보정: `policy.calibrated`면 fit의 `temperature`를 `s1 apply` → `parse_record(use_calib=True)`와 같은 단계로 적용. **calibrated 정책인데 fit이 없으면 `ValueError`**(noul의 조용한 raw fallback은 그대로). choice/score `escalation_band` 정책은 `ValueError`
    - score: 기대 점수 `Σ k·p_k`는 보고만 하고 판정은 최빈 등급 확률로 함(초안의 "기대 점수 기반 판정"에서 바꿈: 튜너가 top-label 확률로 threshold를 고르므로)
    - `GateDecision`에 `probabilities`, `expected_score` 필드 추가(noul은 `null`)
    - primitive가 다른 질문에 호출하면 `ValueError`(exit 1). 이전엔 noul gate를 choice 질문에 부르면 `ModelServedMismatchError`(exit 3)
  - [x] **정규화를 `math.fsum`으로** (`records.probability_total`, 파서와 gate 공용): Python 3.11의 `sum()`은 순서에 따라 마지막 비트가 달라, 같은 분포가 백엔드 순서(raw)와 정렬 순서(`s1 apply` 출력)에서 다른 확률이 됨. 실측(3.11.14, 무작위 분포 2만 개, 순서만 섞음): 정규화한 최고 확률이 K=3에서 5.3%, K=5에서 19%, K=10에서 36% 달라짐(4자리 반올림 입력). 3.12+의 `sum()`은 보정 합산이라 0%이고 `fsum`과 12만 건 모두 같음 → 3.11에서 0.1.0 이하로 튜닝한 choice/score 정책은 재튜닝
  - [x] **공통 판정 규칙 `runtime/routing.py::route`** (소유자 결정: "공통 routing 함수로 통일"): gate, `s1 drift`의 escalation 비율, Langfuse/Phoenix exporter의 `policy_action`이 같은 함수를 씀. 이전엔 exporter와 drift가 각자 다르게 판정했음:
    - noul `error_budget`: exporter는 `p ≥ τ`면 ALLOW 아니면 BLOCK(ESCALATE 없음), `predicted = p ≥ τ`; drift는 `p < τ`를 escalate로 셈 → gate와 같은 `max(p, 1-p) ≥ τ`, `predicted = p ≥ 0.5`
    - noul band: exporter `predicted`가 `p ≥ band_hi`(band 안이면 항상 False) → gate와 같이 band 안에선 `p ≥ 0.5`
    - `threshold: null`: exporter는 SELECT/ALLOW·BLOCK, drift는 escalate 0 → ESCALATE
    - choice/score 동률: exporter는 사전순 가장 큰 라벨 선택 → ESCALATE
    - 판정 대상 분포(raw/보정)는 호출자가 고름. drift와 exporter가 이를 `policy.calibrated`에 맞춰 고르지 않는 점은 그대로 남음
    - export에 `routing_rule: "2"` 표시(Phoenix `s1.routing_rule`, Langfuse `metadata.routing_rule`, 정책으로 판정한 레코드만). `policy_sha256`은 같은데 판정이 달라질 수 있어 이전 export와 구분하기 위함
  - [x] **`parse_record`이 `calib.probabilities` 키가 raw `probabilities` 키와 다르면 `RecordError`**(줄 번호 포함): 이전엔 `s1 drift`(기본 `use_calib`) 등이 `KeyError` traceback으로 죽었고, exporter가 parse된 레코드로 판정하게 되면서 export에도 번질 뻔함
  - [x] **`s1 gate --probs '{"A":0.7,"B":0.3}'`**: `--p`와 상호 배타. 질문의 primitive와 맞지 않는 플래그는 exit 1로 올바른 플래그를 안내. `--probs`는 JSON 객체만, NaN·중복 키 거부
  - [x] **noul 확률 검사** (#15): `evaluate_noul`이 `p_raw`를 검사하지 않아 `1.5`는 ALLOW, NaN은 `error_budget`에서 BLOCK(모든 비교가 거짓)되고 band에선 `s1 gate`가 잘못된 JSON(`NaN`)을 출력했음 → choice/score와 같은 `_probability`로 거부(`ValueError`, exit 1). 정책 조회가 먼저라 `model_served` 불일치는 여전히 exit 3. 숫자 타입은 `numbers.Real`(NumPy 스칼라, `Fraction`)을 받고 bool은 거부, 결과는 plain `float`. 같은 결함이 exporter에도 있었음: noul `calib.p`를 `float()`로만 읽어 `"nan"`이 BLOCK, `1.5`가 ALLOW로 export됨 → choice/score처럼 `parse_record(use_calib=True)`로 다시 파싱해 `RecordError`(줄 번호). 정상 레코드의 export 값은 그대로
  - [x] **score 레벨 규칙 하나** (#13, #18): `records.score_level` / `score_levels`를 파서, gate, `compare_records`, exporter가 함께 씀. 키는 음이 아닌 int 또는 1~20자 ASCII 숫자 문자열, 값은 `0..2**53`, 같은 레벨을 가리키는 두 키는 거부
    - 이전: 파서가 `"0"`과 `"00"`을 한 레벨로 접어 확률 합이 줄어듦(`{"0":0.2,"00":0.3,"1":0.5}` → 합 0.8); 4300자리를 넘는 키는 줄 번호 없이 Python의 `int()` 메시지로 실패; 파서는 401자리 레벨도 받아 `s1 fit`·`policy tune`·`drift`·`export`가 그대로 썼고 `s1 audit`는 `OverflowError` traceback(gate는 #12부터 float 최댓값까지만 받음); `compare_records`에 dict를 넘기면 score 키를 검사하지 않음
    - 상한 `2**53`: 모든 레벨이 정확한 float이라 두 레벨이 같은 float로 반올림되지 않고(`float(2**53+1) == float(2**53)`), 레벨 합산이 overflow하지 않음(float 최댓값 근처 레벨 세 개로 MAE가 inf가 됨). gate의 #12 상한(float 최댓값)보다 좁아짐(미배포)
    - 20자 제한: 0을 길게 덧붙인 키를 정규화하는 대신 거부. `int()`가 20자를 넘는 문자열을 보지 않으므로 `sys.get_int_max_str_digits()` 설정과 무관
  - [x] **리포트도 정책이 튜닝된 분포로 판정** (#14, `routing_rule: "3"`): gate는 `policy.calibrated`일 때만 보정 분포로 판정하는데, exporter는 레코드에 `calib`가 있으면 항상, drift는 `--raw-only` 여부에 따라 골랐음(이슈 재현: export ESCALATE, gate ALLOW) → `routing.policy_distribution` / `route_record`가 정책 기준으로 고름
    - 보정 정책인데 `calib`가 없는 레코드는 거부(`RoutingInputError`, 줄 번호, drift는 reference/current 표시, web은 해당 파일을 지목한 400)
    - drift가 정책을 `question_id`로만 찾아 데모 canary의 `jev-1.14.0-rc1` 레코드를 `jev-1.13.0` 정책으로 판정(escalation 0.525)했음 → group key로 찾고, 정책이 없는 레코드는 비율에서 제외(없으면 `null`)
    - 정책으로 판정한 choice/score export의 `p_effective`·`brier`는 판정한(재정규화된) 분포 기준, gate와 같은 값(이전엔 저장된 값이라 합이 1이 아니면 달랐음: `{"A":0.6,"B":0.39}` → 0.6 대신 0.606061). `p_raw`·`p_calib`는 저장된 값 그대로
    - 데모 canary의 `jev-1.13.0` 80건에 `calib`(fits.json)를 넣음. raw 값은 그대로(calib 뺀 digest가 main과 같음). 현재 쪽이 보정 80 + raw 40이 되어 기본 PSI 값이 바뀜(혼합 비교 자체는 #20)
    - 후속(별도 이슈): drift/export의 `--fit`(calib 없는 레코드를 gate처럼 보정), in-sample 항목 처리, 모델이 섞인 파일에서 `s1 apply`가 exit 3으로 멈추는 문제
  - [x] **데모 choice/score 레코드** (#16): 별도 파일 `records_categorical_train.jsonl` / `records_categorical.jsonl`(choice `support_intent`, score `urgency` 각 150 + 150, `jev-1.13.0`)과 `fits_categorical.json` / `policy_categorical.json`. README walkthrough가 categorical fit → apply → audit → tune → `gate --probs` → export를 실제 정책으로 실행. `questions.json`에 score 질문 `urgency` 추가(`s1 probe` 호출 160건 증가). 별도 난수열이라 기존 noul 파일은 그대로(파일별 digest가 main과 같음). web workbench는 noul 파일만 읽음

---

### Track 4. TypeScript 경량 런타임 SDK (`@memtomem/sys1`) 및 `agent-harness` 연동

- **목표**: Python(`s1ops`)에서 오프라인으로 컴파일한 `fits.json`과 `policy.json`을 Node.js / Edge 런타임이나 `agent-harness` 생태계(`decision-plane`, `tracegraph`)에서 즉시 소비할 수 있도록 연결합니다.
- **세부 설계**:
  - [ ] **`packages/ts` (`@memtomem/sys1`) 스캐폴딩**: (npm `@memtomem` scope 소유 여부는 미확인; 2026-10-25 기준 `@memtomem/sys1`은 404, scope 내 패키지 0개)
    - 외부 무거운 의존성 없이 `fits.json`(`temperature` / `platt` / `isotonic`)과 `policy.json`(`s1-policy/1`, `policy_sha256` 검증)을 로드하는 TypeScript `DecisionGate` 클래스 구현
    - `import { System1, DecisionGate } from '@memtomem/sys1'` 지원
  - [ ] **`agent-harness` 연동 어댑터**:
    - `decision-plane`: 라우팅 정책 로더에서 `s1-policy/1` 아티팩트 로드 및 `model_served` 불일치 시 서킷 브레이커 작동
    - `tracegraph`: `s1 export phoenix`가 출력하는 `safe-v1` OpenInference 스팬(`s1.question_id`, `s1.policy_sha256`, `s1.escalated`) 수집 및 시각화 연동
  - [ ] **Closed-Loop Annotation Pull (`s1 labelaudit pull`)**:
    - Langfuse Annotation Queue에서 검수 완료된 항목을 가져와 로컬 골드셋 패치 → `s1 fit` → `s1 policy tune` 재컴파일 루프 자동화

---

### 후보: 로드 시 `policy_sha256` 대조

- 현재 `load_policies`는 `policy.json`에 저장된 `policy_sha256`을 버리고 재계산하므로, 파일 내 필드 변조는 호출자가 `expected_policy_sha256`을 따로 넘기지 않는 한 감지되지 않습니다.
- [ ] 저장된 지문과 재계산 지문을 로드 시 비교하고 불일치 시 `PolicyIntegrityError` (기존 파일 호환성: 지문 없는 행 처리 방침 결정 필요)

---

### 후보: 행 단위 입력 스키마 (features · probe · calibrated 분포)

파일 단위 손상(깨진 JSON, 객체가 아닌 행, 필드 누락·타입 오류)은 CLI exit 1 / 웹 400으로 보고되지만, 아래 세 경우는 형식상 유효한 행이 계산 단계에서 터져 CLI 트레이스백 또는 웹 500이 됩니다(2026-10-25 리뷰에서 재현, 브랜치 이전부터 존재). 리뷰 라운드마다 새 모양의 잘못된 입력이 발견되므로, 개별 가드 추가보다 features 행과 probe 레코드의 스키마를 정의하는 쪽이 맞습니다.

- [ ] **calibrated 분포 키 불일치** (`contracts/records.py:185`): `calib.probabilities` 키를 원본 `probabilities`의 클래스와 대조하지 않음. 원본 `{"yes","no"}` + calib `{"other","no"}` → `/api/slices`에서 `KeyError('yes')`. `parse_record`에서 `RecordError`로 거부해야 함
- [ ] **probe 역할-프리미티브 호환성** (`lint/lint.py` `audit_probe_records`): `probe_role="twin"` 레코드가 `"yes"` 키를 가진 choice라고 가정. 같은 항목의 `noul` 역할 레코드와 짝을 이룬 noul 레코드에 `twin` 태그 → `KeyError('probabilities')`. 역할별 프리미티브·필수 키 검증 필요
- [ ] **features 행 식별자 타입** (`cli.py` `_cmd_slice`, `web/app.py` `api_slices`): `{"dataset": [], "item_id": "x", ...}` → dict 키 생성에서 `TypeError`. CLI·웹 공용 features 행 검증기 필요
