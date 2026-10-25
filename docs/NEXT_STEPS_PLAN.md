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
  - [x] PyPI·TestPyPI 이름 확보 (2026-10-25): 코드 없는 `s1ops 0.0.1`을 올리고 두 곳 모두 yank. pip는 yank된 버전을 건너뛰지만 `uv`는 고정하지 않은 `s1ops`에도 0.0.1을 고르므로, 0.2.0 전까지 uv 사용자는 빈 패키지를 받음
  - [x] `.github/workflows/ci.yml` 추가 (2026-10-25, PR #2 run 40169216043 녹색: 3.11–3.13 각 92 passed, `package` job `release_check: dist ok`):
    - Python 3.11–3.13 매트릭스 (`eval/judge.py`가 쓰는 `cohen_kappa_score(replace_undefined_by=...)`가 scikit-learn 1.9부터 있고, 1.9는 Python 3.11 이상 필요), `uv sync --all-extras --locked` (lock이 낡으면 실패)
    - `uv run pytest -v` (100% 오프라인 무의존성 검증)
    - `examples/data/`는 gitignore된 생성물이므로 `uv run python examples/make_synthetic.py`로 먼저 만든 뒤 `uv run s1 lint check examples/data/questions.json` 스모크 체크 (데모 문항에 의도된 경고가 있어 `--strict` 없이 실행)
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
  - [ ] 첫 실제 릴리스(0.2.0) 전 체크리스트:
    - README에 인덱스 사용자용 설치 안내(`pip install s1ops`)를 클론 기반 데모와 분리해 추가. 지금 넣으면 yank된 빈 0.0.1로 안내하게 됨
    - 저장소 공개 여부와 시점 결정 (공개 시 `attestations: false` 제거)

---

### Track 2. 라이브 HTTP 백엔드 어댑터 (`HttpSystemOneBackend` & `OpenAIDecisionsBackend`)

- **목표**: `src/s1ops/monitor/runner.py`의 `Backend` 프로토콜을 구현하는 실제 HTTP 클라이언트 어댑터를 추가하여, `s1 probe`가 모의 백엔드(`DeterministicMockBackend`)뿐 아니라 라이브 API 엔드포인트도 예산 상한(`cap_usd`) 하에서 안전하게 호출할 수 있도록 확장합니다.
- **대상 파일**:
  - `src/s1ops/monitor/backends.py` (신규)
  - `src/s1ops/monitor/runner.py` 및 `src/s1ops/cli.py` (`--backend mock|systemone|openai` 플래그 추가)
- **세부 설계**:
  - [ ] **`HttpSystemOneBackend`**:
    - 엔드포인트: `POST {base_url}/v1/systemone` (기본 환경변수 `TYPESAFE_BASE_URL`, `TYPESAFE_API_KEY`)
    - 요청 페이로드: 정규화된 `parse_wire_question(question)` + 런타임 메모리 내 `state` 전달 (디스크 레코드에는 `state` 미저장, `state_sha256`만 기록)
    - 응답 매핑: 실제 서빙된 `model` 헤더/필드를 `model_served`로 캡처하여 무통보 모델 교체 감지와 직결
  - [ ] **`OpenAIDecisionsBackend`**:
    - 엔드포인트: `POST {base_url}/v1/decisions` (기본 환경변수 `OPENAI_BASE_URL`, `OPENAI_API_KEY`)
    - 요청 페이로드: `to_openai_decision(question)` 변환기로 `noul` → `predicate`, `choice` → `choice` 자동 매핑
  - [ ] **재시도 및 원장 정산 안전성**:
    - 429/5xx 발생 시 지수 백오프 후 최종 실패 시 `ledger.settle(res, actual_usd=0.0)` 및 `no_response` 레코드(`model_served=None`, `error_code=...`) 기록

---

### Track 3. `DecisionGate`의 `choice` / `score` 런타임 평가 확장

- **목표**: 현재 `evaluate_noul()`만 지원하는 `src/s1ops/runtime/gate.py`의 `DecisionGate`에 다지선다(`choice`) 및 순서형 루브릭(`score`) 런타임 보정·게이팅 메서드를 추가합니다.
- **대상 파일**:
  - `src/s1ops/runtime/gate.py`
  - `src/s1ops/cli.py` (`s1 gate`에서 `--probs '{"A":0.7,"B":0.3}'` 지원)
  - `tests/test_drift_runner_and_gate.py`
- **세부 설계**:
  - [ ] **`DecisionGate.evaluate_choice(...)`**:
    - 입력: `question_id`, `model_served`, `probs_raw: dict[str, float]`, `classes: Sequence[str]`
    - 처리: `fits.json`에 해당 그룹의 다중클래스 `temperature` 피팅이 있으면 `transform("temperature", "choice", fit.params, P)` 적용 → 최대 확률 선택지 `top_label`과 신뢰도 `conf = max(p_eff)` 산출
    - 라우팅: `conf >= policy.threshold`이면 `SELECT:<option>` 확정(`escalated=False`), 미달이거나 `threshold is None`이면 `ESCALATE`(`escalated=True`)
  - [ ] **`DecisionGate.evaluate_score(...)`**:
    - 입력: `question_id`, `model_served`, `probs_raw: dict[str, float]` (`"0"`..`"L-1"`)
    - 처리: 사후 온도 보정 적용 후 기대 점수 $\mathbb{E}[S] = \sum_{k=0}^{L-1} k \cdot p_k$ 및 최빈 등급 신뢰도 산출 → 임계값 기반 자동 통과/에스컬레이션 판정

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
