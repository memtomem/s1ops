# `s1ops` 아키텍처 설계 및 단계별 구현 계획서 (Architecture & Roadmap)

> [!IMPORTANT]
> **핵심 설계 의사결정 (Resolved Design Decisions)**
> 1. **포지셔닝 (벤더 중립 범용 System 1 LLMOps)**: `/v1/systemone` (`jev-1.13.0`, `kev-1.0`, `laya`, `von`), OpenAI Decisions (`/v1/decisions`, `gpt-6-luna`), 로컬 분류/리랭커 헤드 등 **정형 확률 분포(`noul` / `choice` / `score`)를 반환하는 모든 초저지연 의사결정 모델**을 지원합니다.
> 2. **시각화 및 관제 연동 (익스포터 + 자체 로컬 웹 워크벤치)**: 기존 System 2 도구와의 통합을 위한 **Langfuse / Arize Phoenix (OpenInference `safe-v1`) 익스포터**와, System 1 전용 분석(Noise Floor 오버레이 Reliability Diagram, 임계값/에스컬레이션 밴드 비용 시뮬레이터, 문항 린트·프로브 일관성, 블라인드 라벨 감사, 백엔드 Shadow 비교)을 제공하는 **자체 5-탭 로컬 웹 UI (`s1 web`)**를 함께 제공합니다.
> 3. **저장소 및 네이밍**: `pandas-studio/s1ops` (`/Users/pdstudio/Work/s1ops`), PyPI 패키지 `s1ops`, Python 임포트 `import s1ops` 및 `import s1`, 글로벌 CLI `s1` 및 `s1ops`.
> 4. **단계별 우선순위**: 외부 유료 API 호출 없이 100% 오프라인으로 재현·검증 가능한 **오프라인 성능 평가 · 개선 컴파일러 파이프라인 + 익스포터 + 로컬 웹 워크벤치 (Phase 1 MVP)**를 우선 완결하고, 이후 온라인 드리프트 수집기(Phase 2)와 런타임 정책 핫리로드(Phase 3)로 확장합니다.

---

## 1. 벤더 중립 데이터 계약 (Vendor-Neutral Contracts)

기존 `jev-utils`의 `jev-record/1` 및 `jev-system1` 평가 하네스와 100% 호환되면서 다중 백엔드를 수용하는 4대 핵심 데이터 계약입니다.

### 1.1 `QuestionModel` & `question_id` (`src/s1ops/contracts/questions.py`)

| 프리미티브 | 의미 | 백엔드 매핑 (`/v1/systemone` · `/v1/decisions` · `local`) | 정답(`label`) 타입 | 응답 확률 구조 |
| --- | --- | --- | --- | --- |
| `noul` (`predicate`) | 단일 긍정 명제 참/거짓 판단 | TypeSafe `noul` · OpenAI `predicate` · Binary Head | `bool` (`true`/`false`) | `p`: $[0, 1]$ (`(1-p, p)`) |
| `choice` | 순서가 고정된 배타적 선택지 ($2 \le K \le 255$) | TypeSafe `choice` · OpenAI `choice` · Multiclass Head | `str` (옵션 키 중 하나) | `probabilities`: `{key: float}`, $\sum p = 1 \pm 0.02$ |
| `score` | 순서형 루브릭 등급 ($2 \le L \le 10$) | TypeSafe `score` · Ordinal/Logprob Head | `int` ($0 \dots L-1$) | `probabilities`: `{"0": float, ...}`, `score`: 기댓값 |

- **`question_id` 불변식**: 정규화된 `/v1/systemone` wire JSON(`type`, `instructions`, 옵션 순서가 보존된 `criteria`)의 SHA-256 앞 16자리(`q-<16hex>`). 선택지 순서나 문구가 한 글자라도 바뀌면 확률 분포가 달라지므로 새로운 `question_id`가 부여됩니다.

### 1.2 `Record` (`jev-record/1` & `s1-record/1`, `src/s1ops/contracts/records.py`)

- **엄격한 그룹 키 (`GroupKey`)**: `(question_id, model_served, primitive)`. 서로 다른 질문 문구, 선택지 순서, 또는 실제 서빙된 모델 버전(`model_served`)은 어떤 통계·보정·임계값·익스포터에서도 절대 합산(pool)하지 않습니다.
- **Zero-State 프라이버시 기본값 (`safe-v1`)**: 레코드 파일에는 원문 `state` 텍스트를 절대 기록하지 않으며 `(dataset, item_id)` 식별자와 `state_sha256` 해시만 유지합니다.
- **응답 무결성 검증**: `NaN`/`Inf`, JSON `bool`이 숫자로 둔갑한 경우, 확률 합이 $1 \pm 0.02$를 벗어난 경우 1-based 줄 번호와 함께 즉시 `RecordError`를 발생시킵니다.

### 1.3 `Fit` (`jev-calib-fit/2` & `s1-calib-fit/1`, `src/s1ops/improve/calibrate.py`)

- 그룹 `(question_id, model_served, primitive)`별 사후 확률 보정 파라미터(`temperature`, `platt`, `isotonic`)와 피팅에 사용된 모든 `[dataset, item_id]` 명단 및 원본 파일 `source_sha256`을 기록합니다.
- **In-Sample & Model-Mismatch 차단 (`s1 apply`)**:
  - 피팅에 사용된 `(dataset, item_id)`에 보정을 적용하려 하면 **Exit Code 2**로 즉시 거부합니다 (`--allow-in-sample` 명시 시에만 허용).
  - 피팅된 `model_served`와 다른 모델이 응답한 레코드에 적용하려 하면 **Exit Code 3**으로 즉시 거부합니다.

### 1.4 `PolicyEntry` (`s1-policy/1`, `src/s1ops/contracts/schemas.py`)

- **Mode A — `error_budget`**: 목표 오류 예산 `error_budget` (예: `0.10`) 및 선택적 Clopper-Pearson 유한표본 상한(`--use-cp-upper`) 하에서 동점 그룹(tie-group) 단위로 컴파일된 단일 임계값 `threshold` (`None` = 전량 위임).
- **Mode B — `escalation_band`**: 이진 `noul` 게이트에서 $p < \text{band\_lo}$ (자동 거절/음성 확정), $p \ge \text{band\_hi}$ (자동 수용/양성 확정), $\text{band\_lo} \le p < \text{band\_hi}$ (System 2 LLM 또는 인간 검수 큐로 에스컬레이션).
- **`policy_sha256`**: 핵심 정책 필드의 정규화 JSON SHA-256 앞 16자리 지문으로, 런타임 라우터(`decision-plane`)와 관제 스팬(`s1.policy_sha256`)에서 버전 일치 여부를 검증합니다.

---

## 2. 디렉터리 구조 및 모듈 매핑

```text
s1ops/
  README.md
  pyproject.toml                       # hatchling, s1 & s1ops CLI, src/s1ops + src/s1
  docs/
    ARCHITECTURE_AND_PLAN.md           # 아키텍처 명세 및 Phase 1~3 로드맵 (본 문서)
    REVIEW_AND_COMPARISON.md           # Langfuse/Phoenix 비교 분석, 실증 실패 모드, 네이밍 검토
  src/
    s1/__init__.py                     # `import s1` 단축 별칭 패키지
    s1ops/
      __init__.py
      contracts/
        questions.py                   # Noul/Choice/Score 모델, question_id, /v1/decisions 변환
        records.py                     # jev-record/1 & s1-record/1 엄격 파서 및 그룹핑
        schemas.py                     # PolicyEntry (s1-policy/1) 및 policy_sha256
      ledger/
        journal.py                     # Reserve-then-settle 예산 원장 (flock, migrate, recover)
      lint/
        text.py                        # 부정어·문장 경계·비ASCII 토크나이저
        loader.py                      # 질문 파일 로더 (x-s1-lint / x-jev-lint 억제 지원)
        rules.py                       # 정적 규칙 JL000~JL060 및 UNSUPPRESSIBLE 가드
        lint.py                        # 정적 린트 실행기 및 오프라인 프로브 일관성 감사
      eval/
        metrics.py                     # LogLoss, Brier, ECE(width/mass), ece_floor, tie_mass, bootstrap
        audit.py                       # 그룹별 보정 전/후 감사 및 신뢰구간 산출
        compare.py                     # 다중 백엔드 Shadow 비교 (|Δp|, TVD, Flip, McNemar-Holm)
        judge.py                       # 루브릭 심사자 합의도 (exact, within-1, quadratic κ, bias)
      improve/
        calibrate.py                   # Temperature / Platt / Isotonic 피팅 및 Out-of-Sample 적용
        policy.py                      # Tie-safe 임계값, Clopper-Pearson 상한, Escalation Band 파레토 튜너
        slices.py                      # Wilson CI 답변 가능성 진단 및 Fisher + BH-FDR 오류 슬라이스 마이닝
        labelaudit.py                  # 온도 보정 불일치 랭킹, 컨트롤 암 포함 블라인드 시트, Wilson Lift
      bridges/
        langfuse.py                    # safe-v1 Langfuse Dataset Run / Score / Annotation Queue 익스포터
        openinference.py               # safe-v1 OpenInference / Arize Phoenix JSONL 스팬 익스포터
      monitor/
        drift.py                       # 확률 분포 PSI, Jensen-Shannon 드리프트 및 무통보 model_served 교체 감지
        runner.py                      # Ledger(reserve->settle) 연동 예산 안전 프로브/배치 실행기 (`safe-v1`)
      runtime/
        gate.py                        # 서브 밀리초 DecisionGate (사후 보정 적용 + 임계값/밴드 라우팅 + 해시 검증)
      web/
        app.py                         # FastAPI 로컬 워크벤치 서버 (127.0.0.1 전용, `/api/drift` 포함)
        static/index.html              # 무외부통신 6-탭 인터랙티브 SPA
      cli.py                           # 단일 CLI 진입점 (`s1` / `s1ops`)
  examples/
    make_synthetic.py                  # 시드 고정 데모 데이터셋 생성기 (`examples/data/`, 카나리 드리프트 포함)
  tests/                               # 100% 오프라인 단위·통합 테스트 스위트 (19 tests)
```

---

## 3. 단계별 로드맵 (Phase 1 → Phase 2 → Phase 3)

### Phase 1: 오프라인 평가 · 개선 컴파일러 · 익스포터 · 로컬 웹 워크벤치 (✅ 완료)
- [x] `contracts` & `ledger`: `s1-record/1` / `jev-record/1` 파서, `question_id` 해시, `s1-policy/1`, Reserve-then-settle 원장
- [x] `lint`: 정적 규칙 `JL000`–`JL060` 및 Noul/Choice Twin · Inverse 합 오프라인 감사
- [x] `eval`: Brier, LogLoss, ECE(`width`/`mass`), Monte Carlo `ece_floor`, `tie_mass`, McNemar-Holm `compare`, 루브릭 `judge`
- [x] `improve`: Temperature/Platt/Isotonic `calibrate`, Tie-safe `policy` & Escalation Band 파레토 컴파일러, Wilson CI + BH-FDR `slices`, 블라인드 `labelaudit`
- [x] `bridges`: `safe-v1` Langfuse 및 Arize Phoenix (OpenInference) 익스포터
- [x] `web` & `cli`: 로컬 웹 워크벤치(`127.0.0.1`) 및 통합 CLI (`s1`), 시드 고정 데모 생성기, 100% 오프라인 테스트 검증

### Phase 2: 온라인/카나리 드리프트 모니터 & 예산 안전 프로브 러너 (✅ 완료)
- [x] **Rolling Distribution Drift (`s1 drift`, `src/s1ops/monitor/drift.py`)**: 운영/카나리 로그의 `(question_id, model_served, primitive)`별 예측 확률 분포 **PSI(Population Stability Index)**, **Jensen-Shannon Divergence (JSD)**, 분위수(`p10`/`p50`/`p90`) 이동, 엔트로피 드리프트, 에스컬레이션 밴드 유입률(`escalation_rate_delta`) 및 기권/실패율(`fail_rate_delta`) 급증 감지
- [x] **Silent Model-Served Drift Alert**: 동일 `question_id`에서 기존 베이스라인 `model_served`가 사라지거나 신규 모델 버전(`unexpected_models_in_target`, 예: `jev-1.14.0-silent`)이 유입될 때 즉시 `CRITICAL` 경보 발생 (Exit Code 4 지원)
- [x] **Budget-Guarded Probe & Batch Runner (`s1 probe`, `src/s1ops/monitor/runner.py`)**: `Ledger` (`reserve` → `settle`) 하에서 상한 예산(`cap_usd`)을 절대 초과하지 않고 Twin/Inverse 프로브를 자동 확장·실행하며 원문 `state`를 저장하지 않는(`safe-v1`) 실행기

### Phase 3: 런타임 `DecisionGate` 정책 평가기 & 6-탭 워크벤치 통합 (✅ 완료)
- [x] **Sub-millisecond `DecisionGate` (`s1 gate`, `src/s1ops/runtime/gate.py`)**: `s1 policy tune`이 컴파일한 `policy.json` (`policy_sha256` 무결성 검증)과 `s1 fit`이 생성한 `fits.json`을 로드하여 런타임에서 즉각 사후 보정(`temperature` / `platt` / `isotonic`) + 임계값/에스컬레이션 밴드 라우팅(`ALLOW` / `BLOCK` / `ESCALATE`) 수행 및 `model_served` 불일치 시 즉시 차단(`ModelServedMismatchError`)
- [x] **6-탭 로컬 웹 워크벤치 확장 (`src/s1ops/web/`)**: 6번째 탭 **`6. Distribution & Model Drift`** 추가 및 `/api/drift` 연동
- [ ] **Phase 4 후속 확장 계획**: 상세 명세 및 체크리스트는 **[docs/NEXT_STEPS_PLAN.md](NEXT_STEPS_PLAN.md)** 참고 (GitHub 원격/CI 설정, 라이브 HTTP 백엔드 어댑터, `DecisionGate` `choice`/`score` 확장, `@pandas-studio/sys1` TypeScript SDK 및 `agent-harness` 연동)
