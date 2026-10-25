# System 1 LLMOps 비교 검토 및 네이밍·아키텍처 분석 보고서

본 문서는 `s1ops` 저장소 설계 과정에서 수행한 **1) 기존 System 2 LLMOps 도구(Langfuse, Arize Phoenix 등)와의 구조적 차이점 검토**, **2) System 1 실증 연구(`jev-utils`, `jev-system1`)에서 드러난 핵심 실패 모드와 방어 설계**, **3) 패키지/CLI 네이밍 및 글로벌 레지스트리 충돌 실측 결과**를 기록합니다.

---

## 1. 기존 LLMOps (Langfuse / Arize Phoenix) vs `s1ops` 검토

Langfuse, Arize Phoenix, LangSmith, Braintrust 등 기존 LLMOps 플랫폼은 **System 2 (자기회귀 텍스트 생성, 멀티스텝 에이전트 트레이스, 프롬프트 엔지니어링)**를 중심으로 설계되어 있습니다. 반면 System 1 모델(`/v1/systemone`, `/v1/decisions`, 분류/리랭커 헤드)은 **수십 밀리초 내에 정형 확률 분포(`noul`, `choice`, `score`)를 출력하고 즉시 라우팅/게이팅 결정을 내리는 반사신경(Reflex) 계층**입니다.

### 1.1 7대 구조적 공백과 `s1ops`의 해결 방식

| # | 문제 영역 | 기존 System 2 LLMOps의 한계 | `s1ops`의 해결 방식 |
| --- | --- | --- | --- |
| 1 | **집계 단위 오류 (Pooling Hazard)** | Trace 이름이나 모델 이름으로만 묶어 질문 문구가 조금 바뀌어도 스코어를 평균냄 | 정규화된 wire-form 해시 **`question_id` (`q-<16hex>`)**를 강제하고, **`(question_id, model_served, primitive)`** 단위로만 지표를 산출 (이종 그룹 간 합산 원천 금지) |
| 2 | **소표본 ECE 착시 (Small-$n$ Bias)** | 샘플 수 $n=100 \sim 300$에서 계산한 단순 ECE가 이항 표집 분산 때문에 $0.05 \sim 0.08$로 높게 나오는 현상을 모델 오보정으로 오판 | 동일한 예측 확률에서 베르누이 샘플링을 반복해 완벽히 보정된 모델이 가질 기대 ECE 분포(**Monte Carlo `ece_floor` `p95`**)를 함께 산출하여 진성 오보정만 식별 |
| 3 | **확률 반올림 동점 클러스터 (Tie-Mass)** | 상용 System 1 API(예: `jev-1.13.0`)가 소수점 2자리 반올림 확률을 반환해 `p=1.00`, `p=0.50`에 수백 건이 묶일 때 분위수(quantile) 분할과 순위 기반 임계값이 깨짐 | **`tie_mass`** 진단기로 최빈 동점 비율을 감지하고, 임계값 선택([`select_threshold`](../src/s1ops/improve/policy.py)) 시 동일 확률 그룹 전체를 원자적으로 수용/거절하도록 강제 |
| 4 | **In-Sample 보정 누수 (Calibration Leakage)** | Temperature / Isotonic 피팅에 사용한 데이터셋에 그대로 보정을 적용해 평가 지표를 부풀림 | `fits.json`에 피팅에 사용된 모든 `[dataset, item_id]`와 `model_served`를 기록하고, `s1 apply` 시 In-sample 중복은 **Exit 2**, 모델 불일치는 **Exit 3**으로 차단 |
| 5 | **질문 문구 민감도 (Lyon / Negation Trap)** | 부정문(`"Is it not unsafe?"`), 복합 질문(`"A and B"`), 모호한 선택지 등 System 1에서 확신 오답(Confident Error)을 유발하는 문구를 배포 전에 걸러내지 못함 | **정적 질문 린터 (`JL000`–`JL060`)** 및 **오프라인 프로브 일관성 감사 (`s1 lint probe-audit`)**로 Noul vs Yes/No Choice 쌍둥이 괴리율과 반전 합($\|p + p_{\text{inv}} - 1\|$)을 측정 |
| 6 | **벤치마크 골드 라벨 노이즈** | 평가셋의 `label`을 100% 참값으로 가정하여, 실제로는 라벨 오류(예: YNAT 인간 합의율 34.8%)인데도 모델 튜닝을 잘못된 방향으로 진행 | 온도 보정된 불일치 점수로 후보를 랭킹하고 **무작위 Control 문항을 섞은 블라인드 검수 시트 (`s1 labelaudit`)**와 **Wilson Conservative Lift**로 실제 라벨 오염률을 정량 추정 |
| 7 | **원문 텍스트 유출 위험 (Privacy)** | 기본적으로 입력 프롬프트/문서 원문(`state`)을 관제 서버에 전송·저장 | **`safe-v1` Zero-State 기본값**: 레코드, 예산 원장, Langfuse/Phoenix 익스포터 어디에도 원문 `state`를 기록하지 않으며 `state_sha256`과 확률·메타데이터만 취급 |

### 1.2 상호운용 전략: "대체"가 아닌 "컴파일러 + 브리지"
- 조직 전체의 System 2 에이전트 트레이싱은 기존처럼 **Langfuse** 또는 **Arize Phoenix (`tracegraph`)**에 유지합니다.
- System 1 게이트/분류/리랭커 계층의 **문항 린트 → 확률 보정 → 에스컬레이션 밴드 컴파일 → 슬라이스/라벨 감사**는 `s1ops` 로컬 워크벤치에서 수행하고, `s1 export langfuse` 및 `s1 export phoenix`를 통해 `safe-v1` 스팬/스코어/어노테이션 큐로 내보냅니다.

---

## 2. 패키지명 및 CLI 네이밍 검토 기록 (2026-10-25 실측)

System 1의 아이덴티티(**서브 100ms 반응성, 반사신경(Reflex), 직관(Instinct), 초고속 루프**)를 반영한 후보군에 대해 PyPI, npm, GitHub 전체 저장소 충돌 여부를 실측하여 최종 확정했습니다.

### 2.1 후보군 레지스트리 및 생태계 충돌 실측표

| 추천 후보 | 핵심 메타포 | PyPI 상태 | npm 상태 | GitHub 충돌 실측 및 검토 결과 |
| --- | --- | --- | --- | --- |
| **`s1ops`** (최종 선정) | 가장 직관적이고 군더더기 없는 System 1 LLMOps 약어 | **`404` (미점유)** | **`404` (미점유)** | **GitHub 전체 검색 `0건` (`[]`)**. CLI `s1` / `s1ops`, Python `import s1ops` / `import s1` 모두 간결하며 기존 `agent-ops`, `nemo-ops`와도 일관됨 |
| `sys1` | System 1 약어 | `404` | `404` | GitHub에 **`alvarobartt/sys1`** (동일한 TypeSafe AI `/v1/systemone` 호환 Rust 서버 프로젝트)가 이미 존재하여 도메인 내 혼동 우려 |
| `reflex` / `reflex-ops` / `reflex1` | 자율신경계/반사(Reflex) | `reflex` **점유 (`200`)** / `reflex-ops`, `reflex1` `404` | `reflex` **점유 (`200`)** / `reflex-ops` `404` | 단일어 `reflex`는 대형 Python 웹 프레임워크(`reflex-dev/reflex`, 20k+ stars)가 PyPI 패키지·`import reflex`·`$ reflex` CLI를 모두 점유 중이므로 같은 venv 설치 시 엔트리포인트 충돌 발생 |
| `fastloop` | Fast Loop vs Slow Loop | **점유 (`200`)** | **점유 (`200`)** | PyPI(`0.4.1`) 및 npm(`1.0.5`) 모두 기점유 |
| `instinct-ai` | 인지과학의 직관(Instinct) | **점유 (`200`)** | **점유 (`200`)** | PyPI 및 npm 모두 기점유 |
| `axon-ops` | 신경 축삭(Axon) | `404` | `404` | Cassandra 상용 관제 솔루션 **AxonOps (`axonops.com`)**가 상표·도메인·CLI(`axonops`)를 활발히 사용 중 |
| `synapse-ops` | 시냅스 신호 전달 | `404` | `404` | Matrix 홈서버(`element-hq/synapse`), Apache Synapse, Azure Synapse 등과 이름 중복도가 높음 |

### 2.2 최종 채택 사양 (`s1ops`)
- **GitHub 저장소**: `memtomem/s1ops` (비공개, 로컬 경로: `/Users/pdstudio/Work/s1ops`). 최초 채택안은 `pandas-studio/s1ops`였으나 2026-10-25 `tracegraph`·`syncmill` 등 관련 도구와 같은 `memtomem` 계정으로 변경
- **PyPI 패키지**: `s1ops` (`pip install s1ops` / `uv add s1ops`)
- **Python 임포트**: `import s1ops` 및 단축 별칭 `import s1` 동시 지원
- **CLI 명령어**: `s1` 및 `s1ops` 동시 등록 (`$ s1 audit`, `$ s1 fit`, `$ s1 policy tune`, `$ s1 web` 등)
- **향후 Node/TS 패키지 (Phase 3)**: `@memtomem/sys1` (최초 채택안 `@pandas-studio/sys1`에서 변경)

---

## 3. 관련 저장소 연동 구조 (`pandas-studio` 생태계)

- **`jev-utils` (`/Users/pdstudio/work/jev-utils`)**:
  - `jev-calib`, `jev-lint`, `jev-labelaudit`, `jev-shadow`, `jev-escalate`, `jev-ops`, `jev-answerable`, `jev-judge`에 흩어져 있던 프로토타입의 수학적 불변식과 데이터 계약을 `s1ops` 단일 SDK 및 CLI로 통합 계승했습니다.
- **`decision-plane` (`/Users/pdstudio/Work/agent-harness/decision-plane`)**:
  - 런타임 라우터/게이트 역할을 담당하며, `s1ops`가 오프라인에서 컴파일한 `fits.json`과 `policy.json` (`policy_sha256` 지문 포함)을 소비하여 온라인 트래픽을 제어합니다.
- **`tracegraph` (`/Users/pdstudio/Work/agent-harness/tracegraph`)**:
  - `s1 export phoenix`가 출력하는 `safe-v1` OpenInference JSONL 스팬(단일 통화 비용 가드, `OK`/`ERROR`/`UNSET` 상태 정규화)을 그대로 수집·시각화할 수 있습니다.
