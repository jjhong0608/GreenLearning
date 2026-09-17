# 네 수치예제의 Reference Source 진단과 학습된 초기값 효용 검증

## 1. 목표와 확정 범위

**순서 1도 네 예제 모두에서 수행한다.** FEniCSx reference source의 balance 위반을 측정하고, 원본 및 balance 보정 source를 동일한 학습 Green 연산자로 재구성하여 정확도를 비교한다. :codex-annotation{index="1"}

**순서 2는 learned 모델을 원래 학습 \(K\)에서 고정한다.** \(f/2\) 초기값만 \(K=0,\ldots,64\)로 확장하여 learned 기준 정확도에 도달하는 최소 \(K\)와 온라인 비용을 측정한다.

| 예제 | 기준 checkpoint 그룹 | 학습·기준 평가 \(K\) | Seed | Test |
|---|---|---:|---|---:|
| Unit square | `unit_square_trunk_on_seed*` | 2 | 0–3 | 100 |
| Disk | `disk_separable_seed*` | 2 | 0–3 | 50 |
| Annulus | `annulus_reconstruction_seed*` | 4 | 0–3 | 100 |
| Pentagram | `pentagram_k9_seed*` | 9 | 0–3 | 100 |

현재 16개 best-energy checkpoint와 tangent context, 네 dataset의 `generation_summary.json` 및 reference 배열을 확인했다. Pentagram은 **K9-trained checkpoint**를 사용하며 기존 K10 audit를 덮어쓰지 않는다.

사용자가 선택한 실행 조건은 **GPU:1에서 순차 실행**, 산출물은 **CSV·Markdown 보고서·Plotly 그림**이다. GPU:0, 새 학습, ODE 비교, 다른 결합 solver, 해상도 변경은 이번 범위에서 제외한다. `PLAN.md`는 사용자가 작성하며 이 계획 단계에서는 수정하지 않는다.

## 2. 구현과 입력 검증

1. **공통 audit 모듈과 CLI를 추가한다.** `src/greenonet/complex_source_initialization_audit.py`, `cli/audit_source_initialization.py`에 typed request/dataclass와 단계별 audit 클래스를 둔다. 기존 frozen CSV audit의 모델 로딩·context 검증·독립 시간 측정 패턴, production tangent·reconstruction을 재사용한다. 기존 equal-split CLI의 K10 전용 검증이나 첫 batch 전용 benchmark를 복제하지 않는다.
2. **명시적인 16-run manifest를 작성한다.** Run 경로, 예제·seed, 기준 \(K\), checkpoint·GreenNet·geometry·계수·test·원본 artifact 경로를 기록한다. CLI는 `--manifest`, `--outdir`, `--stage preflight|reference|initialization|all`, `--device`, `--batch-size`, `--max-k`, `--warmup-repeats`, `--timing-repeats`를 받는다.
3. **기본값은 고정한다.** `cuda:1`, float64, eager/no-grad, batch 5, CPU threads 4, 최대 \(K=64\), warm-up 3회·측정 5회로 실행한다. 기존 모델의 preconditioner·epsilon·정규화를 유지한다. Disk의 response 정규화를 다른 세 예제의 legacy 정규화로 바꾸거나 그 반대로 바꾸지 않는다.
4. **Preflight에서 전체 입력을 검증한다.** Test 파일 전체의 필수 배열·shape·finite valid values·active-point 순서, source/solution 단위, CDR 항 분배와 경계조건을 확인한다. 생성 metadata의 과거 경로와 현재 경로가 다르면 명시적 대응을 기록하며 파일명 유사성만으로 대체하지 않는다.
5. **현재 생성 코드와 과거 provenance를 구분한다.** 현재 FEniCSx 코드는 방향별 미분식에 반응항 절반을 더해 별도 projection 후 axial point에서 평가한다. 당시 생성 설정과의 일치를 확인하되, 현재 코드가 당시 실행 코드와 동일했다는 보증은 하지 않는다. Disk 생성 요약의 전체 train/valid/test 통계를 test 통계로 사용하지 않는다.
6. **입력은 읽기 전용으로 보호한다.** Checkpoint·context·config·GreenNet·geometry·계수·test·생성 요약·비교 artifact의 SHA256을 전후 확인한다. Context는 기존 검증 경로로 재사용하고, 장비 차이로 재구성이 필요하면 메모리 또는 새 출력 디렉터리에서만 수행한다. 손상·의미 불일치를 조용히 fallback하지 않는다.

출력 루트는 `docs/analysis/paper_source_initialization_audit/`로 한다. 기존 디렉터리가 있으면 덮어쓰지 않고 실패한다. 수정된 실행은 별도의 `--outdir`로 남기며 예전 결과와 섞지 않는다.

## 3. 단계별 실험과 판정

### A. 순서 1: Reference source 진단

네 예제의 전체 test에서 다음 두 조건을 **tangent와 초기값 네트워크 없이** 평가한다.

\[
r=\phi_{\rm ref}+\psi_{\rm ref}-f,\qquad
(\phi_{\rm bal},\psi_{\rm bal})
=(\phi_{\rm ref}-r/2,\ \psi_{\rm ref}-r/2).
\]

- 원본과 보정본 각각에 동일한 Green response·적분·weak reconstruction을 적용하고, directional/equal/weak solution error의 sample별 값과 mean·P95·maximum을 기록한다.
- Balance 절대 \(L^2_M\) norm, 상대 norm, 최대 절대 residual, pair 보정 norm, directional mismatch와 energy를 보고한다. 상대 분모 norm이 \(10^{-12}\) 이하이면 NA와 명시적 사유를 기록하고 절대값은 유지한다.
- \(\|r\|_M/\sqrt2\)는 전제 조건이 맞을 때 source-pair 오차의 하한이며 solution-error 하한이 아님을 보고서에 명시한다. 원본 source가 balance를 만족해야 한다는 assertion은 두지 않는다.
- 동일한 Green·geometry·계수·적분·reconstruction·test fingerprint는 한 번만 평가하고 해당 run들과 연결한다. 다른 fingerprint를 seed만 같다는 이유로 공유하지 않는다.
- 큰 residual 자체는 실패 조건이 아니다. Source 계약 불일치나 비유한 결과는 해당 조건을 오류로 보고하며, 자동으로 dataset을 재생성하거나 정의를 바꾸지 않는다.

### B. 순서 2: Learned 기준 정확도까지의 보정 비용

각 checkpoint를 원래 \(K\)의 production 경로로 재평가하고 기존 best-energy artifact를 재현하는지 먼저 확인한다. Learned 평가는 기준 \(K\)에서 고정하며 \(f/2\)만 전체 \(K\)를 확장한다.

\[
K_{\rm match}
=\min\{K:\overline E_{f/2}(K)\le\overline E_{\rm learned}(K_{\rm base})\}.
\]

- 주 지표는 weak solution 평균 상대 \(L^2\) error이다. 같은 \(K_{\rm match}\)의 P95·maximum을 함께 제시하고, **평균과 P95를 동시에 만족하는 최소 \(K\)**도 별도 기록한다.
- 전체 \(K\)의 curve를 보존해 도달 후 악화를 표시한다. 비교식은 저장된 float64 원시값으로 계산하고 반올림된 표 값으로 판정하지 않는다.
- 범위 내 미도달은 `not_reached`로 기록한다. \(K>64\) 자동 확장, threshold 변경, 안정화 변경은 하지 않는다. 수치 오류는 미도달과 구분한다.
- 정확도 sweep은 nested prefix를 사용하되, learned 기준 \(K\) 및 모든 도달 \(K\)에서 독립 호출과의 일치를 확인한다. K0은 보정 없음, K1은 기존 uncapped 보정 의미를 유지한다.
- Actual active dimension과 방향별 activity를 기록한다. 방향 탈락은 있는 그대로 보고하며 configured \(K\)만으로 유효 차원을 주장하지 않는다.
- 동일 fingerprint의 \(f/2\) 경로는 재사용하되 seed마다 다른 learned 목표값과 대응시킨다. 복제된 baseline을 독립 관측으로 세지 않는다.

### C. 독립적인 온라인 시간 측정

예제별로 learned 기준 \(K\), \(f/2\)의 기준 \(K\), \(K=64\), 모든 seed의 mean-match·mean+P95-match \(K\)의 합집합을 측정한다.

- 전체 test를 batch 5로 순회하는 prediction-forward를 warm-up 3회 후 5회 측정한다. 조건 순서는 반복마다 반전하며 CUDA를 전후 동기화한다.
- 입력 batch는 사전 준비한다. Network inference·physical 변환·balance projection·보정·response 재구성·weak blend를 포함하고, 데이터 I/O·context/model loading·setup·metric 계산·학습은 제외했다고 명시한다.
- 각 \(K\)는 처음부터 독립 실행한다. Prefix sweep 시간을 분할하거나 저장된 최종 correction을 가져오는 시간을 benchmark로 쓰지 않는다.
- 원시 반복 시간, median·범위, sample당 시간, GPU peak allocated/reserved memory와 측정 직전 baseline을 기록한다. 초기값·보정·재구성 단계별 시간은 이번 필수 산출물에서 제외한다.
- GPU:1에 다른 연산 작업이 있거나 OOM이 발생하면 시간 측정을 중단하고 보고한다. GPU:0·CPU로 자동 전환하거나 batch를 조용히 줄이지 않는다.
- Test로 찾은 도달 \(K\)는 사후 비교 지표일 뿐 reference-free stopping rule이나 배포용 선택값이 아니다. 실제 시간 이득이 없으면 정확도 또는 보정 차원의 이점으로만 결론을 제한한다.

## 4. 산출물·테스트·완료 기준

**산출물:** 입력 manifest와 hash/provenance, effective evaluation 설정, `run.log`, preflight·검증 JSON, reference sample/summary CSV, learned baseline CSV, \(f/2\) 전체-K sample/summary CSV, seed별 도달-K CSV, timing 원시·요약 CSV를 생성한다. CSV 오차는 fraction으로 저장하고 보고서에서 %로 변환한다.

Plotly는 예제별 정확도–K, 정확도–시간, reference 원본/보정본 비교를 만든다. Learned 기준선과 도달점을 표시하고 서로 다른 예제의 오차를 하나의 평균으로 합치지 않는다. Markdown 보고서는 수행 범위·결과·예외·해석 한계·남은 질문을 포함한다.

**영향 파일:** 신규 audit 모듈·CLI·manifest 및 전용 `test/` 테스트를 추가한다. 공통 production helper는 재사용을 우선하고 필요한 경우에만 동작 보존형으로 최소 수정한다. Storyline에 후속 결과 문서와 링크를 추가하고 README·`docs/memory.md`를 갱신한다. 기존 paper CSV·provenance·checkpoint는 변경하지 않는다.

**테스트 순서와 내용:**
- 먼저 합성 fixture로 원본 source 보존, symmetric projection·보정 norm, 불균형 원본 허용, \(f/2\)의 network 우회, K0/K1, nonmonotone 최초 도달·미도달·mean/tail 분리, 중복 baseline 연결을 검증한다.
- Legacy/response 정규화, mask·배열 순서, fingerprint 불일치, zero denominator, 누락·손상 입력, 독립 timing이 매 호출 재계산하는지 검사한다.
- 구현 시도마다 가장 작은 관련 테스트부터 실행한 다음 전체 `pytest`를 실행한다. `ruff check src`, `ruff format --check src`, `mypy src`, 문서 링크 및 diff 검사를 수행한다.
- 예제당 seed0의 소규모 smoke 검증 후 네 예제 전체 reference 진단, 16개 learned 기준 평가와 \(f/2\) sweep, 시간 측정 순으로 실행한다.
- Native metric과 독립 prefix의 일치는 기본 `rtol=1e-8, atol=1e-12`, balanced source 합은 `rtol=atol=1e-12`로 검증한다. Response cost의 비증가는 기존 production audit의 허용오차를 재사용한다. 불일치 시 tolerance를 자동 완화하지 않는다.
- 모든 run/test coverage, 누락·중복 부재, hash 불변, 도달-K 재계산과 완료 상태를 확인한다. 일반 미도달은 정상 결과지만 수치·입력 오류가 남은 작업을 완료로 표시하지 않는다.

**롤백:** 새 코드·manifest·결과 디렉터리와 이번 문서 변경만 개별적으로 되돌린다. 기존 dirty worktree를 reset하지 않는다. 실패 결과는 삭제·성공 표시 대신 실패 상태로 보존하고, 재실행은 새 출력 디렉터리에 남긴다. Checkpoint 계약 변경이 필요하면 구현을 중단하고 tensor/key 불일치, 영향 checkpoint, 최소 migration 방안을 보고한다.

**확신도:** 약 **90%**. 16개 checkpoint와 reference 입력의 존재, 기존 재사용 경로는 확인했다. 남은 불확실성은 과거 데이터 생성과 현재 계수·geometry의 의미적 일치, 큰 \(K\)의 실제 수치 동작 및 실행시간에 관한 정보 부족이다. 범위와 판단 규칙의 모호함은 아니다.

## 5. 실행 가능한 `/goal` 초안

```text
/goal

프로젝트 루트의 PLAN.md를 읽고 그 계획에 따라 순서 1과 순서 2를 구현·실행·분석하라.

순서 1은 Unit square, Disk, Annulus, Pentagram 전체 test에서
FEniCSx reference source의 정의와 balance를 확인하고,
원본 및 symmetric balance 보정 source를 같은 학습 Green 연산자로
재구성하여 정확도를 비교하는 작업이다.

순서 2는 네 예제의 네 seed best-energy checkpoint를 각각 학습 당시
K=2,2,4,9로 평가한 정확도를 고정 기준으로 삼고,
physical phi=psi=f/2를 K=0..64로 보정하여 그 기준에 도달하는
최소 K와 온라인 prediction 시간을 비교하는 작업이다.
Pentagram은 반드시 K9-trained checkpoint를 사용하라.

GPU:1에서 float64 eager로 한 작업씩 순차 실행하라.
GPU:0, 재학습, ODE 비교, 다른 결합 solver, 해상도 변경은 수행하지 마라.
각 checkpoint의 preconditioner와 정규화 설정을 보존하라.

완료는 다음으로 검증한다:
- 네 예제 전체 reference 진단과 16개 learned 기준 평가 완료
- 기준 metric 재현 및 prefix와 독립 평가의 일치
- 평균 도달 K, 평균과 P95 동시 도달 K, 미도달·실패의 명확한 구분
- inference와 projection을 포함한 독립 전체-test 온라인 시간 측정
- CSV, Plotly 그림, 상세 Markdown 보고서와 provenance 생성
- 기존 checkpoint, context, config, dataset과 결과 hash 불변
- 관련 테스트, 전체 pytest, 린트·타입 검사 통과

각 구현 시도 후 가장 작은 관련 테스트부터 실행하고 전체 회귀 테스트를 실행하라.
PLAN.md는 수정하지 말고 README와 docs/memory.md에는 최종 결정과 결과를 기록하라.
기존 결과를 덮어쓰거나 유리한 결과만 선택하지 마라.

Checkpoint 호환성을 유지할 수 없으면 중단하고
정확한 tensor/key 계약 불일치, 영향 checkpoint, 최소 migration 방안을 보고하라.
입력 계약 불일치나 native 재현 실패는 숨기지 말고 해당 평가를 중단하라.
K64 내 정확도 미도달은 정상 결과로 보고하되 자동 범위 확장은 하지 마라.
결과를 확인한 뒤 실제 근거에 맞게 주장 수준을 정리하라.
```
