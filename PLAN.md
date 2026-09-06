# Sequential Frozen-Checkpoint CSV Post-Hoc Audit 구현 계획

**1. 합의 사항**
- 목적은 **동일한 frozen checkpoint에서 K를 증가시켰을 때, structural reach 포화 이후에도 response mismatch와 실제 solution error가 얼마나 개선되는지** 확인하는 것이다. 새로운 학습이나 최적 K의 자동 선택은 수행하지 않는다.
- 첫 적용 대상은 Pentagram의 **K=10으로 학습한 seed 0·1·2·3의 best-energy checkpoint**다. 여러 checkpoint를 명시적으로 지정한 하나의 장비에서 순차 평가한다.
- 기본 평가 범위는 `K=10,11,...,64`다. 각 sample의 **원래 symmetric-balanced proposal에서 1부터 K까지** 계산한다. 이미 보정된 K=10 결과에서 추가 correction을 시작하지 않는다.
- 사용자 선택에 따라 **모든 평가와 시간 측정을 Eager, float64로 통일**한다. 학습 당시 compile 설정은 기록만 한다. 모델, Green response, preconditioner, loss, weak reconstruction 설정과 checkpoint는 변경하지 않는다.
- “CSV 전용”은 **그림과 pointwise field archive를 생성하지 않는다**는 의미다. 재현성을 위한 JSON metadata, 검증 기록, 로그와 출력 설명서는 함께 저장한다. 이번에는 계획만 제공하며 `PLAN.md`는 사용자가 작성한다.

**2. 입력과 출력**
새 CLI는 `--run-dirs`의 순서대로 실행한다. 학습 config에 새로운 옵션은 추가하지 않는다.

| 입력 | 기본값과 동작 |
|---|---|
| `--run-dirs PATH...` | 필수. 각 폴더의 `config_used.json`과 `complex_coupling_model_best_energy.safetensors`를 사용한다. |
| `--outdir`, `--device` | 모두 필수. 장비는 `cpu` 또는 명시적인 CUDA device이며 자동 fallback하지 않는다. |
| `--baseline-k`, `--max-k` | 각각 `10`, `64`. `2 ≤ baseline_k ≤ max_k`이며 각 checkpoint의 학습 K는 baseline과 일치해야 한다. |
| `--batch-size`, `--num-threads` | 각각 `10`, `4`. 모든 run과 K에 동일하게 적용하고 실제 thread 설정을 기록한다. |
| `--benchmark` | 기본 off. 지정하면 정확도 평가와 별도로 각 K의 독립적인 시간 측정을 수행한다. |
| `--warmup-repeats`, `--timing-repeats` | 각각 `3`, `5`. 한 반복은 같은 순서의 전체 testset을 한 번 처리하는 것이다. |
| 경로 override | `--green-checkpoint`, `--geometry`, `--test-path`, `--coefficients`. 생략하면 config를 따른다. |
| `--overwrite` | 기본 off. 이전에 이 CLI가 생성한 출력만 명시적으로 덮어쓸 수 있다. |

| 출력 | 내용과 행 단위 |
|---|---|
| `posthoc_per_sample.csv` | checkpoint × K × sample. `run_id`, seed, 학습 K, 평가 K, sample ID와 file stem을 포함한다. |
| `posthoc_per_seed.csv` | checkpoint × K. sample 평균·중앙값·p95·최댓값, baseline 대비 개선량과 개선 sample 비율. |
| `posthoc_summary.csv` | K별 seed 평균들의 평균과 표본 표준편차. 사용 seed 수를 기록하며 seed가 하나이면 표준편차는 NA. |
| `posthoc_timing.csv` | checkpoint × K × 측정 범위 × 반복. 초 단위 시간과 측정 가능한 메모리. Benchmark off이면 header-only. |
| `metadata.json`, `verification.json` | 입력 hash, 실행 환경, 정의·단위, context provenance, baseline 재현 및 수치 검증 결과. |
| `run.log`, `README.md` | 터미널과 동일한 실행 로그, CSV column·집계 방식·해석상 주의사항. |

핵심 response cost는 **half factor 없는 물리적 squared norm**으로 유지한다.
\[
J_{b,K}=h_xh_y\sum_p\bigl(u_{\phi,b,K}(p)-u_{\psi,b,K}(p)\bigr)^2,\qquad
\Delta_{s,K}=1-\frac{\overline J_{s,K}}{\overline J_{s,K-1}},\qquad
G_{s,K}=1-\frac{\overline J_{s,K}}{\overline J_{s,K_0}}.
\]
- `response_cost`, 이전 단계 대비 ratio·개선율, baseline 대비 ratio·개선율을 저장한다. **Ratio of means와 mean of per-sample ratios를 구분**하며 논문용 기본값은 seed 내부 ratio of means다. Baseline 단계의 직전 cost도 내부 계산에서 확보하고, 분모가 0이면 ratio는 NA로 기록한다.
- `rel_sol`은 config의 최종 weak prediction 기준이며, `rel_sol_equal_mean`, `rel_u_phi`, `rel_u_psi`, `rel_flux`, bulk/boundary/optimized energy도 함께 저장한다. `rel_flux`는 기존 directional-source pair 오차 정의를 유지한다.
- Correction norm, symmetric pair 대비 correction ratio, 단계별 coefficient·active 여부, effective dimension, balance 최대 오차와 response orthogonality를 기록한다. Geometry-only global·lower-5%·minimum reach와 full-reach K도 기존 topology helper로 한 번 계산한다.
- CSV의 오차·개선율은 fraction으로 저장한다. 공유 testset을 seed 수만큼 독립 표본으로 취급하지 않으며, reference error가 가장 낮은 K를 자동으로 추천하지 않는다.

**3. 단계별 구현**
1. **입력 및 provenance 검증:** 새 request dataclass와 순차 runner를 추가한다. 중복 run/seed, checkpoint 누락, 서로 다른 test sample·Green checkpoint·geometry·coefficient·모델 설정을 검증한다. Seed와 원래 학습 장비 등 허용된 차이는 분리 기록한다. Linux/Mac 경로 변경은 override로 처리하고 실제 파일 hash를 남긴다.
2. **기존 계산 재사용:** [기존 subspace audit](/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/src/greenonet/complex_tangent_subspace_audit.py)의 준비·metric 계산에서 필요한 비시각화 부분만 공통화한다. Production `matrix_free_krylov_subspace_step`과 reconstruction helper를 재사용하며, 기존 그림 생성 CLI의 동작은 보존한다.
3. **Context 보호 및 baseline 재현:** 검증된 기존 sidecar는 읽기 전용으로 사용한다. 없으면 메모리에 build하고, 존재하지만 무결성 검증에 실패하면 중단한다. 동일한 static operator identity에서는 checkpoint 간에도 재사용한다. 각 run의 baseline을 production evaluator 및 `artifacts_best_energy`의 동일 sample 지표와 대조한 뒤 sweep을 진행한다. 학습 종료 모델의 `metrics/`를 대신 사용하지 않는다.
4. **정확도 sweep:** Batch별로 frozen model forward와 초기 proposal을 한 번 계산하고 Kmax까지의 nested 결과를 얻는다. 각 K의 reconstruction과 metric은 순차 처리해 큰 candidate field 복제를 피한다. Reference는 metric 계산에만 사용하며, inactive direction 처리·두 번의 MGS·기존 epsilon 규칙을 변경하지 않는다.
5. **독립 benchmark:** 각 K를 처음부터 다시 실행한다. `tangent_only`는 준비된 mismatch·gradient부터 correction까지, `prediction_forward`는 장비에 준비된 입력부터 network·projection·reconstruction·weak blend까지 측정한다. 데이터 읽기·전송·context setup·reference metric·CSV 쓰기는 제외한다. CUDA에서는 지정 device를 측정 전후 동기화하며 warmup을 제외한다. [PyTorch benchmark 원칙](https://docs.pytorch.org/docs/2.14/benchmark_utils.html)
6. **시간·메모리 집계:** 전체 test pass의 batch별 측정 시간을 합산하고 반복별 원자료 및 median/p95를 저장한다. Production helper 내부의 MGS·안전 검사·작은 K×K 직교성 진단 비용은 포함한다. CUDA peak allocated memory와 측정 시작 시 allocation을 기록하고 CPU peak memory는 미측정 NA로 표시한다. 서로 다른 평가 장비의 시간을 합치지 않는다.
7. **저장 및 문서:** 실패 시 부분 결과와 실패 사유는 남기되 완료 aggregate로 표시하지 않는다. 모든 검증을 통과한 뒤 완료 상태를 기록한다. 입력 checkpoint/config/sidecar/artifact는 수정하지 않는다. 신규 파일은 `cli/audit_frozen_tangent_csv.py`, `src/greenonet/complex_frozen_tangent_csv.py`, 대응 focused test이며, 프로젝트 `README.md`와 `docs/memory.md`에 실행·해석 convention을 추가한다.

**4. 테스트와 롤백**
- **수학·동등성:** 작은 float64 fixture에서 Kmax prefix와 각 K 독립 실행의 일치, K=10 baseline 재현, K=64의 finite 출력, exact balance tolerance, J의 tolerance 내 비증가, 퇴화 direction과 effective dimension을 검증한다. `rel_sol`과 energy의 단조 감소는 요구하지 않는다.
- **CLI·CSV:** 여러 checkpoint의 순차 실행, 동일 sample pairing, batch 크기 변경 시 정확도 일치, seed별 집계, ratio 정의, NA 처리, benchmark on/off, 그림·raw NPZ 부재, 입력 hash 불변을 검증한다. Reference 변경이 correction을 바꾸지 않는지도 검사한다.
- **실패·시간:** 잘못된 경로·서로 다른 입력·누락된 reference/기준 artifact·손상 sidecar·중복 seed·baseline 불일치·부적절한 overwrite가 명확히 실패하는지 확인한다. Fake clock과 CUDA mock으로 warmup·동기화·측정 제외 범위를 검증하고, 실제 CUDA smoke는 사용 가능할 때만 수행한다.
- **검증 순서:** 신규 focused tests → 기존 tangent subspace/projection/context 및 artifact tests → 전체 `pytest test` → `ruff check src cli test` → `ruff format src cli test` → `mypy src` → `git diff --check`. 구현 검증 중 실제 네 checkpoint의 K=10..64 본 실험은 실행하지 않는다.
- **롤백:** 새 CLI와 runner 사용을 중단하면 기존 workflow로 즉시 복귀한다. 공통화한 helper에 regression이 있으면 해당 추출만 되돌린다. Model migration, training config 변경, 원본 실험 결과의 복원 작업은 필요 없어야 한다.
- **확신도:** 구현 계획 **0.97**. 실행 모드와 분석 목적은 명확하다. 남은 불확실성은 다른 장비로 복사된 파일의 완전성 및 실제 장비에서의 고차 K 시간·메모리에 관한 **정보 부족**이며, 규칙의 모호성은 아니다.

**5. 실행 가능한 `/goal`**
```text
/goal

프로젝트 루트 PLAN.md의
"Sequential Frozen-Checkpoint CSV Post-Hoc Audit 구현 계획"을 기준으로,
여러 frozen best-energy checkpoint를 같은 장비에서 순차 평가하는
CSV 전용 post-hoc CLI를 끝까지 구현한다.

완료는 다음 조건으로 검증한다.
- 동일한 입력·float64·Eager 환경에서 checkpoint를 순차 평가할 것,
- 기본 K=10..64를 원래 symmetric-balanced proposal부터 계산할 것,
- baseline이 production evaluator와 기존 best-energy artifact를 재현할 것,
- response cost, 단계별 개선량, solution/directional error와 reach를 기록할 것,
- sample별·seed별·seed 간 집계와 optional 독립 benchmark가 정확할 것,
- active direction, balance, finite 결과와 response-cost 비증가를 검증할 것,
- reference는 정확도 metric에만 사용하고 그림이나 raw field NPZ를 만들지 않을 것,
- focused tests, 전체 pytest, Ruff, mypy와 git diff check가 통과할 것.

수정은 새 CSV CLI/runner, 필요한 audit helper 공통화, tests와 관련 문서로 제한한다.
Model, training, tangent 수식, checkpoint key, 기존 artifact behavior는 보존한다.
실제 네 checkpoint의 본 평가와 재학습은 실행하지 않는다. 실행은 사용자가 한다.
각 단계에서 가장 작은 관련 테스트를 먼저 실행한 뒤 regression 검증을 진행한다.

Baseline 수치 또는 입력 provenance를 보존할 수 없다면 중단하고 보고한다.
1. 충돌하는 checkpoint, 입력 또는 metric contract,
2. 영향을 받는 run과 CSV 항목,
3. 기존 수치와 원본 파일을 보존하는 가장 작은 수정 또는 롤백 전략.
```
