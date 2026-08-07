# Optional Response-Trust Loss 구현 계획

## Summary

`response-trust` loss는 현재 tangent correction이 실제 학습 forward path에서 만든 **post-correction directional response mismatch**를 직접 줄이면서, 지나치게 큰 correction response에는 작은 trust penalty를 부과하는 reference-free objective다.

현재 `tangent_post_line_search_stationarity` loss는 uncapped exact line-search 지점에서 gradient가 얼마나 stationary한지를 정규화된 비율로 측정한다. 이 값은 mismatch의 절대 크기에 둔감하므로, 네트워크가 큰 pre-mismatch와 큰 correction을 만들면서도 stationarity ratio만 낮추는 우회가 가능하다. 새 loss는 실제 적용된 capped tangent correction 이후의 mismatch 크기를 고정된 source-response energy로 정규화하여 이 문제를 막는다.

두 loss는 다음 원칙으로 통합한다.

- `response_trust`와 기존 `post_line_search_stationarity` loss는 **상호 배타적**이다.
- `response_trust`가 활성화되면 기존 stationarity ratio는 loss가 아니라 diagnostic으로 계속 계산한다.
- trust coefficient 기본값은 \(\mu=0.01\)로 둔다.
- `sol`, target \(\phi\), target \(\psi\)는 사용하지 않는다.
- full response matrix나 Gram matrix를 만들지 않고 기존 matrix-free Green-response operator를 재사용한다.
- 모델 구조와 checkpoint tensor key는 변경하지 않는다.
- 이 계획 응답에서는 `PLAN.md`를 수정하지 않는다. 사용자가 프로젝트 루트의 `PLAN.md`를 직접 작성한다.

## Mathematical Contract

Symmetric projection이 만든 balanced directional source를

\[
\widetilde\phi+\widetilde\psi=f
\]

라고 하고, frozen directional Green-response operator를 각각 \(H_x,H_y\)라고 둔다.

Tangent correction은 balance plane 안에서

\[
\phi=\widetilde\phi+\delta,
\qquad
\psi=\widetilde\psi-\delta
\]

로 적용되므로 balance는 항상 보존된다.

\[
\phi+\psi=f.
\]

Correction 전 directional solution mismatch는

\[
m_0
=
H_x\widetilde\phi-H_y\widetilde\psi
\]

이고,

\[
S=H_x+H_y
\]

라고 하면 correction 후 mismatch는

\[
m_{\mathrm{post}}
=
H_x(\widetilde\phi+\delta)
-
H_y(\widetilde\psi-\delta)
=
m_0+S\delta
\]

이다.

현재 closed-loop tangent step의 실제 forward correction은

\[
z=D^{-1}g,
\qquad
\delta=-\eta_{\mathrm{applied}}z,
\]

\[
\eta_{\mathrm{applied}}
=
\min(\eta^\star,\eta_{\mathrm{cap}})
\]

이다. 새 loss는 uncapped \(\eta^\star\)가 아니라 **실제 projection에 적용된**
\(\eta_{\mathrm{applied}}\)와 \(m_{\mathrm{post}}\)를 사용한다.

Source-dependent이지만 network-independent인 normalization은

\[
u_{f,x}=H_x\left(\frac{f}{2}\right),
\qquad
u_{f,y}=H_y\left(\frac{f}{2}\right),
\]

\[
E_f
=
\left\|u_{f,x}\right\|_{M_\Omega}^2
+
\left\|u_{f,y}\right\|_{M_\Omega}^2
\]

로 정의한다. \(M_\Omega\)는 현재 tangent context가 사용하는 physical valid-point mass다.

각 sample \(b\)에 대해 다음 세 값을 계산한다.

\[
\ell_{\mathrm{post},b}
=
\frac{
\left\|m_{\mathrm{post},b}\right\|_{M_\Omega}^2
}{
E_{f,b}+\varepsilon
},
\]

\[
\ell_{\mathrm{correction},b}
=
\frac{
\left\|S\delta_b\right\|_{M_\Omega}^2
}{
E_{f,b}+\varepsilon
},
\]

\[
\ell_{\mathrm{response\text{-}trust},b}
=
\ell_{\mathrm{post},b}
+
\mu\ell_{\mathrm{correction},b}.
\]

여기서

\[
S\delta=m_{\mathrm{post}}-m_0
\]

이므로 correction-response term을 위해 Green operator를 다시 적용하지 않는다.

Batch loss는

\[
\mathcal L_{\mathrm{response\text{-}trust}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\ell_{\mathrm{response\text{-}trust},b}
\]

이고 최종 objective는

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{current}}
+
\lambda_{\mathrm{RT}}
\mathcal L_{\mathrm{response\text{-}trust}}
\]

가 된다. \(\mathcal L_{\mathrm{current}}\)는 현재 설정에 따른 optimized canonical energy, boundary weight, 기타 활성화된 기존 objective를 그대로 의미한다.

## Public Configuration

`CouplingTrainingConfig`에 complex-only block을 추가한다.

```json
"response_trust": {
  "enabled": false,
  "weight": 1.0,
  "trust_weight": 0.01,
  "eps": 1e-12
}
```

필드 의미는 다음과 같이 고정한다.

- `enabled`: response-trust를 total objective에 포함할지 결정한다.
- `weight`: 전체 response-trust loss에 곱하는 \(\lambda_{\mathrm{RT}}\).
- `trust_weight`: correction-response 항에 곱하는 \(\mu\). 기본값은 `0.01`.
- `eps`: source-response denominator의 zero guard.

Validation은 다음과 같이 구현한다.

- `enabled`는 정확한 boolean만 허용한다.
- `weight`와 `trust_weight`는 finite, numeric, nonnegative여야 한다.
- `eps`는 finite, numeric, positive여야 한다.
- unknown key는 fail fast한다.
- unit-square CouplingNet에서 `enabled=true`이면 complex-only 오류를 낸다.
- complex mode에서는 `balance_projection.mode="symmetric_tangent_green_response"`가 필요하다.
- `eta_strategy="closed_loop_exact_line_search"`가 아니면 오류를 낸다.
- `response_trust.enabled=true`와 `post_line_search_stationarity.enabled=true`를 동시에 설정하면 두 loss를 분리해 실험하라는 오류를 낸다.
- 두 block이 모두 disabled이면 기존 계산과 metric schema를 그대로 유지한다.

첫 실험에서는 auxiliary loss의 급격한 지배를 피하기 위해 다음처럼 명시하는 것을 권장한다.

```json
"post_line_search_stationarity": {
  "enabled": false
},
"response_trust": {
  "enabled": true,
  "weight": 0.001,
  "trust_weight": 0.01,
  "eps": 1e-12
}
```

`weight=0.001`은 API default가 아니라 첫 paired ablation을 위한 보수적인 실험값이다. Canonical config의 기존 활성 objective를 자동으로 변경하지 않는다.

## Implementation Steps

1. **Strict config 추가**
   - `src/greenonet/config.py`에 `ComplexResponseTrustConfig`를 추가한다.
   - `CouplingTrainingConfig.response_trust`에 연결하고 `from_raw(...)` round-trip을 지원한다.
   - unit-square rejection과 complex tangent-mode validation을 추가한다.
   - response-trust와 stationarity-loss의 상호 배타성을 같은 validation 경로에서 검사한다.

2. **Core response-trust 계산 구현**
   - `src/greenonet/complex_tangent_projection.py`에 `TangentResponseTrustResult` dataclass와 계산 helper를 추가한다.
   - result에는 batch mean, per-sample total, post-mismatch ratio, correction ratio, source-response energy와 source directional responses를 보관한다.
   - `H_x(f/2),H_y(f/2)`는 기존 cached `FrozenBidirectionalResponseOperator.forward_pair(...)`로 한 번 계산한다.
   - `m_0`, `m_post`, 실제 capped \(\eta_{\mathrm{applied}}\)는 production projection diagnostics에서 가져온다.
   - denominator와 frozen operator data는 reference target과 무관하게 유지한다.
   - full matrix materialization, linear solve, 새로운 adjoint action은 추가하지 않는다.

3. **Stationarity diagnostic 분리**
   - response-trust가 활성화되면 현재 normalized post-line-search stationarity ratio도 계속 계산한다.
   - 이 diagnostic은 기존처럼 uncapped \(\eta^\star\)를 사용하고 response-trust의 `eps`를 사용한다.
   - 한 batch당 기존 stationarity diagnostic용 adjoint action 한 번은 유지한다.
   - `loss_tangent_post_line_search_stationarity`는 생성하지 않고 `tangent_post_line_search_stationarity_ratio`만 기록한다.
   - 기존 stationarity loss만 활성화한 config의 수치와 metric contract는 그대로 보존한다.

4. **Shared objective 연결**
   - `src/greenonet/complex_coupling_objective.py`에서 response-trust result를 per-sample total objective에 더한다.
   - trainer와 evaluator가 동일한 helper와 동일한 cached tangent context를 사용하도록 연결한다.
   - training eta-cap schedule에서는 해당 epoch에 실제 적용된 cap을 사용한다.
   - validation/evaluation에서는 현재 production 규칙에 따른 final cap을 사용한다.
   - `best_energy_checkpoint`는 기존 energy 기준을 유지한다.
   - `best_physics_checkpoint`는 response-trust를 포함한 total validation loss를 사용한다.
   - reference `sol/phi/psi` 유무 또는 값이 training loss와 checkpoint 선택에 영향을 주지 않도록 유지한다.

5. **Logging 및 metrics 추가**
   - 다음 metric 이름으로 trainer log, CSV, evaluator per-sample metrics를 통일한다.
   - `loss_tangent_response_trust`: \(\lambda_{\mathrm{RT}}\mathcal L_{\mathrm{RT}}\).
   - `tangent_response_trust_ratio`: weight 적용 전 combined ratio.
   - `tangent_response_post_mismatch_ratio`: \(\ell_{\mathrm{post}}\).
   - `tangent_response_correction_ratio`: \(\ell_{\mathrm{correction}}\).
   - `tangent_source_response_energy`: \(E_f\).
   - `tangent_post_line_search_stationarity_ratio`: 비교용 기존 diagnostic.
   - 시작 로그에는 enabled, weight, trust weight, eps, capped eta 사용, source normalization 수식, 추가 forward/adjoint action 수를 기록한다.

6. **Artifact provenance 확장**
   - `src/greenonet/complex_coupling_artifacts.py`의 summary에 response-trust config, 수식, `eta_source="capped_eta_applied"`, matrix-free 여부, reference-free 여부를 기록한다.
   - per-sample CSV에는 위의 response-trust component metric을 추가한다.
   - selected raw NPZ에는 `tangent_source_response_phi`, `tangent_source_response_psi`, `tangent_response_correction`, source-response energy와 각 ratio를 저장한다.
   - 기존 `tangent_mismatch_pre`, `tangent_mismatch_post`, `tangent_delta` field는 유지한다.
   - response-trust가 활성화된 경우 source-response energy density, actual post mismatch, correction response를 기존 Plotly valid-point style로 시각화한다.
   - feature가 disabled이면 기존 artifact field와 figure count를 바꾸지 않는다.

7. **문서 및 example 정리**
   - `README.md`에 수식, config 예시, 기존 stationarity loss와의 차이, metric 의미를 추가한다.
   - `docs/memory.md`에 response-trust가 capped production response를 직접 제어하고 stationarity는 diagnostic으로만 유지된다는 durable convention을 기록한다.
   - 기존 canonical experiment config는 자동으로 변경하지 않는다.
   - 필요하면 별도 paired-ablation config를 추가하되 기존 stationarity config를 덮어쓰지 않는다.
   - 프로젝트 루트 `PLAN.md`는 사용자가 이 계획을 토대로 직접 작성한다.

## Affected Files

- Config 및 validation: `src/greenonet/config.py`
- Core math: `src/greenonet/complex_tangent_projection.py`, `src/greenonet/complex_projection.py`
- Objective/runtime: `src/greenonet/complex_coupling_objective.py`, `src/greenonet/complex_coupling_trainer.py`, `src/greenonet/complex_coupling_evaluator.py`
- Artifact: `src/greenonet/complex_coupling_artifacts.py`
- Tests: `test/test_complex_tangent_projection.py`, `test/test_complex_coupling_trainer.py`, `test/test_complex_coupling_artifacts.py`, `test/test_io_config.py`, `test/test_cli_train.py`
- Documentation: `README.md`, `docs/memory.md`
- Model, dataset, geometry/sample NPZ, GreenNet checkpoint 및 CouplingNet state-dict key는 변경하지 않는다.

## Test Plan

- **Config**
  - disabled defaults와 JSON/load/save round-trip을 검증한다.
  - invalid boolean, negative/non-finite weight, negative trust weight, nonpositive eps와 unknown key를 거부한다.
  - unit-square 사용, 잘못된 projection mode와 non-closed-loop eta strategy를 거부한다.
  - response-trust와 stationarity loss의 동시 활성화를 거부한다.

- **Math**
  - 작은 analytic response operator에서 \(m_{\mathrm{post}}=m_0+S\delta\)를 검증한다.
  - source-response denominator와 두 loss component를 직접 계산한 값과 비교한다.
  - cap에 걸린 경우 uncapped \(\eta^\star\)가 아니라 actual \(\eta_{\mathrm{applied}}\)를 사용하는지 확인한다.
  - \(\delta=0\)이면 correction ratio가 0인지 확인한다.
  - \(m_{\mathrm{post}}=0\)이면 post-mismatch ratio가 0인지 확인한다.
  - near-zero source에서도 eps를 통해 finite한 결과와 gradient를 반환하는지 확인한다.
  - response mismatch를 확대하면 기존 stationarity ratio와 달리 response-trust가 증가하는 synthetic scale test를 추가한다.
  - 모든 경우 \(\phi+\psi=f\)가 유지되는지 확인한다.

- **Trainer/Evaluator**
  - one-step complex training에서 total loss가 기존 objective와 weighted response-trust의 합인지 검증한다.
  - model parameter까지 finite gradient가 전달되는지 확인한다.
  - stationarity ratio는 기록되지만 weighted stationarity loss key는 생성되지 않는지 확인한다.
  - trainer와 evaluator가 동일한 response-trust 값과 cached context를 사용하는지 확인한다.
  - target `sol/phi/psi`를 변경하거나 제거해도 training loss가 변하지 않는지 확인한다.
  - best-energy와 best-physics checkpoint 기준이 서로 유지되는지 확인한다.

- **Artifact/Regression**
  - summary, per-sample CSV, selected raw NPZ와 figure field를 검증한다.
  - disabled config에서 기존 artifact schema가 그대로인지 확인한다.
  - 기존 stationarity-only path의 수치와 metric contract를 보존한다.
  - unit-square CouplingNet, GreenNet, optimizer, scheduler, reconstruction과 checkpoint key에 regression이 없는지 확인한다.

검증 명령은 다음 순서로 실행한다.

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_complex_tangent_projection.py \
  test/test_complex_coupling_trainer.py \
  test/test_complex_coupling_artifacts.py \
  test/test_io_config.py \
  test/test_cli_train.py

PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest test

ruff check src cli test
ruff format src cli test
~/.conda/envs/green_net/bin/python -m mypy src
git diff --check
```

## Rollback Strategy

- 가장 작은 runtime rollback은 `response_trust.enabled=false`로 변경하는 것이다.
- 기존 stationarity loss로 돌아가려면 response-trust를 끄고 `post_line_search_stationarity.enabled=true`로 설정한다.
- 모델 architecture와 state-dict key가 바뀌지 않으므로 checkpoint migration은 필요하지 않다.
- Code rollback은 response-trust config, core result/helper, objective branch, metrics/artifact fields와 관련 tests만 제거한다.
- Tangent projection, line search, eta-cap schedule, Green reconstruction, canonical/boundary energy와 기존 stationarity 구현은 rollback 과정에서 수정하지 않는다.
- 기존 disabled-path 수치나 stationarity-only 결과를 보존할 수 없다면 integration을 중단하고 차이가 발생한 objective/metric contract를 먼저 보고한다.

## Acceptance Criteria

- response-trust가 config에서 명시적으로 활성화된 경우에만 total loss에 포함된다.
- loss는 actual capped tangent correction 이후의 \(m_{\mathrm{post}}\)를 사용한다.
- normalization은 network output이 아니라 frozen source response \(E_f\)를 사용한다.
- correction trust term은 \(\mu=0.01\)을 기본값으로 사용한다.
- response-trust와 stationarity loss를 동시에 최적화할 수 없다.
- response-trust mode에서도 stationarity ratio는 diagnostic으로 기록된다.
- balance, matrix-free execution과 reference-free training 원칙이 유지된다.
- model/checkpoint tensor contract와 기존 disabled behavior가 변경되지 않는다.
- focused tests, 전체 regression, Ruff, mypy와 `git diff --check`가 통과한다.

## Confidence

- 구현 계획 및 현재 코드와의 통합 가능성에 대한 확신도: **0.98**.
- 기존 stationarity loss보다 Pentagram/Annulus solution 및 flux quality를 개선할 가능성에 대한 경험적 확신도: **0.82**.
- 구현 규칙에는 정보 부족이나 모호성이 없다. 상호 배타성, \(\mu=0.01\), stationarity diagnostic 유지가 모두 확정되었다.
- 남은 불확실성은 규칙 문제가 아니라 `weight`의 최적값과 response-trust 감소가 실제 test solution error에 얼마나 정렬되는지에 대한 실험적 정보 부족이다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Optional Response-Trust Loss 구현 계획"을 기준 문서로 참고하여
response-trust loss의 optional integration을 끝까지 구현한다.

완료는 다음 조건으로 검증한다.

- `coupling_training.response_trust`가 strict config parsing과
  save/load round-trip을 지원할 것,
- response-trust가 actual capped tangent correction 이후의
  directional response mismatch를 사용할 것,
- source-response normalization이 frozen Green-response operator의
  Hx(f/2), Hy(f/2)로 계산될 것,
- correction trust term이 ||S delta||^2를 사용하고 기본 trust weight가
  0.01일 것,
- response-trust와 기존 post-line-search stationarity loss의 동시
  활성화가 거부될 것,
- response-trust 학습 중 기존 stationarity ratio가 diagnostic으로
  계속 기록될 것,
- phi+psi=rhs balance가 모든 valid point에서 유지될 것,
- reference sol, target phi, target psi가 loss, gradient 또는 checkpoint
  선택에 사용되지 않을 것,
- trainer, evaluator와 artifact exporter가 같은 response-trust helper와
  cached Green-response context를 사용할 것,
- logs, metrics CSV, summary JSON과 selected raw NPZ에 config와 component
  metrics가 기록될 것,
- response-trust가 disabled이면 기존 objective, metrics와 artifact
  behavior가 유지될 것,
- CouplingNet/GreenNet architecture와 model checkpoint key가 변경되지
  않을 것,
- unit-square CouplingNet, optimizer, scheduler, projection,
  reconstruction과 canonical energy에 regression이 없을 것,
- focused tests와 전체 pytest, Ruff, mypy, git diff check가 통과할 것.

수정 범위는 complex training config, tangent response-trust math helper,
shared complex objective, trainer/evaluator integration, logging, artifacts,
관련 tests, README 및 docs/memory.md로 제한한다.

Model backbone, dataset, GreenNet 학습, geometry/sample NPZ schema,
full response matrix, Gram solve, learnable trust weight와 reference-target
loss는 추가하지 않는다. 기존 stationarity loss 구현은 삭제하지 않는다.

각 구현 단계 후 가장 작은 config/math/objective tests를 먼저 실행하고,
통과한 뒤 trainer/artifact integration tests와 전체 regression suite를
실행한다. 실제 장기 retraining은 실행하지 않는다.

기존 disabled-path 수치, stationarity-only path 또는 checkpoint architecture
호환성을 유지할 수 없다면 작업을 중단하고 다음을 보고한다.

1. 정확히 달라지는 objective, metric 또는 tensor contract,
2. 영향을 받는 config, checkpoint, artifact와 tests,
3. 기존 동작을 보존하는 가장 작은 rollback 또는 migration 전략.
```

## Open Questions

- 없음.
