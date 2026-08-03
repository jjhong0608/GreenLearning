# Symmetric Tangent Green-Response Method Optional Integration Plan

## Summary

Complex CouplingNet에 **symmetric-balanced directional source를 기준점으로 삼는 one-step tangent Green-response correction**을 optional projection mode로 추가한다.

핵심 목적은 raw output \(p,q\)를 곧바로 directional source의 최종 후보로 간주하지 않고, 먼저 symmetric projection으로 balance-feasible pair \((\widetilde p,\widetilde q)\)를 만든 뒤, reconstructed directional solution의 mismatch를 줄이는 방향으로 source split을 한 번 보정하는 것이다.

새 mode는 다음 원칙을 따른다.

- Complex CouplingNet 전용 opt-in 기능이다.
- 기존 `physical_symmetric`와 `column_diagonal_green_response` mode는 그대로 보존한다.
- \(\phi+\psi=f\) balance는 모든 점에서 정확히 유지한다.
- reference `sol/phi/psi`를 사용하지 않는다.
- global matrix를 구성하거나 matrix equation을 풀지 않는다.
- 학습 가능한 step size, gate, surrogate network를 추가하지 않는다.
- 기존 CouplingNet과 GreenNet architecture 및 checkpoint tensor key를 변경하지 않는다.
- 구현 후 focused smoke test까지만 수행하고 장기 retraining은 실행하지 않는다.
- 기존 config는 수정하지 않고 tangent 전용 실험 config를 별도로 추가한다.

## Tangent Method

### 1. Physical symmetric base pair

현재 CouplingNet raw response를 physical directional-source proposal \(p,q\)로 변환한 후 raw difference를 정의한다.

\[
d_{\mathrm{raw}}=p-q.
\]

먼저 physical symmetric projection을 적용한다.

\[
\widetilde p
=
\frac12\left(f+d_{\mathrm{raw}}\right),
\qquad
\widetilde q
=
\frac12\left(f-d_{\mathrm{raw}}\right).
\]

따라서 다음 balance가 정확히 성립한다.

\[
\widetilde p+\widetilde q=f.
\]

Tangent method는 raw \(p,q\)가 아니라 이 symmetric-balanced pair를 기준점으로 사용한다.

### 2. Frozen directional Green-response operators

Physical directional source에서 reconstructed solution으로 가는 연산자를 다음처럼 정의한다.

\[
u_\phi=H_x\phi,
\qquad
u_\psi=H_y\psi.
\]

여기서

\[
H_x=K_xW_xL_x^2,
\qquad
H_y=K_yW_yL_y^2
\]

이며 각각 다음 요소를 포함한다.

- frozen GreenNet kernel
- segment-local reconstruction quadrature
- physical source에서 reference response로 가는 \(L_x^2,L_y^2\) pull-back
- valid-point assembly
- disconnected axial segment와 endpoint hard-zero 처리

GreenNet은 학습 중 고정되어 있으므로 \(H_x,H_y\)도 geometry와 coefficient가 고정된 run에서는 한 번만 구성해 재사용한다.

### 3. Balance-preserving tangent direction

Symmetric pair에 하나의 feasible correction field \(\delta\)를 적용한다.

\[
\phi(\delta)=\widetilde p+\delta,
\qquad
\psi(\delta)=\widetilde q-\delta.
\]

그러면 임의의 \(\delta\)에 대해

\[
\phi(\delta)+\psi(\delta)=f
\]

가 자동으로 유지된다. 즉 tangent correction은 balance constraint plane 내부에서만 움직인다.

### 4. Response mismatch objective

Directional reconstruction mismatch를 다음처럼 둔다.

\[
m(\delta)
=
H_x(\widetilde p+\delta)
-
H_y(\widetilde q-\delta).
\]

\(\delta=0\)에서의 mismatch는

\[
m_0=H_x\widetilde p-H_y\widetilde q.
\]

고려하는 reference-free response objective는

\[
J(\delta)
=
\frac12
\left\|m(\delta)\right\|_{M_\Omega}^{2},
\]

이며 \(M_\Omega\)는 valid-grid physical quadrature mass다. 현재 uniform Cartesian valid grid에서는 point mass \(h_xh_y\)를 사용한다.

\(\delta=0\)에서의 gradient는

\[
g
=
\nabla_\delta J(0)
=
(H_x+H_y)^\top M_\Omega m_0.
\]

이 계산에는 `sol`, target \(\phi\), target \(\psi\)가 사용되지 않는다.

### 5. Column-diagonal tangent preconditioner

각 physical source coordinate \(j\)의 directional response gain을

\[
\gamma_{x,j}^{2}
=
e_j^\top H_x^\top M_\Omega H_xe_j,
\qquad
\gamma_{y,j}^{2}
=
e_j^\top H_y^\top M_\Omega H_ye_j
\]

로 정의한다.

Preconditioner base와 global relative damping scale은

\[
G_j=\gamma_{x,j}^{2}+\gamma_{y,j}^{2},
\qquad
\overline G=\frac1P\sum_{j=1}^{P}G_j
\]

이고, 실제 denominator는

\[
D_j
=
G_j+
\left(\lambda_{\mathrm{rel}}+\varepsilon_D\right)\overline G
\]

로 둔다.

한 번의 fixed Jacobi-preconditioned tangent step은

\[
\delta_j
=
-\eta\frac{g_j}{D_j}
\]

이다. 최종 physical directional source는

\[
\phi=\widetilde p+\delta,
\qquad
\psi=\widetilde q-\delta.
\]

그 후 기존 pull-back convention에 따라

\[
\Phi=L_x^2\phi,
\qquad
\Psi=L_y^2\psi
\]

를 만들고 Green reconstruction에 사용한다.

이 방법은 다음과 명확히 구분한다.

- **Row-norm method가 아니다.**
- Column-diagonal gain을 balance correction weight로 직접 사용하는 기존 column-diagonal projection이 아니다.
- Full Gram matrix \((H_x+H_y)^\top M_\Omega(H_x+H_y)\)를 구성하지 않는다.
- Cross-column term을 포함한 정확한 Newton step이 아니다.
- Global linear system을 풀지 않는 one-step diagonal-preconditioned gradient correction이다.

## Public Configuration

다음 complex-only projection mode를 추가한다.

```json
"balance_projection": {
  "enabled": true,
  "mode": "symmetric_tangent_green_response",
  "symmetric_tangent_green_response": {
    "eta": 0.01,
    "relative_lambda": 0.01,
    "denominator_relative_eps": 1e-12
  }
}
```

설정 의미는 다음과 같다.

- `eta`: fixed tangent step size. 기본값 `0.01`.
- `relative_lambda`: mean response gain에 대한 relative damping. 기본값 `0.01`.
- `denominator_relative_eps`: zero/near-zero response gain을 위한 relative numerical floor. 기본값 `1e-12`.
- `eta=0`은 symmetric projection과 정확히 같은 결과를 내는 no-op ablation으로 허용한다.
- 모든 값은 sample-independent fixed scalar다.
- `eta`와 `relative_lambda`는 finite, nonnegative여야 한다.
- `denominator_relative_eps`는 finite, positive여야 한다.
- unknown nested key와 잘못된 타입은 fail fast한다.
- Unit-square CouplingNet에서 이 mode를 요청하면 complex-only option이라는 오류를 낸다.

별도 실험 config를 추가한다.

```text
configs/complex_coupling_soap_tangent.json
```

기존 `configs/complex_coupling_soap.json`과 canonical config는 변경하지 않는다. Tangent config는 기존 SOAP 학습 조건을 복제하되 projection block만 tangent mode로 변경한다.

공정한 paired baseline은 같은 config에서 projection만 `physical_symmetric`으로 바꾼 run으로 정의한다. 장기 paired run 자체는 이번 구현에서 실행하지 않는다.

## Implementation Plan

1. `src/greenonet/config.py`에 strict nested tangent config dataclass를 추가하고 `BalanceProjectionConfig.mode`에 `symmetric_tangent_green_response`를 등록한다.
2. 기존 diagnostic용 `complex_axial_response_operator.py`를 production 공용 연산자로 정리해 segment-local \(H_x,H_y\) forward 및 adjoint matvec를 제공한다.
3. Response operator의 cached blocks에서 \(\gamma_x^2,\gamma_y^2\)를 계산하는 column-gain helper를 추가한다. Green kernel을 gain 계산과 reconstruction을 위해 중복 평가하지 않는다.
4. Tangent 전용 immutable context와 cache를 추가한다. Context에는 response operator, point mass, gain squares, \(G\), \(\overline G\), \(D\), build statistics를 저장한다.
5. `complex_projection.py` dispatcher에 tangent branch를 추가한다. Raw response를 physical \(p,q\)로 변환하고 symmetric pair, \(m_0\), \(g\), \(\delta\), 최종 physical/response pair를 순서대로 계산한다.
6. Tangent 계산에서 tensor를 detach하지 않는다. Frozen \(H_x,H_y\) matrix는 parameter가 아니지만 explicit forward/adjoint matvec를 통해 CouplingNet parameter까지 일반적인 first-order autograd가 전달되게 한다.
7. Tangent mode의 최종 reconstruction은 tangent 계산에 사용한 동일 cached \(H_x,H_y\)로 수행한다. GreenNet kernel을 같은 batch에서 다시 평가하지 않으며 기존 reconstruction result contract로 반환한다.
8. Trainer, evaluator, artifact exporter가 같은 projection/reconstruction runtime helper와 context cache를 공유하도록 연결한다. Context는 각 runtime에서 정확히 한 번만 생성한다.
9. Existing canonical energy loss와 checkpoint 선택 기준은 변경하지 않는다. Tangent response objective \(J\)는 correction 방향 계산에만 사용하고 별도의 weighted loss로 추가하지 않는다.
10. Optional cross-axis reconstruction blend는 tangent reconstruction 이후의 기존 downstream diagnostic/final-prediction 단계로 유지한다. Tangent correction 자체는 unblended \(u_\phi-u_\psi\)를 기준으로 계산한다.
11. Model output contract version과 state-dict key는 변경하지 않는다. 기존 complex CouplingNet checkpoint를 post-hoc evaluation 또는 explicit fine-tuning에 로드할 수 있지만, tangent 효과의 결론은 tangent mode로 처음부터 재학습한 paired run에서만 내린다.
12. `README.md`와 `docs/memory.md`에 수식, complex-only 범위, fixed hyperparameter, reference-free 원칙, no-global-solve 정책, post-hoc 결과의 한계를 기록한다.

## Runtime Metrics And Artifacts

Training/evaluation log에는 다음 tangent metadata와 detached diagnostics를 추가한다.

- `tangent_eta`
- `tangent_relative_lambda`
- `tangent_response_mismatch_pre`
- `tangent_response_mismatch_post`
- `tangent_response_mismatch_ratio`
- `tangent_gradient_rms`
- `tangent_delta_rms`
- `tangent_delta_max_abs`
- `tangent_correction_rel_symmetric_pair`
- response context build count와 operator block statistics
- physical balance residual

Artifact raw NPZ에는 다음을 저장한다.

- `symmetric_physical_phi`, `symmetric_physical_psi`
- `symmetric_u_phi`, `symmetric_u_psi`
- `tangent_mismatch_pre`
- `tangent_gradient`
- `tangent_preconditioner_base`
- `tangent_denominator`
- `tangent_delta`
- `projected_physical_phi`, `projected_physical_psi`
- `projected_response_phi`, `projected_response_psi`
- `tangent_u_phi`, `tangent_u_psi`
- `tangent_mismatch_post`

Artifact summary에는 formula, fixed parameters, context provenance, no-reference/no-solve 정책을 기록한다. Existing solution/coefficient/cross-axis figures와 column-diagonal artifacts는 보존한다.

## Affected Files

- Config: `src/greenonet/config.py`
- Projection: `src/greenonet/complex_projection.py`
- Response operator/context: `src/greenonet/complex_axial_response_operator.py` 및 새 tangent context module
- Runtime: `src/greenonet/complex_coupling_trainer.py`, evaluator, artifact exporter
- Experiment config: 새 `configs/complex_coupling_soap_tangent.json`
- Tests: projection, response operator, trainer, evaluator, artifact, config/IO tests
- Documentation: `README.md`, `docs/memory.md`

Model backbone, GreenNet training, optimizer, scheduler, source generator, loss definition, geometry/sample NPZ schema는 수정하지 않는다.

## Test Plan

- **Config:** default/explicit parsing, JSON round-trip, unknown-key rejection, invalid `eta/lambda/eps`, unit-square rejection.
- **Projection:** exact \(\phi+\psi=f\), `eta=0` symmetric equivalence, zero mismatch에서 \(\delta=0\), low-gain point에서 finite denominator와 output.
- **Math:** segment-local forward/adjoint duality, tangent gradient finite-difference 검증, column gain fixture 검증.
- **Autograd:** tangent correction을 포함한 loss gradient가 CouplingNet parameter에 도달하며 response operator와 GreenNet에는 gradient/state change가 없는지 확인.
- **Reconstruction:** cached response operator가 production Green reconstruction과 float64 tolerance 내에서 일치하고, 추가 \(L^2\) scaling이 중복 적용되지 않는지 확인.
- **Integration:** trainer/evaluator/artifact가 같은 fixed context를 사용하고 context가 한 번만 생성되는지 확인.
- **Reference-free:** `sol/phi/psi` target을 변경하거나 제거해도 training loss와 best-energy checkpoint 선택이 바뀌지 않는지 확인.
- **Artifacts:** tangent raw schema, summary formula, figures, exact balance residual, cross/gauge key 부재 검증.
- **Regression:** `physical_symmetric`, `column_diagonal_green_response`, unit-square CouplingNet, complex GreenNet 결과가 유지되는지 확인.
- **Smoke:** 작은 complex fixture에서 tangent mode one-step training/evaluation/export를 실행한다. 장기 retraining은 실행하지 않는다.

검증 명령은 다음과 같이 구성한다.

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_complex_symmetric_tangent_audit.py \
  test/test_complex_projection.py \
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

- Runtime rollback은 `balance_projection.mode`를 `physical_symmetric` 또는 기존 `column_diagonal_green_response`로 변경하는 것으로 완료된다.
- Model architecture와 checkpoint key를 변경하지 않으므로 checkpoint migration은 필요하지 않다.
- Code rollback은 tangent config, tangent context/cache, dispatcher branch, tangent artifact fields와 관련 tests만 제거한다.
- Existing response operator diagnostic과 frozen post-hoc audit 결과는 production mode와 독립적으로 유지할 수 있다.
- 기존 projection 또는 reconstruction의 numerical result가 변하면 tangent integration을 중단하고 shared runtime refactor를 되돌린 뒤 tangent branch를 완전히 격리한다.

## Acceptance Criteria

- Tangent mode가 complex-only optional config로 strict하게 parse된다.
- Symmetric projection 후 fixed tangent correction이 적용된다.
- 모든 valid point에서 \(\phi+\psi=f\)가 float64 tolerance 내에서 유지된다.
- `eta=0`이 기존 physical symmetric 결과와 일치한다.
- Tangent update가 reference target과 global matrix solve 없이 계산된다.
- CouplingNet gradient가 tangent forward/adjoint matvec를 통과한다.
- 동일 cached Green-response operator가 tangent update와 최종 reconstruction에 사용된다.
- Trainer, evaluator, artifact exporter가 동일한 수식과 parameter를 사용한다.
- Model checkpoint key와 output contract는 변경되지 않는다.
- 기존 projection mode와 unit-square/GreenNet regression이 없다.
- 장기 학습은 실행하지 않고 tangent 전용 config와 smoke 검증까지만 완료한다.

## Confidence

- 구현 계획과 코드 적용 가능성에 대한 확신도: **0.97**
- Tangent retraining이 symmetric/column-diagonal training보다 평균 solution error를 개선할 가능성에 대한 경험적 확신도: **0.80**
- Tangent method가 transition-local error까지 개선할 가능성에 대한 경험적 확신도: **0.58**

구현 규칙에는 모호성이 없다. 남은 불확실성은 정보 부족에 해당한다. Frozen-checkpoint post-hoc 결과에서는 global response mismatch와 `rel_sol` 개선이 확인됐지만 transition error가 함께 개선되지는 않았으므로, 최종 효과는 동일 조건의 paired retraining으로 검증해야 한다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Symmetric Tangent Green-Response Method Optional Integration Plan"을 기준
문서로 참고하여 tangent projection의 optional integration을 끝까지 구현한다.

완료는 다음 조건으로 검증한다.

- complex CouplingNet에 `symmetric_tangent_green_response` projection mode가
  optional로 추가될 것,
- symmetric-balanced physical pair를 기준으로 response mismatch gradient와
  fixed diagonal-preconditioned tangent correction을 계산할 것,
- eta, relative lambda, denominator epsilon이 strict config로 parse되고
  save/load round-trip 될 것,
- eta 0이 기존 physical symmetric projection과 정확히 일치할 것,
- 모든 valid point에서 phi+psi=rhs가 float64 tolerance 내에서 보존될 것,
- tangent correction에 reference sol/phi/psi, learned gate, sample-dependent
  parameter, global matrix 또는 linear solve를 사용하지 않을 것,
- frozen segment-local Green-response operator의 forward/adjoint matvec를 통해
  CouplingNet parameter까지 first-order autograd가 전달될 것,
- 동일 cached response operator를 tangent update와 최종 reconstruction에
  사용하고 runtime당 context를 한 번만 생성할 것,
- canonical energy loss, optimizer, scheduler, checkpoint selection과 optional
  cross-axis reconstruction의 기존 의미를 변경하지 않을 것,
- trainer, evaluator, artifact exporter가 동일한 tangent 수식과 context를
  사용할 것,
- summary, raw NPZ, figures와 logs에 tangent parameter, gradient, denominator,
  delta와 pre/post response mismatch가 기록될 것,
- 기존 physical symmetric와 column-diagonal projection, unit-square CouplingNet,
  GreenNet에 regression이 없을 것,
- model architecture, output contract version과 safetensors checkpoint key가
  변경되지 않을 것,
- tangent 전용 paired-experiment config를 별도로 추가하고 기존 config를
  변경하지 않을 것,
- focused tests와 전체 pytest, Ruff, mypy, git diff check가 통과할 것.

수정 범위는 complex projection config, frozen axial response operator,
tangent context/cache, complex trainer/evaluator/artifact integration,
별도 tangent experiment config, 관련 tests, README와 docs/memory.md로 제한한다.

Row-norm projection, full-Gram solve, learnable eta/lambda, sample-dependent tangent
network, 새로운 loss, reference-supervised training, model backbone, GreenNet
training, geometry/sample NPZ schema는 변경하지 않는다.

각 구현 단계 후 가장 작은 config/math/projection tests를 먼저 실행하고,
통과한 뒤 trainer/evaluator/artifact smoke와 전체 regression suite를 실행한다.
실제 장기 paired retraining은 실행하지 않는다.

기존 projection numerical behavior 또는 checkpoint architecture compatibility를
유지할 수 없다면 작업을 중단하고 다음을 보고한다.

1. 정확히 달라지는 projection, reconstruction 또는 tensor contract,
2. 영향을 받는 config, checkpoint, artifact와 tests,
3. 기존 mode를 보존하면서 tangent path를 격리하는 가장 작은 rollback 전략.
```
