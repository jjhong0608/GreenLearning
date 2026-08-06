# Normalized Post-Line-Search Stationarity Loss 구현 계획

## Summary

현재 Complex CouplingNet의 symmetric-tangent projection은 symmetric balance plane에서 Jacobi-preconditioned direction \(z=D^{-1}g\)를 만들고, 그 **고정된 1차원 직선 위에서만** sample별 exact line search를 수행한다. 새 loss는 exact line search 이후에도 남는 **full stationarity residual**을 정규화해 최소화함으로써, CouplingNet이 만든 tangent line이 이상적인 response-mismatch minimizer를 지나도록 유도한다.

이 기능은 complex CouplingNet 전용 opt-in training objective로 추가한다. Projection, Green reconstruction, model architecture, canonical energy, checkpoint tensor key는 변경하지 않는다. Reference `sol/phi/psi`도 사용하지 않는다. Exact line search가 search direction 위의 최적 step만 정한다는 해석은 표준 quadratic exact-line-search 이론과 일치한다. [Exact line-search gradient method 참고](https://arxiv.org/abs/1606.09365)

## Current CouplingNet Contract

Network는 두 directional reference-response proposal \(P,Q\)를 출력한다. Physical source proposal은

\[
p=\frac{P}{L_x^2},
\qquad
q=\frac{Q}{L_y^2}
\]

이고, symmetric balance projection은

\[
\widetilde\phi
=
\frac12\left[f+(p-q)\right],
\qquad
\widetilde\psi
=
\frac12\left[f-(p-q)\right]
\]

로 정확한 balance를 만든다.

Frozen directional response operator를 \(H_x,H_y\)라고 두면 symmetric response mismatch는

\[
m_0
=
H_x\widetilde\phi-H_y\widetilde\psi
\]

이다. Balance plane의 tangent correction은

\[
\phi=\widetilde\phi+\delta,
\qquad
\psi=\widetilde\psi-\delta
\]

이므로 \(\phi+\psi=f\)는 항상 보존된다. 다음을 정의한다.

\[
S=H_x+H_y,
\qquad
J(\delta)=\frac12\lVert m_0+S\delta\rVert_{M_\Omega}^2,
\]

\[
g=S^\top M_\Omega m_0,
\qquad
A=S^\top M_\Omega S.
\]

현재 Jacobi-preconditioned direction과 closed-loop line search는

\[
z=D^{-1}g,
\qquad
\eta_b^\star
=
\frac{g_b^\top z_b}
{\lVert Sz_b\rVert_{M_\Omega}^2+\epsilon_{\mathrm{line},b}},
\]

\[
\eta_{b,\mathrm{applied}}
=
\min(\eta_b^\star,\eta_{\mathrm{cap}}),
\qquad
\delta_b=-\eta_{b,\mathrm{applied}}z_b
\]

이다. 따라서 현재 “exact”는 full-dimensional \(\delta^\star\)가 아니라 고정된 직선 \(\{-\eta z\}\) 위에서 exact하다는 의미다.

## New Loss Contract

Exact line search는 direction \(z\)에 대한 stationarity만 만족시킨다. 이상적인 full stationarity 조건은

\[
\nabla_\delta J(-\eta_b^\star z_b)
=
g_b-\eta_b^\star A z_b
=
0
\]

이다. 새 residual을

\[
r_{\mathrm{stat},b}
=
g_b-\eta_b^\star A z_b
=
S^\top M_\Omega
\left(m_{0,b}-\eta_b^\star Sz_b\right)
\]

로 정의한다.

Normalized post-line-search stationarity loss는

\[
\mathcal L_{\mathrm{stat},b}
=
\frac{
r_{\mathrm{stat},b}^{\top}D^{-1}r_{\mathrm{stat},b}
}{
g_b^{\top}D^{-1}g_b+\epsilon_{\mathrm{stat}}
},
\qquad
\mathcal L_{\mathrm{stat}}
=
\frac1B\sum_{b=1}^{B}\mathcal L_{\mathrm{stat},b}
\]

로 고정한다.

- Loss 계산에는 **uncapped** \(\eta_b^\star\)를 사용한다. 이는 safety cap이 아니라 tangent line 자체의 적합성을 측정하기 위해서다.
- 실제 forward correction에는 기존처럼 capped \(\eta_{b,\mathrm{applied}}\)를 사용한다.
- \(Az\)는 matrix를 만들지 않고 기존 operator action으로 `tangent_gradient(Sz)`를 한 번 더 호출해 계산한다.
- \(\eta^\star\), \(Az\), stationarity residual은 detach하지 않아 CouplingNet output까지 gradient가 전달되게 한다.
- \(A\)가 positive definite이면 loss 0은 tangent line이 full quadratic minimizer를 지난다는 뜻이다. Positive semidefinite이면 response null space를 제외한 stationary minimizer를 뜻한다.
- 이 loss는 learned Green-response surrogate에 대한 조건이며 exact PDE solution을 직접 보장하지 않으므로 canonical energy를 대체하지 않는다.

Total objective는

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{existing}}
+
\lambda_{\mathrm{stat}}\mathcal L_{\mathrm{stat}}
\]

으로 구성한다. `existing`은 현재 설정에 따라 canonical energy 또는 기존 optional relative/weak 항을 포함한 objective다.

## Public Configuration

`coupling_training`에 다음 complex-only block을 추가한다.

```json
"post_line_search_stationarity": {
  "enabled": true,
  "weight": 0.001,
  "eps": 1e-12
}
```

Dataclass 기본값은 기존 실행을 보존하도록 다음과 같이 고정한다.

```text
enabled = false
weight  = 1.0
eps     = 1e-12
```

Validation 규칙:

- `enabled`는 strict boolean이어야 한다.
- `weight`는 finite nonnegative numeric이어야 한다.
- `eps`는 finite positive numeric이어야 한다.
- unknown key는 fail fast한다.
- enabled 상태는 complex geometry, `balance_projection.mode="symmetric_tangent_green_response"`, `eta_strategy="closed_loop_exact_line_search"`에서만 허용한다.
- Unit-square, physical symmetric, column-diagonal, fixed-eta tangent에서 enabled이면 정확한 요구 조건을 포함한 오류를 낸다.
- `weight=0`은 계산과 diagnostic은 유지하지만 optimizer objective에는 영향을 주지 않는 audit mode로 허용한다.

기존 tangent config는 수정하지 않는다. 별도 paired pilot config `configs/complex_coupling_soap_tangent_stationarity.json`을 추가하고, `configs/complex_coupling_soap_tangent.json`과 동일한 조건에서 stationarity block만 `weight=1e-3`으로 활성화한다. Pilot은 `best_energy_checkpoint`와 `best_physics_checkpoint`를 모두 활성화한다.

## Implementation Steps

1. `src/greenonet/config.py`에 strict `ComplexPostLineSearchStationarityConfig`를 추가하고 `CouplingTrainingConfig` parsing, unit-square rejection, projection-mode compatibility validation에 연결한다.
2. `src/greenonet/complex_tangent_projection.py`에 immutable result dataclass와 matrix-free loss helper를 추가한다. 기존 cached response operator, \(D\), `eta_star`, `response_direction=Sz`를 재사용하고 추가 adjoint action으로 \(Az\)를 계산한다.
3. Projection diagnostics에 필요한 uncapped `eta_star`, \(g\), \(Sz\)는 기존 contract를 재사용한다. Forward projection과 capped correction 수식은 변경하지 않는다.
4. `src/greenonet/complex_coupling_objective.py`가 optional stationarity result를 받아 weighted per-sample contribution을 total objective에 더하도록 확장한다.
5. Trainer와 evaluator가 projection 직후 동일 helper를 호출하도록 연결한다. Disabled이면 추가 operator action을 수행하지 않는다.
6. 로그와 CSV에 `loss_tangent_post_line_search_stationarity`와 `tangent_post_line_search_stationarity_ratio`를 추가한다. 전자는 weighted contribution, 후자는 unweighted normalized ratio다.
7. Best-energy checkpoint는 계속 `loss_energy_optimized`로 선택하고, best-physics checkpoint는 새 항을 포함한 total `loss`로 선택한다.
8. Artifact summary에는 config, 수식, uncapped-\(\eta^\star\) convention, reference-free 여부, matrix-free one-adjoint implementation을 기록한다.
9. Per-sample CSV에 unweighted ratio와 weighted contribution을 기록한다. Selected raw NPZ에는 `tangent_hessian_direction`, `tangent_stationarity_residual`, `tangent_stationarity_ratio`를 추가한다. 이번 범위에서는 새 Plotly figure를 추가하지 않는다.
10. README와 `docs/memory.md`에 이 loss가 canonical energy의 대체물이 아니라 tangent-line alignment regularizer라는 durable convention을 기록한다.
11. Model output contract와 safetensors state dict는 변경하지 않는다. 기존 tangent checkpoint도 새 loss를 disabled 또는 enabled 상태로 평가하거나 fine-tune할 수 있다.
12. 장기 학습은 실행하지 않고 작은 trainer smoke와 paired pilot config 생성까지만 수행한다.

## Affected Files

- Config/math/objective: `src/greenonet/config.py`, `src/greenonet/complex_tangent_projection.py`, `src/greenonet/complex_coupling_objective.py`
- Runtime/export: complex trainer, evaluator, artifact exporter
- Experiment/config: 기존 tangent config는 보존하고 별도 stationarity pilot config 추가
- Tests: config, projection math, trainer/evaluator, artifact tests
- Documentation: `README.md`, `docs/memory.md`

현재 작업 트리의 다른 변경사항과 사용자가 작성할 root `PLAN.md`는 구현 과정에서 되돌리거나 덮어쓰지 않는다.

## Test Plan

- **Config:** disabled default, enabled round-trip, invalid boolean/weight/eps/unknown key, unit-square rejection, 잘못된 projection mode와 fixed eta rejection을 검증한다.
- **Math:** synthetic diagonal \(A=D\)에서는 ratio가 0에 가까워지고, non-diagonal \(A\)에서 tangent line이 minimizer를 지나지 않으면 양수가 되는지 확인한다.
- **Normalization:** source/mismatch amplitude scaling에 대한 비율의 불변성, zero-gradient sample의 finite zero result, batch별 독립 계산을 검증한다.
- **Cap separation:** 같은 raw output에서 cap만 바꾸면 forward correction은 달라지지만 uncapped stationarity ratio는 동일한지 확인한다.
- **Autograd:** loss가 CouplingNet parameter까지 finite gradient를 전달하고 `sol/target_phi/target_psi` 변경으로 값이 달라지지 않는지 검증한다.
- **Runtime:** disabled path에서는 추가 adjoint가 없고 기존 objective 결과가 유지되며, enabled path에서는 context를 재생성하지 않고 adjoint action만 한 번 추가되는지 확인한다.
- **Objective/checkpoint:** total loss 항등식, weighted/unweighted metric 구분, best-energy와 best-physics의 독립 선택을 검증한다.
- **Artifacts:** summary, per-sample CSV, selected NPZ schema와 기존 tangent fields 보존을 확인한다.
- **Regression:** projection balance, Green reconstruction, SOAP, scheduler, unit-square CouplingNet, checkpoint tensor keys에 변화가 없는지 확인한다.

검증 순서는 다음과 같다.

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
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

- Runtime rollback은 `post_line_search_stationarity.enabled=false`로 설정하거나 기존 `complex_coupling_soap_tangent.json`을 사용하는 것이다.
- Code rollback은 새 config, matrix-free stationarity helper, objective contribution, metrics/artifact fields, pilot config만 제거한다.
- Projection, tangent line search, canonical energy, model architecture, Green reconstruction에는 rollback 변경이 없어야 한다.
- Model state dict가 변하지 않으므로 checkpoint migration은 필요하지 않다.
- 학습 불안정성이 나타나면 먼저 `weight=0` audit로 gradient 영향과 metric을 분리하고, 이후 paired weight sweep을 수행한다.

## Confidence

- 구현 계획 및 수학적 연결에 대한 확신도: **0.97**.
- 이 loss가 tangent line을 full response minimizer에 더 잘 정렬할 가능성에 대한 확신도: **0.91**.
- 실제 Annulus solution/flux/transition error를 개선할 가능성에 대한 경험적 확신도: **0.74**.
- 규칙과 구현 정보는 충분하다. 남은 불확실성은 규칙 모호성이나 정보 부족이 아니라, learned Green-response stationarity와 실제 PDE accuracy가 얼마나 정렬되는지에 관한 실험적 불확실성이다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Normalized Post-Line-Search Stationarity Loss 구현 계획"을 기준 문서로
참고하여 complex CouplingNet에 optional stationarity loss를 끝까지 구현한다.

완료는 다음 조건으로 검증한다.

- 기존 config에서 새 option을 생략하면 training objective와 projection 결과가
  그대로 유지될 것,
- 새 config가 strict하게 parse되고 save/load round-trip 될 것,
- stationarity loss는 symmetric-tangent closed-loop exact-line-search
  projection에서만 활성화될 것,
- loss 계산에는 uncapped eta_star를 사용하고 실제 projection에는 기존 capped
  eta_applied를 사용할 것,
- Az는 cached segment-local Green response operator의 forward/adjoint action으로
  계산하고 global matrix 또는 linear solve를 만들지 않을 것,
- normalized loss가 sample별로 계산되고 amplitude scaling과 zero-gradient
  edge case에서 finite할 것,
- total loss에 configured weight만큼 추가되며 canonical energy metric과
  best-energy checkpoint 기준은 변경되지 않을 것,
- best-physics checkpoint는 stationarity 항을 포함한 validation total loss를
  사용할 것,
- reference sol, phi, psi는 loss, gradient 또는 checkpoint 선택에 사용되지 않을 것,
- trainer, evaluator와 artifact exporter가 동일한 helper와 수식을 사용할 것,
- logs, CSV, summary와 selected raw NPZ에 stationarity provenance와 diagnostics가
  기록될 것,
- model architecture, safetensors key, GreenNet, unit-square CouplingNet,
  optimizer, scheduler와 reconstruction에 regression이 없을 것,
- 기존 tangent config는 유지되고 별도 weight=1e-3 paired pilot config가
  추가될 것,
- focused tests와 전체 pytest, Ruff, mypy, git diff check가 통과할 것.

수정 범위는 complex training config, tangent stationarity math helper,
shared complex objective, trainer/evaluator wiring, artifact provenance,
paired pilot config, 관련 tests와 README 및 docs/memory.md로 제한한다.

Projection formula, eta-cap scheduler, canonical energy, model backbone,
Green reconstruction, geometry/sample NPZ schema와 기존 checkpoint tensor
contract를 변경하지 않는다. 실제 장기 retraining은 실행하지 않는다.

각 구현 단계 후 가장 작은 config/math tests를 먼저 실행하고, 통과한 뒤
trainer/evaluator/artifact integration tests와 전체 regression suite를 실행한다.

기존 disabled-path 수치 또는 checkpoint architecture compatibility를 유지할 수
없다면 작업을 중단하고 다음을 보고한다.

1. 정확히 달라지는 objective, projection 또는 tensor contract,
2. 영향을 받는 config, checkpoint, artifact와 tests,
3. 기존 tangent behavior를 보존하는 가장 작은 rollback 또는 migration 전략.
```
