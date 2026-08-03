# Closed-Loop Exact-Line-Search Tangent Eta 구현 계획

## Summary

현재 `symmetric_tangent_green_response` projection의 fixed tangent step

\[
\delta_b=-\eta D^{-1}g_b
\]

을 그대로 보존하면서, sample별 response mismatch가 결정하는 closed-loop exact line search를 optional로 추가한다.

\[
m_{0,b}=H_x\widetilde\phi_b-H_y\widetilde\psi_b,
\qquad
g_b=(H_x+H_y)^\top M_\Omega m_{0,b},
\]

\[
z_b=D^{-1}g_b,
\qquad
v_b=(H_x+H_y)z_b.
\]

Tangent direction \(-z_b\)에서 response mismatch를 최소화하는 step은

\[
\eta_b^\star
=
\frac{
\langle m_{0,b},v_b\rangle_{M_\Omega}
}{
\langle v_b,v_b\rangle_{M_\Omega}+\varepsilon_b
}
=
\frac{
g_b^\top D^{-1}g_b
}{
\langle v_b,v_b\rangle_{M_\Omega}+\varepsilon_b
}
\]

로 계산한다. 실제 적용값은 epoch별 safety cap을 사용한다.

\[
\eta_{b,e}
=
\min\left(\eta_b^\star,\eta_{\max,e}\right),
\qquad
\delta_{b,e}=-\eta_{b,e}z_b.
\]

이 방법은 reference `sol/phi/psi`, learnable \(\eta\), batch-global step, full Gram matrix 또는 linear solve를 사용하지 않는다. 모든 sample에서

\[
\phi_b=\widetilde\phi_b+\delta_{b,e},
\qquad
\psi_b=\widetilde\psi_b-\delta_{b,e},
\qquad
\phi_b+\psi_b=f_b
\]

를 정확히 보존한다.

## Public Configuration

기존 `eta`는 그대로 유지하되 역할을 strategy에 따라 구분한다.

```json
"balance_projection": {
  "enabled": true,
  "mode": "symmetric_tangent_green_response",
  "symmetric_tangent_green_response": {
    "eta": 0.01,
    "eta_strategy": "closed_loop_exact_line_search",
    "line_search_relative_eps": 1e-12,
    "relative_lambda": 0.01,
    "denominator_relative_eps": 1e-12
  }
}
```

- `eta_strategy="fixed"`가 기본값이며 기존 `delta=-eta*g/D`를 bitwise하게 보존한다.
- `eta_strategy="closed_loop_exact_line_search"`에서는 `eta`가 최종 safety cap \(\eta_{\max}\)을 의미한다.
- `line_search_relative_eps`는 finite positive float만 허용한다.
- 지원 strategy는 `"fixed"`와 `"closed_loop_exact_line_search"`만 허용한다.
- 기존 config에 새 field가 없으면 자동으로 `"fixed"`가 적용된다.
- `configs/complex_coupling_soap_tangent.json`은 새 실험용으로 `closed_loop_exact_line_search`, `eta=0.01`, `line_search_relative_eps=1e-12`를 명시한다.

Safety cap은 새 warmup field를 추가하지 않고 사용자가 결정한 대로 기존 LR warmup 설정을 공유한다.

- `coupling_training.use_lr_schedule=false` 또는 effective warmup이 0이면 첫 epoch부터 \(\eta_{\max,e}=\eta\)를 사용한다.
- LR schedule이 활성화되면 `coupling_training.warmup_epochs`와 같은 effective warmup 길이를 사용한다.
- LR의 후반 cosine decay는 tangent cap에 적용하지 않는다.
- Zero-based epoch \(k\), effective warmup \(W>0\)에서

\[
s_k=\min\left(\frac{k+1}{W},1\right),
\qquad
\eta_{\max,k}
=
\eta\frac{1-\cos(\pi s_k)}{2}
\]

를 사용하고, warmup 이후에는 \(\eta_{\max,k}=\eta\)로 고정한다.

Training batch에는 scheduled cap을 적용한다. Validation, best-checkpoint 선택, standalone evaluation과 artifact export에는 처음부터 최종 cap \(\eta\)를 적용하여 checkpoint 비교와 deployment inference를 일관되게 유지한다.

## Implementation Steps

1. **Strict config와 schedule contract**
   - `SymmetricTangentGreenResponseProjectionConfig`에 `eta_strategy`와 `line_search_relative_eps`를 추가한다.
   - Unknown key, invalid literal, boolean masquerading as numeric, non-finite 또는 non-positive epsilon을 거부한다.
   - Existing fixed config의 serialize/load round-trip과 수치 결과를 유지한다.
   - 기존 `CouplingLearningRateSchedule`의 `enabled`와 `effective_warmup_epochs`를 재사용하는 immutable tangent-cap schedule helper를 추가한다.

2. **Sample-wise exact line search core**
   - Frozen response context의 \(D\), \(H_x\), \(H_y\), \(M_\Omega\) caching은 그대로 유지한다.
   - `z=g/D`와 directional response \(H_xz,H_yz\)를 계산하고

\[
v=H_xz+H_yz
\]

를 구성한다.
   - 안정화 항은 sample scale을 보존하도록

\[
E_{0,b}=\langle m_{0,b},m_{0,b}\rangle_{M_\Omega},
\qquad
V_b=\langle v_b,v_b\rangle_{M_\Omega},
\]

\[
\varepsilon_b
=
\texttt{line\_search\_relative\_eps}
\max(E_{0,b},V_b)
+
\operatorname{tiny}(\texttt{dtype})
\]

로 둔다.
   - Numerator는 수치적으로 nonnegative인 \(g_b^\top D^{-1}g_b\)로 계산하고, roundoff를 위해 0 이상으로 clamp한다.
   - `eta_star`, cap과의 `minimum`, `delta` 계산을 detach하지 않는다. Frozen Green operator와 \(D\)만 detached 상태를 유지하고 closed-loop projection 전체를 differentiable forward map으로 취급한다.
   - \(E_{0,b}=V_b=0\)이면 `eta_star=0`, `delta=0`을 반환한다.
   - `eta=0` 또는 applied cap이 0이면 physical symmetric result를 bitwise하게 보존한다.

3. **Projection과 reconstruction 연결**
   - `apply_complex_balance_projection(...)`에 optional internal tangent cap override를 추가한다.
   - Override가 없으면 evaluator/export 의미인 최종 cap `config.eta`를 사용한다.
   - Adaptive path는 symmetric response와 \(H_xz,H_yz\)를 이용해 projected response를 대수적으로 조립하여 현재 fixed path보다 response-operator application 횟수를 늘리지 않는다.
   - Final source는 항상 `phi=symmetric_phi+delta`, `psi=symmetric_psi-delta`로 구성한다.
   - Existing Green reconstruction reuse, response pull-back, canonical energy objective와 cross-axis reconstruction은 변경하지 않는다.
   - Full response matrix, row norm, inverse, `torch.linalg.solve`와 iterative linear solver는 추가하지 않는다.

4. **Trainer, validation과 checkpoint 정책**
   - 각 training epoch 시작 시 LR schedule의 effective warmup으로 `eta_cap_train`을 계산하고 모든 training batch에 동일한 cap을 전달한다.
   - Validation은 scheduled training cap과 무관하게 항상 final `eta` cap을 사용한다.
   - `best_energy`와 `best_physics` checkpoint도 final-cap validation metric으로 선택한다.
   - Periodic/final checkpoint는 계속 model-only safetensors이며 optimizer, scheduler 또는 sample eta state를 저장하지 않는다.
   - Projection strategy와 final cap은 `config_used.json`으로 재현한다. Model architecture와 state-dict key는 변경하지 않는다.

5. **Evaluator, artifacts와 telemetry**
   - Standalone evaluator와 artifact exporter는 final cap으로 sample별 exact line search를 재계산한다.
   - Training CSV/log에 다음 항목을 추가한다:
     - `tangent_eta_cap`
     - `tangent_eta_star_mean`
     - `tangent_eta_applied_mean`
     - `tangent_eta_cap_fraction`
     - `tangent_line_search_numerator_mean`
     - `tangent_line_search_denominator_mean`
   - Per-sample evaluation CSV에는 `tangent_eta_star`, `tangent_eta_applied`, `tangent_eta_capped`를 기록한다.
   - Selected raw NPZ에는 `tangent_response_direction`, line-search numerator/denominator, `eta_star`, `eta_applied`, `eta_cap`을 추가한다.
   - Artifact summary에는 strategy, final cap, shared LR-warmup convention, validation final-cap policy, differentiable status와 exact-line-search 수식을 기록한다.
   - Context NPZ는 geometry-dependent \(D\)와 gain field를 계속 한 번만 저장하고, sample-dependent eta 값은 selected/per-sample artifact에만 저장한다.
   - Artifact aggregate에는 eta-star/applied min, median, mean, p95, max와 cap-hit fraction을 기록한다.

6. **Documentation**
   - `README.md`에 fixed와 closed-loop strategy config, safety-cap warmup과 validation 정책을 추가한다.
   - `docs/memory.md`에 다음 durable convention을 기록한다:
     - closed-loop eta는 sample별 deterministic physics correction이다.
     - reference target과 batch composition을 사용하지 않는다.
     - LR warmup 기간만 공유하고 LR cosine decay는 공유하지 않는다.
     - validation/evaluation/export는 final cap을 사용한다.
     - fixed strategy는 기존 behavior를 보존한다.

## Affected Files

- Config/schedule: `src/greenonet/config.py`, `src/greenonet/coupling_lr_scheduler.py` 또는 tangent 전용 schedule helper.
- Core math: `src/greenonet/complex_tangent_projection.py`, `src/greenonet/complex_projection.py`.
- Runtime: `src/greenonet/complex_coupling_trainer.py`, `src/greenonet/complex_coupling_evaluator.py`.
- Provenance/artifacts: `src/greenonet/complex_coupling_artifacts.py`.
- Experiment config: `configs/complex_coupling_soap_tangent.json`.
- Tests: `test/test_complex_tangent_projection.py`, `test/test_complex_coupling_trainer.py`, `test/test_complex_coupling_artifacts.py`, `test/test_io_config.py`.
- Documentation: `README.md`, `docs/memory.md`.

## Test Plan

- **Config**
  - Existing config omission이 `eta_strategy="fixed"`로 round-trip 되는지 확인한다.
  - Adaptive strategy와 epsilon이 serialize/load 되는지 확인한다.
  - Invalid strategy, zero/negative/non-finite epsilon, unknown key와 boolean numeric을 거부한다.

- **Closed-form math**
  - Small deterministic \(H_x,H_y,D,M_\Omega\) fixture에서 analytic \(\eta^\star\)와 구현값을 비교한다.
  - Uncapped adaptive result가 chosen Jacobi direction에서 최소가 되고

\[
\langle m_{\mathrm{post}},v\rangle_{M_\Omega}\approx0
\]

인지 확인한다.
  - Applied eta에서 mismatch가 pre-mismatch보다 증가하지 않는지 확인한다.
  - Cap이 작으면 정확히 `min(eta_star, eta_cap)`이 적용되는지 확인한다.
  - Source amplitude scaling에 대해 `eta_star`가 invariant인지 확인한다.
  - Zero mismatch, zero response direction과 tiny source에서 NaN/Inf가 발생하지 않는지 확인한다.

- **Projection invariants**
  - 모든 strategy와 cap에서 `phi+psi==rhs`를 float64 tolerance 내에서 검증한다.
  - `eta=0`과 fixed strategy의 기존 projection 결과를 bitwise regression으로 검증한다.
  - Sample을 단독 또는 다른 batch 구성으로 평가해도 `eta_star`, `eta_applied`, output이 동일한지 확인한다.
  - `eta_star`를 포함한 projection graph에서 CouplingNet raw response까지 finite gradient가 전달되는지 확인한다.
  - `torch.linalg.solve`를 monkeypatch로 금지해 full solve가 호출되지 않는지 확인한다.

- **Schedule/trainer**
  - LR schedule off에서 첫 epoch부터 final cap을 사용하는지 확인한다.
  - LR warmup \(W\)에서 half-cosine cap sequence와 warmup 이후 hold를 검증한다.
  - Training은 scheduled cap, validation은 final cap을 사용하는지 확인한다.
  - Best-energy/best-physics checkpoint가 final-cap validation으로 선택되는지 확인한다.
  - Frozen response context가 trainer/evaluator에서 한 번만 생성되는지 확인한다.
  - Compile, gradient clipping, SOAP/AdamW와 LR scheduler 순서가 유지되는지 확인한다.

- **Artifacts/regression**
  - Summary, CSV와 NPZ에 strategy, eta distribution, cap policy와 line-search tensors가 기록되는지 확인한다.
  - Fixed tangent artifact의 기존 fields와 의미를 유지한다.
  - Reference `sol/phi/psi`를 변경해도 projection과 training objective가 변하지 않는지 확인한다.
  - Unit-square CouplingNet, physical symmetric, column diagonal, GreenNet, optimizer와 checkpoint keys에 regression이 없는지 확인한다.

검증 순서는 focused config/math tests, trainer/evaluator/artifact integration, 전체 suite, static checks로 한다.

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_complex_tangent_projection.py \
  test/test_complex_coupling_trainer.py \
  test/test_complex_coupling_artifacts.py \
  test/test_io_config.py

PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest test

ruff check src cli test
ruff format src cli test
~/.conda/envs/green_net/bin/python -m mypy src
git diff --check
```

## Rollback Strategy

- Runtime rollback은 `eta_strategy`를 `"fixed"`로 바꾸거나 field를 제거하는 것이다.
- Fixed strategy는 기존 `eta=0.01` 수식과 artifact 의미를 그대로 유지한다.
- Model architecture와 checkpoint tensor key가 바뀌지 않으므로 checkpoint migration은 필요하지 않다.
- Adaptive branch에 문제가 있으면 line-search config, cap helper, adaptive diagnostics와 관련 tests만 제거한다.
- Frozen response operator, fixed Jacobi tangent path, projection balance, Green reconstruction, optimizer, loss와 dataset schema는 rollback 대상이 아니다.
- Existing fixed numerical regression이나 checkpoint architecture compatibility를 유지할 수 없으면 구현을 중단하고 정확한 차이와 최소 rollback을 보고한다.

## Confidence

- 구현 계획과 수학적 타당성에 대한 확신도: **0.98**
- Fixed \(\eta=0.01\)보다 실제 학습과 solution quality가 개선될 가능성에 대한 경험적 확신도: **0.84**
- 규칙 모호성이나 필수 정보 부족은 없다. 남은 불확실성은 sample별 response-\(L^2\) exact line search가 canonical-energy 최적화 및 detached solution metric과 얼마나 잘 정렬되는지에 관한 실험적 불확실성이다.
- 이번 구현은 response-\(L^2\) line search만 포함하며 canonical-energy line search는 추가하지 않는다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Closed-Loop Exact-Line-Search Tangent Eta 구현 계획"을 기준 문서로 참고하여
sample-wise adaptive tangent eta integration을 끝까지 구현한다.

완료는 다음 조건으로 검증한다.

- 기존 tangent config에서 `eta_strategy`를 생략하면 fixed eta 결과가
  bitwise하게 유지될 것,
- `eta_strategy="closed_loop_exact_line_search"`가 sample별 analytic
  `eta_star`를 response-L2 exact line-search 수식으로 계산할 것,
- 실제 eta는 `min(eta_star, eta_cap)`이고 LR warmup 기간을 공유하는
  half-cosine safety cap 이후 final eta로 고정될 것,
- training에는 scheduled cap, validation/checkpoint/evaluation/artifact에는
  final cap이 적용될 것,
- eta 계산을 detach하지 않고 CouplingNet까지 finite gradient가 전달될 것,
- batch composition과 무관하게 같은 sample의 eta와 projection 결과가 같을 것,
- 모든 sample에서 `phi+psi=rhs`가 정확히 보존될 것,
- reference sol/phi/psi, learnable eta, batch-global eta, full matrix,
  row norm 또는 linear solve가 사용되지 않을 것,
- frozen Green-response context가 한 번만 생성되고 기존 reconstruction과
  canonical-energy objective가 유지될 것,
- logs, training/evaluation CSV, raw NPZ와 summary에 eta strategy, cap,
  eta-star/applied 통계와 line-search provenance가 기록될 것,
- `configs/complex_coupling_soap_tangent.json`이 adaptive strategy와
  final cap 0.01을 명시할 것,
- model architecture, state-dict key, unit-square CouplingNet, GreenNet,
  optimizer, scheduler와 dataset schema가 변경되지 않을 것,
- focused tests와 전체 pytest, Ruff, mypy, git diff check가 통과할 것.

수정 범위는 tangent projection config와 fixed response context, sample-wise
line-search math, LR-warmup-linked cap, complex trainer/evaluator/artifact,
tangent experiment config, 관련 tests와 문서로 제한한다.

Canonical-energy line search, learnable/global/sample-network eta, full-Gram
solve, iterative solver, row-norm projection, model backbone, training loss,
Green reconstruction 및 geometry/sample NPZ schema는 변경하지 않는다.

각 구현 단계 후 가장 작은 config/math/projection tests를 먼저 실행하고,
통과한 뒤 trainer/evaluator/artifact integration과 전체 regression suite를
실행한다. 실제 장기 retraining은 실행하지 않는다.

기존 fixed tangent 수치 또는 checkpoint architecture compatibility를 유지할 수
없다면 작업을 중단하고 다음을 보고한다.

1. 정확히 달라지는 projection 수식, floating-point result 또는 tensor contract,
2. 영향을 받는 config, checkpoint, artifact와 tests,
3. fixed eta path를 보존하는 가장 작은 rollback 또는 migration 전략.
```
