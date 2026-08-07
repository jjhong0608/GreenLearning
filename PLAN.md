# Source-Normalized Stationarity + Response-Trust Joint Objective 구현 계획

## Summary

기존 post-line-search stationarity loss의 목적은 유지하되, initial-gradient 기준의 scale-invariant ratio를 **source-response 기준의 absolute-relative residual**로 교체한다. 기존 ratio는 tangent-line alignment diagnostic으로 계속 기록하고, 새로운 source-normalized stationarity와 response-trust를 동시에 total objective에 포함할 수 있도록 현재의 상호 배타 제약을 제거한다.

기존 수식은

\[
S=H_x+H_y,\qquad
A=S^\top M_\Omega S,\qquad
g=S^\top M_\Omega m_{\mathrm{pre}},\qquad
z=D^{-1}g,
\]

\[
r_{\mathrm{stat}}
=
g-\eta^\star Az
\]

에 대해

\[
\mathcal L_{\mathrm{stat,old}}
=
\frac{
r_{\mathrm{stat}}^\top D^{-1}r_{\mathrm{stat}}
}{
g^\top D^{-1}g+\varepsilon
}
\]

를 사용한다. 이 값은 mismatch gradient의 전체 크기가 변해도 거의 변하지 않으므로 loss에서는 폐기하고 diagnostic으로만 유지한다.

새 stationarity loss는 response-trust와 동일한 source-response normalization을 사용한다.

\[
E_f
=
\left\|H_x(f/2)\right\|_{M_\Omega}^{2}
+
\left\|H_y(f/2)\right\|_{M_\Omega}^{2},
\]

\[
\boxed{
\mathcal L_{\mathrm{stat,new}}
=
\frac{
r_{\mathrm{stat}}^\top D^{-1}r_{\mathrm{stat}}
}{
E_f+\varepsilon
}
}
\]

Stationarity는 계속 **uncapped** \(\eta^\star\)를 사용하여 tangent-line alignment를 측정한다. Response-trust는 기존처럼 실제 production correction의

\[
\eta_{\mathrm{applied}}
=
\min(\eta^\star,\eta_{\mathrm{cap}})
\]

를 사용한다.

Combined objective는

\[
\boxed{
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{split\ objective}}
+
\mathcal L_{\mathrm{weak}}
+
\lambda_{\mathrm{RT}}\mathcal L_{\mathrm{RT}}
+
\lambda_{\mathrm{stat}}\mathcal L_{\mathrm{stat,new}}
}
\]

로 고정한다. Combined Pentagram pilot에서는 현재 설정을 보존하여 weak/relative-split은 비활성 상태이고,

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{energy}}
+
10^{-3}\mathcal L_{\mathrm{RT}}
+
10^{-4}\mathcal L_{\mathrm{stat,new}}
\]

를 사용한다. Response-trust 내부 `trust_weight`는 기존 `0.01`을 유지한다.

## Public Contract

- 기존 `coupling_training.post_line_search_stationarity` schema를 그대로 유지한다.

```json
"post_line_search_stationarity": {
  "enabled": true,
  "weight": 0.0001,
  "eps": 1e-12
}
```

- `normalization` 또는 legacy-mode option을 추가하지 않는다. `enabled=true`이면 앞으로 항상 source-normalized stationarity를 최적화한다.
- 기존 config를 다시 실행하면 stationarity loss 의미가 새 수식으로 바뀐다. 기존 model checkpoint tensor와 safetensors key는 바뀌지 않는다.
- 기존 `tangent_post_line_search_stationarity_ratio`는 다음 old ratio diagnostic으로 유지한다.

\[
\frac{E_{\mathrm{res}}}{E_{\mathrm{init}}+\varepsilon},
\qquad
E_{\mathrm{res}}=r_{\mathrm{stat}}^\top D^{-1}r_{\mathrm{stat}},
\qquad
E_{\mathrm{init}}=g^\top D^{-1}g.
\]

- 새 metric contract는 다음으로 고정한다.
  - `loss_tangent_post_line_search_stationarity`: config `weight`가 곱해진 새 optimized loss
  - `tangent_post_line_search_stationarity_source_normalized`: 새 unweighted source-normalized loss
  - `tangent_post_line_search_stationarity_ratio`: 기존 initial-gradient-relative diagnostic
  - `tangent_stationarity_initial_source_ratio`: \(E_{\mathrm{init}}/(E_f+\varepsilon)\)
  - `tangent_source_response_energy`: sample-mean \(E_f\)
- `response_trust.enabled=true`와 `post_line_search_stationarity.enabled=true`를 동시에 허용한다.
- 두 항 모두 complex `symmetric_tangent_green_response`와 `eta_strategy="closed_loop_exact_line_search"`를 계속 필수로 요구한다.
- Unit-square CouplingNet에서는 두 항을 계속 거부한다.
- Auxiliary weight는 fixed scalar로 적용한다. Scheduler, learnable weight, GradNorm은 추가하지 않는다.

## Implementation Changes

1. **공통 source-response normalization**
   - `f/2`에 대한 \(H_x,H_y\) response와 sample별 \(E_f\)를 반환하는 shared dataclass/helper를 `complex_tangent_projection`에 추가한다.
   - Response-trust와 stationarity가 동일 helper 결과를 받도록 하여 \(H_x(f/2),H_y(f/2)\)를 중복 계산하지 않는다.
   - 두 loss가 모두 비활성화되면 source-response forward를 수행하지 않는다.
   - Stationarity-only는 source normalization을 위해 forward-pair 한 번, response-trust 또는 combined mode는 현재 response-trust와 동일하게 forward-pair 한 번만 사용한다.

2. **Stationarity 수식 교체**
   - `NormalizedPostLineSearchStationarityResult`의 optimized `loss/loss_per_sample`을 \(E_{\mathrm{res}}/(E_f+\varepsilon)\)로 변경한다.
   - 기존 \(E_{\mathrm{res}}/(E_{\mathrm{init}}+\varepsilon)\)는 별도 `relative_ratio` 필드로 보존한다.
   - `initial_source_ratio=E_{\mathrm{init}}/(E_f+\varepsilon)`와 source energy를 결과 contract에 추가한다.
   - \(r_{\mathrm{stat}}=g-\eta^\star Az\), uncapped \(\eta^\star\), Jacobi \(D\), cached matrix-free adjoint 계산은 변경하지 않는다.
   - Global matrix, full Gram matrix, linear solve를 추가하지 않는다.

3. **Joint trainer/evaluator objective**
   - Config validation에서 response-trust와 stationarity의 mutual-exclusion 오류를 제거한다.
   - Trainer와 evaluator의 현재 `if/elif` stationarity 분기를 독립적인 두 optional 계산으로 바꾼다.
   - Stationarity residual은 stationarity가 optimized이거나 response-trust diagnostic이 필요한 경우 한 번만 계산한다.
   - Shared objective가 response-trust와 stationarity weighted term을 각각 total loss에 더하도록 변경한다.
   - `best_energy_checkpoint`는 계속 `loss_energy_optimized`만 사용하고, `best_physics_checkpoint`는 두 auxiliary term을 포함한 total `loss`를 사용한다.
   - `sol`, target `phi/psi`, `rel_sol`, `rel_flux`는 loss나 checkpoint 선택에 사용하지 않는다.

4. **Logging과 artifacts**
   - Training/evaluation log에 stationarity normalization, uncapped eta source, response-trust와의 joint-enabled 여부, fixed weights, shared source-response reuse를 기록한다.
   - CSV에 새 optimized/unweighted/legacy diagnostic metric을 모두 기록한다.
   - Artifact summary에는 새 formula, old ratio의 diagnostic-only 상태, simultaneous optimization 여부, shared \(E_f\) convention을 기록한다.
   - Selected raw NPZ에 source-normalized stationarity, old relative ratio, initial/source ratio, \(E_f\), 기존 stationarity residual과 Hessian-direction을 저장한다.
   - `tangent_source_response_*` field는 response-trust 또는 stationarity 중 하나라도 활성화되면 한 번만 생성한다.
   - Existing response-trust figures/raw fields와 cross-key 부재 contract를 유지한다.

5. **Config와 문서**
   - 현재 사용자 수정 파일 `configs/complex_coupling_soap_tangent_stationarity.json`은 되돌리거나 덮어쓰지 않는다.
   - 현재 Pentagram 설정을 기준으로 새 `configs/complex_coupling_soap_tangent_response_trust_stationarity.json`을 추가한다.
   - 새 config는 `response_trust.weight=1e-3`, `trust_weight=0.01`, `post_line_search_stationarity.weight=1e-4`로 두 항을 모두 활성화한다.
   - Geometry, source provider, optimizer, scheduler, tangent cap, boundary-off 및 cross-axis reconstruction 설정은 원본과 동일하게 유지한다.
   - README와 `docs/memory.md`를 새 stationarity 의미, joint objective, metric 구분, fixed-weight pilot 설정에 맞게 갱신한다.
   - `PLAN.md`는 사용자가 이 계획을 프로젝트 루트에 작성하며, 구현 과정에서는 해당 문서를 기준 문서로 사용한다.

## Affected Files

- Core math와 projection: `src/greenonet/complex_tangent_projection.py`, `src/greenonet/complex_projection.py`
- Config와 objective: `src/greenonet/config.py`, `src/greenonet/complex_coupling_objective.py`
- Runtime integration: `src/greenonet/complex_coupling_trainer.py`, `src/greenonet/complex_coupling_evaluator.py`
- Artifact provenance: `src/greenonet/complex_coupling_artifacts.py`
- Experiment config: 새 `configs/complex_coupling_soap_tangent_response_trust_stationarity.json`
- Tests: `test/test_complex_tangent_projection.py`, `test/test_complex_coupling_trainer.py`, `test/test_complex_coupling_artifacts.py`, `test/test_io_config.py`, `test/test_cli_train.py`
- Documentation: `README.md`, `docs/memory.md`

## Test Plan

- **Stationarity math**
  - 새 loss가 \(E_{\mathrm{res}}/(E_f+\varepsilon)\)와 정확히 일치하는지 검증한다.
  - 기존 ratio가 동일한 값으로 diagnostic에 보존되는지 검증한다.
  - Source를 고정하고 mismatch/gradient를 \(c\)배 하면 새 loss가 \(c^2\)배 되고 old ratio는 유지되는지 확인한다.
  - Source와 prediction response를 함께 \(c\)배 하면 새 loss가 scale invariant한지 확인한다.
  - Exact stationary case는 새 loss와 old ratio가 모두 0에 가까운지 확인한다.
  - Zero/near-zero source에서 `eps`로 finite loss와 finite gradient가 유지되는지 검증한다.

- **Shared computation**
  - Stationarity-only에서 source-response forward-pair가 정확히 한 번 호출되는지 확인한다.
  - Response-trust-only와 combined mode에서도 source-response forward-pair가 한 번만 호출되는지 확인한다.
  - Combined mode의 stationarity adjoint 계산이 중복되지 않는지 확인한다.
  - `torch.linalg.solve` 또는 global matrix materialization이 호출되지 않는지 검증한다.

- **Config와 total objective**
  - Response-trust와 stationarity 동시 enabled config가 strict parse/round-trip 되는지 확인한다.
  - 두 항의 tangent-mode/eta-strategy 요구사항과 unit-square rejection을 유지한다.
  - Total loss가 energy, weighted response-trust, weighted source-normalized stationarity의 정확한 합인지 확인한다.
  - 각 항을 독립적으로 끌 수 있고 모두 끄면 기존 energy-only objective가 복원되는지 확인한다.
  - Reference `sol/phi/psi`를 바꾸어도 training loss와 best-checkpoint 선택이 변하지 않는지 검증한다.

- **Trainer/evaluator/artifact**
  - Trainer와 evaluator가 같은 source normalization과 loss 값을 반환하는지 확인한다.
  - Log/CSV에 weighted new loss, unweighted new loss, old ratio, initial/source ratio, source energy가 모두 기록되는지 확인한다.
  - `best_energy`와 `best_physics`가 각각 기존 energy와 combined total 기준으로 저장되는지 확인한다.
  - Artifact summary, raw NPZ와 Plotly field schema가 joint mode를 정확히 기록하는지 검증한다.
  - Model checkpoint key, output contract, GreenNet, unit-square behavior와 기존 cross-key 부재에 regression이 없는지 확인한다.

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

실제 장기 Pentagram/Annulus 재학습은 구현 범위에 포함하지 않는다.

## Rollback Strategy

- Runtime rollback은 `response_trust.enabled=false`와 `post_line_search_stationarity.enabled=false`로 energy-only objective를 복원하는 것이다.
- 개별 ablation은 둘 중 하나만 비활성화하여 energy+response-trust 또는 energy+stationarity로 실행한다.
- Model architecture와 checkpoint tensor가 바뀌지 않으므로 checkpoint migration은 필요하지 않다.
- Code rollback은 stationarity denominator를 `E_initial`로 되돌리고 mutual-exclusion validation을 복원하면 된다.
- 기존 ratio 계산과 artifact field를 계속 보존하므로 새 normalization이 실패해도 legacy 수치 진단을 잃지 않는다.
- AdamW/SOAP, scheduler, projection correction, reconstruction, geometry/sample NPZ 및 GreenNet에는 rollback 변경이 없어야 한다.

## Acceptance Criteria

- Optimized stationarity가 source-response-normalized residual을 사용한다.
- Existing initial-gradient-relative ratio는 동일한 diagnostic 값으로 유지된다.
- Response-trust와 stationarity를 동시에 활성화할 수 있다.
- Combined mode의 total loss가 energy + fixed-weight response-trust + fixed-weight stationarity와 정확히 일치한다.
- Source-response 계산과 stationarity residual 계산이 중복되지 않는다.
- 모든 loss와 checkpoint 선택은 reference-free다.
- 새 Pentagram combined pilot config가 `RT weight=1e-3`, `stationarity weight=1e-4`를 명시한다.
- Model/GreenNet checkpoint key와 geometry/sample schema가 변경되지 않는다.
- Focused/full tests와 Ruff, mypy, `git diff --check`가 통과한다.

## Confidence

- 구현 계획과 수식 정합성에 대한 확신도: **0.98**
- 세 항의 joint optimization이 기존 response-trust-only보다 실제 solution quality를 개선할 가능성에 대한 경험적 확신도: **0.76**
- 규칙 모호성이나 구현에 필요한 정보 부족은 없다.
- 남은 불확실성은 fixed auxiliary weight의 실제 gradient 비율과 Pentagram/Annulus에서의 개선 폭에 관한 실험적 불확실성이다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Source-Normalized Stationarity + Response-Trust Joint Objective 구현 계획"을
기준 문서로 참고하여 stationarity loss 교체와 joint objective integration을
끝까지 구현한다.

완료는 다음 조건으로 검증한다.

- 기존 post-line-search stationarity config schema를 유지하면서 optimized loss가
  source-response-normalized residual을 사용할 것,
- 기존 initial-gradient-relative stationarity ratio가 diagnostic으로 동일하게
  유지될 것,
- response-trust와 새 stationarity를 동시에 활성화할 수 있을 것,
- combined total loss가 energy, fixed-weight response-trust 및 fixed-weight
  stationarity의 정확한 합일 것,
- response-trust와 stationarity가 Hx(f/2), Hy(f/2) source-response 계산을
  중복하지 않을 것,
- uncapped eta-star stationarity와 capped applied-eta response-trust 의미가
  각각 유지될 것,
- reference sol/phi/psi가 loss, gradient 또는 checkpoint selection에 사용되지
  않을 것,
- best-energy와 best-physics checkpoint 기준이 서로 구분되어 유지될 것,
- 새 Pentagram combined pilot config가 response-trust weight 1e-3,
  stationarity weight 1e-4, trust weight 0.01을 명시할 것,
- model architecture, checkpoint tensor key, GreenNet, unit-square CouplingNet,
  geometry/sample NPZ schema가 변경되지 않을 것,
- focused tests와 전체 pytest, Ruff, mypy, git diff check가 통과할 것.

수정 범위는 tangent stationarity/source normalization core, complex config와
objective, trainer/evaluator, artifact provenance, 새 paired config, 관련 tests,
README 및 docs/memory.md로 제한한다.

Learnable/adaptive loss weight, auxiliary-weight scheduler, legacy normalization
option, full response Gram matrix, linear solve, row-norm projection, model
backbone, Green reconstruction 및 장기 retraining은 추가하지 않는다.

각 구현 단계 후 가장 작은 stationarity/config tests를 먼저 실행하고, joint
trainer/artifact tests를 거쳐 전체 regression suite를 실행한다.

새 normalization을 적용하면서 기존 model checkpoint compatibility 또는
reference-free 원칙을 유지할 수 없다면 작업을 중단하고 다음을 보고한다.

1. 정확히 충돌하는 loss, tensor 또는 config contract,
2. 영향을 받는 config, checkpoint, metrics와 artifacts,
3. 기존 ratio diagnostic과 energy-only 경로를 보존하는 가장 작은 rollback 전략.
```
