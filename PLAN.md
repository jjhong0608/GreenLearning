# Complex Pre-Projection Single Nonlinear Residual MLP 구현 계획

## Summary

Complex CouplingNet의 optional pre-projection fuser를 기존 linear/nonlinear 이중 branch 구조에서 **하나의 nonlinear residual MLP**로 교체한다.

각 valid physical point에서 base reference response를 physical directional source proposal로 변환한다.

\[
p_{\mathrm{base}}=\frac{P_{\mathrm{base}}}{L_x^2},
\qquad
q_{\mathrm{base}}=\frac{Q_{\mathrm{base}}}{L_y^2},
\]

\[
d_{\mathrm{base}}=p_{\mathrm{base}}-q_{\mathrm{base}}.
\]

Pointwise source scale은 기존 정의를 유지한다.

\[
A=
\sqrt{\frac{A_x^2+A_y^2}{2}},
\qquad
A_{\mathrm{safe}}=\max(A,\varepsilon).
\]

새 MLP는 geometry와 length를 직접 받지 않고 normalized physical values만 입력받는다.

\[
z=
\left[
\frac{d_{\mathrm{base}}}{A_{\mathrm{safe}}},
\frac{f}{A_{\mathrm{safe}}}
\right],
\qquad
r_\theta=\operatorname{MLP}_\theta(z),
\]

\[
d_{\mathrm{fused}}
=
d_{\mathrm{base}}
+
A_{\mathrm{safe}}r_\theta.
\]

Physical pre-projection pair는 source balance를 사용해 직접 구성한다.

\[
\phi_{\mathrm{pre}}
=
\frac{f+d_{\mathrm{fused}}}{2},
\qquad
\psi_{\mathrm{pre}}
=
\frac{f-d_{\mathrm{fused}}}{2}.
\]

이후 기존 physical symmetric projection과 reference-response pull-back, Green reconstruction, canonical energy loss를 그대로 사용한다. 기존 pre-projection checkpoint/config/artifact 호환이나 migration은 구현하지 않는다.

## Public Config

기존 `coupling_model.pre_projection_fusion` 위치와 `enabled` switch는 유지하되, config schema를 다음으로 교체한다.

```json
"pre_projection_fusion": {
  "enabled": true,
  "hidden_dim": 16,
  "depth": 1,
  "eps": 1e-12
}
```

- `enabled=false`이면 fuser를 생성하지 않고 현재 base response를 바로 projection에 전달한다.
- `hidden_dim=16`, `depth=1`을 기본값으로 사용해 작은 `2 -> 16 -> 1` MLP를 구성한다.
- `depth`는 hidden layer 개수로 정의한다.
- hidden activation과 Linear bias 정책은 각각 상위 `coupling_model.activation`, `coupling_model.use_bias`를 재사용한다.
- output layer weight와 bias는 config와 무관하게 항상 zero-initialize한다.
- `hidden_dim`과 `depth`는 양의 정수, `eps`는 finite positive float여야 한다.
- 다음 legacy fields는 완전히 제거하며, 포함된 config는 strict parser에서 unknown-key 오류로 거부한다.
  - `mode`
  - `combination`
  - `nonlinear_hidden_dim`
  - `nonlinear_depth`
  - `nonlinear_final_init_scale`
  - `gate_initial_value`
- `configs/complex_coupling.json`의 fuser는 기존처럼 disabled 상태를 유지하되 새 schema를 사용한다.
- `configs/complex_coupling_soap.json`의 fuser는 기존처럼 enabled 상태를 유지하되 새 schema를 사용한다.

## Network Implementation

- `ComplexPreProjectionFusionConfig`와 `ComplexPreProjectionFusion` 이름 및 optional model attachment 위치는 유지한다.
- 내부의 `linear_correction`, geometry-rich `nonlinear_correction`, `gate_logit`, component-combination helper를 모두 제거한다.
- 새 학습 module은 `residual_mlp` 하나만 갖는다.
- MLP input dimension은 항상 2이고 output dimension은 항상 1이다.
- `x_local_t`, `y_local_t`, `log L_x`, `log L_y`, `log(L_x/L_y)`, \(\kappa\)를 계산하거나 MLP에 전달하는 코드를 제거한다.
- Final layer zero initialization으로 초기 상태에서
  \[
  r_\theta=0,\qquad d_{\mathrm{fused}}=d_{\mathrm{base}}
  \]
  를 정확히 만족시킨다.
- Zero-initialized final layer 특성상 첫 backward에서는 final layer가 먼저 학습되고 hidden layer gradient는 0일 수 있음을 정상 동작으로 테스트에 고정한다.
- \(A=0\)인 point에서는 correction을 명시적으로 0으로 mask해 exact zero-source homogeneity를 유지한다. \(A>0\)에서는 요청한 \(A_{\mathrm{safe}}r_\theta\) 공식을 그대로 적용한다.
- Fuser가 만드는 physical pair는 base common mode가 아니라 \(f\)를 common mode로 사용하므로
  \[
  \phi_{\mathrm{pre}}+\psi_{\mathrm{pre}}=f
  \]
  를 construction으로 만족한다.
- `fused_response`는
  \[
  P_{\mathrm{fused}}=L_x^2\phi_{\mathrm{pre}},
  \qquad
  Q_{\mathrm{fused}}=L_y^2\psi_{\mathrm{pre}}
  \]
  로 구성해 기존 projection API에 전달한다.
- 기존 physical symmetric projection은 제거하지 않는다. 새 pre-projection output에 대해서는 수치적으로 idempotent한 balance contract 검증 단계가 된다.
- Complex CouplingNet output contract version은 `6`으로 유지한다. Fuser 외부의 tensor 의미가 바뀌지 않기 때문이다.
- Legacy fuser checkpoint는 새 `residual_mlp` state key와 맞지 않으므로 strict load에서 실패하게 하며 별도 migration을 제공하지 않는다.
- Fuser가 disabled인 기존 v6 checkpoint는 architecture가 동일하면 계속 load될 수 있지만 이를 위한 별도 compatibility code는 추가하지 않는다.

## Diagnostic Result Contract

`ComplexPreProjectionFusionResult`는 다음 field로 재구성한다.

- `base_response`
- `base_physical`
- `fused_response`
- `fused_physical`
- `base_difference`
- `normalized_difference`
- `normalized_rhs`
- `normalized_residual`
- `physical_residual`
- `fused_difference`
- `source_scale`
- `safe_source_scale`
- `pre_projection_balance_residual`

다음 legacy field와 alias는 제거한다.

- `mode`
- `combination`
- `linear_component`
- `nonlinear_component`
- `combined_component`
- `gate`
- `linear_correction`
- `nonlinear_correction`
- `blended_correction`

Evaluator는 새 result contract를 전달하되 projection, reconstruction, objective, evaluation metric 계산은 변경하지 않는다.

## Training And Logging

- Trainer에서 `pre_projection_fusion_gate` metric과 gate 추출 helper를 제거한다.
- `complex_training_metrics.csv`에서 `pre_projection_fusion_gate` column을 제거한다.
- 학습 시작 로그에는 다음 static architecture metadata만 한 번 기록한다.
  - enabled 여부
  - `architecture=single_nonlinear_residual_mlp`
  - `space=physical_directional_source`
  - `input_dim=2`
  - hidden dimension
  - depth
  - activation
  - bias 사용 여부
  - `final_initialization=zeros`
  - `identity_skip=true`
  - `explicit_geometry_features=false`
- Loss, optimizer, scheduler, checkpoint 선택, reference diagnostic 정책은 수정하지 않는다.

## Artifact Changes

Complex CouplingNet artifact summary의 `pre_projection_fusion` section을 새 구조로 교체한다.

```text
enabled
architecture = single_nonlinear_residual_mlp
space = physical_directional_source
input = [base_difference_over_safe_source_scale, rhs_over_safe_source_scale]
hidden_dim
depth
activation
use_bias
identity_skip = true
final_layer_initialization = zeros
explicit_geometry_features = false
learned_linear_branch = false
learned_gate = false
source_scale = sqrt((A_x^2+A_y^2)/2)
formula = d_fused=d_base+A_safe*r_theta(z)
pre_projection_balance_constructed = true
uses_reference_targets = false
```

`selected_raw_arrays.npz`에서는 기존 linear/nonlinear decomposition field를 제거하고 다음 per-sample diagnostic suffix를 저장한다.

- `fusion_base_physical_p`
- `fusion_base_physical_q`
- `fusion_base_difference`
- `fusion_normalized_difference`
- `fusion_normalized_rhs`
- `fusion_residual_normalized`
- `fusion_residual_physical`
- `fusion_fused_difference`
- `fusion_pre_projection_phi`
- `fusion_pre_projection_psi`
- `fusion_source_scale`
- `fusion_safe_source_scale`
- `fusion_pre_projection_balance_residual`

다음 legacy artifact field와 해당 figure를 제거한다.

- `linear_difference_component`
- `nonlinear_difference_component`
- `combined_difference_component`
- `linear_difference_correction`
- `nonlinear_difference_correction`
- `blended_difference_correction`
- `fusion_gate`

새 signed Plotly diagnostic figure는 다음 세 field에 대해 생성한다.

- `fusion_base_difference`
- `fusion_residual_physical`
- `fusion_fused_difference`

기존 solution, flux, coefficient, error, energy artifact와 figure contract는 그대로 유지한다.

## Documentation

다음 문서의 기존 linear/nonlinear split 설명을 single nonlinear residual MLP contract로 교체한다.

- `README.md`
- `docs/memory.md`
- complex geometry instruction/design 문서
- complex GreenNet/CouplingNet technical report
- optimizer applicability 문서에서 기존 fuser parameter shape를 참조하는 부분

문서에는 다음 사항을 명시한다.

- fuser는 physical directional-source space에서 작동한다.
- 입력은 normalized physical difference와 RHS뿐이다.
- geometry/length feature는 fuser에 직접 들어가지 않는다.
- geometry dependence는 upstream axis-conditioned CouplingNet의 \(d_{\mathrm{base}}\)를 통해 간접 전달된다.
- identity skip은 deterministic하며 학습되는 linear branch가 아니다.
- final layer zero initialization으로 초기 projected output이 fuser-disabled path와 같다.
- reference `sol/phi/psi`를 사용하지 않는다.
- legacy fuser config와 enabled checkpoint는 호환하지 않는다.

`PLAN.md`는 사용자가 이 계획을 바탕으로 project root에 직접 작성하며, 이번 계획 단계에서는 파일을 생성하거나 수정하지 않는다.

## Test Plan

- **Config:** 새 schema default/JSON round-trip, positive `hidden_dim/depth`, finite positive `eps`, strict unknown-key rejection을 검증한다.
- **Legacy rejection:** 기존 `mode`, `combination`, `nonlinear_*`, `gate_initial_value`가 포함된 config가 fail fast하는지 확인한다.
- **Architecture:** state dict에 `residual_mlp`만 존재하고 `linear_correction`, `nonlinear_correction`, `gate_logit`이 없는지 확인한다.
- **Input contract:** first layer input dimension이 2이며 explicit geometry feature helper가 존재하지 않는지 확인한다.
- **Initialization:** final weight/bias가 정확히 0이고 initial residual이 0인지 확인한다.
- **Initial equivalence:** fuser-enabled와 disabled model이 동일한 base weights를 가질 때 physical projection 이후 output이 동일한지 확인한다.
- **Math:** \(d_{\mathrm{fused}}=d_{\mathrm{base}}+A_{\mathrm{safe}}r_\theta\), \(\phi_{\mathrm{pre}}-\psi_{\mathrm{pre}}=d_{\mathrm{fused}}\), \(\phi_{\mathrm{pre}}+\psi_{\mathrm{pre}}=f\)를 검증한다.
- **Scaling:** source와 base difference를 같은 상수로 scale했을 때 normalized input과 residual scaling이 일관되는지 확인한다.
- **Zero source:** \(A=0\)에서 correction과 fused response가 finite하고 zero-source homogeneity를 보존하는지 확인한다.
- **Gradient:** 첫 backward에서 final layer gradient가 finite/nonzero이고, optimizer update 후 hidden layer까지 gradient가 전달되는지 확인한다.
- **Model integration:** output shape `(B,2,P)`, physical symmetric projection, `torch.compile`, trainer one-step smoke를 검증한다.
- **Logging:** gate metric이 console, `training.log`, CSV에서 제거되고 새 architecture metadata가 기록되는지 확인한다.
- **Artifacts:** 새 summary/raw/figure field가 생성되고 모든 legacy linear/nonlinear/gate field가 없는지 확인한다.
- **Checkpoint:** legacy enabled-fuser state dict가 strict load에서 실패하고 새 checkpoint가 save/load 되는지 확인한다.
- **Regression:** projection, reconstruction, canonical energy, source provider, SOAP/AdamW, GreenNet, unit-square CouplingNet 동작이 변하지 않는지 확인한다.

검증 순서는 다음과 같이 고정한다.

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_complex_pre_projection_fusion.py \
  test/test_complex_coupling_model.py \
  test/test_complex_coupling_trainer.py \
  test/test_complex_coupling_artifacts.py \
  test/test_cli_train.py \
  test/test_io_config.py

PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest test

ruff check src cli test
ruff format src cli test
~/.conda/envs/green_net/bin/python -m mypy src
git diff --check
```

## Rollback Strategy

- Runtime rollback은 `pre_projection_fusion.enabled=false`로 fuser를 비활성화하는 것이다.
- Code rollback은 새 `residual_mlp` 구현과 config/artifact 변경을 함께 되돌리는 단일 commit revert로 수행한다.
- Projection, reconstruction, loss, optimizer, scheduler, dataset 및 GreenNet에는 rollback 변경이 없어야 한다.
- 기존 enabled-fuser checkpoint는 호환 대상으로 삼지 않으므로 rollback 또는 migration artifact를 만들지 않는다.
- 새 fuser가 성능을 개선하지 못해도 fuser-disabled base architecture는 영향을 받지 않아야 한다.

## Acceptance Criteria

- 학습 가능한 pre-projection module은 하나의 nonlinear `residual_mlp`뿐이다.
- MLP는 오직 \([d_{\mathrm{base}}/A_{\mathrm{safe}},f/A_{\mathrm{safe}}]\)만 입력받는다.
- Linear branch, geometry feature input, combination mode 및 learned gate가 코드와 config에서 제거된다.
- 초기화 시 projection 이후 결과가 fuser-disabled path와 동일하다.
- 모든 valid point에서 pre-projection pair와 최종 projection이 physical balance를 만족한다.
- Artifact에 새 residual decomposition만 남고 기존 linear/nonlinear/gate field는 없다.
- Pre-projection 이외의 output contract v6, projection, Green reconstruction, objective와 checkpoint selection은 변경되지 않는다.
- Legacy enabled-fuser checkpoint/config migration은 제공하지 않는다.
