# Complex Pre-Projection Fuser Absolute-Difference Optional Mode 구현 계획

## Summary

- 기존 `residual_correction` 방식은 그대로 보존한다.

\[
d_{\mathrm{fused}}
=
d_{\mathrm{base}}
+
\delta_{\mathrm{blend}},
\]

\[
\delta_{\mathrm{blend}}
=
(1-g)\delta_{\mathrm{linear}}
+
g\delta_{\mathrm{nonlinear}}.
\]

- 새 `absolute_difference` mode에서는 외부의 `+d_base` skip connection을 제거하고 fuser가 만든 difference만 사용한다.
- Linear candidate도 기존 fuser와 동일한 source-amplitude normalization을 유지한다.

\[
\boxed{
d_{\mathrm{linear}}
=
A h_{\mathrm{linear}}
\left(
\frac{d_{\mathrm{base}}}{A},
\frac{f}{A}
\right)
}
\]

- Nonlinear branch도 같은 normalized difference/source에 geometry feature를 추가하는 현재 contract를 유지한다.
- 새 mode는 `linear_plus_nonlinear`과 `convex_average`를 모두 지원한다.
- Canonical 새 설정은 다음으로 고정한다.
  - `mode="absolute_difference"`
  - `combination="linear_plus_nonlinear"`
  - `nonlinear_final_init_scale=0.01`
  - `gate_initial_value=0.5`
- 대안으로 standard nonlinear final-layer initialization과 작은 learnable gate를 지원한다.
  - `nonlinear_final_init_scale=1.0`
  - `gate_initial_value=0.05`
- Complex CouplingNet 외의 GreenNet, unit-square CouplingNet, projection, reconstruction, objective, dataset 및 NPZ contract는 변경하지 않는다.

## Public Configuration

```json
"pre_projection_fusion": {
  "enabled": true,
  "mode": "absolute_difference",
  "combination": "linear_plus_nonlinear",
  "nonlinear_hidden_dim": 16,
  "nonlinear_depth": 1,
  "nonlinear_final_init_scale": 0.01,
  "gate_initial_value": 0.5,
  "eps": 1e-12
}
```

지원 값은 다음과 같다.

```text
mode:
  residual_correction
  absolute_difference

combination:
  convex_average
  linear_plus_nonlinear
```

기존 config에 새 필드가 없으면 다음처럼 해석한다.

```text
mode = residual_correction
combination = convex_average
nonlinear_final_init_scale = 0.0
```

Validation은 다음으로 고정한다.

- `mode`와 `combination`은 지원 enum만 허용한다.
- `nonlinear_final_init_scale`은 finite하고 `>=0`이어야 한다.
- `gate_initial_value`는 기존과 같이 `(0,1)` 범위여야 한다.
- `residual_correction` mode는 기존 수식을 보존하기 위해 `convex_average`만 허용한다.
- `absolute_difference` mode에서는 두 combination을 모두 허용한다.
- Unknown key는 fail fast한다.
- Unit-square CouplingNet에서는 pre-projection fusion을 계속 거부한다.

## Mathematical Contract

### Source Scale과 Normalized Input

현재 fuser와 같은 physical source scale을 사용한다.

\[
A
=
\sqrt{
\frac{A_x^2+A_y^2}{2}
}.
\]

수치적으로는

\[
A_{\mathrm{safe}}
=
\max(A,\varepsilon)
\]

를 사용한다.

Normalized input은

\[
\widetilde d
=
\frac{d_{\mathrm{base}}}{A_{\mathrm{safe}}},
\qquad
\widetilde f
=
\frac{f}{A_{\mathrm{safe}}}
\]

이다.

Linear/nonlinear branch 모두 같은 normalized pair를 공유한다.

### 기존 `residual_correction`

기존 linear correction은 그대로 유지한다.

\[
\delta_{\mathrm{linear}}
=
A h_{\mathrm{linear}}
\left(
\widetilde d,\widetilde f
\right).
\]

Nonlinear correction도 현재 geometry feature contract를 유지한다.

\[
\delta_{\mathrm{nonlinear}}
=
A h_{\mathrm{nonlinear}}
\left(
\widetilde d,
\widetilde f,
\mathcal G
\right),
\]

\[
\mathcal G
=
[
t_x,t_y,
\log(L_x/L_{\mathrm{ref}}),
\log(L_y/L_{\mathrm{ref}}),
\log(L_x/L_y),
\kappa
].
\]

\[
\delta_{\mathrm{blend}}
=
(1-g)\delta_{\mathrm{linear}}
+
g\delta_{\mathrm{nonlinear}}.
\]

\[
d_{\mathrm{fused}}
=
d_{\mathrm{base}}
+
\delta_{\mathrm{blend}}.
\]

Initialization은 기존과 동일하다.

- Linear correction weight: zero.
- Nonlinear final layer weight/bias: zero.
- 초기 `d_fused=d_base`.

### 새 Absolute Linear Candidate

새 mode에서도 linear map의 수식은 기존 normalized formulation을 따른다.

\[
d_{\mathrm{linear}}
=
A h_{\mathrm{linear}}
\left(
\widetilde d,\widetilde f
\right).
\]

Linear layer는

\[
h_{\mathrm{linear}}
\left(
\widetilde d,\widetilde f
\right)
=
w_d\widetilde d+w_f\widetilde f
\]

이다.

초기값은

\[
w_d=1,\qquad w_f=0
\]

으로 둔다.

따라서 \(A\ge\varepsilon\)인 일반 sample에서는

\[
d_{\mathrm{linear}}
=
A\frac{d_{\mathrm{base}}}{A}
=
d_{\mathrm{base}}
\]

가 정확히 성립한다.

\(A<\varepsilon\)에서는 기존 safe-scale convention을 따른다.

\[
d_{\mathrm{linear}}
=
\frac{A}{\varepsilon}d_{\mathrm{base}}.
\]

특히 zero source amplitude에서는

\[
A=0
\quad\Longrightarrow\quad
d_{\mathrm{linear}}=0.
\]

이는 zero source에서 zero directional difference를 만드는 물리적으로 일관된 behavior로 채택한다.

### 새 Absolute Nonlinear Component

Nonlinear branch도 기존 normalized source/geometry input을 유지한다.

\[
r_{\mathrm{nonlinear}}
=
A h_{\mathrm{nonlinear}}
\left(
\widetilde d,
\widetilde f,
\mathcal G
\right).
\]

Hidden layer는 표준 initialization을 사용한다.

Final layer는 PyTorch 표준 initialization 후 다음 scale을 적용한다.

\[
W_{\mathrm{final}}
\leftarrow
s_{\mathrm{init}}W_{\mathrm{final}},
\]

\[
b_{\mathrm{final}}
\leftarrow
s_{\mathrm{init}}b_{\mathrm{final}},
\]

\[
s_{\mathrm{init}}
=
\texttt{nonlinear\_final\_init\_scale}.
\]

따라서:

- `scale=0.01`: 작은 nonlinear component.
- `scale=1.0`: 표준 final-layer initialization.
- `scale=0.0`: exact-zero nonlinear component.

### `linear_plus_nonlinear`

Canonical absolute mode는 다음 수식을 사용한다.

\[
\boxed{
d_{\mathrm{fused}}
=
d_{\mathrm{linear}}
+
g r_{\mathrm{nonlinear}}
}
\]

여기서

\[
g=\operatorname{sigmoid}(\gamma)
\]

는 learnable scalar gate이다.

외부에서 `d_base`를 다시 더하지 않는다.

Canonical initialization은

\[
w_d=1,\qquad w_f=0,
\]

\[
s_{\mathrm{init}}=0.01,
\qquad
g_0=0.5
\]

로 고정한다.

대안 initialization은

\[
s_{\mathrm{init}}=1.0,
\qquad
g_0=0.05
\]

로 설정할 수 있다.

### `convex_average`

Convex mode에서는 nonlinear output을 absolute candidate로 해석한다.

\[
d_{\mathrm{nonlinear}}
=
A h_{\mathrm{nonlinear}}
\left(
\widetilde d,
\widetilde f,
\mathcal G
\right).
\]

최종 difference는 true convex average로 정의한다.

\[
\boxed{
d_{\mathrm{fused}}
=
(1-g)d_{\mathrm{linear}}
+
g d_{\mathrm{nonlinear}}
}
\]

이 mode에서는 nonlinear candidate가 linear candidate와 같은 값으로 초기화되지 않는 한 전체 fuser가 exact identity로 시작하지 않는다. 이 차이를 config documentation과 artifact summary에 명시한다.

### Physical Components와 Projection

Base common mode를 보존해 temporary physical pair를 구성한다.

\[
s_{\mathrm{base}}
=
p_{\mathrm{base}}+q_{\mathrm{base}}.
\]

\[
p_{\mathrm{fused}}
=
\frac12
\left(
s_{\mathrm{base}}+d_{\mathrm{fused}}
\right),
\]

\[
q_{\mathrm{fused}}
=
\frac12
\left(
s_{\mathrm{base}}-d_{\mathrm{fused}}
\right).
\]

기존 model return contract를 유지한다.

\[
P_{\mathrm{fused}}
=
L_x^2p_{\mathrm{fused}},
\qquad
Q_{\mathrm{fused}}
=
L_y^2q_{\mathrm{fused}}.
\]

Physical-symmetric projection은 변경하지 않는다.

\[
\phi
=
\frac12(f+d_{\mathrm{fused}}),
\qquad
\psi
=
\frac12(f-d_{\mathrm{fused}}).
\]

따라서 모든 mode에서

\[
\phi+\psi=f
\]

가 정확히 유지된다.

이번 작업에서는 model forward의 reference-response tensor contract와 중간 physical/reference 왕복을 유지한다. 이 API 리팩터링은 absolute-mode 수식 변경과 분리한다.

## Implementation Changes

### 1. Config와 Serialization

영향 파일:

- `src/greenonet/config.py`
- `src/greenonet/io.py`
- complex CouplingNet config examples

작업:

- `ComplexPreProjectionFusionConfig`에 `mode`, `combination`, `nonlinear_final_init_scale`을 추가한다.
- Strict enum/numeric validation을 추가한다.
- 기존 config는 residual mode로 복원한다.
- Canonical config는 현재 residual behavior를 유지한다.
- 새 absolute-mode example config를 추가하거나 README에 완전한 example을 제공한다.
- Standard-final/small-gate 대안도 문서화한다.

### 2. Mode-Aware Initialization

영향 파일:

- `src/greenonet/complex_pre_projection_fusion.py`

작업:

- 기존 initialization을 residual/absolute helper로 분리한다.
- Residual mode는 현재 zero-correction initialization을 그대로 사용한다.
- Absolute mode의 linear weight를 `[1,0]`으로 초기화한다.
- Linear bias는 계속 사용하지 않는다.
- Nonlinear hidden layers는 현재 activation-compatible 표준 initialization을 유지한다.
- Nonlinear final layer는 표준 initialization 후 `nonlinear_final_init_scale`을 곱한다.
- Small nonzero scale에서 첫 backward부터 nonlinear hidden layer에 gradient가 전달되어야 한다.

### 3. Mode와 Combination Dispatcher

영향 파일:

- `src/greenonet/complex_pre_projection_fusion.py`
- `src/greenonet/complex_coupling_model.py`

작업:

- 공통 source scale과 normalized input을 한 번 계산한다.
- Residual mode는 현재 수식 그대로 실행한다.
- Absolute mode는 동일한 normalized linear formulation을 사용한다.
- `linear_plus_nonlinear`과 `convex_average`를 별도 private helper로 구현한다.
- 최종 absolute difference로 base common mode를 보존하는 physical pair를 만든다.
- 기존 `(B,2,P)` fused response를 반환한다.
- `forward()`와 `forward_with_diagnostics()`가 같은 core tensor helper를 사용한다.
- `torch.compile`이 mode별로 정상 동작하도록 mode branch를 module construction 시 고정한다.

### 4. Diagnostic와 Artifact Contract

현재 correction 중심 명칭을 absolute mode에서 그대로 사용하지 않는다.

공통 diagnostic field를 추가한다.

```text
base_physical_difference
linear_difference_component
nonlinear_difference_component
combined_difference_component
fused_physical_difference
fusion_gate
source_scale
```

Mode별 semantics:

- Residual:
  - linear/nonlinear component는 correction.
  - combined component는 blended correction.
  - 기존 `linear_difference_correction`,
    `nonlinear_difference_correction`,
    `blended_difference_correction` key와 figure를 유지한다.
- Absolute + linear-plus:
  - linear component는 absolute candidate.
  - nonlinear component는 residual.
  - combined component는 final absolute difference.
- Absolute + convex:
  - linear/nonlinear component는 각각 absolute candidate.
  - combined component는 convex-averaged final difference.

Artifact summary에 다음을 기록한다.

```text
mode
combination
linear_input_normalization
linear_initialization
nonlinear_final_initialization
nonlinear_final_init_scale
gate_initial_value
gate_value
nonlinear_component_semantics
outer_base_residual_used
```

Absolute mode에서는

```text
outer_base_residual_used = false
```

로 기록한다.

### 5. Trainer와 Evaluator

- Existing projection/reconstruction 호출 순서를 유지한다.
- Startup log에 resolved mode, combination, final init scale와 gate를 기록한다.
- Existing `pre_projection_fusion_gate` metric을 유지한다.
- Canonical energy, optimizer, scheduler, checkpoint selection은 변경하지 않는다.
- `sol/phi/psi` reference는 계속 detached evaluation-only로 유지한다.
- Absolute/residual checkpoint의 embedded config를 artifact exporter가 그대로 사용한다.

### 6. Checkpoint Compatibility

- Output shape와 model response space가 유지되므로 output contract version은 `6`으로 유지한다.
- Parameter names와 tensor shapes를 유지한다.
- 기존 residual checkpoint는 새 field가 없는 config를 residual mode로 해석해 계속 로드한다.
- 새 absolute checkpoint는 embedded `model_config`에 mode, combination과 initialization을 저장한다.
- Absolute checkpoint를 residual config로 재해석하지 않는다.
- GreenNet checkpoint에는 변화가 없다.

## Test Plan

### Config

- 기존 config가 residual mode로 parse되는지 확인한다.
- Absolute linear-plus config round-trip.
- Absolute convex config round-trip.
- `scale=0`, `0.01`, `1.0` 검증.
- Invalid enum, negative/nonfinite scale, invalid gate, unknown key rejection.
- Unit-square fuser rejection 유지.

### Initialization

- Residual mode가 기존 exact identity인지 확인한다.
- Absolute linear weight가 `[1,0]`인지 확인한다.
- \(A\ge\varepsilon\)에서 `d_linear == d_base`인지 확인한다.
- \(A=0\)에서 `d_linear == 0`인지 확인한다.
- `scale=0.01` final parameter가 같은 seed의 standard parameter 대비 정확히 0.01배인지 확인한다.
- `scale=1.0`이 standard initialization과 일치하는지 확인한다.
- Small nonzero final initialization에서 hidden-layer gradient가 첫 step부터 finite/nonzero인지 확인한다.

### Math

- Residual mode 결과가 기존 구현과 일치한다.
- Linear-plus가

\[
d_{\mathrm{linear}}+g r_{\mathrm{nonlinear}}
\]

을 정확히 계산한다.

- Convex mode가

\[
(1-g)d_{\mathrm{linear}}+g d_{\mathrm{nonlinear}}
\]

을 정확히 계산한다.

- Absolute mode에서 외부 `+d_base`가 없는지 확인한다.
- Base common mode가 temporary pair에서 보존되는지 확인한다.
- Projection 후 `phi+psi=rhs`를 확인한다.
- Unequal line lengths, short segment, zero source scale에서 finite한지 확인한다.

### Integration

- 두 mode와 두 absolute combination에서 output `(B,2,P)` 유지.
- One-step trainer smoke.
- `torch.compile` smoke.
- Artifact summary/raw schema 검증.
- Existing residual artifact keys 유지.
- Reference target이 loss/checkpoint graph에 들어가지 않음.
- Existing complex GreenNet 및 unit-square CouplingNet regression 통과.

### Verification

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_complex_pre_projection_fusion.py \
  test/test_complex_coupling_model.py \
  test/test_complex_coupling_trainer.py \
  test/test_complex_coupling_artifacts.py \
  test/test_io_config.py \
  test/test_cli_train.py
```

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest test
ruff check src cli test
ruff format src cli test
~/.conda/envs/green_net/bin/python -m mypy src
git diff --check
```

## Rollback Strategy

- Config에서 `mode="residual_correction"`을 사용하거나 새 field를 제거하면 기존 동작으로 돌아간다.
- Default가 residual mode이므로 기존 config migration은 없다.
- Model backbone과 parameter shape를 바꾸지 않으므로 architecture rollback은 필요 없다.
- Code rollback은 absolute dispatcher, 새 config fields, mode-specific diagnostics와 관련 tests만 제거한다.
- Projection, reconstruction, objective, optimizer, scheduler, dataset은 rollback 대상이 아니다.
- Residual regression이 발견되면 absolute implementation을 중단하고 기존 residual formula와 zero initialization을 먼저 복원한다.
- Absolute checkpoint는 residual mode로 재해석하지 않고 해당 embedded config로만 사용한다.

