# Complex Pre-Projection Fuser Optional Absolute Mode 구현 계획

## Summary

현재 physical pre-projection fuser의 `residual` 동작을 그대로 보존하면서, 같은 입력과 같은 MLP가 최종 directional difference 전체를 출력하는 `absolute` mode를 추가한다.

```text
residual:
d_fused = d_base + A_safe * MLP([d_base / A_safe, f / A_safe])

absolute:
d_fused = A_safe * MLP([d_base / A_safe, f / A_safe])
```

두 mode 모두 다음 balance-preserving physical pair를 구성한다.

\[
\phi_{\mathrm{pre}}
=
\frac{f+d_{\mathrm{fused}}}{2},
\qquad
\psi_{\mathrm{pre}}
=
\frac{f-d_{\mathrm{fused}}}{2}.
\]

따라서 항상

\[
\phi_{\mathrm{pre}}+\psi_{\mathrm{pre}}=f
\]

를 만족하며, 이후 physical symmetric projection, reference-response pull-back, Green reconstruction은 변경하지 않는다.

권장 기본 결정은 다음과 같이 고정한다.

- 기존 config 기본값: `mode="residual"`
- final-layer initialization 기본값: `final_layer_init_scale=0.0`
- initialization scale 허용 범위: `[0.0, 1.0]`
- 기존 `coupling3` residual checkpoint: 호환 유지
- 다음 absolute 실험: 별도 config에서 `mode="absolute"`를 명시
- Complex CouplingNet output contract: version 6 유지

## Public Config

`coupling_model.pre_projection_fusion`을 다음과 같이 확장한다.

```json
"pre_projection_fusion": {
  "enabled": true,
  "mode": "absolute",
  "hidden_dim": 16,
  "depth": 1,
  "eps": 1e-12,
  "final_layer_init_scale": 0.0
}
```

각 field의 의미는 다음으로 고정한다.

- `mode`: `"residual"` 또는 `"absolute"`
- `final_layer_init_scale=0.0`: final-layer weight와 bias를 모두 zero initialization
- `final_layer_init_scale=1.0`: `torch.nn.Linear`가 생성할 때 사용하는 기본 initialization을 그대로 유지
- `0.0 < final_layer_init_scale < 1.0`: 기본 initialization으로 생성된 weight와 bias를 해당 계수로 곱해 축소
- `use_bias=false`: scale은 final-layer weight에만 적용
- `pre_projection_fusion` block에 새 field가 없으면 기존 동작과 동일하게 `mode="residual"`, `final_layer_init_scale=0.0`으로 해석

Validation은 다음과 같이 구현한다.

- `mode`가 string이 아니면 `TypeError`
- 지원하지 않는 mode는 `ValueError`
- scale이 boolean 또는 numeric이 아니면 `TypeError`
- scale이 non-finite이거나 `[0,1]` 밖이면 `ValueError`
- unknown config key는 기존처럼 fail fast

## Implementation Changes

### 1. Config와 Initialization

`ComplexPreProjectionFusionConfig`에 다음 field를 추가한다.

```python
mode: Literal["residual", "absolute"] = "residual"
final_layer_init_scale: float = 0.0
```

MLP topology는 현재와 동일하게 유지한다.

```text
2 inputs
-> hidden_dim
-> configured activation
-> 1 output
```

Final layer는 `nn.Linear` 생성 시 이미 적용된 PyTorch 기본 initialization을 기준으로 scale한다.

```python
with torch.no_grad():
    final_layer.weight.mul_(final_layer_init_scale)
    if final_layer.bias is not None:
        final_layer.bias.mul_(final_layer_init_scale)
```

따라서 scale `0`은 현재 zero initialization과 정확히 같고, scale `1`은 생성 시 기본 initialization을 보존한다. 별도의 random draw나 custom initializer는 추가하지 않는다.

### 2. Mode별 Fuser 계산

현재 normalized input은 변경하지 않는다.

\[
A
=
\sqrt{\frac{A_x^2+A_y^2}{2}},
\qquad
A_{\mathrm{safe}}=\max(A,\varepsilon),
\]

\[
z=
\left[
\frac{d_{\mathrm{base}}}{A_{\mathrm{safe}}},
\frac{f}{A_{\mathrm{safe}}}
\right].
\]

MLP output을 공통으로

\[
h_\theta=\operatorname{MLP}_\theta(z),
\qquad
d_{\mathrm{network}}=A_{\mathrm{safe}}h_\theta
\]

로 계산한 뒤 mode만 분기한다.

```python
if mode == "residual":
    fused_difference = base_difference + network_output_physical
else:
    fused_difference = network_output_physical
```

Source amplitude가 정확히 zero인 point에서는 기존 정책을 확장해 network physical output과 absolute fused difference를 `0`으로 강제한다.

`absolute + final_layer_init_scale=0`의 초기 상태는 다음과 같다.

\[
d_{\mathrm{fused}}=0,
\qquad
\phi_{\mathrm{pre}}=\psi_{\mathrm{pre}}=\frac{f}{2}.
\]

`residual + final_layer_init_scale=0`의 초기 상태는 현재와 같다.

\[
d_{\mathrm{fused}}=d_{\mathrm{base}}.
\]

### 3. Checkpoint Contract

Response tensor의 의미와 shape는 변경되지 않으므로 global output-contract version은 `6`으로 유지한다.

기존 checkpoint compatibility를 위해 현재 state-dict prefix인

```text
pre_projection_fusion.residual_mlp.*
```

는 내부 명칭이 다소 좁더라도 그대로 유지한다. 이를 `fusion_mlp`로 rename하지 않는다.

Residual checkpoint를 absolute config로 조용히 재해석하는 것은 금지한다. Fuser module에 persistent mode marker를 추가한다.

```text
residual -> 0
absolute -> 1
```

Checkpoint load 정책은 다음과 같다.

- 기존 v6 single-residual checkpoint에 mode marker가 없으면 legacy current-fuser checkpoint로 인식
- target config가 `mode="residual"`이면 residual marker를 주입하고 load 허용
- target config가 `mode="absolute"`이면 명확한 mode mismatch 오류 발생
- marker가 있는 새 checkpoint는 checkpoint mode와 target config mode가 같아야 함
- `final_layer_init_scale`은 fresh initialization 설정이므로 checkpoint load compatibility 조건에 포함하지 않음
- retired split linear/nonlinear checkpoint rejection은 그대로 유지

### 4. Diagnostics와 Artifacts

기존 residual 전용 용어를 mode-independent audit 정보로 확장한다.

공통 raw field:

```text
fusion_base_difference
fusion_network_output_normalized
fusion_network_output_physical
fusion_fused_difference
fusion_delta_from_base
fusion_pre_projection_phi
fusion_pre_projection_psi
fusion_pre_projection_balance_residual
```

여기서

\[
\mathrm{fusion\_delta\_from\_base}
=
d_{\mathrm{fused}}-d_{\mathrm{base}}.
\]

Residual mode에서는 `network_output_physical == delta_from_base`다. Absolute mode에서는 둘이 다르므로 반드시 별도 저장한다.

기존 residual artifact reader를 위해 residual mode에서는 현재의 다음 key를 alias로 유지한다.

```text
fusion_residual_normalized
fusion_residual_physical
```

Absolute mode에서는 residual이라는 이름으로 MLP absolute output을 잘못 저장하지 않는다.

Artifact summary와 training log에는 다음을 기록한다.

- `mode`
- mode별 formula
- `identity_skip`: residual이면 `true`, absolute이면 `false`
- `final_layer_init_scale`
- `final_layer_initialization="scaled_torch_linear_default"`
- `input=["base_difference_over_safe_source_scale", "rhs_over_safe_source_scale"]`
- `explicit_geometry_features=false`
- `learned_gate=false`
- `pre_projection_balance_constructed=true`

Length-response diagnostic도 동일한 generic field와 mode formula를 사용하도록 갱신한다.

### 5. Config와 Documentation

기존 canonical complex config는 backward compatibility를 위해 다음 설정을 사용한다.

```json
"pre_projection_fusion": {
  "enabled": true,
  "mode": "residual",
  "hidden_dim": 16,
  "depth": 1,
  "eps": 1e-12,
  "final_layer_init_scale": 0.0
}
```

다음 실험용 paired config는 별도 파일로 둔다.

```text
configs/complex_coupling_soap_absolute.json
```

이 파일은 residual baseline과 다음 두 값만 다르게 한다.

```json
"mode": "absolute",
"final_layer_init_scale": 0.0
```

README와 `docs/memory.md`에는 다음 내용을 기록한다.

- residual과 absolute mode의 수식
- scale `0`, 중간값, `1`의 의미
- absolute zero-init이 symmetric split `f/2, f/2`에서 시작한다는 점
- absolute mode는 continuity를 구조적으로 보장하지 않지만 `d_base` seam을 강제 전달하지 않는다는 점
- reference target은 여전히 training에 사용하지 않는다는 점
- residual과 absolute checkpoint mode mismatch는 거부된다는 점

## Affected Files

주요 변경 영역은 다음과 같다.

- Config 및 core computation:
  `src/greenonet/config.py`,
  `src/greenonet/complex_pre_projection_fusion.py`,
  `src/greenonet/complex_coupling_model.py`
- Runtime provenance와 export:
  `src/greenonet/complex_coupling_trainer.py`,
  `src/greenonet/complex_coupling_artifacts.py`,
  `src/greenonet/complex_length_response_diagnostics.py`
- Config와 문서:
  `configs/complex_coupling.json`,
  `configs/complex_coupling_soap.json`,
  새 absolute pilot config,
  `README.md`,
  `docs/memory.md`
- Tests:
  pre-projection fuser, complex model, trainer, artifact, config/CLI tests

Dataset, GreenNet, projection helper, reconstruction, canonical energy loss, SOAP optimizer, scheduler와 geometry/sample NPZ schema는 변경하지 않는다.

## Test Plan

### Config Tests

- 새 field가 없는 config가 `residual + scale=0`으로 parse되는지 확인
- residual/absolute round-trip 확인
- scale `0`, `0.5`, `1` 허용 확인
- invalid mode, boolean scale, NaN, infinity, 음수, `1` 초과, unknown key 거부 확인
- embedded checkpoint config load에서도 nested field가 복원되는지 확인

### Initialization Tests

동일한 `torch.manual_seed`로 module을 생성해 다음을 검증한다.

- scale `0`: final weight/bias가 정확히 zero
- scale `1`: final weight/bias가 PyTorch 기본 initialization 값
- scale `0.5`: scale `1`과 동일한 initial draw의 정확히 절반
- hidden-layer initialization은 scale에 따라 바뀌지 않음
- `use_bias=false` 동작 확인

### Forward Math Tests

- residual scale `0`이 현재 identity-initialized behavior와 정확히 일치
- absolute scale `0`이 `d_fused=0`, `phi_pre=psi_pre=f/2` 생성
- manually fixed MLP output에서 residual mode가 `d_base+A*h`를 반환
- 같은 MLP output에서 absolute mode가 `A*h`만 반환
- 두 mode 모두 `phi_pre+psi_pre=rhs`를 float64 tolerance 내에서 만족
- zero source amplitude에서 absolute fused difference가 zero
- output shape `(B,2,P)`와 output-contract v6 유지
- 두 mode 모두 `torch.compile` forward/backward 통과
- zero-init absolute mode에서 final layer가 gradient를 받고, final layer update 이후 upstream gradient가 전달되는지 확인

### Checkpoint Tests

- 기존 unmarked v6 single-residual checkpoint를 residual config가 load
- 기존 residual checkpoint를 absolute config가 거부
- 새 residual checkpoint는 residual config에서 load
- 새 absolute checkpoint는 absolute config에서 load
- residual/absolute cross-load 모두 fail fast
- 기존 `residual_mlp` parameter key 유지
- retired split-fuser checkpoint rejection 유지

### Trainer와 Artifact Tests

- residual one-step regression
- absolute one-step training smoke
- log에 mode, scale, identity-skip 상태 기록
- artifact summary의 mode별 formula 확인
- generic raw fields 존재 확인
- residual alias는 residual mode에서만 기존 의미로 유지
- absolute artifact에서 network output과 delta-from-base를 구분
- pre-projection balance residual이 round-off 수준인지 확인
- solution/flux figures와 canonical energy metric contract 유지
- reference `sol/phi/psi`가 loss 또는 checkpoint selection에 들어가지 않는지 확인

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

Runtime rollback은 config만 다음과 같이 되돌리면 된다.

```json
"mode": "residual",
"final_layer_init_scale": 0.0
```

이는 현재 `coupling3` behavior를 정확히 복원한다.

Code rollback이 필요하면 absolute formula branch, mode marker, generic absolute artifact fields와 absolute pilot config만 제거한다. Residual computation, current `residual_mlp` state keys, projection, reconstruction, loss와 existing checkpoint load path는 그대로 남아야 한다.

Absolute checkpoint는 rollback 이후 지원하지 않아도 된다. 실험 checkpoint directory는 삭제하지 않고, unsupported mode checkpoint라는 명확한 오류를 출력한다.
