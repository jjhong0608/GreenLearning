# Tangent Green-response context 저장 및 재사용 구현 지침

## 0. 문서 지위

이 문서는 complex CouplingNet의
`balance_projection.mode="symmetric_tangent_green_response"`에서 runtime당 한 번
만드는 frozen tangent Green-response context를 디스크에 저장하고, 이후 evaluation,
artifact export 및 다른 CouplingNet checkpoint 평가에서 재사용하기 위한 **구현 기준
문서**다.

- 기준 저장소: `/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry`
- 작성 기준일: 2026-08-26 (Asia/Seoul)
- 기준 구현:
  - `src/greenonet/complex_axial_response_operator.py`
  - `src/greenonet/complex_tangent_projection.py`
  - `src/greenonet/complex_projection.py`
  - `src/greenonet/complex_coupling_trainer.py`
  - `src/greenonet/complex_coupling_evaluator.py`
  - `src/greenonet/complex_coupling_artifacts.py`
- 문서 상태: schema-v2 serialization/load 및 runtime integration 구현 완료

후속 구현 중 설계 결정을 변경해야 한다면 코드보다 먼저 이 문서를 갱신한다. 이
문서는 기존 model checkpoint, optimizer state 또는 source sample을 context에 넣는
것을 허용하지 않는다.

## 1. 결정 요약

다음 계약으로 구현한다.

1. CouplingNet model checkpoint는 지금처럼 model parameter만 저장한다.
2. Tangent context는 별도 `tangent_response_context.safetensors` sidecar에 저장한다.
3. Sidecar는 frozen segment-local response operator `H_x`, `H_y`, Jacobi gain과
   denominator, point mass 및 검증 metadata를 포함한다.
4. CouplingNet raw output, source, mismatch, tangent direction, subspace coefficient,
   `eta`, `delta`와 auxiliary loss 값은 저장하지 않는다.
5. Geometry, GreenNet state, actual Green branch input, reconstruction contract,
   floating dtype와 preconditioner-defining config fingerprint가 모두 일치할 때만
   sidecar를 사용한다.
6. 기존 runtime-only lazy cache가 backward-compatible default다. Persistence는
   명시적인 opt-in이다.
7. Existing sidecar가 손상되었거나 fingerprint가 다르면 조용히 rebuild하지 않고
   오류를 낸다. Missing sidecar만 policy에 따라 build할 수 있다.
8. Trainer, evaluator, artifact exporter와 tangent audit가 동일 serializer와
   validator를 사용한다.
9. 저장/복원은 global response matrix, full Gram matrix 또는 linear solve를 만들지
   않는다.
10. Model tensor key와 architecture contract는 바뀌지 않는다.

## 2. 범위

### 2.1 포함

- Frozen x/y axial response block serialization
- Jacobi preconditioner tensor serialization
- Strict compatibility fingerprint와 schema validation
- Atomic save와 safe load
- Trainer first-build save
- Evaluator/artifact/audit load
- Runtime cache의 build/load/save telemetry
- Config, CLI override, artifact provenance
- Round-trip, corruption, mismatch 및 numerical-equivalence tests

### 2.2 제외

- CouplingNet parameter checkpoint에 context tensor를 섞는 것
- Optimizer/scheduler/RNG state 저장과 interrupted-training resume
- Dynamic source/sample/CouplingNet output 저장
- Per-sample coefficient operator를 하나의 context로 대표하는 것
- GreenNet joint training 중 stale context를 계속 사용하는 것
- Full response/Gram matrix 조립
- Full matrix factorization 또는 solve
- `torch.save`/pickle 기반 context format
- Legacy sidecar의 silent migration
- Preconditioner formula 변경 또는 새로운 tangent algorithm 도입

## 3. 수학적 계약

### 3.1 Symmetric balance와 tangent coordinate

Raw physical directional proposal을 `p`, `q`, source를 `f`라 하면 먼저

\[
\widetilde\phi=\frac{f+p-q}{2},
\qquad
\widetilde\psi=\frac{f-p+q}{2}
\]

를 만든다. 따라서

\[
\widetilde\phi+\widetilde\psi=f.
\]

Balance plane의 tangent update는

\[
\phi=\widetilde\phi+\delta,
\qquad
\psi=\widetilde\psi-\delta
\]

이고 모든 `delta`에 대해 `phi + psi = f`를 보존한다.

### 3.2 Frozen response operator

Production Green reconstruction의 physical-source response operator를

\[
H_s=K_s W_s L_s^2,
\qquad s\in\{x,y\}
\]

로 둔다. `K_s`는 learned frozen Green kernel, `W_s`는 unit-interval quadrature,
`L_s^2`는 physical source의 reference-response pull-back이다. 구현에서는 connected
axial segment별 square block으로 저장하며 global matrix를 만들지 않는다.

Symmetric source의 directional solution과 mismatch는

\[
u_\phi=H_x\widetilde\phi,
\qquad
u_\psi=H_y\widetilde\psi,
\qquad
m_0=u_\phi-u_\psi.
\]

`S=H_x+H_y`라 두면 correction 후 mismatch는

\[
m(\delta)=m_0+S\delta.
\]

### 3.3 Gradient와 cached Jacobi denominator

Weighted response objective는

\[
J(\delta)=\frac12\|m_0+S\delta\|_{M_\Omega}^2,
\qquad
M_\Omega=(h_xh_y)I.
\]

Initial gradient는

\[
g_0=S^T M_\Omega m_0
=(H_x+H_y)^T M_\Omega m_0.
\]

현재 production separable Jacobi gain은

\[
\gamma_x^2=\operatorname{diag}(H_x^T M_\Omega H_x),
\qquad
\gamma_y^2=\operatorname{diag}(H_y^T M_\Omega H_y)
\]

이고

\[
D_{\mathrm{base}}=\gamma_x^2+\gamma_y^2,
\]

\[
D=D_{\mathrm{base}}
+(\lambda_{\mathrm{rel}}+\epsilon_{D,\mathrm{rel}})
\operatorname{mean}(D_{\mathrm{base}})
\]

를 사용한다.

### 3.4 Static과 dynamic tensor의 경계

다음 값은 geometry, coefficient, GreenNet과 reconstruction rule이 고정되어 있으면
source와 CouplingNet parameter에 무관하다.

| Static, sidecar 저장 대상 | 이유 |
|---|---|
| segment-local `H_x`, `H_y` block | frozen Green response operator |
| block valid-point indices | local block과 global valid point의 매핑 |
| `gamma_x_squared`, `gamma_y_squared` | fixed operator column gain |
| `preconditioner_base` | fixed Jacobi base |
| `gain_scale` | fixed damping scale |
| `denominator` | fixed preconditioner denominator |
| `point_mass` | fixed geometry quadrature mass |

다음 값은 sample과 현재 CouplingNet output에 의존하므로 저장하거나 재사용하지 않는다.

| Dynamic, 매 forward 재계산 | 이유 |
|---|---|
| `p`, `q`, `symmetric_physical` | network/source dependent |
| `mismatch_pre`, `gradient` | current proposal dependent |
| `direction_k`, `response_direction_k` | current residual dependent |
| `coefficient_k`, `eta_star`, `eta_applied` | sample dependent |
| `delta`, `projected_physical` | current tangent result |
| stationarity/response-trust values | current training objective |

K=1에서는 cached `D`에 대해 `z=D^-1 g`를 계산한다. K=2,3,4에서도 같은 `D`와
같은 `H_x,H_y`를 residual gradient마다 반복 적용할 뿐, preconditioner를 다시 만들지
않는다.

## 4. 구현 전 상태와 현재 해소된 부분

`SymmetricTangentGreenResponseContextCache`는 trainer/evaluator instance마다
context를 lazy build하거나 schema-v2 sidecar에서 strict load한 뒤 한 번만
재사용한다. GreenNet은 trainer 생성 시 `eval()`과 `requires_grad_(False)`로
고정된다. Cache telemetry는 build/load/save count와 시간, context ID, schema,
path 및 file size를 구분한다.

Artifact의
`data/symmetric_tangent_green_response_fields.npz`에는 다음만 저장한다.

- `gamma_x_squared`
- `gamma_y_squared`
- `preconditioner_base`
- `denominator`
- `point_mass`
- K/eta 관련 scalar provenance

이 NPZ에는 response block matrix와 indices가 없으므로 `H_x`, `H_y`, adjoint 및
K-dimensional tangent step을 복원할 수 없다. 따라서 diagnostic archive로만
유지한다. Runtime 복원은 `src/greenonet/complex_tangent_context_io.py`의 별도
`tangent_response_context.safetensors`와 동일 cache의 load-or-build lifecycle만
사용한다.

## 5. 파일과 ownership

### 5.1 파일 배치

한 fixed operator당 다음 파일 한 쌍을 사용한다.

```text
<run-directory>/tangent_response_context.safetensors
<run-directory>/tangent_response_context.json
```

- `.safetensors`: authoritative payload와 machine-readable metadata
- `.json`: 사람이 확인하기 위한 manifest mirror
- Loader는 `.safetensors`만으로 완전하게 검증하고 복원할 수 있어야 한다.
- JSON이 없거나 stale이면 context load 결과를 바꾸지 않는다. Export 시 다시 만들 수
  있다.
- 같은 run의 best-energy, best-physics, final CouplingNet checkpoint는 sidecar 하나를
  공유한다.

### 5.2 Model checkpoint와 분리하는 이유

Context는 model parameter가 아니며 CouplingNet state dict에 의존하지 않는다.
Model checkpoint에 context key를 추가하면 architecture validation, old model loading,
checkpoint size와 ownership을 불필요하게 결합한다. 따라서

```text
complex_coupling_model_best_energy.safetensors
tangent_response_context.safetensors
```

는 서로 독립적인 파일이다.

### 5.3 새 core module

다음 새 파일을 추가하는 것을 기준으로 한다.

```text
src/greenonet/complex_tangent_context_io.py
```

권장 public symbols는 다음과 같다.

```python
@dataclass(frozen=True)
class TangentContextIdentity: ...

@dataclass(frozen=True)
class TangentContextIoResult: ...

class SymmetricTangentContextStore:
    def save(...) -> TangentContextIoResult: ...
    def load(...) -> tuple[SymmetricTangentGreenResponseContext, TangentContextIoResult]: ...
```

Pure packing/unpacking helpers는 module-private 함수로 둔다. Trainer, evaluator와
artifact exporter가 직접 safetensors key를 다루지 않는다.

## 6. Public config 계약

Persistence policy는 model architecture가 아니라 runtime 정책이므로
`coupling_training` 아래에 둔다.

```json
"coupling_training": {
  "tangent_context_checkpoint": {
    "enabled": true,
    "path": null,
    "load_policy": "if_available",
    "save_after_build": true
  }
}
```

새 dataclass는 다음 계약을 사용한다.

```python
@dataclass
class TangentContextCheckpointConfig:
    enabled: bool = False
    path: Path | None = None
    load_policy: Literal["never", "if_available", "required"] = "if_available"
    save_after_build: bool = True
```

### 6.1 Field 의미

- `enabled=false`: 현재와 동일한 in-memory lazy build만 사용한다. Disk I/O가 없다.
- `load_policy="never"`: sidecar가 있어도 읽지 않고 current inputs로 build한다.
- `load_policy="if_available"`: sidecar가 있으면 strict load하고, 없으면 build한다.
- `load_policy="required"`: valid sidecar가 없으면 즉시 실패한다. GreenNet kernel을
  평가해 대체하지 않는다.
- `save_after_build=true`: 이번 runtime에서 build한 context를 atomic save한다.
- `path=null`: caller가 아래의 deterministic default path를 resolve한다.

### 6.2 Path resolution

- Training: `<work_dir>/tangent_response_context.safetensors`
- Evaluation/artifact/audit: 기본적으로
  `<coupling_checkpoint.parent>/tangent_response_context.safetensors`
- CLI `--tangent-context PATH`: config/default path보다 우선한다.
- Explicit relative config path는 config file이 아니라 process working directory를
  기준으로 하지 않는다. Config loader가 사용하는 existing path policy를 따라
  resolved absolute path로 core에 전달한다.

### 6.3 Validation

- Unknown key는 거부한다.
- Boolean field에는 실제 JSON boolean만 허용한다.
- `enabled=false`에서 `path!=null`, non-default load policy 또는
  `save_after_build=false`를 조용히 무시하지 않는다. 명시적 validation error를 낸다.
- `load_policy="required"`와 `save_after_build=true` 조합은 거부한다. Required mode는
  load-only다.
- Tangent projection mode가 아니면 `enabled=true`를 거부한다.
- Unit-square CouplingNet은 이 complex-only option을 거부한다.

### 6.4 Recommended usage

첫 training run:

```json
"tangent_context_checkpoint": {
  "enabled": true,
  "path": null,
  "load_policy": "if_available",
  "save_after_build": true
}
```

Frozen production evaluation:

```json
"tangent_context_checkpoint": {
  "enabled": true,
  "path": null,
  "load_policy": "required",
  "save_after_build": false
}
```

Current behavior rollback:

```json
"tangent_context_checkpoint": {
  "enabled": false,
  "path": null,
  "load_policy": "if_available",
  "save_after_build": true
}
```

## 7. Safetensors schema

### 7.1 Format identifier

다음 값을 고정한다.

```text
format_name = "greenonet_symmetric_tangent_context"
schema_version = 1
```

Unknown `format_name`, unknown schema version 또는 missing required key는 fail fast다.
Version migration을 추측하지 않는다.

### 7.2 Segment block packing

Axis `s`의 block 개수를 `B_s`, 각 block 크기를 `n_{s,j}`라 한다. Variable-size square
matrix를 다음 flat tensors로 저장한다.

| Tensor key | dtype | shape | 의미 |
|---|---:|---:|---|
| `x_block_sizes` | int64 | `(B_x,)` | 각 x block의 `n_j` |
| `x_index_ptr` | int64 | `(B_x+1,)` | concatenated index offset |
| `x_valid_indices` | int64 | `(sum n_j,)` | global valid-point indices |
| `x_matrix_ptr` | int64 | `(B_x+1,)` | flattened matrix offset |
| `x_matrix_values` | context float | `(sum n_j^2,)` | row-major block matrices |
| `y_block_sizes` | int64 | `(B_y,)` | 각 y block의 `n_j` |
| `y_index_ptr` | int64 | `(B_y+1,)` | concatenated index offset |
| `y_valid_indices` | int64 | `(sum n_j,)` | global valid-point indices |
| `y_matrix_ptr` | int64 | `(B_y+1,)` | flattened matrix offset |
| `y_matrix_values` | context float | `(sum n_j^2,)` | row-major block matrices |

Packing은 block tuple의 기존 순서를 보존한다. Matrix는 `contiguous().reshape(-1)`로
저장하고 load 시 `(n_j,n_j)`로 복원한다.

### 7.3 Preconditioner tensors

| Tensor key | dtype | shape |
|---|---:|---:|
| `gamma_x_squared` | context float | `(P,)` |
| `gamma_y_squared` | context float | `(P,)` |
| `cross_axis_inner_product` | context float | `(P,)` |
| `normalized_correlation` | context float | `(P,)` |
| `normalized_quadratic_cross_axis` | context float | `(P,)` |
| `separable_preconditioner_base` | context float | `(P,)` |
| `exact_preconditioner_base` | context float | `(P,)` |
| `absolute_preconditioner_base` | context float | `(P,)` |
| `quadratic_preconditioner_base` | context float | `(P,)` |
| `separable_denominator` | context float | `(P,)` |
| `exact_denominator` | context float | `(P,)` |
| `absolute_denominator` | context float | `(P,)` |
| `quadratic_denominator` | context float | `(P,)` |
| `gain_scale` | context float | `(1,)` |
| `q_epsilon`, `damping` | context float | `(1,)` each |
| `point_mass` | context float | `(1,)` |
| `cauchy_violation` | context float | `(P,)` |
| `cauchy_violation_max` | context float | `(1,)` |
| `exact_roundoff_clamp_mask` | int8 | `(P,)` |
| `exact_roundoff_clamp_count` | int64 | `(1,)` |

`P`는 `geometry.num_points`다. Scalar는 rank-0 대신 shape `(1,)`로 저장한다.

### 7.4 저장하지 않는 tensors

- `direction_0` ... `direction_3`
- `response_direction_*`
- `directional_response_*`
- `coefficient_*`
- `mismatch_*`, `gradient`, `delta`
- `rhs`, `p`, `q`, `phi`, `psi`
- `u_phi`, `u_psi`, `u_pred`
- reference `sol/target_phi/target_psi`
- CouplingNet 또는 GreenNet model parameter

### 7.5 Safetensors metadata

Safetensors string metadata에 `manifest_json` 하나를 canonical JSON으로 넣는다.
Canonical JSON은 UTF-8, sorted keys, compact separators를 사용한다. Required fields는
다음과 같다.

```json
{
  "format_name": "greenonet_symmetric_tangent_context",
  "schema_version": 2,
  "formula_suite_id": "tangent_diagonal_preconditioner_suite_v2",
  "context_id": "sha256:...",
  "created_utc": "...",
  "point_count": 0,
  "x_block_count": 0,
  "y_block_count": 0,
  "floating_dtype": "float64",
  "stored_device": "cpu",
  "operator_formula": "H_s=K_s*W_s*L_s^2",
  "gradient_formula": "g=(H_x+H_y)^T*M_Omega*m",
  "created_with_preconditioner_variant": "separable",
  "relative_lambda": 0.01,
  "denominator_relative_eps": 1e-12,
  "cross_axis_relative_eps": 1e-12,
  "identity": {},
  "tensor_payload_sha256": "..."
}
```

`subspace_dimension`, `eta`, `eta_strategy`, `line_search_relative_eps`와 selected
`preconditioner_variant`는 context file의 compatibility key가 아니다. 이 값들은
저장된 `H_x,H_y`와 네 denominator suite를 바꾸지 않으므로 current runtime config에서
주입한다. `created_with_preconditioner_variant`는 생성 당시 provenance일 뿐 load할
variant를 제한하지 않는다.

## 8. Compatibility identity와 hashing

### 8.1 Authoritative identity fields

`TangentContextIdentity`는 다음 digest를 가져야 한다.

| Field | 역할 |
|---|---|
| `geometry_semantic_sha256` | 실제 loaded geometry tensors의 identity |
| `geometry_file_sha256` | 원본 NPZ provenance, path가 있을 때 |
| `green_state_dict_sha256` | 실제 loaded frozen GreenNet parameter identity |
| `green_checkpoint_file_sha256` | checkpoint file provenance, path가 있을 때 |
| `x_green_branch_sha256` | `H_x` 생성에 직접 사용한 branch tensor identity |
| `y_green_branch_sha256` | `H_y` 생성에 직접 사용한 branch tensor identity |
| `reconstruction_contract_id` | kernel/quadrature/pull-back semantics |
| `floating_dtype` | response arithmetic dtype |
| `point_count` | valid-point count |

Green branch digest가 coefficient module path나 source code hash보다 권위가 높다.
실제로 `evaluate_segment_green_kernel`에 들어간 `[a_unit, ap_unit, b_unit, c_unit]`
tensor를 hash하면 coefficient 값, segment order와 unit scaling을 동시에 검증할 수 있기
때문이다.

### 8.2 Tensor hash algorithm

Tensor digest는 다음 canonical byte stream에 SHA-256을 적용한다.

1. logical tensor name
2. normalized dtype string
3. rank와 shape
4. detached CPU contiguous raw bytes

State dict는 sorted parameter key 순서로 각 tensor stream을 이어서 hash한다. Python
`hash()`와 pickle bytes를 사용하지 않는다.

Geometry semantic hash에는 적어도 다음 loaded tensors가 포함되어야 한다.

- `coords_valid`, `hx`, `hy`
- `x_recon_ptr`, `x_recon_t`, `x_recon_weight`, `x_recon_valid_index`
- `y_recon_ptr`, `y_recon_t`, `y_recon_weight`, `y_recon_valid_index`
- `x_segment_length`, `y_segment_length`

### 8.3 Reconstruction contract ID

Schema v1은 다음 fixed string을 사용한다.

```text
segment_green_response_interior_kwl2_v1
```

이는 다음을 의미한다.

- connected segment별 kernel
- true boundary endpoint source hard zero
- interior valid points만 response block에 포함
- production `x_recon_*`/`y_recon_*` node와 weights 사용
- physical-source scale `node_weight * segment_length^2`
- `green_branch[0, segment_index]` coefficient contract

위 의미가 하나라도 바뀌면 contract ID 또는 schema version을 바꾼다.

### 8.4 Context ID

`context_id`는 compatibility metadata와 모든 required payload tensor의 canonical hash로
만든다. Human-readable path나 timestamp는 context ID에 포함하지 않는다. 같은
operator/context를 다른 directory에서 만들면 같은 ID가 나와야 한다.

## 9. Save algorithm

저장은 `@torch.no_grad()`에서 다음 순서로 수행한다.

1. Context와 identity의 point count, dtype, device를 검증한다.
2. 모든 block에서 matrix shape `(n,n)`, index shape `(n,)`, finite value와 valid index
   range를 검증한다.
3. 각 axis에서 모든 valid point가 정확히 한 block에 한 번 배정되는지 기존
   `FrozenAxialResponseOperator` validation을 다시 통과시킨다.
4. Context tensors가 finite인지 검사한다.
5. 다음 algebraic invariants를 검사한다.

   \[
   D_{\mathrm{base}}=\gamma_x^2+\gamma_y^2
   \]

   \[
   \mathrm{gain\_scale}=\operatorname{mean}(D_{\mathrm{base}})
   \]

   \[
   D=D_{\mathrm{base}}+
   (\lambda_{\mathrm{rel}}+\epsilon_{D,\mathrm{rel}})\mathrm{gain\_scale}
   \]

6. Tensors를 detached CPU contiguous representation으로 pack한다.
7. Payload digest와 context ID를 계산한다.
8. Destination과 같은 directory에 unique temporary safetensors file을 쓴다.
9. Temporary file을 다시 safe-open하여 metadata와 required keys를 검증한다.
10. `os.replace`로 authoritative sidecar를 atomic replace한다.
11. Human-readable JSON manifest를 별도 temporary file에 쓴 뒤 atomic replace한다.
12. File size, context ID, save duration과 destination을 log한다.

Parent directory는 생성할 수 있지만 기존 unrelated file을 삭제하지 않는다. Save 실패
시 기존 valid sidecar를 보존한다.

## 10. Load algorithm

Load는 다음 순서로 수행한다.

1. Path가 regular file인지 확인한다.
2. Safetensors로만 연다. `torch.load` fallback을 사용하지 않는다.
3. `manifest_json`을 strict JSON object로 parse한다.
4. Format name, schema version와 required metadata를 검증한다.
5. Required tensor key set이 정확한지 확인한다. Unknown tensor key도 schema 오류로
   거부한다.
6. Integer pointer/index tensors가 int64인지 검사한다.
7. Floating tensors가 manifest dtype과 일치하는지 검사한다.
8. Pointer가 0에서 시작하고 monotone이며 마지막 offset이 payload length와 일치하는지
   검사한다.
9. 각 `matrix_ptr[j+1]-matrix_ptr[j] == block_sizes[j]^2`와
   `index_ptr[j+1]-index_ptr[j] == block_sizes[j]`를 검사한다.
10. Block objects와 frozen x/y operators를 CPU에서 복원한다.
11. Payload digest와 context ID를 다시 계산해 manifest와 비교한다.
12. Runtime `TangentContextIdentity`의 모든 authoritative field를 stored identity와
    비교한다.
13. Runtime `relative_lambda`, `denominator_relative_eps`와 formula ID가 저장 값과
    정확히 일치하는지 확인한다.
14. Runtime target device로 move한다. Dtype cast는 하지 않는다. Runtime dtype과
    stored dtype이 다르면 실패한다.
15. Runtime의 K/eta/line-search fields를 주입해
    `SymmetricTangentGreenResponseContext`를 생성한다.
16. Loaded context의 기존 `validate_for(...)`와 algebraic invariants를 다시 검사한다.
17. Source=`loaded`, load duration, file bytes와 context ID를 log한다.

CPU에서 저장한 float64 context는 float64를 유지하면서 CPU 또는 GPU에서 사용할 수
있다. Device 변경은 compatibility mismatch가 아니다. Dtype 변경은 mismatch다.

## 11. Cache lifecycle

### 11.1 Required state

`SymmetricTangentGreenResponseContextCache`는 다음 telemetry를 유지한다.

```text
context
source: "none" | "built" | "loaded"
build_count
load_count
save_count
build_seconds
load_seconds
save_seconds
context_path
context_id
file_bytes
```

### 11.2 State transitions

`enabled=false`:

```text
empty -> build -> in-memory reuse
```

`if_available` and file exists:

```text
empty -> strict load -> in-memory reuse
```

`if_available` and file missing:

```text
empty -> build -> optional atomic save -> in-memory reuse
```

`required`:

```text
empty -> strict load -> in-memory reuse
```

`never`:

```text
empty -> build -> optional atomic save -> in-memory reuse
```

어떤 mode에서도 한 instance가 valid context를 가진 뒤 두 번째 build/load를 수행하지
않는다.

### 11.3 Invalid existing file policy

`if_available`에서 file이 **없는 경우만** build한다. File이 있으나 corrupt,
incompatible 또는 schema mismatch이면 즉시 실패한다. 다음과 같은 broad exception
fallback은 금지한다.

```python
try:
    return load_context(path)
except Exception:
    return build_context(...)
```

명시적인 rebuild는 `load_policy="never"`와 `save_after_build=true`로 수행한다.

## 12. Trainer integration

1. `ComplexCouplingTrainer.__init__`에서 config를 parse하고 context path와 identity
   inputs를 준비한다.
2. GreenNet은 지금처럼 `eval()`과 `requires_grad_(False)`로 고정한다.
3. 첫 `_forward_batch`에서 `get_or_load_or_build(...)`를 호출한다.
4. Load가 성공하면 `FrozenAxialResponseOperatorBuilder.build(...)`와 GreenNet kernel
   evaluation을 호출하지 않는다.
5. Build한 경우 first batch의 x/y Green branch를 hash하고 context를 생성한 뒤 policy에
   따라 저장한다.
6. 첫 batch에 여러 rows가 있으면 모든 row의 x/y Green branch가 row 0과 동일한지
   검증한다. 현재 response builder가 `green_branch[0, segment]`를 사용하므로 이
   검증 없이 하나의 context를 저장하면 안 된다.
7. 이후 모든 train/validation batch에서도 Green branch digest 또는 fixed-operator
   identity가 같은지 검증한다. 전체 bytes hash를 매 batch 계산하는 비용을 피하려면
   dataset/provider가 제공하는 immutable operator identity를 사용하되, first batch
   direct tensor hash와 일치함을 한 번 확인해야 한다.
8. Best-energy, best-physics, periodic/final CouplingNet checkpoint마다 sidecar를
   복제하지 않는다.
9. Context 저장 실패는 training 성공으로 숨기지 않는다. Persistence가 enabled이면
   명시적으로 training을 실패시킨다.

Context tensor는 detached/frozen이어야 하지만 `H_x/H_y` matvec을 통한 CouplingNet
output gradient는 유지되어야 한다. 즉 context parameter에 gradient는 필요 없지만
`forward(source)`에서 source를 detach하면 안 된다.

## 13. Evaluator, artifact와 audit integration

### 13.1 Evaluator

- `ComplexCouplingEvaluator`는 trainer와 동일 store/cache를 사용한다.
- `required` load 성공 시 `context_build_count=0`, `context_load_count=1`이어야 한다.
- Loaded context와 runtime-built context는 같은 input에 대해 K=1..4 projection 결과가
  tolerance 안에서 동일해야 한다.
- CouplingNet checkpoint hash는 context compatibility에 포함하지 않는다.

### 13.2 Artifact exporter

`cli/export_coupling_artifacts.py`에 다음 optional override를 추가한다.

```text
--tangent-context PATH
```

Artifact summary에는 다음을 기록한다.

```json
{
  "tangent_context": {
    "persistence_enabled": true,
    "source": "loaded",
    "path": "...",
    "context_id": "sha256:...",
    "schema_version": 1,
    "build_count": 0,
    "load_count": 1,
    "save_count": 0,
    "build_seconds": 0.0,
    "load_seconds": 0.0,
    "save_seconds": 0.0,
    "file_bytes": 0,
    "identity_verified": true
  }
}
```

기존 `symmetric_tangent_green_response_fields.npz`와 figures는 유지한다. Sidecar는
selected-sample artifact가 아니라 run-level frozen operator asset이다.

### 13.3 Audit CLI

Tangent audit CLI도 같은 `--tangent-context` override를 받을 수 있어야 한다. Audit가
자체 serializer를 구현하거나 artifact NPZ에서 context를 추측해 만들면 안 된다.

## 14. Error contract

다음은 서로 다른 오류 유형으로 명확히 보고한다.

| 오류 | 메시지에 포함할 정보 |
|---|---|
| Missing required file | resolved path, load policy |
| Unknown schema | stored/expected version |
| Missing/unknown tensor key | key 목록 |
| Corrupt payload | context ID 또는 payload digest mismatch |
| Geometry mismatch | stored/runtime geometry digest |
| GreenNet mismatch | stored/runtime state digest |
| Green branch mismatch | axis와 stored/runtime digest |
| Reconstruction mismatch | stored/runtime contract ID |
| Dtype mismatch | stored/runtime dtype |
| Config mismatch | formula ID와 mismatched field/value |
| Invalid block | axis, block index, expected/actual shape or offset |

Absolute path를 log할 수 있지만 error에 source values, model tensor contents 또는
reference fields를 dump하지 않는다.

## 15. Storage와 성능

Block matrix payload entry 수는

\[
N_H=\sum_{j=1}^{B_x} n_{x,j}^2+
\sum_{j=1}^{B_y} n_{y,j}^2
\]

이고 대략적인 byte 수는

\[
\mathrm{bytes}\approx
N_H\times\mathrm{sizeof(dtype)}
+(\sum n_{x,j}+\sum n_{y,j})\times8
+O(P).
\]

따라서 저장 전 `matrix_entry_count`, estimated bytes와 actual file bytes를 log한다.
이 sidecar는 diagonal field NPZ보다 크지만, repeated evaluator/artifact process에서
segment별 GreenNet kernel evaluation을 제거한다.

Load 후에도 tangent step의 block matvec, adjoint, K-dimensional response MGS와
sample-specific coefficient 계산 비용은 남는다. Persistence는 tangent correction을
없애는 최적화가 아니라 **static operator construction만 재사용하는 최적화**다.

## 16. Reuse boundary

다음 변화에는 같은 sidecar를 재사용할 수 있다.

- Source sample, train/validation/test split 변경
- CouplingNet checkpoint 또는 parameter 변경
- Batch size와 batch order 변경
- K=1,2,3,4 변경
- K=1 `eta`, eta strategy와 line-search cap 변경
- Response-trust/stationarity loss on/off 또는 weight 변경
- CPU/GPU device 변경, 단 dtype 유지

다음 변화에는 재사용할 수 없다.

- Geometry 또는 resolution 변경
- Axial reconstruction nodes/weights/segment ordering 변경
- GreenNet state 변경
- Actual x/y Green coefficient branch 변경
- Physical/reference pull-back convention 변경
- Floating dtype 변경
- Jacobi denominator를 정의하는 `relative_lambda` 또는
  `denominator_relative_eps` 변경
- Formula-suite 또는 `cross_axis_relative_eps` 변경. 단, schema v2에 저장된 네
  `preconditioner_variant` 사이의 선택 변경은 허용한다.

Coefficient field가 sample마다 달라지는 operator family에서는 하나의 sidecar가 전체
dataset을 대표할 수 없다. 그 문제를 지원하려면 coefficient/Green-branch identity별
context를 별도로 저장하는 keyed cache가 필요하며 이 문서의 v1 범위 밖이다.

## 17. Preconditioner variant와 schema v2

Separable 전용 schema v1은 production에 도입하지 않았다. 최초 production format은
네 preconditioner가 공유하는 schema v2다. Sidecar는
`a,b,c,rho,q`, 네 undamped base, 네 damped denominator, common damping,
`q_epsilon`, Cauchy violation과 exact-diagonal roundoff-clamp audit를 모두 저장한다.
여기서 저장되는 `q`는

\[
q=\frac{c^2}{a+b+\varepsilon_q}
\]

이고 `D_q`를 고를 때만 `4q`를 더한다. Selected variant, K와 eta는 static operator
identity에 포함하지 않으므로 하나의 schema-v2 sidecar를 네 variant와 K=1..4가
공유한다. 대신 `relative_lambda`, `denominator_relative_eps`,
`cross_axis_relative_eps`, geometry, GreenNet state, actual Green branch,
reconstruction contract와 dtype mismatch는 strict rejection한다. Missing required
tensor를 0으로 간주하거나 알 수 없는 schema를 migration하는 fallback은 없다.

## 18. Test plan

### 18.1 Serialization unit tests

- Variable-size x/y blocks의 pack/unpack round-trip
- Empty/malformed pointer, non-square matrix, duplicate/missing valid index rejection
- Float64 tensor bitwise round-trip
- CPU save/load와 optional CUDA load smoke
- Unknown/missing tensor key rejection
- Unknown schema/format rejection
- Corrupted payload/context ID rejection
- Atomic overwrite에서 기존 valid file 보존

### 18.2 Math equivalence tests

Loaded와 built context에 대해 다음을 비교한다.

- `H_x.forward`, `H_y.forward`
- `H_x.adjoint`, `H_y.adjoint`
- `tangent_gradient`
- `gamma_x_squared`, `gamma_y_squared`, `D`
- K=1 fixed eta
- K=1 closed-loop exact line search
- K=2, K=3, K=4 final `delta`, mismatch, coefficients와 activity masks
- stationarity와 response-trust loss/diagnostics
- final `phi+psi=rhs`

Float64에서는 기존 implementation tolerance를 사용하고, 저장 전후 산술 순서가
같다면 exact equality를 우선 검증한다.

### 18.3 Identity tests

각 field를 하나씩 바꿔 strict rejection을 검증한다.

- Geometry tensor/hash
- GreenNet parameter/hash
- x/y Green branch tensor/hash
- reconstruction contract ID
- dtype
- point count
- `relative_lambda`
- `denominator_relative_eps`
- formula ID

반대로 CouplingNet checkpoint, source, K와 eta를 바꿔도 load가 허용되는지 검증한다.

### 18.4 Integration tests

- Trainer first forward: build=1, save=1
- Trainer subsequent forward/validation: additional build/load 없음
- Evaluator required load: build=0, load=1
- Existing `runtime_only`: 현재 K=1..4 수치와 build count 유지
- Artifact exporter CLI override와 default sibling resolution
- Summary/log/manifest provenance
- Sidecar가 없어도 persistence disabled config가 그대로 동작
- CouplingNet model checkpoint keys와 safetensors format 불변
- Unit-square rejection
- Reference `sol/phi/psi`가 context identity 또는 projection 계산에 들어가지 않음

### 18.5 Static/full verification

Focused tests를 먼저 실행한 뒤 repository convention을 따른다.

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_complex_tangent_context_io.py \
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

## 19. 단계별 구현 순서

1. `TangentContextCheckpointConfig`와 strict config tests를 추가한다.
2. Tensor/state/geometry SHA-256 canonical helper를 existing hash utility와 비교하고,
   중복 없이 shared helper로 정리한다.
3. `complex_tangent_context_io.py`의 block pack/unpack과 schema validator를 구현한다.
4. Small synthetic context round-trip/math tests를 통과시킨다.
5. `TangentContextIdentity` builder를 geometry, GreenNet state와 Green branch에 연결한다.
6. `SymmetricTangentGreenResponseContext`에 serialized static payload로부터 runtime K/eta
   fields를 주입하는 strict constructor를 추가한다.
7. `SymmetricTangentGreenResponseContextCache`에 load/build/save lifecycle과 telemetry를
   통합한다.
8. Trainer first-forward save와 fixed-operator batch validation을 연결한다.
9. Evaluator load와 path resolution을 연결한다.
10. Artifact/audit CLI의 `--tangent-context` override를 연결한다.
11. Summary, logs와 existing context NPZ provenance를 확장한다.
12. Focused tests, full tests, Ruff, mypy와 diff check를 실행한다.

각 단계에서 runtime-only path를 먼저 regression test하고 persistence path를 추가한다.

## 20. Acceptance criteria

- Persistence를 생략한 기존 config의 projection 수치가 바뀌지 않는다.
- One fixed operator context가 training에서 한 번 생성되고 한 번만 저장된다.
- Evaluation은 valid sidecar에서 context를 복원하며 GreenNet kernel builder를 호출하지
  않는다.
- Built와 loaded context가 K=1..4에서 같은 tangent 결과를 만든다.
- `H_x,H_y,D`만 재사용하고 sample-dependent direction/correction은 매번 계산한다.
- Geometry/GreenNet/Green branch/reconstruction/dtype/config mismatch가 모두 fail fast다.
- Invalid existing sidecar가 silent rebuild되지 않는다.
- CouplingNet model checkpoint tensor key와 format이 변하지 않는다.
- Reference target이 context 저장, load, projection, loss 또는 checkpoint selection에
  사용되지 않는다.
- Artifact와 log만으로 context source, identity, schema, path, timing과 validation
  결과를 재현할 수 있다.
- Full test/lint/type/diff verification이 통과한다.

## 21. Rollback

Runtime rollback은

```json
"tangent_context_checkpoint": {
  "enabled": false,
  "path": null,
  "load_policy": "if_available",
  "save_after_build": true
}
```

로 수행한다. 이 설정은 현재 in-memory lazy build와 동일해야 한다.

Code rollback은 다음만 제거한다.

- `complex_tangent_context_io.py`
- context checkpoint config
- cache load/save integration
- CLI override
- persistence telemetry와 tests

Projection, Green reconstruction, tangent K algorithm, objective, model architecture,
dataset/geometry NPZ와 model checkpoint loader는 rollback에서 수정하지 않는다.
Sidecar는 model checkpoint와 독립이므로 삭제해도 model migration이 필요 없다.

## 22. 구현 시 금지 사항

- Model state dict에 `tangent_context.*` key 추가
- `torch.load` fallback
- Broad `except Exception: rebuild`
- Fingerprint mismatch를 warning만 남기고 계속 사용
- Runtime dtype으로 implicit cast
- CouplingNet checkpoint hash를 operator identity로 요구
- Source/reference field를 sidecar에 저장
- Every epoch 또는 every batch save
- Best checkpoint마다 같은 context 복제
- Loaded context의 source input을 detach하여 CouplingNet gradient 차단
- Existing diagnostic NPZ를 runtime sidecar로 오인
- Context load를 이유로 balance, reconstruction 또는 loss 수식 변경

## 23. 구현 완료 후 문서 갱신

실제 구현이 완료되면 다음을 같은 변경에서 갱신한다.

- `README.md`: config와 CLI 사용 예
- `docs/memory.md`: persistence contract와 default
- `docs/couplingnet_complex_geometry_design.md`: runtime lifecycle
- Artifact summary schema 설명
- 이 문서의 상태를 `구현 완료`로 변경하고 실제 schema/version/file key와 tests를
  다시 대조한다.
