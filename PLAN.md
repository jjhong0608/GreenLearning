# Geometry-Only Automatic Tangent Dimension \(K\) Selection Plan

## Summary

- Complex CouplingNet의 `symmetric_tangent_green_response`에서 geometry의 connected axial-line incidence만으로 tangent subspace dimension \(K\)를 자동 선택한다.
- Active point 사이의 point-graph 거리 \(d_L(i,j)\)와 tangent structural distance를
  \[
  d_A(i,j)=\left\lceil\frac{d_L(i,j)}{2}\right\rceil
  \]
  로 정의하고,
  \[
  C_i(K)=\frac1P\sum_{j=1}^P\mathbf 1[d_A(i,j)\le K-1],
  \qquad
  C_{\mathrm{global}}(K)=\frac1P\sum_{i=1}^P C_i(K)
  \]
  를 계산한다.
- 자동 선택값은
  \[
  K_{\Omega,h}
  =
  \min\left\{
  K:
  C_{\mathrm{global}}(K)\ge\tau_{\mathrm{global}},
  \quad
  Q_{0.05}(C_i(K))\ge\tau_{\mathrm{tail}}
  \right\}
  \]
  로 고정한다. 기본값은 \(\tau_{\mathrm{global}}=\tau_{\mathrm{tail}}=0.99\)이고, quantile \(0.05\)는 고정한다.
- 이 계산에는 geometry의 `coords_valid`, `x_segment_id`, `y_segment_id`만 사용한다. PDE coefficients, source, GreenNet/CouplingNet prediction, reference `sol/phi/psi`는 사용하지 않는다.
- 현재 확인된 기본 geometry 결과 `Square=2`, `Disk=2`, `Annulus=4`, `Pentagram=4`를 regression 기준으로 사용한다.

## Public Configuration

기존 explicit 설정은 그대로 유지한다.

```json
"symmetric_tangent_green_response": {
  "subspace_dimension": 4,
  "max_subspace_dimension": 8,
  "geometry_k_selection": {
    "enabled": false,
    "global_reach_threshold": 0.99,
    "pointwise_tail_reach_threshold": 0.99
  }
}
```

자동 선택은 다음처럼 활성화한다.

```json
"symmetric_tangent_green_response": {
  "subspace_dimension": 4,
  "max_subspace_dimension": 8,
  "geometry_k_selection": {
    "enabled": true,
    "global_reach_threshold": 0.99,
    "pointwise_tail_reach_threshold": 0.99
  }
}
```

- `enabled=false`가 backward-compatible default이며 `subspace_dimension`을 그대로 사용한다.
- `enabled=true`이면 geometry가 계산한 \(K_{\Omega,h}\)가 `subspace_dimension`을 override한다. 이때 명시적 값은 fallback으로 사용하지 않는다.
- `max_subspace_dimension`은 explicit/automatic mode에 공통으로 적용되는 configurable safety limit이며 기본값은 `8`이다.
- 자동으로 필요한 \(K\)가 safety limit보다 크면 clamp하지 않고, 필요한 최소 \(K\), limit에서의 두 reach metric과 geometry diameter를 포함해 fail fast한다.
- Threshold는 boolean이 아닌 finite numeric `(0,1]`, dimension과 maximum은 boolean이 아닌 양의 정수로 strict validation한다. Unknown key도 거부한다.
- \(K\ge2\)로 해석되면 기존처럼 `eta_strategy="closed_loop_exact_line_search"`가 필수다.

## Implementation Changes

1. Plotly/Scipy에 의존하지 않는 production geometry-selection core를 분리한다. 기존 topology analyzer와 visualization CLI는 이 core를 재사용해 graph 정의가 달라지지 않게 한다.
2. Selector는 `ComplexGeometryMetadata`를 받아 topology counts, selected \(K\), global/tail reach, graph diameter, geometry identity와 setup time을 반환한다.
3. Train/eval/artifact config-loading boundary에서 geometry를 한 번 읽고 model/trainer/evaluator 생성 전에 auto 설정을 explicit resolved config로 변환한다. Unresolved auto config가 runtime까지 들어오면 fail fast한다.
4. `config_used.json`에는 실제 실행한 정수 `subspace_dimension`, `geometry_k_selection.enabled=false`를 저장한다. 별도 `tangent_subspace_dimension_provenance`에는 원래 auto 설정, thresholds, fixed quantile, configured/selected/max \(K\), geometry SHA-256, reach metric과 setup time을 기록한다.
5. 원본 auto config를 직접 evaluation/export에 주는 경우에도 동일 resolver를 실행한다. Resolved `config_used.json`을 주면 topology를 재계산하지 않는다.
6. Tangent core의 `Literal[1,2,3,4]`와 `{2,3,4}` validation을 동적 positive \(K\)로 일반화한다. Existing K1 special path와 K2 construction은 유지하고, K3 이상은 기존 MGS loop를 그대로 확장한다.
7. `KrylovSubspaceStepResult`의 stacked directions/responses/coefficients/costs를 authoritative dynamic contract로 사용한다. 기존 direction-0/1 alias는 호환성을 위해 유지하되 K>2 자료는 dynamic result에서 읽는다.
8. Trainer metrics, CSV field ordering, evaluator rows, raw NPZ, Plotly figures와 artifact summary를 resolved \(K\)만큼 동적으로 생성한다. 기존 K1–K4 key와 수치는 유지한다.
9. K가 증가하다 numerical direction이 퇴화하면 기존처럼 해당 방향을 zero/inactive로 표시하고 이후 cost가 증가하지 않도록 한다. Balance \(\phi+\psi=f\)는 모든 \(K\)에서 그대로 보존한다.
10. Tangent response-context sidecar는 \(K\)와 독립적인 frozen operator/preconditioner이므로 schema와 identity를 변경하지 않는다. Model architecture와 safetensors tensor key도 변경하지 않는다.
11. 기존 audit CLI의 기본 비교 범위는 K1–K4로 유지하되, explicit 요청 시 `max_subspace_dimension`까지 사용할 수 있도록 공통 dynamic core를 적용한다.
12. `README.md`, `docs/memory.md`, tangent design 문서에 auto/explicit precedence, geometry-only 의미, fixed 5% quantile, safety limit, provenance와 \(O(K)\) operator action 및 \(O(K^2)\) MGS 비용을 기록한다. `PLAN.md`는 사용자가 이 계획을 바탕으로 작성한다.

## Test Plan

- Config: legacy explicit K round-trip, auto nested config, independent thresholds, unknown/invalid keys, explicit \(K>\) maximum, auto-selected \(K>\) maximum, K2+ eta-strategy validation을 검사한다.
- Geometry math: synthetic chain에서 known \(C_i(K)\), \(C_{\mathrm{global}}(K)\), 최소 K와 threshold 변화 효과를 검증하고 disconnected incidence graph를 거부한다.
- Real geometry regression: 현재 NPZ에서 Square/Disk/Annulus/Pentagram이 각각 `2/2/4/4`로 선택되는지 확인한다.
- Dynamic projection: K5와 K8에서 finite result, exact physical balance, nested response-cost non-increase, inactive-direction fallback과 autograd를 검증한다.
- Backward compatibility: K1–K4 projection 결과와 existing metric key가 변경 전 tolerance를 유지하고 explicit mode에서는 topology selector가 호출되지 않는지 확인한다.
- Integration: train/eval/export가 같은 resolved K를 사용하고, auto resolution이 runtime당 한 번만 실행되며, `config_used.json`, logs, CSV, artifact summary에 동일 provenance가 기록되는지 확인한다.
- 전체 검증은 focused config/topology/projection tests 이후 전체 `pytest test`, `ruff check src cli test`, touched-file `ruff format`, `mypy src`, `git diff --check` 순서로 수행한다.

## Rollback And Acceptance

- Runtime rollback은 `geometry_k_selection.enabled=false`와 explicit `subspace_dimension` 설정만으로 가능하다.
- Auto run의 `config_used.json`은 resolved explicit K를 저장하므로 selector 코드를 제거해도 해당 checkpoint의 evaluation/export가 가능하다.
- Code rollback은 geometry resolver, dynamic-K extension과 provenance만 제거하며 K1–K4 projection, tangent context sidecar, model checkpoint와 dataset/geometry NPZ는 유지한다.
- 완료 기준은 auto/default thresholds, independent threshold override, explicit K, K>4 dynamic execution, fail-fast safety limit, exact balance, current four-geometry regression과 reference-free 원칙이 모두 검증되는 것이다.

## Confidence

- 구현 계획 확신도: **0.97**.
- Geometry-only selector가 정의한 수식을 정확히 구현할 확신도: **0.99**.
- 선택된 \(K\)가 모든 PDE에서 accuracy-optimal일 경험적 확신도: **0.78**.
- 규칙은 명확하다. 남은 불확실성은 규칙 모호성이 아니라, 사용자가 safety limit를 크게 높였을 때 K별 메모리·시간 증가와 PDE별 최적 \(K\)가 geometry-only 값과 얼마나 일치하는지에 대한 경험적 정보 부족이다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Geometry-Only Automatic Tangent Dimension K Selection Plan"을 기준 문서로
참고하여 geometry 기반 automatic K selection과 dynamic K>4 integration을
끝까지 구현한다.

완료는 다음 조건으로 검증한다.

- geometry_k_selection.enabled=false인 기존 config가 explicit K 동작과
  K1-K4 수치를 그대로 유지할 것,
- enabled=true이면 PDE, source, model prediction과 reference target을 사용하지
  않고 C_global 및 lower-5% pointwise reach 기준으로 최소 K를 선택할 것,
- 두 reach threshold가 각각 config에서 독립적으로 설정될 것,
- Square, Disk, Annulus, Pentagram의 canonical geometry가 각각 K=2,2,4,4로
  선택될 것,
- explicit K와 automatic K가 configurable max_subspace_dimension 기본값 8을
  공유하고 limit 초과는 clamp 없이 fail fast할 것,
- tangent projection, trainer, evaluator, artifact와 audit가 K>4 dynamic
  directions, coefficients, costs와 activity를 처리할 것,
- 모든 K에서 phi+psi=rhs와 nested response-cost non-increase가 유지될 것,
- config_used.json에는 resolved explicit K가 고정되고 원래 auto policy,
  geometry identity와 reach metrics가 provenance로 기록될 것,
- tangent response-context sidecar, model checkpoint tensor key, GreenNet,
  geometry/sample NPZ schema와 reference-free training 원칙이 변경되지 않을 것,
- focused tests와 전체 pytest, Ruff, mypy, git diff check가 통과할 것.

수정 범위는 tangent geometry-selection core, config parsing/resolution,
dynamic tangent subspace core, train/eval/export/audit provenance, 관련 tests와
문서로 제한한다.

Full response matrix, global linear solve, PDE-dependent K selector, learnable K,
sample-dependent K, silent K clamping과 장기 retraining은 추가하지 않는다.

각 구현 단계 후 가장 작은 config/topology/projection tests를 먼저 실행하고,
통과한 뒤 trainer/evaluator/artifact integration과 전체 regression suite를
실행한다.

기존 K1-K4 수치 또는 checkpoint compatibility를 유지할 수 없다면 작업을
중단하고 다음을 보고한다.

1. 정확히 달라지는 projection, tensor, config 또는 metric contract,
2. 영향을 받는 config, checkpoint, context sidecar와 artifact,
3. explicit K1-K4 경로를 보존하는 가장 작은 rollback 전략.
```
