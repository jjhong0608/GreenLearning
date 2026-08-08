# Optional Matrix-Free \(K=2\) Tangent Subspace Integration Plan

## Summary

현재 `symmetric_tangent_green_response`는 symmetric-balanced directional source에서 시작해 Jacobi-preconditioned 한 방향 \(z_0=D^{-1}g_0\) 위에서 response mismatch를 최소화하는 \(K=1\) correction을 적용한다. 새 기능은 두 번째 preconditioned Krylov direction을 추가해 2차원 부분공간에서 mismatch를 최소화하는 **matrix-free \(K=2\)** 경로다. 이는 full response Gram matrix를 구성하거나 선형 시스템을 푸는 방법이 아니다. Krylov 방법이 점진적으로 확장된 부분공간에서 해를 개선한다는 일반 원칙과 일치한다. [Netlib Templates for the Solution of Linear Systems](https://www.netlib.org/templates/templates.html)

Frozen-checkpoint audit에서는 \(K=2\)가 \(K=1\) 대비 평균 response mismatch를 약 33.0%, `rel_sol`을 약 25.8%, `rel_u_phi`를 약 17.9%, `rel_u_psi`를 약 22.4% 줄였다. 반면 correction norm은 약 13.2% 증가했고 일부 sample에서 canonical energy가 악화됐다. 따라서 \(K=2\)는 기본값이 아니라 명시적인 opt-in 실험으로 추가한다.

결정 사항은 다음과 같다.

- 기존 \(K=1\)을 `subspace_dimension=1` 기본값으로 그대로 보존한다.
- \(K=2\)는 `subspace_dimension=2`로 활성화한다.
- Frozen audit에서 검증한 무제약 \(c_0,c_1\) 수식을 그대로 production에 적용한다.
- \(K=2\)에서는 기존 `eta` cap과 tangent eta warmup을 적용하지 않는다.
- Stationarity와 response-trust는 최종 \(K=2\) residual/correction을 기준으로 일반화한다.
- CouplingNet architecture, model parameter, checkpoint tensor key는 변경하지 않는다.

## \(K=2\) 수식

Symmetric-balanced source를

\[
\widetilde\phi=\frac{f+p-q}{2},
\qquad
\widetilde\psi=\frac{f-p+q}{2}
\]

라고 둔다. 이때 항상 \(\widetilde\phi+\widetilde\psi=f\)이다. Directional response operator와 mismatch를

\[
S=H_x+H_y,
\qquad
m_0=H_x\widetilde\phi-H_y\widetilde\psi,
\qquad
g_0=S^\top M_\Omega m_0
\]

로 정의한다. \(D\)는 현재 cached column-diagonal Jacobi denominator다.

첫 번째 방향은 현재 uncapped \(K=1\)과 동일하다.

\[
z_0=D^{-1}g_0,
\qquad
v_0=Sz_0,
\qquad
c_0=
\frac{\max(\langle g_0,z_0\rangle,0)}
{\langle v_0,v_0\rangle_{M_\Omega}+\varepsilon_0}.
\]

첫 correction 이후 gradient residual에서 두 번째 방향을 만든다.

\[
r_1=g_0-c_0S^\top M_\Omega v_0,
\qquad
z_{1,\mathrm{raw}}=D^{-1}r_1.
\]

Response-space에서 첫 방향과 직교화한다.

\[
\beta=
\frac{\langle v_0,Sz_{1,\mathrm{raw}}\rangle_{M_\Omega}}
{\langle v_0,v_0\rangle_{M_\Omega}+\varepsilon_0},
\]

\[
z_1=z_{1,\mathrm{raw}}-\beta z_0,
\qquad
v_1=Sz_1,
\qquad
c_1=
\frac{\langle g_0,z_1\rangle}
{\langle v_1,v_1\rangle_{M_\Omega}+\varepsilon_1}.
\]

최종 correction과 directional source는

\[
\delta_{K=2}=-c_0z_0-c_1z_1,
\]

\[
\phi=\widetilde\phi+\delta_{K=2},
\qquad
\psi=\widetilde\psi-\delta_{K=2}.
\]

따라서 모든 valid point에서 \(\phi+\psi=f\)가 정확히 보존된다. 최종 mismatch는

\[
m_2=m_0-c_0v_0-c_1v_1
\]

이며, \(K=2\)는 \(\operatorname{span}\{z_0,z_1\}\)에서 \(\|m_0+S\delta\|_{M_\Omega}^2\)를 최소화한다. 두 번째 response direction이 numerical tolerance 이하이면 `second_direction_active=false`, \(c_1=0\)으로 두어 uncapped \(K=1\)과 정확히 일치시킨다.

## Public Configuration

기존 config block에 strict integer field를 추가한다.

```json
"balance_projection": {
  "enabled": true,
  "mode": "symmetric_tangent_green_response",
  "symmetric_tangent_green_response": {
    "subspace_dimension": 2,
    "eta_strategy": "closed_loop_exact_line_search",
    "line_search_relative_eps": 1e-12,
    "relative_lambda": 0.01,
    "denominator_relative_eps": 1e-12
  }
}
```

- `subspace_dimension`은 boolean이 아닌 정수 `1` 또는 `2`만 허용한다.
- 생략하면 `1`로 해석해 기존 설정과 수치를 보존한다.
- `subspace_dimension=2`는 `eta_strategy="closed_loop_exact_line_search"`만 허용한다.
- \(K=2\)의 \(c_0,c_1\)은 exact subspace minimizer이므로 `eta`와 tangent eta schedule은 적용하지 않는다.
- Shared schema 때문에 `eta`가 존재할 수는 있지만 provenance와 log에 `eta_applicability="k1_only"`를 명시하고 \(K=2\) 계산에는 사용하지 않는다.
- `line_search_relative_eps`를 \(c_0,c_1\) denominator 안정화와 두 번째 방향의 degeneracy 판정에 공통 사용한다.
- Unit-square CouplingNet의 허용 projection mode는 변경하지 않는다.

## Implementation Steps And Affected Files

1. `src/greenonet/config.py`에 `subspace_dimension` parsing, round-trip, strict validation과 \(K=2\)/`eta_strategy` 조합 검증을 추가한다. 기존 config는 자동으로 \(K=1\)이 된다.

2. Frozen audit의 `matrix_free_krylov_k2_step(...)`와 결과 dataclass를 `src/greenonet/complex_tangent_projection.py`의 production core로 이전한다. `src/greenonet/complex_tangent_subspace_audit.py`는 이 공통 helper를 import하도록 바꿔 audit와 production 수식 중복을 제거한다.

3. `SymmetricTangentGreenResponseContext.tangent_step(...)`를 `subspace_dimension` dispatcher로 확장한다. \(K=1\) branch는 기존 코드를 그대로 호출하고, \(K=2\) branch는 두 번의 matrix-free forward response action과 필요한 adjoint action만 사용한다. Global matrix, full Gram matrix, dense \(2\times2\) solve는 만들지 않는다.

4. `src/greenonet/complex_projection.py`에서 \(K=2\)의 최종 `delta`, `projected_physical`, `projected_solution`, `mismatch_post`를 연결한다. \(H_xz_0,H_yz_0,H_xz_1,H_yz_1\)을 재사용해 projected solution을 구성하고 불필요한 Green reconstruction을 반복하지 않는다.

5. Stationarity loss는

\[
r_2=S^\top M_\Omega m_2,
\qquad
\mathcal L_{\mathrm{stat}}
=
\frac{r_2^\top D^{-1}r_2}{E_f+\varepsilon}
\]

로 일반화한다. 기존 initial-gradient-relative diagnostic은

\[
\frac{r_2^\top D^{-1}r_2}
{g_0^\top D^{-1}g_0+\varepsilon}
\]

로 유지한다. Response-trust는 최종 \(m_2\)와 \(S\delta_{K=2}=m_2-m_0\)를 사용한다. 두 loss는 기존처럼 \(H_x(f/2),H_y(f/2)\) source normalization을 공유하며 reference `sol/phi/psi`를 사용하지 않는다.

6. `src/greenonet/complex_coupling_trainer.py`와 evaluator에서 동일 dispatcher를 사용한다. \(K=2\)에서는 tangent eta schedule을 생성하거나 step하지 않고, `subspace_dimension`, \(c_0/c_1\) 평균, second-direction 활성 비율, \(K=1\)/\(K=2\) response cost와 correction norm을 로그와 CSV에 기록한다. Best-energy와 best-physics checkpoint 기준은 그대로 유지한다.

7. Artifact exporter는 기존 `tangent_delta`를 최종 correction으로 유지하면서 `tangent_direction_0/1`, `tangent_response_direction_0/1`, `tangent_coefficient_0/1`, `tangent_second_direction_active`, `tangent_mismatch_k1`, `tangent_mismatch_post`, `tangent_response_cost_k1/k2`, post-\(K=2\) stationarity residual을 raw NPZ와 summary에 추가한다. \(K=1\) artifact schema와 eta fields는 기존 동작을 유지한다.

8. Frozen audit 결과를 재현하는 regression test를 추가하고, coupling8 설정을 기준으로 projection 차수만 `2`로 바꾼 별도 paired experiment config `configs/complex_coupling_soap_tangent_k2_pentagram.json`을 추가한다. 기존 config와 현재 사용자 변경은 덮어쓰지 않는다.

9. `README.md`와 `docs/memory.md`에 \(K=1/K=2\) 수식, exact balance, matrix-free 정책, \(K=2\)에서 eta cap이 적용되지 않는 이유, frozen evidence와 retraining 필요성을 기록한다.

## Test Plan

- Config: 기본값 `1`, `1/2` round-trip, bool/float/0/3 거부, `K=2+fixed eta_strategy` 거부.
- Math: \(K=2\) response cost가 uncapped \(K=1\)보다 tolerance 밖에서 증가하지 않는지 검증한다.
- Degeneracy: 두 번째 direction이 0이면 \(c_1=0\)이고 \(K=2=K=1\)인지 확인한다.
- Projection: 모든 sample과 point에서 \(\phi+\psi=f\), projected response contract, float64 finite output을 검증한다.
- Regression: `subspace_dimension`을 생략하거나 `1`로 두었을 때 기존 \(K=1\) tensor와 metric이 동일한지 확인한다.
- Autograd: \(K=2\), stationarity, response-trust의 gradient가 finite하고 reference target 변경에 영향을 받지 않는지 확인한다.
- Integration: trainer/evaluator/export가 같은 cached context를 한 번만 만들고 동일한 \(K=2\) 결과를 사용하는지 확인한다.
- Artifact: 새 NPZ/CSV/summary fields와 `second_direction_active` schema를 검증한다.
- Audit: 공통 production helper로 리팩터링한 뒤 기존 frozen-checkpoint aggregate 수치가 tolerance 내에서 유지되는지 확인한다.
- Compile: complex one-step training에서 `torch.compile` 경로와 SOAP/AdamW optimizer가 모두 동작하는지 smoke test한다.

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_complex_tangent_projection.py \
  test/test_complex_projection.py \
  test/test_complex_tangent_subspace_audit.py \
  test/test_complex_coupling_trainer.py \
  test/test_complex_coupling_artifacts.py \
  test/test_io_config.py

PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest test
ruff check src cli test
ruff format src cli test
~/.conda/envs/green_net/bin/python -m mypy src
git diff --check
```

실제 장기 retraining은 구현 검증 범위에 포함하지 않는다.

## Rollback Strategy

- Runtime rollback은 `subspace_dimension=1`로 바꾸거나 해당 field를 제거하는 것이다.
- Model architecture와 safetensors key가 바뀌지 않으므로 checkpoint migration은 필요하지 않다.
- Code rollback은 \(K=2\) dispatcher, diagnostics, artifact fields와 paired config만 제거하고 기존 \(K=1\) implementation을 유지한다.
- \(K=1\) tensor 또는 frozen audit 수치를 보존하지 못하면 구현을 중단하고 수치 차이, 영향받는 config/artifact/test, 최소 rollback을 보고한다.

## Confidence

- 수학 및 구현 계획 확신도: **0.98**.
- Frozen-checkpoint 개선이 paired retraining에서도 유지될 가능성: **0.84**.
- 규칙 모호성이나 필요한 정보 부족은 없다. 남은 불확실성은 \(K=2\)가 correction norm을 증가시키면서 일부 sample의 canonical energy를 악화시킨 현상이 재학습을 통해 얼마나 해소되는지에 관한 경험적 불확실성이다.
- 이번 단계에서는 repository 파일을 수정하지 않으며, 사용자가 이 계획을 root `PLAN.md`에 작성한다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Optional Matrix-Free K=2 Tangent Subspace Integration Plan"을 기준 문서로
참고하여 K=2 tangent subspace integration을 끝까지 구현한다.

완료는 다음 조건으로 검증한다.

- 기존 config에서 subspace_dimension을 생략하면 K=1 동작과 수치가 유지될 것,
- subspace_dimension=2가 frozen audit의 무제약 matrix-free K=2 수식을 사용할 것,
- K=2가 두 개의 Jacobi-preconditioned response-orthogonal direction만 사용하고
  global matrix, full Gram matrix 또는 linear solve를 만들지 않을 것,
- 두 번째 direction이 퇴화하면 c1=0으로 K=1에 안전하게 fallback할 것,
- 모든 valid point에서 phi+psi=rhs가 float64 tolerance 내에서 보존될 것,
- K=2 response mismatch cost가 uncapped K=1보다 tolerance 밖에서 증가하지 않을 것,
- K=2에서는 eta cap과 tangent eta schedule이 적용되지 않고 이 사실이 log와
  provenance에 명시될 것,
- stationarity는 post-K2 residual gradient를 사용하고 response-trust는 최종
  K=2 mismatch와 correction response를 사용할 것,
- 두 auxiliary loss가 source-response normalization을 공유하고 reference
  sol/phi/psi를 loss, gradient, scheduler 또는 checkpoint 선택에 사용하지 않을 것,
- trainer, evaluator, artifact exporter와 frozen audit CLI가 동일한 production
  K=2 helper와 한 번 생성된 cached response context를 사용할 것,
- raw NPZ, CSV, figures와 summary에 K=2 directions, coefficients, activity,
  response costs와 final residual이 기록될 것,
- CouplingNet architecture, model checkpoint tensor key, GreenNet, unit-square
  CouplingNet, geometry/sample NPZ schema가 변경되지 않을 것,
- focused tests와 전체 pytest, Ruff, mypy, git diff check가 통과할 것.

수정 범위는 tangent projection config/core, complex projection dispatcher,
stationarity와 response-trust의 K=2 일반화, trainer/evaluator/artifact provenance,
기존 post-hoc audit의 공통 helper 재사용, paired experiment config, 관련 tests,
README 및 docs/memory.md로 제한한다.

Learnable K, K>2, eta network, K=2 correction cap/scheduler, full matrix,
full-Gram solve, row-norm projection, model backbone 변경과 장기 retraining은
추가하지 않는다.

각 구현 단계 후 가장 작은 config/math/projection tests를 먼저 실행하고,
통과한 뒤 trainer/evaluator/artifact tests와 전체 regression suite를 실행한다.

기존 K=1 수치, frozen K=2 audit 수치 또는 checkpoint architecture compatibility를
유지할 수 없다면 작업을 중단하고 다음을 보고한다.

1. 정확히 달라지는 수식, tensor, config 또는 floating-point 결과,
2. 영향을 받는 checkpoint, config, metric, artifact와 test,
3. subspace_dimension=1 경로를 보존하는 가장 작은 rollback 전략.
```
