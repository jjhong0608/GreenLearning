# Tangent Subspace Saturation, Adaptive Stopping, and Restart Design Note

## Status

- 이 문서는 향후 구현을 위한 기준 설계 문서다.
- 현재 production code, config, checkpoint, artifact schema는 이 문서 작성으로
  변경하지 않는다.
- exhausted mask, adaptive stopping, restart, block tangent method는 아직
  구현되지 않았다.
- 구현 전에는 frozen-checkpoint audit으로 현재 saturation 양상을 먼저
  측정해야 한다.

## 1. 목적

현재 matrix-free tangent correction은 최대 subspace dimension \(K\)를 설정하고,
매 단계에서 남은 response mismatch로부터 새로운 preconditioned source
direction을 만든다. Response-space modified Gram-Schmidt(MGS) 이후 새
direction의 response norm이 너무 작으면 해당 direction을 inactive로 처리한다.

이 문서는 다음 질문에 대한 현재 결론을 정리한다.

1. Orthogonalized response direction이 작다는 것은 무엇인가?
2. Inactive direction은 configured \(K\)와 effective \(K\)에서 어떻게
   해석해야 하는가?
3. 한 direction이 inactive가 된 뒤 같은 recurrence를 계속 적용하면 무엇이
   발생하는가?
4. Direction activity와 실제 convergence를 어떻게 구분해야 하는가?
5. 현재 수식을 보존하면서 어떤 저위험 개선을 먼저 적용해야 하는가?
6. 현재 tangent subspace가 포화되었지만 mismatch가 큰 경우 어떤 restart 또는
   block extension을 후속으로 검토할 수 있는가?

## 2. 현재 Tangent Subspace Recurrence

Symmetric-balanced directional sources를

\[
\widetilde\phi+\widetilde\psi=f
\]

라고 한다. Balance plane 안의 correction은

\[
\phi=\widetilde\phi+\delta,
\qquad
\psi=\widetilde\psi-\delta
\]

로 적용되므로 모든 단계에서 \(\phi+\psi=f\)가 정확히 유지된다.

Directional response mismatch와 tangent response operator는

\[
m_0=H_x\widetilde\phi-H_y\widetilde\psi,
\qquad
S=H_x+H_y
\]

이고,

\[
m(\delta)=m_0+S\delta
\]

이다. 목적함수는 physical response \(L^2\)-norm의 이산 근사다.

\[
J(\delta)=\frac12\|m_0+S\delta\|_{M_\Omega}^2.
\]

Gradient와 Hessian은

\[
g(\delta)=S^\top M_\Omega(m_0+S\delta),
\qquad
A=S^\top M_\Omega S
\]

이다. Production path는 global \(A\)를 조립하거나 normal equation을 풀지
않는다.

### 2.1 Nested direction construction

\(k\)번째 accumulated correction과 mismatch를

\[
\delta_k=-\sum_{j=0}^{k-1}c_jz_j,
\qquad
m_k=m_0-\sum_{j=0}^{k-1}c_jv_j
\]

라고 한다. 현재 residual gradient에서 raw candidate를 만든다.

\[
g_k=S^\top M_\Omega m_k,
\]

\[
z_{k,\mathrm{raw}}=D^{-1}g_k,
\qquad
v_{k,\mathrm{raw}}=Sz_{k,\mathrm{raw}}.
\]

여기서 \(D^{-1}\)은 full inverse-Hessian action \(A^{-1}g_k\)를 근사하기
위한 positive diagonal preconditioner다. 기본 separable \(D\)는 exact
\(\operatorname{diag}(A)\)가 아니며 cross-axis diagonal contribution을
생략한다.

Raw response candidate는 이전 response basis와 중복될 수 있다.

\[
v_{k,\mathrm{raw}}
=
\sum_{j=0}^{k-1}\beta_j^{(k)}v_j+v_{k,\perp},
\]

\[
\beta_j^{(k)}
=
\frac{
\langle v_{k,\mathrm{raw}},v_j\rangle_{M_\Omega}
}{
\langle v_j,v_j\rangle_{M_\Omega}+\varepsilon
}.
\]

Paired response/source orthogonalization은

\[
v_k
=
v_{k,\mathrm{raw}}
-
\sum_{j=0}^{k-1}\beta_j^{(k)}v_j,
\]

\[
z_k
=
z_{k,\mathrm{raw}}
-
\sum_{j=0}^{k-1}\beta_j^{(k)}z_j
\]

로 적용한다. 같은 coefficients를 사용하므로 \(Sz_k=v_k\)가 유지된다.
Production의 dynamic \(K\) extension은 response-space에서 two-pass MGS를
사용한다.

실제 correction coefficient는 orthogonalization coefficient
\(\beta_j^{(k)}\)와 다르다.

\[
c_k
=
\frac{
\langle m_k,v_k\rangle_{M_\Omega}
}{
\langle v_k,v_k\rangle_{M_\Omega}+\varepsilon_k
}.
\]

그다음

\[
\delta_{k+1}=\delta_k-c_kz_k,
\qquad
m_{k+1}=m_k-c_kv_k
\]

로 갱신한다.

## 3. 현재 Activity Test의 정확한 의미

현재 구현은 source vector \(\|z_k\|_2\) 또는 response의 개별 point 값을
threshold와 비교하지 않는다. Orthogonalized response direction의 physical
energy를 sample별로 검사한다.

\[
E_{v_k}
=
\|v_k\|_{M_\Omega}^2
=
m_{\mathrm{point}}\sum_p v_k(p)^2,
\]

\[
E_m=\|m_0\|_{M_\Omega}^2.
\]

Direction threshold는 현재 다음 형태다.

\[
\varepsilon_k
=
\varepsilon_{\mathrm{rel}}
\max(E_m,E_{v_k})
+
\varepsilon_{\mathrm{tiny}},
\]

여기서 \(\varepsilon_{\mathrm{rel}}\)은 config의
line_search_relative_eps다. Activity condition은

\[
E_{v_k}>\varepsilon_k
\]

이다.

Inactive이면 production은 해당 sample에 대해

\[
v_k\leftarrow0,
\qquad
z_k\leftarrow0,
\qquad
c_k\leftarrow0
\]

으로 처리한다. 따라서 correction과 mismatch는 이전 lower-dimensional
결과를 유지한다.

### 3.1 무엇이 너무 작은가

너무 작다는 것은 다음 중 하나일 수 있다.

1. \(v_{k,\mathrm{raw}}\)가 이전 response subspace에 거의 포함되어 있어
   orthogonalization 후 새로운 component가 거의 남지 않는다.
2. \(z_k\neq0\)이지만 \(Sz_k\approx0\)인 response near-null direction이다.
3. 큰 이전 components를 제거하는 과정에서 남은 값이 floating-point
   cancellation 수준이다.

Activity test는 이 원인들을 구별하지 않는다. 공통적으로 line-search
denominator에 사용하기에 response magnitude가 수치적으로 신뢰할 수 있는지를
판정한다.

### 3.2 Unit normalization이 해결책이 아닌 이유

Active direction을

\[
\widehat z_k=\frac{z_k}{\|v_k\|_{M_\Omega}},
\qquad
\widehat v_k=\frac{v_k}{\|v_k\|_{M_\Omega}}
\]

로 paired normalization하는 것은 가능하다. Exact line search에서는 direction
scale이 coefficient에 의해 상쇄되어 final correction이 변하지 않는다.

그러나 inactive direction을 unit-normalize하는 것은 새로운 response
information을 만들지 않는다. 거의 0인 orthogonal remainder 또는 numerical
noise를 크게 증폭할 수 있고, \(v_k=0\)이면 normalization 자체가 정의되지
않는다. 따라서 normalization은 activity validation 이후의 bookkeeping
선택일 수는 있지만 activity test를 대체할 수 없다.

## 4. Configured K와 Effective K

다음을 구분해야 한다.

\[
K_{\mathrm{configured}}
=
\text{생성을 시도할 maximum direction 수},
\]

\[
K_{\mathrm{effective}}^{(b)}
=
\sum_{k=0}^{K_{\mathrm{configured}}-1}
\mathbf 1\{\text{direction }k\text{ is active for sample }b\}.
\]

Inactive direction은 fixed-shape tensor의 bookkeeping slot에는 남지만 실제
correction subspace에는 기여하지 않는다. 해당 direction의 생성과 판정까지의
계산비용은 이미 발생한다. Batch 안에서 sample별
\(K_{\mathrm{effective}}^{(b)}\)는 다를 수 있다.

향후 artifacts와 metrics에서는 최소한 다음을 구분해 기록해야 한다.

- tangent_subspace_dimension_configured
- tangent_subspace_dimension_effective per sample
- tangent_direction_active[k]
- effective-\(K\) histogram과 quantiles

## 5. 확인된 반복 문제

현재 deterministic recurrence에서는 한 sample의 \(k\)번째 direction이
inactive가 되면

\[
c_k=0,
\qquad
m_{k+1}=m_k.
\]

따라서

\[
g_{k+1}=S^\top M_\Omega m_{k+1}=g_k,
\]

\[
z_{k+1,\mathrm{raw}}
=
D^{-1}g_{k+1}
=
z_{k,\mathrm{raw}},
\]

\[
v_{k+1,\mathrm{raw}}
=
Sz_{k+1,\mathrm{raw}}
=
v_{k,\mathrm{raw}}.
\]

Inactive direction은 이후 active orthogonalization basis로 사용되지 않으므로
다음 candidate도 같은 active basis에 대해 같은 projection을 받는다. 별도의
restart seed, 다른 preconditioner, block candidate 또는 perturbation이 없다면
후속 slot도 같은 방식으로 퇴화한다.

따라서 현재 recurrence에서 첫 inactive direction은 해당 sample의 effective
subspace expansion이 종료되었다는 신호다. Configured \(K\)까지 loop를 계속하는
것은 결과를 바꾸지 않으면서 반복 계산을 만들 수 있다.

## 6. Activity와 Convergence는 다르다

Activity test는 numerical reliability를 판단한다. 다음 질문에는 직접 답하지
않는다.

> 현재 tangent least-squares problem이 충분히 수렴했는가?

새 direction이 작다는 사실만으로 final mismatch가 작다고 결론 내릴 수 없다.
다음 세 기준을 분리해야 한다.

### 6.1 Normalized stationarity

\[
g_k=S^\top M_\Omega m_k.
\]

현재 preconditioner와 일관된 stationarity energy는

\[
E_{\mathrm{stat},k}
=
g_k^\top D^{-1}g_k.
\]

Reference-free source-response scale \(E_{\mathrm{ref}}\)로 normalize한 후보
metric은

\[
\rho_{\mathrm{stat},k}
=
\frac{
g_k^\top D^{-1}g_k
}{
E_{\mathrm{ref}}+\varepsilon
}.
\]

기존 source-normalized post-line-search stationarity diagnostic과 normalization
contract를 우선 재사용해야 하며 별도의 중복 정의를 production에 추가해서는
안 된다.

### 6.2 Marginal objective gain

새 direction에서 얻을 수 있는 ideal scalar-line-search 감소량은

\[
\Delta J_k
=
\frac12
\frac{
\langle m_k,v_k\rangle_{M_\Omega}^2
}{
\|v_k\|_{M_\Omega}^2+\varepsilon_k
}.
\]

현재 objective에 대한 상대 gain 후보는

\[
\rho_{\mathrm{gain},k}
=
\frac{\Delta J_k}{J_k+\varepsilon}.
\]

이 값은 \(K\)를 하나 더 증가시켰을 때 실질적으로 얻는 개선을 direction norm보다
직접적으로 측정한다.

### 6.3 Remaining mismatch

\[
J_k=\frac12\|m_k\|_{M_\Omega}^2
\]

또는

\[
\rho_{\mathrm{mismatch},k}
=
\frac{
\|m_k\|_{M_\Omega}^2
}{
\|m_0\|_{M_\Omega}^2+\varepsilon
}
\]

를 별도로 보고해야 한다.

가능한 상태는 다음과 같다.

| Stationarity | Mismatch | 해석 |
|---|---|---|
| 작음 | 작음 | 현재 tangent space에서 원하는 convergence |
| 작음 | 큼 | 현재 tangent response range에서 줄일 수 없는 residual 가능성 |
| 큼 | 큼 | 유효한 correction 가능성이 남음 |
| 큼 | direction inactive | numerical 또는 preconditioner breakdown 가능성 |

\(g_k\approx0\)이지만 mismatch가 크다면 exact solution에 가깝다는 뜻이 아니다.
현재 balance-preserving tangent correction space에서 stationary하다는 뜻이다.

## 7. 우선 구현할 저위험 개선

다음 순서를 권장한다.

### Phase 1: Frozen-checkpoint audit

Production behavior를 바꾸기 전에 frozen checkpoints에서 다음 sample-wise
분포를 측정한다.

- configured \(K\)와 effective \(K\)
- 최초 inactive direction index
- \(\rho_{\mathrm{stat},k}\)
- \(\rho_{\mathrm{gain},k}\)
- \(\rho_{\mathrm{mismatch},k}\)
- inactive 이후 후속 raw/orthogonalized direction의 반복 여부
- effective \(K\)별 rel_sol, rel_flux, canonical energy의 detached evaluation

Reference sol/phi/psi는 audit report에서만 detached evaluation metric으로
사용하고 stopping, correction, loss 또는 checkpoint selection에는 사용하지
않는다.

### Phase 2: Sample-exhausted mask

한 sample에서 first inactive direction이 발생하면

\[
\mathrm{exhausted}^{(b)}=\mathrm{true}
\]

로 표시한다. 이후 slots는

\[
z_\ell^{(b)}=v_\ell^{(b)}=c_\ell^{(b)}=0
\qquad(\ell>k)
\]

으로 유지한다.

- 모든 sample이 exhausted이면 loop를 조기 종료한다.
- 일부 sample만 exhausted이면 active sample은 계속 진행한다.
- Fixed artifact tensor shape이 필요하면 남은 slots를 0으로 pad한다.
- 이 phase는 현재 numerical result를 tolerance 내에서 보존해야 한다.

Vectorized response operator가 batch 전체를 처리하므로 sample mask만으로
wall-clock이 완전히 감소하지 않을 수 있다. Active sample compaction은 별도의
성능 최적화로 다룬다.

### Phase 3: Adaptive stopping

\(K_{\max}\)는 computational budget으로 유지하고 sample별 effective \(K\)를
다음 조건으로 결정한다.

\[
\rho_{\mathrm{stat},k}\le\tau_{\mathrm{stat}}
\]

그리고/또는

\[
\rho_{\mathrm{gain},k}\le\tau_{\mathrm{gain}}.
\]

최종 논리 연산, threshold, consecutive-step requirement는 Phase 1 audit 후
고정한다. Activity threshold를 convergence threshold로 재사용하지 않는다.

## 8. 선택적 후속 확장

Phase 1에서 stationarity는 작지만 normalized mismatch가 큰 sample이 의미 있게
존재할 때만 다음 확장을 고려한다.

### 8.1 Deterministic restart

현재 recurrence와 다른 source seed \(r\)를 선택하고

\[
v_{\mathrm{restart,raw}}=Sr
\]

를 기존 active response basis에 대해 paired orthogonalize한다. Candidate
selection은 다음 response gain criterion을 사용할 수 있다.

\[
r^\star
=
\arg\max_r
\frac{
|\langle m_k,\widehat v_r\rangle_{M_\Omega}|^2
}{
\|\widehat v_r\|_{M_\Omega}^2+\varepsilon
}.
\]

가능한 candidate pool은 deterministic canonical probes, geometry-stratified
probes 또는 작은 fixed probe bank다. 모든 point basis를 전수 조사하는 방식은
비용 때문에 production 기본안으로 두지 않는다.

### 8.2 Block tangent candidates

여러 source candidates를 동시에 만들고

\[
V_{\mathrm{raw}}=SZ_{\mathrm{raw}}
\]

를 기존 basis와 block 내부에서 orthogonalize한다. Full global matrix는
필요하지 않지만 추가 response actions, block MGS, 작은 block coefficient
solve가 필요하다.

### 8.3 Preconditioner audit

Early saturation이 preconditioner와 강하게 연관될 때만 separable,
exact_diagonal, absolute_cross_axis, normalized_quadratic_cross_axis variants를
effective-\(K\), stationarity, marginal gain, mismatch, runtime 기준으로
재비교한다.

Restart, block method, preconditioner 변경은 numerical result를 바꾸는
algorithmic extension이다. Exhausted masking과 별도 change set으로 구현하고
paired retraining 또는 frozen post-hoc evidence를 요구한다.

## 9. 향후 영향 파일

실제 구현 시 우선 확인할 파일은 다음과 같다.

- src/greenonet/complex_tangent_projection.py
  - activity mask, exhausted state, stopping metrics, padding
- src/greenonet/config.py
  - adaptive stopping을 opt-in으로 만들 경우 strict nested config
- complex trainer/evaluator/artifact exporter
  - configured/effective \(K\), stopping reason, sample-wise metrics
- tangent audit CLIs
  - frozen-checkpoint saturation audit
- test/test_complex_tangent_projection.py
  - recurrence, inactive fallback, mask, stopping, gradient tests
- complex trainer/artifact tests
  - logging, CSV/NPZ/summary provenance
- README.md와 docs/memory.md
  - production convention과 실험 결과

## 10. 향후 테스트 요구사항

### Mathematical behavior

- 첫 inactive direction 이후 현재 recurrence의 raw candidate가 반복되는지 확인
- exhausted mask가 inactive 이후 모든 coefficients/directions를 0으로 유지하는지
  확인
- mask 적용 전후 final \(\delta\), mismatch, cost가 tolerance 내에서 동일한지 확인
- 모든 active 단계에서 \(\phi+\psi=f\)가 유지되는지 확인
- paired update가 \(Sz_k=v_k\)를 유지하는지 확인
- active response directions의 \(M_\Omega\)-orthogonality 확인

### Convergence diagnostics

- stationarity, marginal gain, mismatch가 서로 독립 metric인지 확인
- source amplitude scaling에 대한 normalization behavior 확인
- mismatch가 작지 않지만 stationarity가 작은 synthetic range-orthogonal fixture
- gradient가 큰 numerical breakdown fixture
- sample별 effective \(K\)와 stopping reason 확인

### Batch and shape stability

- batch 내 sample별 서로 다른 effective \(K\)
- 모든 sample exhausted 시 early break
- 일부 sample만 exhausted 시 active sample 결과 보존
- fixed-size artifact padding과 active mask round-trip
- torch.compile 및 autograd smoke

### Regression

- adaptive feature가 disabled이면 기존 configured-\(K\) 결과 보존
- K1 legacy path 보존
- K2 compatibility contract 보존
- model architecture와 checkpoint tensor keys 불변
- reference targets가 stopping, loss, gradient, scheduler 또는 checkpoint selection에
  들어가지 않음

## 11. Rollback 경계

- Exhausted mask는 opt-in 또는 결과보존이 증명된 internal optimization으로
  독립 적용한다.
- Adaptive stopping config를 비활성화하면 configured \(K\)를 끝까지 계산하는
  기존 behavior로 돌아가야 한다.
- Restart/block extension은 별도 mode로 추가하고 기본 tangent recurrence를
  덮어쓰지 않는다.
- Model backbone, GreenNet, projection balance algebra, geometry/sample NPZ,
  model-only checkpoint key를 변경하지 않는다.

## 12. 구현 전 결정해야 할 사항

다음 항목은 아직 확정하지 않았다.

1. \(\rho_{\mathrm{stat}}\)에서 재사용할 canonical source-response normalization
2. \(\rho_{\mathrm{gain}}\) threshold와 absolute floor
3. stopping에 stationarity와 gain을 and 또는 or로 결합할지
4. 한 step 판정인지 consecutive-step 판정인지
5. sample-exhausted mask만 사용할지 active sample compaction까지 할지
6. restart가 필요한 evidence threshold
7. restart candidate pool과 deterministic pivot rule
8. block size와 작은 block solve 허용 여부

이 값들은 추측으로 production config에 추가하지 않는다. Phase 1 audit 결과를
기준으로 고정한다.

## 13. 현재 결론

1. Orthogonalized response direction의 작은 norm은 새로운 response information이
   부족하거나 수치적으로 신뢰할 수 없다는 뜻이다.
2. Inactive direction은 configured \(K\) slot에는 남지만 effective subspace
   dimension에는 포함되지 않는다.
3. 현재 deterministic recurrence에서는 첫 inactive direction 이후 같은
   candidate가 반복되므로 sample별 expansion은 사실상 종료된다.
4. Unit normalization은 scale만 바꾸며 새로운 direction을 만들지 않으므로
   inactive direction의 해결책이 아니다.
5. Activity는 numerical reliability이고 convergence는 stationarity, marginal
   gain, remaining mismatch로 별도 판정해야 한다.
6. 첫 구현 우선순위는 frozen saturation audit, exhausted mask, adaptive
   stopping이다.
7. Restart 또는 block tangent method는 stationary-but-large-mismatch sample이
   실제로 확인될 때만 후속 실험으로 도입한다.

