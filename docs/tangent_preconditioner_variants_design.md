# Tangent preconditioner 4종 설계 및 실험 기준

## 0. 문서 지위와 조사 기준

이 문서는 `GreenNetResearch/ComplexGeometry`의 symmetric-tangent Green-response
projection에서 사용할 diagonal preconditioner 4종의 **구현, 검증, 실험, 산출물
계약을 한곳에 고정하는 단일 기준 문서**다. 후속 코드 수정과 실험 설정은 이
문서를 우선 기준으로 삼고, 구현 중 합의가 바뀌면 코드보다 먼저 이 문서를
갱신한다.

- 조사 저장소: `/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry`
- 조사 호스트: `nimsgpu`
- 조사 일자: 2026-08-26 (Asia/Seoul)
- 조사 기준 commit: `97be27fc07bacaccf24f2064bd381b1926c7a6bd`
- 조사 기준: commit 자체가 아니라 **해당 commit 위의 현재 dirty working tree**
- production default preconditioner: `D_sep`
- 문서 상태: 네 variant, 공용 schema-v2 context, frozen 4 x K audit 구현 및 검증 완료

조사 당시 working tree에는 `README.md`, `docs/memory.md`, tangent projection과
artifact 관련 source/test를 포함한 기존 사용자 변경이 있었다. 이 문서를 만드는
작업은 그 변경을 되돌리거나 정리하지 않으며, 후속 구현도 요청 범위 밖의 변경을
보존해야 한다.

## 1. 결정 요약

구현할 후보는 다음 네 개뿐이다. 아래에서 `damping`은 네 후보에 공통으로 더하는
동일한 scale-aware positive shift다.

\[
\boxed{D_{\mathrm{sep},i}=a_i+b_i+\mathrm{damping}}
\]

\[
\boxed{D_{\mathrm{exact},i}=a_i+b_i+2c_i+\mathrm{damping}}
\]

\[
\boxed{D_{\mathrm{abs},i}=a_i+b_i+2|c_i|+\mathrm{damping}}
\]

\[
\boxed{
D_{q,i}=a_i+b_i+\frac{4c_i^2}{a_i+b_i+\varepsilon_q}
+\mathrm{damping}}
\]

`D_sep`는 현재 baseline을 그대로 보존한다. `D_exact`는 tangent mismatch의
normal matrix `A=S^T M_Omega S`의 exact diagonal이며, 동등한 `J/2`
least-squares convention에서는 Hessian diagonal이다. `D_abs`와 `D_q`는 부호에
불변인 cross-axis coupling surrogate이며 Hessian diagonal이 아니다.

다음 후보는 명시적으로 범위 밖이다.

- intermediate `alpha`, `beta`, `gamma` 같은 보간 또는 강도 조절 parameter
- `max(c_i, 0)` 또는 normalized correlation의 ReLU/positive-part variant
- sign-dependent piecewise surrogate
- full Gram matrix 조립 또는 global linear solve
- preconditioner variant별 damping 재튜닝을 포함한 1차 비교

Config에서 위 제외 항목을 암시하는 key나 mode를 조용히 받아들이지 않는다. 알 수
없는 값은 명시적인 validation error로 거부한다.

## 2. 구현된 실제 구조

### 2.1 현재 계산 경로

현재 production 경로는 다음과 같다.

1. `ComplexCouplingNet`의 raw response를 physical source proposal `(p,q)`로
   변환한다.
2. exact-balanced symmetric pair를 만든다.

   \[
   \widetilde p=\frac{f+p-q}{2},\qquad
   \widetilde q=\frac{f-p+q}{2}.
   \]

3. frozen axial Green-response operator로

   \[
   m_0=H_x\widetilde p-H_y\widetilde q
   \]

   를 계산한다.
4. tangent gradient

   \[
   g_0=(H_x+H_y)^\top M_\Omega m_0
   \]

   를 계산한다.
5. 현재 denominator

   \[
   D_i=\gamma_{x,i}^2+\gamma_{y,i}^2
   +(\lambda_{\mathrm{rel}}+\epsilon_{D,\mathrm{rel}})
   \operatorname{mean}_j(\gamma_{x,j}^2+\gamma_{y,j}^2)
   \]

   로 preconditioned direction을 만든다.
6. K=1이면 fixed step 또는 capped exact line search를 적용하고, K=2..4이면
   response-space modified Gram-Schmidt로 nested directions를 추가한다.
7. 모든 경우

   \[
   \phi=\widetilde p+\delta,\qquad
   \psi=\widetilde q-\delta
   \]

   로 갱신하므로 `phi + psi = rhs`를 보존한다.

### 2.2 현재 코드와 symbol 매핑

| 역할 | 현재 파일과 symbol | 현재 계약 | 후속 변경 |
|---|---|---|---|
| axial local response block | `src/greenonet/complex_axial_response_operator.py::AxialResponseBlock` | connected segment별 square response matrix와 valid index 저장 | 그대로 재사용 |
| axis response action | `FrozenAxialResponseOperator.forward`, `adjoint` | global matrix 없이 local block matvec | 그대로 재사용 |
| self column gain | `FrozenAxialResponseOperator.column_gain_squared` | `diag(H_axis^T M H_axis)` 계산 | `a_i`, `b_i`의 권위 있는 계산 경로로 유지 |
| bidirectional gradient | `FrozenBidirectionalResponseOperator.tangent_gradient` | `(H_x+H_y)^T M mismatch` | 변경 없음 |
| tangent context | `src/greenonet/complex_tangent_projection.py::SymmetricTangentGreenResponseContext` | frozen operator, `a,b,c,rho,q`, 네 base/denominator, K/eta 설정과 safeguards 저장 | 구현 완료 |
| denominator 생성 | `src/greenonet/complex_tangent_preconditioner.py` | pure builder가 네 후보와 공통 damping을 한 번에 계산 | 구현 완료 |
| K=1 step | `SymmetricTangentGreenResponseContext.tangent_step` | fixed 또는 capped exact line search | denominator의 출처만 달라지고 산술 순서 유지 |
| K=2 seed | `matrix_free_krylov_k2_step` | 기존 K=2 산술 고정 | variant가 선택한 `D^{-1}`만 사용 |
| K=3/4 extension | `matrix_free_krylov_subspace_step` | two-pass response MGS | algorithm 변경 없음 |
| projection integration | `src/greenonet/complex_projection.py::apply_complex_balance_projection` | symmetric pair, tangent step, reconstruction 재사용 | selected variant diagnostics 노출 |
| config dataclass | `src/greenonet/config.py::SymmetricTangentGreenResponseProjectionConfig` | K/eta fields와 strict `preconditioner_variant`, `cross_axis_relative_eps` | 구현 완료 |
| trainer/evaluator cache | `SymmetricTangentGreenResponseContextCache` | runtime build 또는 schema-v2 strict sidecar load를 한 번 수행 | 구현 완료 |
| production artifact | `src/greenonet/complex_coupling_artifacts.py` | selected arrays, 4종 context NPZ, summary/log와 five spatial maps | 구현 완료 |
| frozen K audit | `src/greenonet/complex_tangent_preconditioner_audit.py` | 같은 raw output/operator에서 네 immutable context와 K=1..4 비교 | 구현 완료 |
| audit CLI | `cli/audit_tangent_preconditioners.py` | 4 x K frozen audit와 optional `--tangent-context` | 구현 완료 |
| core unit tests | `test/test_complex_tangent_projection.py` | K1 bitwise, K2 seed, K3/4 nesting, finite/degenerate cases | dense diagonal reference와 4종 formula test 추가 |
| config tests | `test/test_io_config.py`, `test/test_cli_train.py` | strict config parse/round-trip | 새 fields, default, invalid mode tests 추가 |
| trainer/evaluator tests | `test/test_complex_coupling_trainer.py` | cache reuse, K별 metrics/log | variant별 cache/provenance와 runtime metrics 추가 |
| artifact tests | `test/test_complex_coupling_artifacts.py` | NPZ/summary/figure field contract | schema와 backward aliases 고정 |

현재 `preconditioner_base`는 정확히 `gamma_x_squared + gamma_y_squared`다. 따라서
현재 artifact의 `gamma_x_squared`, `gamma_y_squared`, `preconditioner_base`,
`denominator`는 각각 이 문서의 `a`, `b`, `D_sep`의 undamped base, 최종 damped
denominator에 대응한다.

## 3. 수학적 설정

### 3.1 weighted response space

현재 geometry에서는 valid-point mass가 scalar

\[
M_\Omega=(h_xh_y)I
\]

로 구현되어 있다. 일반적인 표기를 위해

\[
\langle u,v\rangle_{M_\Omega}=u^\top M_\Omega v,
\qquad
\|u\|_{M_\Omega}^2=\langle u,u\rangle_{M_\Omega}
\]

를 사용한다.

`P`개의 physical source coordinate에 대해 `e_i`를 i번째 standard basis vector라
하고 다음 response column을 정의한다.

\[
x_i=H_xe_i,\qquad y_i=H_ye_i.
\]

### 3.2 `a_i`, `b_i`, `c_i`

\[
\boxed{a_i=\|x_i\|_{M_\Omega}^2}
\]

\[
\boxed{b_i=\|y_i\|_{M_\Omega}^2}
\]

\[
\boxed{c_i=\langle x_i,y_i\rangle_{M_\Omega}}
\]

`a_i`와 `b_i`는 각 source coordinate가 x/y Green reconstruction 전체에 만드는
response energy다. `c_i`는 같은 source coordinate의 x-response column과
y-response column 사이 weighted overlap이다. 이는 sample 집단에서 계산한 Pearson
correlation이 아니다.

편의를 위해

\[
s_i=a_i+b_i
\]

로 둔다.

### 3.3 `rho_i`

수학적 normalized correlation은

\[
\rho_i=
\begin{cases}
\dfrac{c_i}{\sqrt{a_ib_i}},&a_ib_i>0,\\[6pt]
0,&a_ib_i=0
\end{cases}
\]

로 정의한다. Cauchy--Schwarz에 의해 `|rho_i| <= 1`이다. 구현과 spatial map에서는
0 근처 division을 피하기 위해

\[
\rho_i^{\mathrm{safe}}
=\frac{c_i}{\max(\sqrt{a_ib_i},\varepsilon_\rho)},
\qquad
\varepsilon_\rho=epsilon_{q,\mathrm{rel}}\,\bar s
\]

를 저장한다. 여기서

\[
\bar s=\operatorname{mean}_j(s_j)>0.
\]

`rho_safe`는 diagnostic 전용이다. 네 preconditioner 중 어느 것도 `rho_safe`를 다시
곱해 `c`를 복원하지 않으며, denominator는 원래의 `a,b,c`로 직접 계산한다.

### 3.4 `q_i`

\[
\boxed{
q_i=\frac{c_i^2}{a_i+b_i+\varepsilon_q}},
\qquad
\varepsilon_q=epsilon_{q,\mathrm{rel}}\,\bar s>0.
\]

`q_i`는 `a_i`, `b_i`, `c_i`와 같은 response-energy 단위를 가진다. `D_q`에 들어가는
correction은 `4q_i`다. `epsilon_q`는 q normalization에만 사용하며 최종 denominator
positive shift와 구분한다.

`epsilon_q=0`인 이상화된 경우에는

\[
q_i=\rho_i^2\frac{a_ib_i}{a_i+b_i}
=\frac12\rho_i^2\operatorname{HM}(a_i,b_i),
\]

즉 squared alignment와 두 directional gain의 harmonic mean을 결합한 scale이다.

### 3.5 tangent Hessian과 exact diagonal 유도

Balance plane에서

\[
\phi=\widetilde p+\delta,\qquad
\psi=\widetilde q-\delta
\]

이므로 reconstruction mismatch는

\[
m(\delta)
=H_x(\widetilde p+\delta)-H_y(\widetilde q-\delta)
=m_0+S\delta,
\qquad S=H_x+H_y.
\]

현재 코드의 response objective convention과 맞추어 `1/2` 없는 비용을 사용한다.

\[
J(\delta)=\|m_0+S\delta\|_{M_\Omega}^2.
\]

`J/2`의 gradient와 Hessian, 또는 동등하게 `J`에서 공통 factor 2를 제외한
normal-equation pair는

\[
g(\delta)=S^\top M_\Omega(m_0+S\delta),
\qquad
A=S^\top M_\Omega S.
\]

따라서

\[
\begin{aligned}
A_{ii}
&=\langle Se_i,Se_i\rangle_{M_\Omega}\\
&=\langle x_i+y_i,x_i+y_i\rangle_{M_\Omega}\\
&=a_i+b_i+2c_i.
\end{aligned}
\]

이다. 따라서 현재 `J` 자체의 gradient와 Hessian은 각각 `2g`, `2A`지만 이 공통
factor는 search direction과 exact scalar line minimizer를 바꾸지 않는다. 합의된
`D_exact` 명칭은 `diag(A)`를 뜻하며, 그 undamped base가 위 식이다. Full `A`의
off-diagonal `A_ij`는 포함하지 않는다.

## 4. 네 preconditioner의 정의와 해석

### 4.1 공통 damping

네 후보 모두 같은 baseline gain scale을 사용한다.

\[
\bar s=\operatorname{mean}_i(a_i+b_i).
\]

현재 코드와 동일하게

\[
d_\lambda=\lambda_{\mathrm{rel}}\bar s,
\qquad
d_{\epsilon}=\epsilon_{D,\mathrm{rel}}\bar s,
\]

\[
\boxed{\mathrm{damping}=d_\lambda+d_\epsilon}
\]

로 정의한다. 첫 controlled ablation에서는 variant별로 `bar_s`,
`relative_lambda`, `denominator_relative_eps`를 바꾸지 않는다.

권장 초기값은 현재 production default를 그대로 쓴다.

| 설정 | 기본값 | 역할 |
|---|---:|---|
| `relative_lambda` | `1e-2` | 의도적인 scale-aware damping |
| `denominator_relative_eps` | `1e-12` | 모든 coordinate에서 strictly positive denominator 보장 |
| `cross_axis_relative_eps` | `1e-12` | `q`와 diagnostic `rho`의 scale-aware zero guard |
| dtype | `torch.float64` | 현재 complex path와 K-orthogonality contract 유지 |

`denominator_relative_eps`와 `cross_axis_relative_eps`는 이름과 artifact field를
분리한다. 하나를 다른 하나의 대체재로 해석하지 않는다.

### 4.2 `D_sep`: current baseline

\[
\boxed{D_{\mathrm{sep},i}=s_i+\mathrm{damping}}
\]

- x/y directional response gain을 분리하여 합한다.
- `c_i`의 constructive coupling과 cancellation을 모두 무시한다.
- positive, separable, line-local surrogate다.
- `diag(A)`가 아니며 일반적으로 `A_ii`의 upper/lower bound도 아니다.
- 새 config field를 생략했을 때 이 경로가 기존 산술과 비트 단위로 같아야 한다.

### 4.3 `D_exact`: exact tangent Hessian diagonal

\[
\boxed{D_{\mathrm{exact},i}=s_i+2c_i+\mathrm{damping}}
\]

- `diag(S^T M S)`를 정확히 사용한다.
- `c_i>0`이면 denominator를 늘리고 `c_i<0`이면 cancellation을 반영해 줄인다.
- anti-correlated response가 거의 상쇄되면 undamped base가 0에 가까워질 수 있다.
- 이 작은 값은 `H_x e_i`, `H_y e_i` 각각이 작다는 뜻이 아니라 `S e_i`가
  cancellation된다는 뜻이다.
- damping이 source update 폭주를 막는 핵심 guard다.

### 4.4 `D_abs`: absolute cross-axis coupling

\[
\boxed{D_{\mathrm{abs},i}=s_i+2|c_i|+\mathrm{damping}}
\]

- aligned와 anti-aligned response를 모두 강한 coupling으로 본다.
- `|rho|`에 선형적으로 반응한다.
- negative `c_i`에서도 denominator를 키우므로 curvature approximation이라기보다
  conservative coupling-strength scaling이다.
- `D_exact`의 absolute value가 아니고, `|diag(A)|`도 아니다.

### 4.5 `D_q`: normalized quadratic cross-axis coupling

\[
\boxed{D_{q,i}=s_i+4q_i+\mathrm{damping}}
\]

\[
4q_i=\frac{4c_i^2}{s_i+\varepsilon_q}.
\]

- `c`의 부호에 불변이다.
- alignment에 대해 quadratic이므로 weak cross-axis overlap을 `D_abs`보다 더
  강하게 억제한다.
- `ab/(a+b)` scale 때문에 한 축만 강한 coordinate를 strong 2D coupling으로
  과대평가하지 않는다.
- factor 4는 별도 tuning parameter가 아니라 `epsilon_q=0`인 이론적 경계에서
  `D_abs`와 동일한 최대 correction scale을 갖도록 정의에 포함한 normalization이다.
- 이 factor는 고정하며 ablation axis로 만들지 않는다.

## 5. positivity, bounds, ordering

### 5.1 기본 부등식

Cauchy--Schwarz와 AM--GM에 의해

\[
|c_i|\le\sqrt{a_ib_i}\le\frac{a_i+b_i}{2}=\frac{s_i}{2}.
\]

따라서 `c_i^2 <= a_i b_i`다.

### 5.2 undamped base bounds

#### `D_sep`

\[
D_{\mathrm{sep},i}^{(0)}=s_i\ge0.
\]

#### `D_exact`

\[
\begin{aligned}
D_{\mathrm{exact},i}^{(0)}
&=s_i+2c_i=\|x_i+y_i\|_{M_\Omega}^2\\
&\ge(\sqrt{a_i}-\sqrt{b_i})^2\ge0,
\end{aligned}
\]

\[
D_{\mathrm{exact},i}^{(0)}
\le(\sqrt{a_i}+\sqrt{b_i})^2\le2s_i.
\]

#### `D_abs`

\[
s_i\le D_{\mathrm{abs},i}^{(0)}
=s_i+2|c_i|
\le(\sqrt{a_i}+\sqrt{b_i})^2\le2s_i.
\]

#### `D_q`

`epsilon_q >= 0`이므로

\[
0\le4q_i
\le\frac{4a_ib_i}{s_i+\varepsilon_q}
\le\frac{s_i^2}{s_i+\varepsilon_q}
\le s_i.
\]

따라서

\[
s_i\le D_{q,i}^{(0)}\le2s_i.
\]

### 5.3 variant 간 ordering

`c_i >= 0`이면

\[
D_{\mathrm{sep},i}^{(0)}
\le D_{q,i}^{(0)}
\le D_{\mathrm{exact},i}^{(0)}
\le D_{\mathrm{abs},i}^{(0)}.
\]

`c_i < 0`이면

\[
D_{\mathrm{exact},i}^{(0)}
\le D_{\mathrm{sep},i}^{(0)}
\le D_{q,i}^{(0)}
\le D_{\mathrm{abs},i}^{(0)}.
\]

특히 `D_q <= D_abs`는 항상 성립한다. 따라서 `D_abs`는 sign-invariant 후보 중
가장 conservative하고, `D_q`는 weak/imbalanced coupling을 더 선별적으로 반영한다.

### 5.4 damping 이후 strict positivity

현재 context는 `bar_s > 0`, `relative_lambda >= 0`,
`denominator_relative_eps > 0`을 요구한다. 그러므로 네 후보 모두

\[
D_{\star,i}\ge
\epsilon_{D,\mathrm{rel}}\bar s>0.
\]

`D_exact`가 cancellation으로 0이 되어도 최종 denominator는 strictly positive다.

### 5.5 balanced-gain 예시

`a_i=b_i=t`, `c_i=rho_i t`, `epsilon_q=0`이면

\[
D_{\mathrm{sep}}^{(0)}=2t,
\]

\[
D_{\mathrm{exact}}^{(0)}=2t(1+\rho),
\]

\[
D_{\mathrm{abs}}^{(0)}=2t(1+|\rho|),
\]

\[
D_q^{(0)}=2t(1+\rho^2).
\]

따라서 `D_abs`는 `|rho|`에 선형, `D_q`는 `rho^2`에 quadratic으로 반응한다.

## 6. 구현 설계

### 6.1 새 config API

`SymmetricTangentGreenResponseProjectionConfig`에 다음 두 flat field를 추가한다.
기존 dataclass의 구조를 유지하고 불필요한 nested object는 만들지 않는다.

```python
TangentPreconditionerVariant = Literal[
    "separable",
    "exact_diagonal",
    "absolute_cross_axis",
    "normalized_quadratic_cross_axis",
]

preconditioner_variant: TangentPreconditionerVariant = "separable"
cross_axis_relative_eps: float = 1.0e-12
```

JSON 예시는 다음과 같다.

```json
{
  "balance_projection": {
    "enabled": true,
    "mode": "symmetric_tangent_green_response",
    "symmetric_tangent_green_response": {
      "subspace_dimension": 4,
      "eta": 0.01,
      "eta_strategy": "closed_loop_exact_line_search",
      "line_search_relative_eps": 1e-12,
      "relative_lambda": 0.01,
      "denominator_relative_eps": 1e-12,
      "preconditioner_variant": "normalized_quadratic_cross_axis",
      "cross_axis_relative_eps": 1e-12
    }
  }
}
```

Validation contract:

- `preconditioner_variant`는 위 네 문자열만 허용한다.
- `cross_axis_relative_eps`는 boolean이 아닌 finite positive number여야 한다.
- field가 없으면 `separable`과 `1e-12`를 materialize한다.
- K, eta, line-search validation은 기존 규칙을 그대로 유지한다.
- `alpha`, `beta`, `relu`, `positive_part`, `cross_axis_weight` 같은 key는 unknown
  key error로 거부한다.
- `cross_axis_relative_eps`는 모든 variant에서 provenance를 위해 resolve하지만
  `D_sep`, `D_exact`, `D_abs`의 denominator 식에는 들어가지 않는다.

### 6.2 raw column-Gram term API

Response operator 계층은 config mode를 알지 않고 raw `a,b,c`만 제공한다.

```python
@dataclass(frozen=True)
class TangentColumnGramTerms:
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor


class FrozenAxialResponseOperator:
    def diagonal_response(self) -> torch.Tensor:
        """Scatter each local response block diagonal to global source order."""


class FrozenBidirectionalResponseOperator:
    def tangent_column_gram_terms(
        self,
        *,
        point_mass: torch.Tensor | float,
    ) -> TangentColumnGramTerms:
        """Return a, b, c without assembling a global response or Gram matrix."""
```

`a`와 `b`는 기존 `column_gain_squared`를 호출한다. `c`는

\[
c_i=(h_xh_y)\sum_{r\in I_x(i)\cap I_y(i)}
(H_x)_{ri}(H_y)_{ri}
\]

로 계산한다. 여기서 `I_x(i)`와 `I_y(i)`는 source `i`가 속한 connected x/y
segment의 valid indices다.

현재 orthogonal axial geometry에서는 두 support의 교집합이 source point 하나인
것이 정상 contract이므로 fast path는

\[
c_i=(h_xh_y)(H_x)_{ii}(H_y)_{ii}
\]

다. 구현은 이 singleton-intersection invariant를 실제 block indices로 검증해야 한다.
Invariant가 깨질 때 diagonal product를 조용히 사용해서는 안 된다. 일반 intersection
accumulation으로 정확히 fallback하거나 명시적인 unsupported-geometry error를 낸다.
권장 구현은 일반 accumulation을 authoritative path로 두고, 검증된 singleton case만
vectorized diagonal fast path로 처리하는 것이다.

이 계산은 다음을 금지한다.

- global `P x P` `H_x`, `H_y` 조립
- global `P x P` `A=S^TMS` 조립
- basis vector `P`개에 대한 global dense batch application
- source sample마다 `a,b,c` 재계산

### 6.3 preconditioner term builder

수식 선택과 numerical audit를 pure function으로 분리한다.

```python
@dataclass(frozen=True)
class TangentPreconditionerTerms:
    variant: TangentPreconditionerVariant
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    rho: torch.Tensor
    q: torch.Tensor
    separable_base: torch.Tensor
    exact_base: torch.Tensor
    absolute_base: torch.Tensor
    quadratic_base: torch.Tensor
    selected_base: torch.Tensor
    gain_scale: torch.Tensor
    q_epsilon: torch.Tensor
    damping: torch.Tensor
    denominator: torch.Tensor
    cauchy_violation_max: torch.Tensor
    exact_roundoff_clamp_count: int


def build_tangent_preconditioner_terms(
    *,
    gram: TangentColumnGramTerms,
    variant: TangentPreconditionerVariant,
    relative_lambda: float,
    denominator_relative_eps: float,
    cross_axis_relative_eps: float,
) -> TangentPreconditionerTerms:
    ...
```

함수는 네 base를 모두 한 번 계산하고 `selected_base`만 variant에 따라 고른다. 이
구조는 spatial map과 frozen 4-way audit가 formula를 중복 구현하지 않게 한다.

### 6.4 context field와 backward alias

`SymmetricTangentGreenResponseContext`는 다음을 저장한다.

- 기존 field 유지: `response_operator`, `gamma_x_squared`,
  `gamma_y_squared`, `preconditioner_base`, `gain_scale`, `denominator`,
  `point_mass`, K/eta 관련 fields
- 새 field: `preconditioner_variant`, `cross_axis_inner_product`,
  `normalized_correlation`, `normalized_quadratic_cross_axis`,
  `separable_preconditioner_base`, `exact_preconditioner_base`,
  `absolute_preconditioner_base`, `quadratic_preconditioner_base`,
  `q_epsilon`, `damping`, numerical-audit counts

호환 의미는 다음처럼 고정한다.

- `gamma_x_squared`는 계속 `a`다.
- `gamma_y_squared`는 계속 `b`다.
- `preconditioner_base`는 **선택된 undamped base**다.
- default `separable`에서는 기존과 정확히 `a+b`다.
- `denominator=preconditioner_base+damping`이다.
- 새 `separable_preconditioner_base`가 variant와 무관한 `a+b`를 항상 보존한다.

### 6.5 K=1..4 algorithm과의 연결

Preconditioner 변경은 `D^{-1}`를 사용하는 direction 생성에만 영향을 준다.

- K=1 fixed: `delta=-eta*D^-1*g`.
- K=1 closed-loop: `z=D^-1*g`, `c0=(g^Tz)/(||Sz||_M^2+eps)`를 사용하고
  production cap을 적용한다.
- K=2: 기존 `z0`, residual gradient, `z1_raw`, response orthogonalization,
  scalar `c1` 산술을 유지한다.
- K=3/4: 기존 two-pass response MGS와 degeneracy fallback을 유지한다.
- K>=2에서 `eta`와 eta schedule이 not-applicable인 현재 contract를 유지한다.
- `phi=ptilde+delta`, `psi=qtilde-delta`의 balance contract를 변경하지 않는다.
- response blocks는 frozen/detached지만 batch matvec와 scalar coefficients는
  detach하지 않아 CouplingNet까지 first-order autograd가 이어진다.

새 variant 구현을 이유로 line-search, MGS, reconstruction, training objective를
동시에 바꾸지 않는다.

### 6.6 cache, reuse, invalidation

현재 trainer/evaluator별 `SymmetricTangentGreenResponseContextCache`의 lazy build
1회 contract를 유지한다.

논리적 cache fingerprint에는 최소한 다음 값이 포함되어야 한다.

- geometry identity: point count, x/y segment partition, `hx`, `hy`
- frozen GreenNet checkpoint identity 또는 resolved model fingerprint
- dtype와 device
- `preconditioner_variant`
- `relative_lambda`
- `denominator_relative_eps`
- `cross_axis_relative_eps`
- `subspace_dimension`, eta/line-search fields

Production runtime은 cache instance 자체가 위 config와 model에 종속되므로 한 context만
허용한다. `get_or_build` 이후 다른 fingerprint를 같은 cache에 넣는 API는 제공하지
않는다. Frozen 4-way audit는 response operator와 raw Gram terms를 한 번 만들고 네 개의
immutable preconditioner context를 만드는 별도 경로를 사용한다.

필수 provenance:

- `context_build_count == 1`
- `context_build_seconds`
- `response_operator_build_count == 1`
- 4-way audit의 `preconditioner_context_count == 4`
- x/y segment block 수와 local matrix entry 수
- `global_matrix_materialized=false`
- `full_gram_solve=false`
- `row_norm_used=false`

## 7. numerical safeguards

### 7.1 공통 입력 검사

- `point_mass`는 finite positive scalar여야 한다.
- `a`, `b`는 finite이고 음수가 아니어야 한다.
- `c`는 finite여야 한다.
- `bar_s=mean(a+b)`는 finite positive여야 한다.
- 모든 tensor는 response operator와 dtype/device가 같아야 한다.
- final `denominator`는 모든 coordinate에서 finite positive여야 한다.

검사 실패를 zero tensor, `D_sep`, 또는 physical-symmetric 결과로 조용히 대체하지
않는다.

### 7.2 Cauchy--Schwarz audit

다음을 저장한다.

\[
v_i=|c_i|-\sqrt{\max(a_ib_i,0)}.
\]

Scale-aware 허용 오차를

\[
\tau_i=C_{cs}\,\epsilon_{mach}\max(s_i,\bar s)
\]

로 두고, 권장 `C_cs=128`을 사용한다.

- `v_i <= tau_i`: roundoff 범위로 허용하고 raw value와 violation statistic을 기록
- `v_i > tau_i`: response-block indexing 또는 cross-term 계산 오류로 보고 context
  build를 실패

### 7.3 `D_exact` roundoff 처리

수학적으로 `s+2c >= 0`이지만 cancellation 근처에서 작은 음수 roundoff가 생길 수
있다.

- raw exact base가 `-tau_i` 이상이면 0으로 clamp하고 count/max magnitude 기록
- `-tau_i`보다 작으면 fail fast
- clamp는 `D_exact`에만 적용되는 수치 보정이며 negative cross term을 제거하는 ReLU가
  아니다.

### 7.4 q/rho zero guard

- `q_epsilon=cross_axis_relative_eps*bar_s`를 float64 tensor로 만든다.
- `q` denominator는 `s+q_epsilon`이며 반드시 positive다.
- `rho` denominator는 `max(sqrt(max(a*b,0)),q_epsilon)`다.
- raw `rho`의 Cauchy violation을 기록한다.
- Plotly color range를 위해 display copy를 `[-1,1]`로 clip할 수 있지만 NPZ의 raw
  diagnostic 값은 보존한다.

### 7.5 dtype-specific tiny

`torch.finfo(dtype).tiny`는 현재 line-search의 sample-local zero division guard처럼
마지막 방어선에만 쓴다. `tiny`를 scale-aware `denominator_relative_eps`나
`cross_axis_relative_eps` 대신 사용하지 않는다.

## 8. diagnostics와 spatial maps

### 8.1 run-level field archive

기존 `data/symmetric_tangent_green_response_fields.npz`를 canonical run-level archive로
유지하고 다음 fields를 추가한다.

```text
tangent_preconditioner_schema_version
preconditioner_variant
gamma_x_squared                    # backward alias: a_i
gamma_y_squared                    # backward alias: b_i
a_i
b_i
c_i
rho_i
rho_i_display_clipped
q_i
q_epsilon
separable_preconditioner_base
exact_preconditioner_base
absolute_preconditioner_base
quadratic_preconditioner_base
preconditioner_base                # selected undamped base
damping
denominator                        # selected base + damping
gain_scale
cauchy_violation
exact_roundoff_clamp_mask
coords_valid
```

Field shape는 point field가 `(P,)`, scalar가 `()`가 되도록 고정한다. 같은 run-level
field를 sample 수만큼 복제하지 않는다.

### 8.2 필수 Plotly maps

현재 exporter가 tangent map을 저장하는 `figures/balance_projection/` 경로와 기존
`tangent_preconditioner_base`, `tangent_denominator` 파일명을 유지한다. 같은 경로에
HTML/JSON/PNG/PDF 정책을 맞춰 `tangent_` prefix의 다음 map을 추가한다. 기존 파일을
새 directory로 이동하지 않는다.

- `a_i`, `b_i`, `a_i+b_i`
- signed `c_i`
- signed `rho_i`
- `q_i`와 `4q_i`
- undamped `D_sep`, `D_exact`, `D_abs`, `D_q`
- damped selected `D`
- `D_exact/D_sep`, `D_abs/D_sep`, `D_q/D_sep`
- `damping/(a+b+damping)`
- `sign(c_i)`는 diagnostic map으로 허용하되 ReLU variant를 만들지 않음

Undamped ratio map은 `a_i+b_i > q_epsilon`인 point에서만 계산하고, 나머지는
`NaN`과 별도 valid mask로 저장한다. `0/0`을 0이나 1로 조용히 치환하지 않는다.
필요하면 항상 positive인 damped ratio를 별도 field로 추가한다.

Signed `c`와 `rho`는 zero-centered diverging range를 사용한다. Positive fields는
robust quantile range와 full min/max/saturation count를 함께 기록한다. Domain boundary와
기존 transition edges를 overlay하되 raw arrays를 clip하지 않는다.

### 8.3 summary statistics

각 point field에 대해 다음을 기록한다.

- finite count, min, p01, p05, median, mean, p95, p99, max
- zero 또는 near-zero count
- `c<0`, `c=0`, `c>0` 비율
- `|rho|`의 p50/p90/p99와 `|rho|>0.5`, `>0.9` 비율
- 각 D의 min/max, p95/p05 spread, selected damping-floor proximity count
- Cauchy violation max/count
- exact roundoff clamp count/max magnitude

### 8.4 sample-level tangent diagnostics

기존 K diagnostics에 다음 identity fields를 추가한다.

- `tangent_preconditioner_variant`
- `tangent_j0`
- `tangent_jk`
- `tangent_jk_over_j0`
- `tangent_context_build_seconds`
- `tangent_correction_seconds`
- `tangent_total_projection_seconds`
- `tangent_forward_action_count`
- `tangent_adjoint_action_count`
- `tangent_peak_memory_bytes` (CUDA에서만, 아니면 null/not_applicable)

기존 `tangent_response_cost_kN`, coefficients, active mask, response orthogonality,
balance residual, eta fields는 유지한다.

## 9. artifact와 provenance schema

### 9.1 production artifact summary

`summary.json`에 다음 구조를 추가한다.

```json
{
  "symmetric_tangent_green_response": {
    "preconditioner": {
      "schema_version": 1,
      "variant": "separable",
      "mathematical_label": "D_sep",
      "formula": "a+b+damping",
      "relative_lambda": 0.01,
      "denominator_relative_eps": 1e-12,
      "cross_axis_relative_eps": 1e-12,
      "gain_scale_policy": "mean(a+b)_shared_across_variants",
      "q_factor": 4.0,
      "intermediate_parameter_used": false,
      "relu_variant_used": false
    },
    "matrix_policy": {
      "global_response_matrix_materialized": false,
      "global_gram_matrix_materialized": false,
      "global_linear_solve": false,
      "segment_local_cross_axis_diagonal": true
    },
    "numerical_checks": {
      "finite": true,
      "strictly_positive_denominator": true,
      "cauchy_violation_count": 0,
      "exact_roundoff_clamp_count": 0
    }
  }
}
```

실제 summary는 추가로 git commit, dirty flag, config path/hash, CouplingNet/GreenNet
checkpoint path와 SHA-256, geometry path/hash, dtype/device, seed provenance를 기록해야
한다. Path만 있고 hash가 없는 상태를 reproducible provenance로 간주하지 않는다.

### 9.2 4 variants x K=1..4 frozen audit artifact

기존 `cli/audit_tangent_subspace.py`를 확장한 audit의 권장 output은 다음과 같다.

```text
<outdir>/
  audit_tangent_preconditioners.log
  summary.json
  diagnosis_report.md
  metrics/per_sample_preconditioner_k1_k4.csv
  metrics/aggregate_preconditioner_k1_k4.csv
  metrics/paired_preconditioner_k1_k4.csv
  data/tangent_preconditioner_fields.npz
  data/selected_preconditioner_k1_k4.npz
  figures/tangent_preconditioner/*.json|html|png|pdf
  figures/aggregate/*.json|html|png|pdf
  figures/selected_samples/*.json|html|png|pdf
```

Method ID는 문자열 정렬과 parser 안정성을 위해 다음처럼 고정한다.

```text
physical_symmetric
sep_k1_capped
sep_k1
sep_k2
sep_k3
sep_k4
exact_k1_capped
exact_k1
...
q_k4
```

`abs`는 config full name `absolute_cross_axis`, artifact method prefix `abs`를 쓰고,
`q`는 config full name `normalized_quadratic_cross_axis`, artifact prefix `q`를 쓴다.
`K1` primary nested comparison은 uncapped exact scalar coefficient이며, capped
production K1은 별도 method로 기록한다.

### 9.3 schema compatibility

- 기존 NPZ field는 삭제/rename하지 않는다.
- `gamma_x_squared`, `gamma_y_squared`, `preconditioner_base`, `denominator`는 유지한다.
- 새 reader는 `schema_version`을 검사한다.
- 구 reader가 새 optional fields를 무시할 수 있어야 한다.
- non-default variant에서 `preconditioner_base` 의미가 selected base임을 summary와
  field description에 명시한다.
- baseline default artifact의 기존 numeric fields는 bitwise 또는 지정된 exact
  tolerance로 같아야 한다.

## 10. backward compatibility와 checkpoint 영향

### 10.1 model checkpoint tensor

Tangent context와 denominator는 `nn.Module` parameter/buffer가 아니므로 새 variant는
CouplingNet/GreenNet `state_dict` key, tensor shape, output-contract version을 바꾸지
않는다.

- 기존 model checkpoint는 그대로 load 가능하다.
- 새 variant를 사용해 학습한 checkpoint도 model tensor 표면은 동일하다.
- architecture compatibility 검사에 새 tensor key를 추가하지 않는다.

### 10.2 config compatibility

- 기존 config가 새 fields를 생략하면 `preconditioner_variant="separable"`과
  `cross_axis_relative_eps=1e-12`를 resolve한다.
- default path의 `preconditioner_base`, `gain_scale`, `damping`, `denominator`, K=1
  output, K=2 seed tensors, gradients가 기존과 같아야 한다.
- `config_used.json`에는 resolved fields를 항상 기록한다.

### 10.3 resume와 post-hoc override 구분

- 동일 run의 strict training resume 또는 continuation에서는 saved config와 runtime
  `preconditioner_variant`가 다르면 error로 처리한다.
- Frozen checkpoint audit는 의도적인 post-hoc variant override를 허용하지만
  `training_or_checkpoint_updated=false`, `posthoc_preconditioner_override=true`를
  기록한다.
- Post-hoc 우수 결과는 해당 variant로 paired retraining한 효과로 해석하지 않는다.
- model-only safetensors가 optimizer/RNG state를 보존하지 않는 현재 한계는 그대로다.

## 11. 테스트 설계

### 11.1 unit tests

#### raw Gram terms

새 `test/test_complex_axial_response_operator.py` 또는 기존 tangent test에 다음을
추가한다.

- 작은 dense synthetic `H_x`, `H_y`, scalar `M`에서 local-block `a,b,c`가
  `diag(H_x^T M H_x)`, `diag(H_y^T M H_y)`,
  `diag(H_x^T M H_y)`와 일치
- singleton intersection fast path와 general overlap path가 같은 결과
- disconnected segment와 source ordering에서도 global index가 맞음
- invalid block overlap/invariant를 조용히 무시하지 않음
- float64 기준 권장 `rtol=1e-12`, `atol=1e-14`

#### formulas와 bounds

- hand-computed aligned, anti-aligned, orthogonal, imbalanced 사례
- `D_sep`, `D_exact`, `D_abs`, `D_q` 식 직접 비교
- `c=0`이면 네 surrogate 중 `sep=exact=abs=q`
- `c>0` ordering과 `c<0` ordering
- `D_q <= D_abs`, 각 base `<=2(a+b)`
- exact cancellation `a=b`, `c=-a`에서 undamped `D_exact=0`
- damping 후 네 denominator가 strictly positive
- `q_epsilon` scale invariance: `H_x,H_y`를 같은 상수로 scale하면 D도 상수 제곱으로
  scale
- nonfinite, negative gain, significant Cauchy violation fail-fast
- roundoff-sized negative exact base만 clamp하고 provenance count 기록

#### config

- field omission -> `separable`
- 네 variant round-trip
- invalid string, boolean epsilon, zero/negative/nonfinite epsilon 거부
- `alpha`, ReLU 관련 unknown key 거부
- old config resolution과 `config_used.json` materialization

### 11.2 integration tests

- 실제 small complex geometry에서 context build 1회, train/eval batch 재사용
- 네 variant x K=1..4 projection shape, balance, autograd finite
- `phi+psi-rhs` max abs가 float64 tolerance 이내
- production reconstruction과 response-operator reconstruction equivalence
- K nested sequence에서 `J_{k+1} <= J_k`가 relative `1e-10` 이내
- response orthogonality max가 relative `1e-10` 이내
- degenerate direction은 `active=false`, coefficient 0, 이전 K 결과로 fallback
- trainer/evaluator log와 CSV에 variant/provenance 기록
- 4-way audit가 response operator 한 번만 build
- artifact export가 필수 NPZ/summary/map을 생성하고 shapes/schema가 일치

### 11.3 regression tests

- omitted variant와 explicit `separable`의 context tensors bitwise equality
- 기존 K=1 fixed output/gradient bitwise equality
- 기존 K=1 closed-loop output/eta diagnostics bitwise equality
- 기존 generic K=2 tensors bitwise equality
- existing K=3/K=4 nested results tolerance equality
- state_dict key/shape equality와 old checkpoint strict load
- existing `test_complex_tangent_projection.py`, trainer/evaluator/artifact/config suite 통과
- `git diff --check`

### 11.4 성능 regression

동일 device, dtype, batch, geometry에서 warmup 후 최소 20회 측정한다.

- context build seconds
- projection forward seconds/sample
- forward/adjoint action count
- peak CUDA allocated bytes
- total evaluation wall time
- training step time와 samples/s

Acceptance guard는 `D_sep` 대비 cross-term context build overhead를 별도로 보고하고,
steady-state projection runtime이 formula 선택 때문에 5% 이상 악화되면 원인을
분석하는 것이다. K 증가로 인한 operator-action 비용은 variant 비용과 분리한다.

## 12. controlled ablation protocol

### 12.1 실험 질문

1. 같은 K에서 cross-axis 정보를 넣는 D가 `D_sep`보다 response mismatch를 더 빨리
   낮추는가?
2. `D_exact`의 cancellation 반영이 과도한 update/flux 악화를 만드는가?
3. `D_abs`의 conservative scaling과 `D_q`의 quadratic selectivity 중 어느 것이
   end-to-end energy/solution/runtime tradeoff가 좋은가?
4. K가 커질 때 D 선택의 영향이 사라지는가, 유지되는가?
5. frozen post-hoc 개선이 paired retraining에서도 재현되는가?

### 12.2 4 x 4 factorial

주 실험 축은 정확히 다음 16 cell이다.

| variant | K=1 | K=2 | K=3 | K=4 |
|---|---:|---:|---:|---:|
| `separable` (`D_sep`) | run | run | run | run |
| `exact_diagonal` (`D_exact`) | run | run | run | run |
| `absolute_cross_axis` (`D_abs`) | run | run | run | run |
| `normalized_quadratic_cross_axis` (`D_q`) | run | run | run | run |

추가 alpha/ReLU cell은 만들지 않는다.

### 12.3 Phase A: formula와 spatial preflight

한 geometry/Green checkpoint에서 `a,b,c,rho,q`와 네 D map을 생성한다.

통과 조건:

- 모든 finite/positivity/bound 검사 통과
- Cauchy violation이 허용 roundoff 이내
- exact clamp가 없거나 roundoff 범위로 설명됨
- map에서 boundary/transition concentration을 확인할 수 있음
- context build가 global matrix 없이 1회

### 12.4 Phase B: frozen-checkpoint 4 x K audit

같은 CouplingNet raw output, 같은 GreenNet, 같은 test sample 순서에서 16개
uncapped nested 후보를 비교한다. K=1 production cap 결과는 별도 safety reference로
함께 기록한다.

이 단계는 D와 K의 즉시 수치 효과를 가장 깨끗하게 분리하지만 training adaptation을
평가하지 않는다. 따라서 screening과 defect discovery 용도다.

#### 12.4.1 완료된 Pentagram audit

2026-08-26에 Pentagram `coupling11` best-energy checkpoint와 고정 GreenNet으로
100개 test sample을 평가했다. 결과 bundle은 다음 경로에 보존한다.

```text
checkpoints/pentagram/coupling11/tangent_preconditioner_4x4_audit/
```

Audit는 batch당 CouplingNet raw output을 한 번만 계산하고, 전체 실행에서 response
context와 segment-local operator를 각각 한 번만 만들었다. 네 variant는 같은
`a,b,c,rho,q`를 가진 immutable context view로 구성되었고 global response/Gram
matrix 또는 linear solve는 사용하지 않았다. Operator-equivalence 최대 절대 오차는
`2.775558e-17`, physical balance 최대 절대 잔차는 `0`이었다.

| method | response mismatch | optimized energy | mean `rel_sol` |
|---|---:|---:|---:|
| physical symmetric | `1.637002e-6` | `1.960605e-3` | `1.672124e-1` |
| separable K=1 uncapped | `1.409522e-7` | `4.751977e-4` | `5.931192e-2` |
| separable K=3 | `6.999244e-9` | `2.338294e-5` | `1.295009e-2` |
| separable K=4 | `5.381861e-9` | `1.741050e-5` | `1.112221e-2` |
| exact/absolute K=4 | `5.373708e-9` | `1.767079e-5` | `1.114672e-2` |
| normalized-quadratic K=4 | `5.380846e-9` | `1.741583e-5` | `1.112297e-2` |

이 frozen operator에서는 `c`가 모든 valid point에서 nonnegative였으므로
`D_exact`와 `D_abs`가 대수적으로 동일했다. `separable K=1 -> K=4`에서 response
mismatch는 `96.182%`, optimized energy는 `96.336%`, mean `rel_sol`은 `81.248%`
감소했다. 반면 K=4에서 variant 차이는 작았다. `D_exact/D_abs`는 `D_sep`보다
response mismatch를 `0.152%` 더 낮췄지만 optimized energy를 `1.495%`, mean
`rel_sol`을 `0.220%` 높였고, `D_q`는 `D_sep`와 사실상 같았다.

따라서 이 checkpoint의 post-hoc 근거는 **variant 변경보다 K 증가가 지배적**임을
보이며, production default는 `separable`로 유지한다. 이는 paired retraining을
대체하지 않으며 다른 geometry에서 variant promotion을 주장하는 근거로 사용하지
않는다.

### 12.5 Phase C: paired retraining

16 cell을 seed `0,1,2,3`으로 반복하여 권장 64 runs를 구성한다. Compute budget이
부족해도 single-seed 결과로 최종 variant를 선택하지 않는다. 축소할 경우 16-cell
single-seed를 pilot로 명시하고, promotion 후보와 `D_sep`를 네 paired seeds로
재실험한다.

모든 paired run에서 다음을 동일하게 고정한다.

- dataset files와 train/valid/test split
- indexed-GP source identity seed
- CouplingNet model seed와 parameter initialization
- GreenNet checkpoint
- geometry와 coefficients
- batch order, batch size, total optimizer calls
- optimizer, LR schedule, warmup, validation frequency
- canonical energy/boundary weight
- response-trust/stationarity enable 및 weights
- final cross-axis reconstruction
- checkpoint selection policy
- dtype/device/software/hardware 범위
- `relative_lambda=0.01`, `denominator_relative_eps=1e-12`,
  `cross_axis_relative_eps=1e-12`

달라지는 것은 `preconditioner_variant`, `subspace_dimension` 두 field뿐이다.

### 12.6 K=1 공정성

K=1은 두 결과를 구분한다.

- `K1_uncapped`: nested K=1..4 response-subspace 비교의 수학적 첫 단계
- `K1_capped`: 실제 production safety cap을 적용한 결과

K=2..4는 현재 계약상 eta cap을 쓰지 않는다. 따라서 `K1_capped`와 K>=2를
monotonic nested sequence라고 부르지 않는다. Monotonicity는 `K1_uncapped -> K2 ->
K3 -> K4`에 대해서만 검사한다.

### 12.7 run naming

권장 config/run ID:

```text
tangent_precond_<variant_short>_k<K>_seed<S>
```

Short name은 `sep`, `exact`, `abs`, `q`다. Config는 기존 flat JSON 관례를 따르되,
canonical base에서 기계적으로 생성하고 resolved diff가 두 실험 field와 seed 외에는
없는지 test로 확인한다.

## 13. 평가 metric 정의

### 13.1 primary response metric

각 sample에 대해

\[
m_K=m_0+S\delta_K,
\qquad
\boxed{J_K=\|m_K\|_{M_\Omega}^2}.
\]

\[
\boxed{R_K=J_K/(J_0+\epsilon_J)}.
\]

`epsilon_J`는 reporting zero guard이며 method selection에 유리하도록 variant별로
바꾸지 않는다. Per-sample `J_K`, `J_K/J_0`와 aggregate mean, median, std, p90,
p95, max를 모두 보고한다.

`J_K`는 tangent response mismatch objective다. Canonical energy, total training loss,
PDE ground-truth error가 아니다.

### 13.2 reference-free physics metrics

- `loss_energy_optimized`
- unweighted `canonical_energy`
- `canonical_bulk_energy`
- `canonical_boundary_energy`
- `tangent_response_trust_*` (enabled인 경우)
- `tangent_post_line_search_stationarity_*` (enabled인 경우)
- `physical_balance_max_abs`
- split transition/regular jump RMS

Training checkpoint selection은 기존 reference-free policy를 유지한다.

### 13.3 evaluation-only reference metrics

Reference가 있을 때만 계산하고 training, correction, checkpoint selection에 쓰지 않는다.

\[
\mathrm{rel\_sol}
=\frac{\|u_{\mathrm{pred}}-u\|_2}{\|u\|_2}.
\]

`u_pred`는 configured production final reconstruction을 사용하고,
`rel_sol_equal_mean`을 별도로 기록한다.

\[
\mathrm{rel\_u\_phi}
=\frac{\|u_\phi-u\|_2}{\|u\|_2},
\qquad
\mathrm{rel\_u\_psi}
=\frac{\|u_\psi-u\|_2}{\|u\|_2}.
\]

현재 artifact/CSV field 이름은 각각 `rel_u_phi`, `rel_u_psi`로 유지한다.

`rel_flux`는 현재 audit contract를 유지한다. 즉 `phi`, `psi` 각각의 target-relative
L2를 계산해 두 축 평균을 보고하고, pair-flattened `pair_rel_flux_target`도 함께
저장한다.

추가로 다음을 보고한다.

- directional split mismatch relative error
- transition/regular zone solution error와 trace-jump RMS
- per-sample paired win/loss/tie counts
- seed aggregate와 paired bootstrap interval

### 13.4 runtime와 resource metrics

- context/Gram-term build seconds
- D formula build seconds
- correction seconds/sample
- total evaluator wall seconds
- training step seconds와 samples/s
- response operator forward/adjoint action count
- CPU peak RSS 또는 CUDA max memory allocated
- artifact export time은 numerical evaluation time과 분리

## 14. acceptance criteria

### 14.1 구현 승인 gate

모두 충족해야 한다.

1. Default `separable` old/new bitwise regression 통과.
2. Small dense reference의 `a,b,c`와 네 D가 float64 tolerance 내 일치.
3. 모든 supported geometry에서 finite/positive/Cauchy checks 통과.
4. `phi+psi=rhs` max error가 `1e-10` 이하 또는 dtype-scaled 더 엄격한 기존
   tolerance 통과.
5. `K1_uncapped -> K2 -> K3 -> K4`의 `J` nonincrease가 relative `1e-10` 이내.
6. active response directions의 normalized off-diagonal Gram이 `1e-10` 이내.
7. cache build count 1, global matrix/solve 없음.
8. old checkpoint strict load와 state_dict surface 불변.
9. 필수 artifact schema와 maps 생성.
10. focused pytest, touched-file Ruff, `mypy src`, `git diff --check` 통과.

### 14.2 frozen audit gate

- 모든 16 cell이 같은 raw outputs를 사용했음을 hash로 확인
- nonfinite/invalid direction 0건; 정당한 degeneracy는 active mask로 설명
- K monotonicity 위반 0건
- variant별 D map과 distribution이 formula ordering을 만족
- `J_K/J_0`, energy, reference metrics, runtime의 sample-paired CSV 완비

### 14.3 scientific promotion gate

새 variant를 production default 후보로 올리려면 paired retraining에서 다음을 모두
충족해야 한다.

- 최소 네 paired seeds
- primary reference-free validation `loss_energy_optimized`가 `D_sep`보다 seed 평균
  2% 이상 개선하거나, 1% non-inferiority 안에서 `rel_sol` 또는 runtime의 명확한
  Pareto 개선
- paired seeds 중 최소 3/4가 primary metric에서 같은 방향
- `rel_sol` seed mean이 `D_sep`보다 1% relative 이상 악화되지 않음
- `rel_flux`와 boundary-energy p95가 각각 5% relative 이상 악화되지 않음
- balance, nonfinite, monotonicity, artifact completeness failure 0건
- steady-state runtime이 같은 K의 `D_sep`보다 10% 이상 느리면 그 비용을 상쇄하는
  accuracy improvement가 명확함

위 2%, 1%, 5%, 10%는 **사전 등록용 기본 threshold**다. 첫 official run 전에 사용자가
변경할 수 있지만 결과를 본 뒤 유리하게 변경해서는 안 된다. Threshold를 충족하지
못한 variant도 연구 결과로 보존하되 default로 승격하지 않는다.

## 15. 단계별 구현 순서

### Step 1. 수학 reference test를 먼저 추가

- dense small-matrix `a,b,c`와 네 D expected 값을 test에 고정
- ordering, bounds, scale invariance, invalid-input tests 작성
- 기존 default bitwise baseline fixture 확보

완료 조건: 새 implementation 없이도 expected contract가 명확하고, 새 tests가 필요한
미구현 지점에서만 실패한다.

### Step 2. axial raw cross term 구현

- `diagonal_response`와 `tangent_column_gram_terms` 구현
- general overlap correctness와 singleton fast path 검증
- global matrix 미생성 통계 추가

완료 조건: dense reference `a,b,c` test 통과.

### Step 3. config와 pure preconditioner builder 구현

- variant literal/fields/strict validation
- `TangentPreconditionerTerms`
- four formulas, damping, Cauchy/roundoff safeguards

완료 조건: formula/config tests 통과.

### Step 4. tangent context에 연결

- context fields와 statistics 확장
- default arithmetic order를 보존
- K=1..4 기존 step code는 변경 최소화

완료 조건: bitwise default, balance, autograd, K regression 통과.

### Step 5. trainer/evaluator cache와 provenance

- resolved variant log
- context build timing/count
- config/checkpoint/geometry hash
- runtime counters

완료 조건: cache integration tests 통과.

### Step 6. artifact와 spatial maps

- run-level NPZ schema
- summary schema/version
- Plotly maps와 ranges
- backward fields 유지

완료 조건: artifact golden/schema tests 통과.

### Step 7. frozen 4 x K audit 확장

- shared response operator/raw Gram terms
- four immutable contexts
- 16 uncapped cells와 four capped K1 references
- CSV/NPZ/report/figures

완료 조건: same-output hash, K monotonicity, method IDs, artifact completeness 통과.

### Step 8. 전체 정적/회귀 검증

권장 환경 선택 순서는 repo root의 `.venv`가 있으면 그것을 우선하고, 없으면
`/home/jjhong0608/.conda/envs/green_net/bin/python`을 사용한다.

```bash
PYTHONPATH=src <python> -m pytest \
  test/test_complex_tangent_projection.py \
  test/test_complex_tangent_subspace_audit.py \
  test/test_io_config.py \
  test/test_cli_train.py \
  test/test_complex_coupling_trainer.py \
  test/test_complex_coupling_artifacts.py

ruff check src cli test
ruff format --check src cli test
mypy src
git diff --check
```

전체 suite는 범위 밖의 실험 완료 marker 같은 외부 상태 실패와 이번 구현 실패를
구분해 보고한다. 실패를 `_SUCCESS` 위조나 unrelated test 수정으로 숨기지 않는다.

### Step 9. Phase A/B 실행 후 paired training 승인

수학/구현 gate와 frozen audit가 통과한 뒤에만 64-run paired training을 시작한다.
공식 run 시작 전 threshold와 canonical geometry/checkpoint를 기록한다.

## 16. 미결정 사항

다음은 코드 구현을 막지 않지만 official experiment 전 확정해야 한다.

1. **첫 official geometry**: 현재 K=2..4 paired config가 있는 Pentagram을 권장하나,
   최종 결론은 Disk/Annulus 같은 두 번째 geometry에서 독립 검증해야 한다.
2. **64-run compute budget**: 16 cells x 4 seeds 전체를 즉시 실행할지, single-seed
   screening 후 `D_sep`와 promotion 후보만 4 seeds로 갈지 결정이 필요하다.
3. **성능 측정 device**: CPU와 CUDA 결과를 섞지 않는다. 논문 표의 canonical device를
   사전 지정해야 한다.
4. **Cauchy tolerance constant**: 기본 `128*eps_machine`을 제안한다. 실제 geometry의
   dense-reference test에서 더 엄격한 값이 안정적이면 결과를 보기 전에 낮춘다.
5. **cross-support fallback**: singleton invariant가 모든 production geometry에서
   test로 확인되면 general overlap fallback을 유지할지 fail-fast로 단순화할지 결정할
   수 있다. 정확성을 낮추는 silent diagonal approximation은 선택지가 아니다.
6. **default 승격 여부**: 본 문서는 네 후보를 구현/비교하도록 승인하지만
   `D_sep`를 production default에서 교체하도록 승인하지 않는다.

## 17. 금지할 해석

- `D_sep`를 standard Jacobi 또는 exact Hessian diagonal이라고 부르지 않는다.
- `D_exact`를 full Hessian, Newton solve, exact PDE Hessian, exact physical source split로
  부르지 않는다.
- `D_abs`와 `D_q`를 Hessian diagonal이라고 부르지 않는다.
- `D_abs`는 PETSc의 `abs(diag(A))`와 동일하지 않다. 여기서는 cross term에만
  absolute value를 적용한다.
- `rho`는 sample 통계 correlation이 아니라 두 response column의 weighted cosine다.
- `q`는 확률, mutual information, 유일한 Schur complement, 증명된 optimal
  preconditioner가 아니다.
- `c<0`을 오류나 약한 coupling으로 간주하지 않는다. 강한 anti-alignment일 수 있다.
- `D_exact`가 작다는 사실만으로 source response 자체가 작다고 결론내리지 않는다.
  x/y cancellation 때문일 수 있다.
- positive denominator가 convergence, lower canonical energy, lower `rel_sol`을 보장한다고
  주장하지 않는다.
- `J_K/J_0` 개선을 canonical energy 또는 ground-truth solution 개선과 동일시하지
  않는다.
- K를 training epoch, network depth, Green iterations로 해석하지 않는다. K는 tangent
  response subspace dimension이다.
- Frozen-checkpoint post-hoc 결과를 paired retraining 결과로 바꾸어 말하지 않는다.
- Reference `sol/phi/psi`를 preconditioner 생성, tangent update, training loss,
  checkpoint selection에 사용하지 않는다.
- GreenNet이 learned/frozen approximation이라는 사실을 빼고 `D_exact`를 continuum
  operator의 exact quantity라고 부르지 않는다.
- intermediate parameter나 ReLU variant를 실험 편의상 몰래 추가하지 않는다.

## 18. 외부 수치선형대수 참고 경계

표준 point Jacobi preconditioner가 matrix diagonal을 사용한다는 명칭 경계는
[Netlib, *Templates for the Solution of Linear Systems*, Section 3.2](https://netlib.org/templates/templates.pdf)와
[PETSc `PCJACOBI`](https://petsc.org/release/manualpages/PC/PCJACOBI/)에 맞춘다.
PETSc가 absolute diagonal option을 제공한다는 사실은 sign-robust scaling의 일반적
예일 뿐, 이 문서의 `D_abs=s+2|c|`를 정당화하거나 동일시하지 않는다. dtype별
`eps`와 `tiny`의 의미는 [PyTorch `torch.finfo` 문서](https://docs.pytorch.org/docs/stable/type_info.html)를
따르되, 구현은 scale-aware relative epsilon을 우선한다.

## 19. 최종 완료 정의

이 설계의 구현 작업은 다음이 모두 성립할 때 완료다.

- 네 variant가 정확한 config 이름과 수식으로 production tangent context에 연결됨
- default `D_sep`가 기존 path와 호환됨
- `a,b,c,rho,q`, 네 base, damping, selected denominator가 artifact에 남음
- matrix-free/axial cross term이 dense reference로 검증됨
- cache/provenance/checkpoint 계약이 test로 고정됨
- K=1..4의 unit/integration/regression 및 nested-cost 검사가 통과함
- frozen 4 x K audit가 reproducible artifact bundle을 생성함
- paired-training config matrix가 두 field 외 조건을 고정함
- acceptance threshold와 미결정 실험 조건이 official run 전에 기록됨
- intermediate parameter와 ReLU variant가 코드, config, artifact 어디에도 없음
