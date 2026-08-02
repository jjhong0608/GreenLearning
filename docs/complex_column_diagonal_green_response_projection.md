# Complex Column-Diagonal Green-Response Projection

## Purpose

Complex CouplingNet은 두 directional reference response proposal
`P_raw`, `Q_raw`를 출력한다. 기존 `physical_symmetric` projection은 balance residual을
두 physical directional source에 동일하게 나눈다. Optional
`column_diagonal_green_response` mode는 frozen GreenNet reconstruction이 각 source-point
correction에 얼마나 크게 반응하는지를 이용해 residual을 pointwise 배분한다.

이 mode는 physical source space에서 projection을 수행한다. CouplingNet backbone,
GreenNet checkpoint, canonical energy objective, geometry/sample NPZ schema는 바꾸지 않는다.

## Physical Projection Variables

각 valid point에서

\[
p=\frac{P_{\mathrm{raw}}}{L_x^2},
\qquad
q=\frac{Q_{\mathrm{raw}}}{L_y^2},
\qquad
r=f-p-q
\]

로 둔다. Projection은 corrections가

\[
\delta\phi+\delta\psi=r
\]

를 만족하도록 구성하고, 최종 physical source는

\[
\phi=p+\delta\phi,
\qquad
\psi=q+\delta\psi
\]

이다. Green reconstruction에 전달할 reference response는 projection 이후에만

\[
\Phi=L_x^2\phi,
\qquad
\Psi=L_y^2\psi
\]

로 만든다.

## Directional Response Operator

Connected segment `s`의 reconstruction node에서 production operator는

\[
H_{s,ij}=G_s(t_i,\eta_j)w_{s,j}L_s^2
\]

이다. `G_s`는 frozen GreenNet kernel, `w_{s,j}`는 production reconstruction과 같은
unit-interval quadrature weight다. Boundary endpoint source는 hard zero이고 valid output과
source column에는 interior valid node만 포함한다.

Valid Cartesian points의 lumped solution mass는

\[
M_\Omega=(h_xh_y)I
\]

로 고정한다. Full response metric은

\[
A_s=H_s^\top M_\Omega H_s
\]

이지만 full `A_s`는 생성하지 않는다.

## Column Diagonal, Not Row Norm

구현하는 gain은 source point `j`에 대한 column diagonal이다.

\[
\gamma_{s,j}^2
=[A_s]_{jj}
=\left\|M_\Omega^{1/2}H_s(:,j)\right\|_2^2
\]

따라서 segment별 계산식은

\[
\gamma_{s,j}^2
=h_xh_y\,L_s^4w_{s,j}^2
\sum_i G_s(t_i,\eta_j)^2.
\]

합은 source index `j`를 고정하고 output index `i`에 대해 수행한다. 이는 다음 row norm과
다르다.

\[
\rho_{s,i}^2=\sum_j H_s(i,j)^2.
\]

Column diagonal은 한 source correction이 전체 directional solution field를 움직이는 비용을
측정한다. Row norm은 한 evaluation point가 모든 source에 얼마나 민감한지를 측정하므로
projection correction coefficient로 사용하지 않는다.

## Pointwise Exact-Balance Rule

Regularized response cost를

\[
\bar\gamma_x^2=\gamma_x^2+\varepsilon,
\qquad
\bar\gamma_y^2=\gamma_y^2+\varepsilon
\]

로 둔다. Fixed gain exponent를 `alpha`라고 하면 production weight는

\[
w_\phi^{(\alpha)}
=
\frac{(\bar\gamma_y^2)^\alpha}
{(\bar\gamma_x^2)^\alpha+(\bar\gamma_y^2)^\alpha},
\qquad
w_\psi^{(\alpha)}=1-w_\phi^{(\alpha)},
\qquad 0\le\alpha\le1
\]

이다. 중간 exponent에서는 같은 식을 overflow/underflow 없이 계산하기 위해

\[
\ell=\log\bar\gamma_y^2-\log\bar\gamma_x^2,
\qquad
w_\phi^{(\alpha)}=\operatorname{sigmoid}(\alpha\ell)
\]

을 사용한다. Endpoint의 의미와 구현은 명시적으로 고정한다.

- `alpha=0`: unequal gain에서도 `w_phi=w_psi=1/2`인 physical symmetric correction.
- `alpha=1`: 기존 floating-point 결과를 보존하는 direct ratio
  `gy_bar/(gx_bar+gy_bar)`.
- `0<alpha<1`: column-diagonal anisotropy의 강도만 완화하는 fixed tempered
  correction.

`alpha`는 sample-independent scalar config이며 model parameter, learned gate 또는
sample-dependent network가 아니다. Correction은

\[
\delta\phi=w_\phi^{(\alpha)} r,
\qquad
\delta\psi=w_\psi^{(\alpha)} r
\]

이다. Difference form에서는

\[
d_{\mathrm{base}}=p-q,
\qquad
d^\star=d_{\mathrm{base}}+
(w_\phi^{(\alpha)}-w_\psi^{(\alpha)})r,
\]

\[
\phi=\frac12(f+d^\star),
\qquad
\psi=f-\phi.
\]

마지막 대입 때문에 floating-point에서도 `phi+psi=f`를 직접 보존한다. 두 directional gain이
같으면 모든 exponent에서 `w_phi=w_psi=1/2`이고 기존 physical symmetric
projection과 일치한다. Tempering은 row norm이나 full-Gram approximation을 새로
도입하는 것이 아니라 기존 column-diagonal correction strength만 조절한다.

## Why No Full Solve

Full quadratic response cost를 그대로 최소화하면

\[
(A_x+A_y)\delta\phi=A_y r,
\qquad
\delta\psi=r-\delta\phi
\]

를 풀어야 한다. 이 기능은 axial-only local computation을 유지하기 위해 off-diagonal source
correlation을 버리고 column diagonal만 사용한다. 전체 `H_s`, 전체 `A_s`, 2D mesh, global
matrix solve는 생성하지 않는다.

## Runtime And Artifacts

- Gain context는 frozen GreenNet, geometry, coefficient branch, reconstruction metadata에만
  의존하므로 sample-independent다.
- Trainer와 evaluator는 첫 batch에서 context를 한 번 만들고 같은 instance의 모든 batch에서
  재사용한다. Gain tensors는 autograd graph에 들어가지 않는다.
- Artifact exporter는 evaluator의 동일 cached context를 사용한다.
- Run-level gain/weight archive는
  `data/column_diagonal_green_response_fields.npz`에 fixed `gain_exponent`와 함께
  저장한다.
- Selected-sample archive는 raw balance residual, `delta_phi`, `delta_psi`, correction weights,
  difference update를 저장한다.
- Plotly gain/weight figures는 `figures/balance_projection/`에 저장한다.
- Context 최초 생성 log, artifact summary, NPZ scalar 및 figure title에 같은 fixed
  exponent를 기록한다.

## Configuration

```json
"balance_projection": {
  "enabled": true,
  "mode": "column_diagonal_green_response",
  "column_diagonal_green_response": {
    "gain_squared_eps": 1e-12,
    "gain_exponent": 0.25
  }
}
```

`gain_squared_eps`는 finite positive float여야 한다. `gain_exponent`는 bool이 아닌
finite numeric `[0,1]`이어야 하며 생략하면 기존 동작을 보존하는 `1.0`이다. `row_norm`
alias는 지원하지 않는다. Unit-square CouplingNet은 이 mode를 거부한다. Canonical
complex config의 기본값은 계속 `physical_symmetric`다.

## Experimental Interpretation

Projection mode는 model state-dict key나 output tensor shape를 바꾸지 않는다. 따라서 기존
checkpoint를 새 mode로 post-hoc 평가하는 것은 기술적으로 가능하지만, symmetric projection을
기준으로 학습된 raw output과의 비교는 공정한 training comparison이 아니다. 성능 결론은 같은
dataset, initialization, GreenNet checkpoint, optimizer, scheduler, epoch budget으로 각 mode를
처음부터 학습한 paired experiment에서 내려야 한다.

같은 이유로 exponent가 다른 checkpoint를 post-hoc으로 바꾸어 비교하지 않는다.
`alpha in {0, 0.25, 0.5, 1}` 비교는 같은 seed와 data를 사용해 각각 처음부터 재학습한다.
