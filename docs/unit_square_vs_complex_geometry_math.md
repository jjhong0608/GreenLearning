# Unit-Square Domain과 Complex Geometry에서의 GreenNet/CouplingNet 수학적 구조 비교

## 1. 목적과 관점

이 문서는 unit-square domain과 complex geometry에서 GreenNet/CouplingNet 구조가
수학적으로 어떻게 달라지는지를 설명한다. 관심의 중심은 사용법이나 저장 형식이 아니라,
동일한 PDE 해석 관점을 서로 다른 domain representation 위에서 어떻게 neural architecture로
표현하는가이다.

두 경우 모두 목표는 같다. 2차원 elliptic PDE의 solution을 직접 하나의 거대한 black-box
operator로 예측하기보다, axial direction에 따른 1차원 Green operator와 direction-split
coupling field를 결합해 solution을 재구성한다. GreenNet은 axial 1차원 operator의 Green
kernel을 제공하고, CouplingNet은 source와 coefficient field를 보고 두 방향의 split source를
만든다. Projection은 이 split source가 원래 PDE의 balance relation을 만족하도록 조정하고,
Green reconstruction은 split source를 다시 solution field로 되돌린다.

Unit-square와 complex geometry의 차이는 단순히 domain 모양이 다르다는 데 있지 않다.
Unit-square에서는 모든 axial line이 동일한 reference interval 위에 놓이므로 line geometry가
거의 암묵적이다. 반면 complex geometry에서는 domain과 axial line의 교차가 line마다 서로 다른
connected interval을 만들고, 각 interval의 길이와 위치가 operator 자체의 일부가 된다. 따라서
network가 받아야 하는 정보의 분해 방식도 달라진다.

이 문서에서는 GreenNet과 CouplingNet을 다음 네 요소를 중심으로 비교한다.

- branch: 함수 또는 geometry 정보를 전역적으로 요약하는 부분
- trunk: 특정 evaluation coordinate에서의 pointwise variation을 표현하는 부분
- projection: neural output을 PDE balance에 맞게 조정하는 부분
- Green reconstruction: split source를 solution으로 되돌리는 부분

## 2. 공통 PDE와 axial decomposition

기본적으로 고려하는 PDE는 다음 형태의 2차원 elliptic equation이다.

\[
-\partial_x(a\,\partial_x u)
-\partial_y(a\,\partial_y u)
+ b_x\,\partial_x u
+ b_y\,\partial_y u
+ c\,u
= f.
\]

여기서 \(u\)는 solution, \(f\)는 source, \(a\)는 diffusion coefficient,
\((b_x,b_y)\)는 convection field, \(c\)는 reaction coefficient이다. Axial decomposition은
이 2차원 operator를 x-direction part와 y-direction part로 나누어 해석한다. 이때 두 split field를
\(\phi\), \(\psi\)로 두면,

\[
\phi
=
-\partial_x(a\,\partial_x u)
+ b_x\,\partial_x u
+ \frac12 c\,u,
\]

\[
\psi
=
-\partial_y(a\,\partial_y u)
+ b_y\,\partial_y u
+ \frac12 c\,u.
\]

따라서 원래 PDE는 다음 balance relation으로 다시 표현된다.

\[
\phi+\psi=f.
\]

CouplingNet의 역할은 source와 coefficient field가 주어졌을 때, 이 balance relation을 만족하면서
Green reconstruction을 통해 올바른 solution을 만들 수 있는 \(\phi\), \(\psi\)를 찾는 것이다.
GreenNet의 역할은 각 axial direction에서 split source를 받아 solution contribution으로 바꾸는
1차원 Green operator를 제공하는 것이다.

수학적으로는 두 방향의 represented solution을 다음처럼 생각할 수 있다.

\[
u_\phi = \mathcal{G}_x[\phi],
\qquad
u_\psi = \mathcal{G}_y[\psi],
\]

여기서 \(\mathcal{G}_x\), \(\mathcal{G}_y\)는 각 방향의 axial Green reconstruction operator이다.
최종 solution prediction은 두 represented solution의 평균으로 해석한다.

\[
u_{\mathrm{final}}
=
\frac12\left(u_\phi+u_\psi\right).
\]

이 관점에서 GreenNet은 \(\mathcal{G}_x\), \(\mathcal{G}_y\)를 제공하는 model이고,
CouplingNet은 \(\phi\), \(\psi\)를 만드는 model이다.

## 3. Unit-Square GreenNet 구조

Unit-square domain에서는 모든 horizontal line과 vertical line이 같은 길이의 interval로 해석된다.
각 axial line은 자연스럽게 같은 reference coordinate 위에 놓이며, boundary condition도 모든 line에서
동일한 방식으로 적용된다. 이 단순성이 unit-square GreenNet 구조를 가능하게 한다.

하나의 axial line을 생각하자. 그 line 위의 coordinate를 \(s\)라 두면, line-wise 1차원 문제는
다음 형태로 볼 수 있다.

\[
-\frac{d}{ds}
\left(
a(s)\frac{dv}{ds}
\right)
+ b(s)\frac{dv}{ds}
+ c(s)v(s)
=
g(s),
\qquad
v=0 \text{ on the endpoints}.
\]

여기서 \(g\)는 해당 line 위의 source이고, \(v\)는 그 line에서 재구성되는 1차원 solution이다.
이 1차원 문제의 Green representation은

\[
v(s)
=
\int G(s,r)\,g(r)\,dr
\]

이다. GreenNet은 바로 이 kernel \(G(s,r)\)를 학습한다.

Unit-square에서 GreenNet branch는 coefficient profile의 axial slice를 encoding한다.
Diffusion, diffusion derivative, convection, reaction 정보는 해당 line의 1차원 operator를
결정한다. 즉 branch는 “이 line에서 어떤 1차원 differential operator를 풀 것인가”를 알려준다.

Trunk는 Green kernel을 평가할 두 coordinate의 관계를 표현한다. 하나는 solution을 알고 싶은
evaluation coordinate \(s\), 다른 하나는 source가 놓이는 coordinate \(r\)이다. Green kernel은
일반적으로 두 coordinate의 상대적 위치, boundary까지의 거리, coefficient variation에 따라 달라진다.
따라서 trunk는 \(G(s,r)\)의 pointwise structure를 담당한다.

Training 관점에서 중요한 점은 GreenNet이 Green kernel 값을 직접 외부 정답으로 맞추는 model이 아니라는
것이다. 핵심 objective는

\[
\int G(s,r)\,g(r)\,dr
\]

로 재구성한 solution이 reference solution과 일치하도록 만드는 것이다. 즉 GreenNet은
source-to-solution reconstruction을 통해 Green kernel을 학습한다.

## 4. Unit-Square CouplingNet 구조

Unit-square CouplingNet은 source-conditioned axial decomposition model이다. 같은 coefficient field가
주어져도 source \(f\)가 달라지면 필요한 split field \(\phi\), \(\psi\)도 달라진다. 따라서 source는
CouplingNet의 핵심 입력이다.

Source branch는 forcing profile의 전역적 형태를 encoding한다. 이 branch는 “현재 source가 어떤 방향으로
solution을 밀어내는가”를 나타내며, CouplingNet이 source-dependent split field를 만들 수 있게 한다.

Coefficient branch는 operator의 물리적 성격을 encoding한다. Diffusion은 smoothing과 energy geometry를,
convection은 방향성 있는 transport 효과를, reaction은 local damping 또는 amplification을 결정한다.
이 정보가 없으면 같은 source라도 어떤 operator 아래에서 어떤 split field가 적절한지 구분할 수 없다.

Trunk는 특정 spatial coordinate에서의 output variation을 표현한다. Branch가 source와 operator의
global context를 제공한다면, trunk는 “그 context 아래에서 지금 이 위치의 \(\phi\), \(\psi\) 값이
어떻게 달라져야 하는가”를 결정하는 역할을 한다.

Raw neural output은 일반적으로 PDE balance를 정확히 만족하지 않는다. 따라서 projection이 필요하다.
Projection은 output의 자유도를 사용해 다음 relation을 강제한다.

\[
\phi+\psi=f.
\]

가장 기본적인 symmetric projection은 residual

\[
r=f-\phi_{\mathrm{raw}}-\psi_{\mathrm{raw}}
\]

를 두 방향에 같은 비율로 나누어 더하는 방식으로 이해할 수 있다.

\[
\phi_{\mathrm{proj}}
=
\phi_{\mathrm{raw}}+\frac12 r,
\qquad
\psi_{\mathrm{proj}}
=
\psi_{\mathrm{raw}}+\frac12 r.
\]

이후 projected split field는 Green reconstruction의 source로 들어간다.

\[
u_\phi=\mathcal{G}_x[\phi_{\mathrm{proj}}],
\qquad
u_\psi=\mathcal{G}_y[\psi_{\mathrm{proj}}],
\qquad
u_{\mathrm{final}}
=
\frac12(u_\phi+u_\psi).
\]

Unit-square에서는 geometry가 단순하기 때문에 source branch, coefficient branch, trunk만으로도
대부분의 spatial context가 표현된다. Domain의 endpoint, line length, transverse location은 모든 line에서
동일하거나 규칙적으로 정렬되어 있으므로 별도의 geometry-aware branch가 필수적이지 않다.

## 5. Complex geometry에서 깨지는 unit-square 가정

Complex geometry에서는 axial line과 domain의 교차가 더 이상 항상 같은 interval이 아니다. 어떤 line은
긴 interval을 만들고, 어떤 line은 짧은 interval을 만들며, 어떤 line은 domain과 만나지 않을 수도 있다.
더 일반적으로는 하나의 axial line이 domain과 여러 개의 disconnected interval로 만날 수도 있다.

이 현상은 단순한 계산상의 불편이 아니라 수학적 표현의 변화이다. 하나의 connected interval은
Dirichlet boundary condition을 갖는 독립적인 1차원 boundary value problem으로 해석되어야 한다.
서로 떨어진 두 interval을 하나의 긴 interval처럼 다루면, 실제로는 domain 밖에 있는 구간을 통해
정보가 연결되는 잘못된 Green problem을 만들게 된다.

따라서 complex geometry에서는 다음 해석이 필요하다.

- Axial line 전체가 아니라, domain과 axial line이 만드는 connected interval이 기본 단위이다.
- 각 connected interval은 자기 endpoint와 length를 갖는다.
- 같은 direction의 interval이라도 physical length와 위치가 다르면 서로 다른 1차원 operator를 갖는다.
- Transverse location은 해당 interval이 2차원 domain 안에서 어디에 놓였는지를 나타내는 중요한 정보이다.
- Boundary는 global square boundary가 아니라 각 connected interval의 endpoint로 나타난다.

Unit-square에서는 line geometry가 거의 보이지 않는 배경 조건이었다. Complex geometry에서는 geometry가
operator와 reconstruction의 일부가 되므로, network architecture 안에서 명시적으로 표현되어야 한다.

## 6. Complex GreenNet 구조

Complex geometry에서 GreenNet은 각 connected axial interval을 독립적인 1차원 Green problem으로 본다.
Physical interval을

\[
I=[s_0,s_1],
\qquad
L=s_1-s_0>0
\]

라 하자. 이 interval의 local coordinate를 \(t\in[0,1]\)로 두면,

\[
s=s_0+Lt.
\]

Source coordinate도 같은 방식으로 \(\eta\in[0,1]\)로 표현한다.

\[
r=s_0+L\eta.
\]

Physical solution \(u(s)\)에 대응하는 normalized solution을

\[
v(t)=u(s_0+Lt)
\]

로 정의한다. 미분 연산은 coordinate transformation에 의해 scale이 바뀐다. Physical 1차원 operator

\[
-\frac{d}{ds}
\left(
a_{\mathrm{phys}}(s)\frac{du}{ds}
\right)
+ b_{\mathrm{phys}}(s)\frac{du}{ds}
+ c_{\mathrm{phys}}(s)u(s)
=
f_{\mathrm{phys}}(s)
\]

를 unit coordinate \(t\) 위의 equivalent problem으로 바꾸면 coefficient와 source는 다음 scaling을
따른다.

\[
a_{\mathrm{unit}}(t)
=
a_{\mathrm{phys}}(s_0+Lt),
\]

\[
a'_{\mathrm{unit}}(t)
=
L\,a'_{\mathrm{phys}}(s_0+Lt),
\]

\[
b_{\mathrm{unit}}(t)
=
L\,b_{\mathrm{phys}}(s_0+Lt),
\]

\[
c_{\mathrm{unit}}(t)
=
L^2\,c_{\mathrm{phys}}(s_0+Lt),
\]

\[
f_{\mathrm{unit}}(t)
=
L^2\,f_{\mathrm{phys}}(s_0+Lt).
\]

이 변환의 의미는 모든 connected interval을 같은 unit interval problem으로 바꾼다는 것이다.
GreenNet은 physical coordinate \((s,r)\)가 아니라 normalized coordinate \((t,\eta)\) 위에서
kernel을 학습한다.

\[
G_{\mathrm{unit}}(t,\eta).
\]

Unit-coordinate reconstruction은 다음과 같다.

\[
v(t)
=
\int_0^1
G_{\mathrm{unit}}(t,\eta)
f_{\mathrm{unit}}(\eta)
\,d\eta.
\]

여기서 추가적인 length factor를 다시 곱하지 않는다. Length effect는 이미
\(f_{\mathrm{unit}}\), \(b_{\mathrm{unit}}\), \(c_{\mathrm{unit}}\), 그리고
\(a'_{\mathrm{unit}}\)에 들어가 있다. Physical Green kernel을 별도로 해석할 때는
unit kernel과 physical kernel 사이의 관계를 따로 둘 수 있지만, normalized reconstruction 자체는
위의 unit integral로 닫힌다.

이 구조 덕분에 GreenNet은 interval의 physical 길이와 위치가 달라도 동일한 trunk coordinate domain에서
작동할 수 있다. Geometry 차이는 branch에 들어가는 scaled coefficient와 source를 통해 operator에 반영된다.

## 7. Complex CouplingNet 구조

Complex CouplingNet은 unit-square CouplingNet과 같은 목표를 갖는다. Source와 coefficient field를 보고
direction-split fields \(\phi\), \(\psi\)를 만든 뒤, projection과 Green reconstruction을 통해 solution을
얻는다. 그러나 complex geometry에서는 branch와 trunk의 역할 분해가 더 세밀해진다.

### Source branch

Source branch는 physical source \(f\)를 connected interval 위의 normalized physical forcing profile로
해석한다. Physical source를 unit coordinate에서 sampling한 뒤 amplitude를

\[
A
=
\left(
\int_0^1
|f_{\mathrm{phys}}(s_0+Lt)|^2\,dt
\right)^{1/2}
\]

로 정의하고, source branch에는 \(f_{\mathrm{phys}}/A\)를 넣는다. Model의 directional output은 다시
\(A\)로 scale되어 physical raw split field가 된다. 이는 CouplingNet이 source-dependent
split field를 만들기 위해 반드시 필요하다. 같은 geometry와 coefficient라도 source profile이 바뀌면
\(\phi\), \(\psi\)의 적절한 분해도 달라지기 때문이다.

Interval length effect는 source amplitude에 \(L^2\)를 미리 곱하는 방식이 아니라 geometry branch와
unit-scaled coefficient context를 통해 network에 제공된다. \(L^2\) source conversion은 physical symmetric
projection이 끝난 뒤 Green reconstruction source를 만들 때 적용된다.

### Coefficient branch

Coefficient branch는 local 1차원 operator의 성격을 encoding한다. Diffusion, convection, reaction은 각각
Green reconstruction과 split field의 형태를 다르게 만든다.

Complex geometry에서 coefficient는 interval-local coordinate 위에서 해석된다. 같은 2차원 coefficient field라도
어떤 axial interval을 따라 보느냐에 따라 다른 1차원 coefficient profile이 된다. 따라서 coefficient branch는
“이 connected interval에서 어떤 differential operator가 작동하는가”를 알려준다.

Convection의 경우에는 primary axial direction의 coefficient만으로는 충분하지 않다. Direction-split source는
한 축 방향 Green reconstruction에 들어가지만, 같은 physical point는 동시에 transverse axial interval 위에도
놓인다. Complex geometry에서는 이 transverse interval의 boundary와 length가 point마다 달라질 수 있으므로,
coefficient branch는 convection을

\[
b_{\mathrm{primary}},
\qquad
b_{\mathrm{transverse}}
\]

로 분리해 전달한다. \(\phi\)-path에서는 primary direction이 \(x\), transverse direction이 \(y\)이므로

\[
\phi\text{-path}:\quad [L_x b_x,\ L_x b_y],
\]

이고, \(\psi\)-path에서는 primary direction이 \(y\), transverse direction이 \(x\)이므로

\[
\psi\text{-path}:\quad [L_y b_y,\ L_y b_x]
\]

로 해석된다. 이 정보는 CouplingNet이 source를 두 directional component로 나누기 위한 operator context이다.
반면 Green reconstruction 자체는 각 connected interval의 primary 1차원 operator만 사용하므로, transverse
convection은 Green operator coefficient로 추가되지 않는다.

Source branch와 coefficient branch의 개념적 차이도 중요하다. Source branch는 문제 instance마다 달라지는
forcing profile을 나타낸다. Coefficient branch는 해당 PDE family와 spatial operator를 나타낸다. 두 branch가
곱해지거나 결합될 때, network는 “이 operator 아래에서 이 source가 만들 split field”를 표현하게 된다.

### Geometry branch

Complex geometry에서는 interval의 endpoint와 length가 operator 해석에 직접 영향을 준다. 같은 normalized
coordinate \(t\)라도 physical interval이 어디서 시작하고 얼마나 긴지에 따라 실제 위치와 differential scaling이
달라진다.

Geometry branch는 이런 정보를 담당한다. 수학적으로는 interval의 시작점, 끝점, 중간 위치, 길이, 그리고 길이에
따른 scale 정보를 통해 local problem이 physical domain 안에서 어떤 크기와 위치를 갖는지 알려준다.

이 branch가 없으면 network는 두 interval이 같은 normalized source와 coefficient profile을 갖는 경우, 그것들이
physical domain에서 서로 다른 위치와 길이에 놓였다는 사실을 구분하기 어렵다.

### Transverse branch

Axial interval에는 진행 방향 coordinate뿐 아니라 transverse location도 있다. Horizontal interval이라면
transverse direction은 vertical coordinate이고, vertical interval이라면 transverse direction은 horizontal
coordinate이다.

Transverse branch는 “이 interval이 다른 parallel interval들과 비교해 어디에 놓여 있는가”를 표현한다.
Unit-square에서는 transverse coordinate가 항상 같은 bounded reference range 안에 있으므로 이 정보가 비교적
단순했다. Complex geometry에서는 domain의 크기와 위치가 달라질 수 있으므로 transverse coordinate를 normalized
quantity로 보아야 한다. 이 branch는 global domain 안에서의 상대적 위치를 알려주는 역할을 한다.

### Local trunk

Complex geometry의 trunk는 physical coordinate 전체를 직접 표현하지 않는다. 대신 connected interval 내부의
local coordinate \(t\)에 따른 pointwise variation을 담당한다. 즉 trunk는 “현재 interval 안에서 endpoint로부터
얼마나 떨어진 위치인가”를 본다.

Physical 위치와 길이 정보는 geometry branch와 transverse branch가 담당하고, trunk는 local 1차원 coordinate의
변화를 담당한다. 이 분리는 중요하다. Physical coordinate를 trunk에 직접 넣으면, GreenNet의 unit-interval
normalization과 CouplingNet의 segment-local output 해석이 섞인다. 반대로 local trunk와 geometry-aware branch를
분리하면, 모든 interval에서 동일한 local coordinate structure를 사용하면서도 physical geometry 차이를 branch가
설명할 수 있다.

## 8. Projection과 reconstruction 비교

Unit-square와 complex geometry 모두 projection의 목적은 같다. CouplingNet이 만든 raw split field가 원래 PDE의
source balance를 만족하도록 조정하는 것이다.

\[
\phi+\psi=f.
\]

Projection은 neural model이 PDE constraint를 완전히 외워야 하는 부담을 줄이고, model output을 물리적으로
해석 가능한 split field로 바꾼다. 기본 symmetric projection은 raw prediction의 residual을 두 방향에 같은
비율로 분배한다.

\[
r=f-\phi_{\mathrm{raw}}-\psi_{\mathrm{raw}},
\]

\[
\phi_{\mathrm{proj}}
=
\phi_{\mathrm{raw}}+\frac12 r,
\qquad
\psi_{\mathrm{proj}}
=
\psi_{\mathrm{raw}}+\frac12 r.
\]

Complex geometry에서 CouplingNet의 두 raw channel은 처음부터 physical
\(\phi_{\mathrm{raw}},\psi_{\mathrm{raw}}\)를 나타낸다. PDE balance \(\phi+\psi=f\)는 physical
domain에서의 equation이므로 symmetric projection도 이 physical raw fields에 직접 적용한다. Equal-half
correction은 raw difference mode를 보존한다.

Projection이 끝난 뒤에만 connected interval Green reconstruction을 위한 unit source를 만든다.

\[
\Phi_{\mathrm{unit}}=L_x^2\phi_{\mathrm{proj}},
\qquad
\Psi_{\mathrm{unit}}=L_y^2\psi_{\mathrm{proj}}.
\]

Projection 이후에는 두 split field를 각각 Green reconstruction source로 본다.

\[
u_\phi
=
\mathcal{G}_x[\phi_{\mathrm{proj}}],
\qquad
u_\psi
=
\mathcal{G}_y[\psi_{\mathrm{proj}}].
\]

두 represented solution이 완전히 같다면 이상적인 axial decomposition이 달성된 것이다. 실제 학습에서는
두 solution의 차이를 줄이고, 그 평균이 target solution에 가까워지도록 model을 학습한다.

\[
u_{\mathrm{final}}
=
\frac12
\left(
u_\phi+u_\psi
\right).
\]

이 final solution은 단순히 두 neural output의 평균이 아니다. Projection을 거친 split source를 Green operator로
적분해 얻은 두 solution representation의 평균이다. 따라서 CouplingNet의 output space와 solution space는
Green reconstruction을 통해 연결된다.

## 9. 핵심 차이 요약

| 비교 항목 | Unit-square domain | Complex geometry |
| --- | --- | --- |
| Domain representation | 모든 axial line이 같은 reference interval 위에 놓인다. | Axial line과 domain의 각 connected intersection이 독립 interval이 된다. |
| GreenNet interval | Line geometry가 암묵적이며 길이 scaling이 필요하지 않다. | Physical interval을 unit interval로 바꾸고 coefficient와 source를 scaling한다. |
| GreenNet branch | Line-wise operator coefficient를 encoding한다. | Connected interval마다 scaled operator coefficient를 encoding한다. |
| GreenNet trunk | Green kernel의 evaluation coordinate와 source coordinate를 표현한다. | Normalized coordinate pair \((t,\eta)\)를 표현한다. |
| Source branch | Source profile이 split field를 결정하므로 source-conditioned decomposition을 가능하게 한다. | Physical source를 interval-local normalized forcing profile로 해석한다. |
| Coefficient branch | Operator family와 spatial coefficient variation을 알려준다. | Primary 1차원 operator와 transverse convection context를 함께 알려준다. |
| Geometry information | Domain geometry가 regular structure에 거의 암묵적으로 들어 있다. | Endpoint, length, scale, transverse position이 explicit mathematical context가 된다. |
| Trunk role | Spatial coordinate에 따른 output variation을 표현한다. | Local coordinate \(t\)에 따른 interval-internal variation을 표현한다. |
| Projection | Split field가 \(\phi+\psi=f\)를 만족하도록 조정한다. | Physical split field 위에서 \(\phi+\psi=f\)를 만족하도록 조정한다. |
| Reconstruction | Projected split source를 axial Green operator로 solution에 되돌린다. | 각 connected interval의 unit Green reconstruction을 통해 represented solution을 만든다. |

## 10. 설계 해석

Unit-square 구조는 domain geometry가 단순하다는 사실에 크게 의존한다. 모든 axial line이 동일한 reference
interval을 공유하므로, branch는 source와 coefficient를 보고 trunk는 coordinate variation을 보는 비교적 단순한
분해가 가능하다.

Complex geometry에서는 같은 구조를 그대로 사용할 수 없다. Axial decomposition의 기본 단위가 line 전체가 아니라
connected interval로 바뀌기 때문이다. Connected interval은 각자 endpoint, length, transverse location을 갖고,
그 정보는 1차원 operator와 Green reconstruction에 직접 영향을 준다.

따라서 complex geometry에서 GreenNet/CouplingNet은 다음 원칙을 따른다.

- GreenNet은 모든 connected interval을 unit interval로 정규화해 같은 coordinate domain에서 Green kernel을 학습한다.
- Length와 physical scale은 coefficient와 source scaling에 반영한다.
- CouplingNet은 source branch와 coefficient branch만으로는 부족하므로 geometry branch와 transverse branch를 추가로 사용한다.
- Trunk는 physical coordinate 전체가 아니라 local coordinate variation을 담당한다.
- Projection은 항상 PDE balance를 보장하기 위한 구조이며, complex geometry에서는 physical split field에 대해 적용된다.
- Final solution은 projected split source를 Green reconstruction으로 되돌린 뒤 얻는 두 represented solution의 평균이다.

이 관점에서 complex geometry 확장은 unit-square 구조를 단순히 더 복잡한 domain에 적용한 것이 아니다.
같은 axial Green decomposition 철학을 유지하되, domain representation이 바뀌면서 branch, trunk, projection,
reconstruction의 역할을 다시 분리한 구조라고 해석해야 한다.
