# Unit-Square Domain과 Complex Geometry에서의 GreenNet/CouplingNet 구조 비교

## 이 문서의 목적

이 문서는 기존 unit-square domain에서의 GreenNet/CouplingNet 구조와 새로 추가한
complex geometry 구조가 무엇이 다르고, 왜 그렇게 분리되어 구현되었는지를 정리한다.
핵심은 단순한 사용법이 아니라 model/data contract의 변화이다. 기존 square path는
`dataset.geometry_mode="unit_square"` 기본값으로 보존되고, complex path는
`dataset.geometry_mode="complex"`에서만 별도로 동작한다.

두 path가 공유하는 연구 목적은 같다. 2D elliptic PDE를 axial direction으로 나누어
1D Green kernel과 coupling field를 사용해 solution을 재구성한다. 그러나 unit-square는
모든 axial line이 동일한 길이와 동일한 grid 구조를 갖는 반면, complex geometry는 line마다
valid interval의 길이, endpoint, point count, 연결성이 달라진다. 이 차이가 GreenNet,
CouplingNet, loss, artifact 구조를 모두 바꾼다.

이 문서에서 unit-square 설명은 complex geometry와 비교하는 데 필요한 core path만 다룬다.
Unit-square에만 존재하는 auxiliary option이나 diagnostic path는 의도적으로 설명하지 않는다.

## 공통 문제 설정

프로젝트의 기본 PDE 관점은 다음 elliptic operator를 푸는 것이다.

```text
-d_x(a d_x u) - d_y(a d_y u) + b_x d_x u + b_y d_y u + c u = f
```

CouplingNet에서는 이 operator를 x-direction component와 y-direction component로 나누어
다음 두 field를 다룬다.

```text
phi = -d_x(a d_x u) + b_x d_x u + 0.5 c u
psi = -d_y(a d_y u) + b_y d_y u + 0.5 c u
phi + psi = f
```

GreenNet은 각 axial 1D operator의 Green kernel을 제공한다. CouplingNet은 source `f`와
operator coefficient 정보를 보고 `phi`, `psi`를 예측한다. 예측된 `phi`, `psi`는 GreenNet을
통해 각각 represented solution `u_phi`, `u_psi`로 재구성되고, 최종 solution prediction은
다음 convention을 사용한다.

```text
u_pred=0.5*(u_phi+u_psi)
```

Unit-square와 complex geometry의 가장 큰 차이는 이 공통 아이디어를 표현하는 data structure이다.
Unit-square에서는 모든 line을 하나의 rectangular tensor로 묶을 수 있다. Complex geometry에서는
그렇지 않으므로 connected segment를 명시적으로 관리해야 한다.

## Unit-Square Core 구조

### Geometry와 tensor contract

Unit-square path는 domain을 `[0, 1] x [0, 1]` regular grid로 본다. x-direction axial
line은 fixed `y` 위에서 `x in [0, 1]`을 움직이는 line이고, y-direction axial line은 fixed
`x` 위에서 `y in [0, 1]`을 움직이는 line이다.

이 구조에서는 모든 axial line이 다음 성질을 공유한다.

- 모든 x-line과 y-line은 같은 unit interval `[0, 1]`을 갖는다.
- 모든 line의 branch/trunk sample point 수가 같다.
- line endpoint는 항상 domain boundary에 대응한다.
- x-line 개수와 y-line 개수는 같은 regular grid convention 아래에서 정렬된다.
- tensor shape가 `(axis, line, point)` 형태로 안정적으로 유지된다.

`AxialDataset`은 이 regular structure를 그대로 사용한다. 좌표는
`coords: (2, n_lines, m_points, 2)`로 공유되고, solution/source/coefficient tensor는
sample batch가 붙은 `(B, 2, n_lines, m_points)` 구조를 갖는다. 첫 번째 axis index는
x-line family와 y-line family를 나타낸다.

### Unit-square GreenNet

Unit-square GreenNet은 각 axial line에 대한 1D source-to-solution reconstruction 문제를
학습한다. Synthetic sampler는 x-line과 y-line coefficient slice를 만들고, 각 line에서
1D source `f`와 solution `u`를 생성한다. GreenNet은 coefficient branch와 trunk coordinate를
입력으로 받아 Green kernel `G(x, xi)`를 예측하고, 다음 reconstruction이 sampled solution을
잘 맞추도록 학습한다.

```text
u_pred(x) = integral G(x, xi) f(xi) dxi
```

Unit-square에서는 line coordinate가 이미 unit coordinate이므로 별도의 length normalization이
필요하지 않다. 모든 interval length가 `1`이고, `x`와 `xi`는 같은 fixed grid 위에 있다.
Coefficient branch는 해당 line의 `a`, axis-derivative `a'`, directional convection `b`,
reaction `c` slice를 사용한다. x-line에서는 `a' = d_x a`, `b = b_x`이고, y-line에서는
`a' = d_y a`, `b = b_y`이다.

GreenNet의 중심 training objective는 Green kernel 자체의 pointwise supervision이 아니라
source를 Green kernel로 적분했을 때 solution을 재구성하는 것이다. Exact/reference kernel과의
비교 metric은 가능한 coefficient family에서 diagnostic으로 사용할 수 있지만, core training
contract는 source-to-solution reconstruction이다.

### Unit-square CouplingNet

Unit-square CouplingNet은 source-conditioned axial decomposition model이다. Regular axial
grid 위에서 source branch는 sampled `rhs` 또는 그 normalized version을 보고, coefficient branch는
operator coefficient를 보고, trunk는 interior coordinate를 본다. Model output은 x/y axial
flux-divergence field에 대응하는 `phi`, `psi`이다.

이 path에서 중요한 점은 CouplingNet이 단순히 coefficient만 보는 model이 아니라 source에 조건화된
model이라는 것이다. 같은 coefficient problem이라도 source `f`가 달라지면 필요한 `phi`, `psi`도
달라진다. 따라서 source branch는 CouplingNet의 core input이다.

예측된 `phi`, `psi`는 balance projection을 거쳐 `phi + psi = f` relation을 강하게 맞추는 방향으로
보정된다. 이후 GreenNet reconstruction을 통해 `u_phi`와 `u_psi`가 만들어지고, solution metric은
`u_pred=0.5*(u_phi+u_psi)`와 ground-truth solution을 비교한다.

Unit-square path가 이런 구조를 간단하게 유지할 수 있는 이유는 모든 line이 같은 좌표계, 같은
endpoint, 같은 tensor size를 공유하기 때문이다.

## Unit-Square 가정이 Complex Geometry에서 깨지는 지점

Complex geometry에서는 unit-square의 rectangular tensor assumption이 더 이상 성립하지 않는다.
원형 domain만 생각해도 fixed `y` chord의 x-interval 길이는 `y`에 따라 달라지고, fixed `x` chord의
y-interval 길이도 `x`에 따라 달라진다. 더 일반적인 disconnected domain에서는 같은 fixed coordinate
위에 서로 떨어진 여러 connected interval이 생길 수 있다.

구체적으로 다음 문제가 생긴다.

- Axial line마다 physical endpoint가 다르다.
- Segment length `L`이 line마다 다르다.
- Some fixed-coordinate line은 valid interior point가 없을 수 있다.
- 같은 fixed coordinate 위에서도 disconnected connected component가 여러 개일 수 있다.
- x-direction segment 수 `Sx`와 y-direction segment 수 `Sy`가 같다는 보장이 없다.
- Full Cartesian grid에는 outside-domain point가 포함된다.
- Boundary endpoint는 Dirichlet hard-zero value로 필요하지만, CouplingNet이 평가할 valid point는 아니다.
- Valid point order와 full-grid `[row=y, col=x]` order를 명시적으로 연결해야 한다.

이 조건에서 unit-square tensor를 억지로 유지하려면 padding, mask, artificial line id가 필요해진다.
그 방식은 구현을 복잡하게 만들 뿐 아니라, connected interval을 독립 1D domain으로 보는 GreenNet
해석을 흐릴 수 있다. 그래서 complex path는 square tensor를 일반화하는 대신 connected segment
contract를 별도로 둔다.

## Complex Geometry 입력 Contract

Complex path의 geometry 정보는 precomputed geometry `.npz`로 들어온다. 이 파일은 model이 학습 중에
geometry를 추출하지 않도록, connected segment와 valid point mapping을 모두 저장한다.

중요한 geometry field는 다음 역할을 갖는다.

- `coords_valid`: 실제 domain 내부 valid point 좌표 `(P, 2)`.
- `valid_grid_y_index`, `valid_grid_x_index`: full-grid sample array에서 valid point를 gather하기 위한 index.
- `x_segment_id`, `y_segment_id`: 각 valid point가 어느 x/y connected segment에 속하는지 나타내는 id.
- `x_local_t`, `y_local_t`: 각 valid point의 segment-local unit coordinate.
- `x_segment_left`, `x_segment_right`, `x_segment_y`, `x_segment_length`: x-direction segment의 physical endpoint, fixed coordinate, length.
- `y_segment_bottom`, `y_segment_top`, `y_segment_x`, `y_segment_length`: y-direction segment의 physical endpoint, fixed coordinate, length.
- `x_recon_ptr`, `x_recon_t`, `x_recon_weight`, `x_recon_valid_index`: x-direction reconstruction용 CSR-like node arrays.
- `y_recon_ptr`, `y_recon_t`, `y_recon_weight`, `y_recon_valid_index`: y-direction reconstruction용 CSR-like node arrays.
- `x_edges`, `y_edges`: same-segment valid-edge energy loss를 위한 valid point edge list.
- `hx`, `hy`: physical grid spacing.

Sample `.npz`는 full-grid array를 저장한다. Complex CouplingNet dataset은 `rhs`, `sol`을 필수로
읽고, optional target으로 `phi`, `psi`를 읽을 수 있다. 모든 full-grid array는 `[row=y, col=x]`
convention을 사용하며, 실제 model input은 geometry의 valid index를 통해 `coords_valid` order로
gather된다.

## Complex GreenNet 구조

### Flat connected interval list

Complex GreenNet은 geometry `.npz`의 모든 connected x/y segment를 하나의 flat list로 펼친다.
이때 interval 개수는 다음과 같다.

```text
N = Sx + Sy
```

여기서 `Sx`는 x-direction connected segment 수이고, `Sy`는 y-direction connected segment 수이다.
이 design은 `Sx != Sy`를 자연스럽게 허용한다. 같은 fixed coordinate 위에 disconnected segment가
여러 개 있어도 각각 독립 interval로 남긴다.

x-segment는 physical coordinate가

```text
x_phys = left + L t
y_phys = fixed
```

이고, y-segment는

```text
x_phys = fixed
y_phys = bottom + L t
```

이다. 여기서 `t in [0, 1]`은 segment-local unit coordinate이고, `L`은 해당 connected segment의
physical length이다.

### Unit interval normalization

Complex GreenNet은 physical interval에서 직접 Green kernel을 학습하지 않는다. 모든 connected
segment를 unit interval로 mapping한 뒤 `G_unit(t, eta)`를 학습한다. 이때 coefficient와 source는
operator scaling을 반영해 다음 convention으로 변환된다.

```text
a_unit=a_phys
ap_unit=L*ap_phys
b_unit=L*b_phys
c_unit=L^2*c_phys
f_unit=L^2*f_phys
```

`ap_phys`와 `b_phys`는 axis-specific 값이다. x-segment에서는 `ap_phys=d_x a`,
`b_phys=b_x`이고, y-segment에서는 `ap_phys=d_y a`, `b_phys=b_y`이다.

이 normalization의 목적은 모든 segment를 동일한 unit-domain 1D Green problem으로 보이게 만드는
것이다. Physical length effect는 coordinate가 아니라 `ap_unit`, `b_unit`, `c_unit`, `f_unit`에
들어간다. 따라서 GreenNet trunk coordinate는 항상 unit pair `(t, eta) in [0, 1]^2`이고, model query도
`forward_pairs(...)`를 통해 arbitrary unit-coordinate pair에 대해 수행된다.

### Reconstruction convention

Complex GreenNet training과 Coupling reconstruction 모두 unit reconstruction에서는 추가 length factor를
곱하지 않는다.

```text
u(t) = integral_0^1 G_unit(t, eta) f_unit(eta) d eta
```

Physical Green kernel을 해석 목적으로 내보낼 때는 `G_phys = L * G_unit` 관계를 사용할 수 있지만,
training loss와 CouplingNet reconstruction path에서는 `G_unit`과 `f_unit`의 unit integral convention이
기준이다.

## Complex CouplingNet 구조

Complex CouplingNet은 unit-square CouplingNet의 핵심 의미, 즉 source-conditioned axial decomposition을
complex geometry에 맞게 다시 구현한 path이다. Sample schema와 GreenNet checkpoint contract는
바꾸지 않고, dataset/model/reconstruction 내부에서 connected segment contract를 사용한다.

### Full-grid sample에서 valid point로 gather

FEniCSx sample generator나 다른 numerical solver가 만든 sample `.npz`는 full-grid `rhs`, `sol`,
optional `phi`, `psi`를 저장한다. Complex CouplingNet은 full-grid array 전체를 model에 직접 넣지 않는다.
먼저 geometry의 `valid_grid_y_index`, `valid_grid_x_index`를 사용해 valid point order로 값을 gather한다.

이렇게 얻은 `rhs_valid`, `sol_valid`는 shape `(P,)`이고, batch에서는 `(B, P)`가 된다. `P`는
`coords_valid`의 point count이다.

### Source branch

Complex CouplingNet의 source branch는 mandatory input이다. `rhs_valid`는 x/y segment별로 다음 과정을
거쳐 branch tensor가 된다.

1. Reconstruction arrays의 segment node를 따라 endpoint와 interior valid point를 정렬한다.
2. `valid_index == -1`인 endpoint source value는 hard-zero로 둔다.
3. Interior node는 `rhs_valid`에서 값을 가져온다.
4. Segment-local node values를 fixed unit branch grid `linspace(0, 1, M)`으로 linear interpolation한다.
5. Physical source를 unit source로 변환한다.

```text
f_unit=L^2*f_phys
```

6. Unit interval에서 source L2 norm을 계산하고, branch input은 normalized source로 넣는다.
7. Model output은 segment별 source norm으로 다시 scale된다.

Source branch에 interpolation이 필요한 이유는 sample source가 full-grid valid point에서만 주어지기
때문이다. CouplingNet branch input은 fixed length `M`의 unit grid를 기대하므로, 각 segment의 irregular
valid/reconstruction node 값을 branch grid로 옮겨야 한다.

### Coefficient branch

Coefficient branch는 source branch와 다르게 sample에서 interpolation하지 않는다. Coefficient는
`a_fun`, `apx_fun`, `apy_fun`, `bx_fun`, `by_fun`, `c_fun`으로 주어진 function이므로, 필요한 unit branch
grid point에서 직접 evaluate할 수 있다.

Complex CouplingNet coefficient branch는 `coefficient_terms`에 따라 `[a,b,c]` 순서로 구성된다.
Diffusion term이 켜지면 `a_unit`, convection term이 켜지면 `b_unit`, reaction term이 켜지면
`c_unit`이 들어간다. 모든 term이 꺼져 있으면 coefficient branch 없이 source-only branch path가 된다.

중요한 구분은 `a'`의 역할이다. `a'`는 GreenNet reconstruction query에는 필요하므로
`x_green_branch`, `y_green_branch`에 보관된다. 하지만 CouplingNet coefficient branch에는 넣지 않는다.
CouplingNet coefficient branch는 unit-square의 generic coefficient branch 의미에 맞춰 `[a,b,c]`만
제어한다.

### Geometry branch, transverse branch, local trunk

Complex CouplingNet은 point prediction을 위해 네 종류의 branch/trunk 정보를 결합한다.

- `source branch`: segment-local normalized `f_unit`.
- `coefficient branch`: 선택된 `[a,b,c]` coefficient samples.
- `geometry branch`: `[s_left, s_right, s_mid, L, L^2, 1/L]`.
- `transverse branch`: globally normalized transverse coordinate `r_hat`의 Fourier features.
- `trunk`: valid point의 segment-local coordinate `t`.

x-segment에서 transverse coordinate는 fixed `y`이고, y-segment에서 transverse coordinate는 fixed `x`이다.
`r_hat`은 geometry global extent로 normalize한다. x-segment의 fixed `y`는 `grid_y` extent를 우선 사용하고,
y-segment의 fixed `x`는 `grid_x` extent를 우선 사용한다. Grid metadata가 없으면 `coords_valid` extent를
fallback으로 사용한다. 이 normalization은 circular domain처럼 transverse coordinate가 `[0, 1]`에 있지
않은 경우에도 stable feature scale을 제공하기 위한 것이다.

Trunk는 항상 local 1D coordinate `t`만 본다. Physical coordinate 자체를 trunk에 넣지 않는 이유는
GreenNet과 CouplingNet의 segment-local unit interval contract를 일관되게 유지하기 위해서다. Physical
위치와 length 정보는 geometry/transverse branch가 담당한다.

### Raw unit output, physical projection, unit reconstruction

Complex CouplingNet model output은 raw unit quantity `(B, 2, P)`이다. 첫 번째 channel은 x-direction
`Phi_raw`, 두 번째 channel은 y-direction `Psi_raw`에 대응한다. 이 raw unit output은 source norm으로
scale된 뒤 segment length를 사용해 physical `phi`, `psi`로 변환된다.

Projection은 physical variable에서 적용된다. 현재 complex mode는 hard symmetric projection만 사용한다.

```text
residual = rhs - phi_raw - psi_raw
phi = phi_raw + 0.5 residual
psi = psi_raw + 0.5 residual
```

이 projection 이후 `phi + psi = rhs`가 valid point에서 강하게 맞춰진다. Projected physical `phi`, `psi`는
다시 unit quantity로 변환되어 GreenNet reconstruction에 들어간다. Reconstruction은 segment별
precomputed nonuniform trapezoid weight를 사용한다. Endpoint는 hard-zero node로 포함되지만 CouplingNet을
endpoint에서 평가하지 않는다.

결과적으로 complex path에서 저장/평가되는 represented solutions는 다음과 같다.

```text
u_phi: projected x-direction source를 GreenNet으로 재구성한 solution
u_psi: projected y-direction source를 GreenNet으로 재구성한 solution
u_pred=0.5*(u_phi+u_psi)
```

## Loss, Evaluation, Artifact 차이

### Unit-square path

Unit-square CouplingNet은 regular grid 위에서 represented solution consistency와 energy consistency를
계산한다. Evaluation은 full square grid와 axial tensor convention을 그대로 사용할 수 있고, artifact도
grid heatmap 중심으로 구성할 수 있다.

### Complex path

Complex path는 valid-point geometry만 신뢰한다. 따라서 metrics, logs, artifacts는 complex-safe field만
남긴다. Training loss의 중심은 same-segment valid edge를 사용하는 physical energy consistency이다.
이 loss는 `x_edges`, `y_edges`, `hx`, `hy`, valid point coefficient `a_valid`를 사용한다. Face coefficient는
edge 양끝의 arithmetic average로 계산한다.

Evaluation metric은 다음처럼 valid point order를 기준으로 한다.

- `loss`: training objective aggregate.
- `loss_energy_consistency`: physical edge energy consistency.
- `rel_sol`: `u_pred=0.5*(u_phi+u_psi)`와 `sol`의 valid-point relative error.
- `rel_flux`: optional target `phi`, `psi`가 sample에 있을 때만 계산되는 projected physical flux-divergence error.

Complex artifact도 `coords_valid` 위의 valid-point scatter가 기본이다. Selected sample archive와 figure는
다음 해석을 따른다.

- input/target: `rhs`, `sol`.
- solution prediction: `u_pred`, `u_phi`, `u_psi`.
- signed solution error: `u_pred - sol`, `u_phi - sol`, `u_psi - sol`.
- split mismatch: `u_phi - u_psi`.
- flux-divergence: projected physical `phi`, `psi`.
- optional target이 있을 때: `target_phi`, `target_psi`, `phi - target_phi`, `psi - target_psi`.

Full-grid outside-domain value는 sample storage에서는 존재할 수 있지만, complex model/evaluation/artifact의
의미 있는 좌표는 `coords_valid`이다.

## Side-by-Side 요약

| 항목 | Unit-square core path | Complex geometry path | 설계 이유 |
| --- | --- | --- | --- |
| Domain 표현 | `[0,1]^2` regular square grid | precomputed geometry `.npz`의 valid points와 connected segments | complex domain은 rectangular tensor로 정확히 표현되지 않음 |
| Axial domain | 모든 line이 unit interval | segment마다 physical endpoint와 length가 다르고 unit interval로 mapping | length effect를 coefficient/source scaling에 넣기 위함 |
| Line 구조 | `(2, n_lines, m_points)` tensor | flat interval list `N = Sx + Sy` | `Sx != Sy`, disconnected interval 허용 |
| GreenNet input | line-wise coefficient/source tensors | connected segment별 unit coefficient/source tensors | 모든 segment를 독립 1D problem으로 처리 |
| GreenNet trunk | unit-square line coordinate pair | unit pair `(t, eta)` | physical trunk coordinate 제거, unit normalization 유지 |
| Scaling | interval length가 항상 1 | `a_unit=a_phys`, `ap_unit=L*ap_phys`, `b_unit=L*b_phys`, `c_unit=L^2*c_phys`, `f_unit=L^2*f_phys` | physical interval operator를 unit operator로 변환 |
| Coupling source | regular axial source branch | full-grid `rhs` gather 후 segment-local source branch | sample source는 valid point에서 주어지므로 branch grid interpolation 필요 |
| Coupling coefficient | coefficient function slices | coefficient function을 branch grid에서 직접 evaluate | coefficient는 function이므로 sample interpolation 불필요 |
| Geometry 정보 | regular grid 자체에 암묵적으로 포함 | geometry branch와 transverse branch로 명시 입력 | endpoint, length, transverse 위치가 segment마다 다름 |
| Trunk | square coordinate 또는 axis-local coordinate | segment-local `t` | physical 위치는 branch 쪽에서 처리 |
| Projection | balance relation을 square grid에서 적용 | physical `phi`, `psi`에 hard symmetric projection 적용 | `phi + psi = rhs`를 valid point에서 보장 |
| Reconstruction | regular line quadrature | segment별 nonuniform trapezoid weights | irregular valid/reconstruction node 처리 |
| Artifact | square grid heatmap 중심 | `coords_valid` scatter 중심 | outside-domain point를 시각적 의미에서 제외 |

## 왜 별도 Complex Path로 구현했는가

Complex geometry를 unit-square path 안에 옵션으로 억지로 넣지 않은 이유는 명확하다. Unit-square path의
장점은 모든 line이 같은 shape를 갖는다는 점이고, complex geometry의 본질은 그 shape가 깨진다는 점이다.
따라서 기존 tensor contract를 padding과 mask로 확장하면 model interface는 겉으로 유지되더라도 다음
문제가 생긴다.

- disconnected interval을 같은 line으로 합치는 오류가 생길 수 있다.
- physical endpoint와 hard-zero endpoint를 valid prediction point와 혼동할 수 있다.
- segment length scaling이 data shape 뒤에 숨겨진다.
- x/y segment 수가 다를 때 불필요한 dummy row가 필요해진다.
- artifact와 metric이 outside-domain point를 포함하는 것처럼 보일 수 있다.

별도 complex path는 이런 문제를 피한다. Geometry `.npz`가 connected segment와 valid point mapping을
명시하고, GreenNet은 `N = Sx + Sy` flat interval을 학습하며, CouplingNet은 source/coefficient/geometry/
transverse/local trunk 정보를 분리해 사용한다. 이 구조는 unit-square model의 핵심 의미를 유지하면서도
general complex geometry를 표현할 수 있다.

## 읽을 때 주의할 점

첫째, complex GreenNet은 full 2D Green function을 직접 학습하는 것이 아니다. Unit-square path와 마찬가지로
axial decomposition에서 생기는 1D Green kernel을 segment별로 학습한다.

둘째, complex CouplingNet의 source branch와 coefficient branch는 입력 생성 방식이 다르다. Source는 sample
array에서 오므로 segment-local branch grid로 interpolation해야 한다. Coefficient는 callable function이므로
branch grid에서 직접 evaluate한다.

셋째, complex reconstruction에서 unit Green kernel에 추가 length factor를 곱하지 않는다. Length effect는
이미 unit coefficient/source scaling에 들어가 있다.

넷째, complex artifact의 기본 좌표계는 `coords_valid`이다. Full-grid sample array는 저장과 gather를 위한
container이고, model output과 metric의 의미 있는 support는 valid point set이다.

