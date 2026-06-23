# GreenNet / CouplingNet 학회 발표 준비 문서

이 문서는 학회 발표 슬라이드를 직접 대체하는 자료가 아니라, 발표자료를 만들기 전에
내용, 수식, 실험 설정, figure 후보, 발표 메시지를 정리하기 위한 준비 문서이다.
발표의 중심 대상은 현재 repo의 GreenNet과 CouplingNet이며, 설명은 실제 baseline에서
사용하는 설정을 중심으로 구성한다.

## 1. 발표 목적과 핵심 메시지

### 1.1 발표 목적

이 발표의 목적은 2D elliptic PDE solution reconstruction 문제를 axial Green function
관점에서 다루는 두 단계 모델을 설명하는 것이다.

첫 번째 단계인 GreenNet은 2D domain 전체의 Green's function을 직접 학습하는 것이
아니라, axial decomposition에서 생기는 x-line/y-line 방향의 1D Green kernel을 학습한다.
두 번째 단계인 CouplingNet은 학습된 axial Green response를 이용해 2D solution과
flux-divergence decomposition을 일관되게 재구성한다.

발표의 핵심 구조는 다음과 같다.

```text
Fixed PDE coefficient family
        |
        v
Line-wise coefficient slices
        |
        v
GreenNet: axial 1D Green kernel learning
        |
        v
CouplingNet: 2D solution / flux-divergence coupling reconstruction
```

### 1.2 발표에서 강조할 핵심 메시지

- GreenNet은 고정된 coefficient problem에서 각 axial line이 보는 1D operator의
  Green kernel을 학습한다.
- 같은 2D coefficient field 안에서도 line 위치가 달라지면 1D coefficient slice가
  달라지고, 따라서 axial Green kernel도 line별로 달라진다.
- CouplingNet은 GreenNet이 제공하는 axial response를 기반으로 x-direction solution
  representation과 y-direction solution representation을 결합한다.
- 현재 baseline CouplingNet은 shared axis-1D trunk, boundary-aware transverse encoding,
  sine smooth-mask balance projection을 사용한다.
- 결과 해석은 Green kernel 자체의 정확도, source-to-solution reconstruction, flux-divergence
  reconstruction, balance residual을 분리해서 보아야 한다.

### 1.3 발표에서 피해야 할 혼동

- GreenNet은 하나의 global 2D Green function을 직접 출력하는 모델이 아니다.
- CouplingNet의 `phi`, `psi`는 단순한 auxiliary scalar가 아니라 axial flux-divergence
  decomposition을 나타내는 물리적 의미의 field이다.
- 현재 발표 baseline에서는 `source_stencil_lift`, `green_response_feature`,
  `trunk_positional_encoding`, `balance_loss`, `symmetric_boundary_loss`를 사용하지 않는다.
- Error heatmap은 절대값이 아니라 signed difference를 기준으로 해석한다.

## 2. 문제 설정과 PDE family

### 2.1 기본 PDE 형식

발표에서 다루는 문제는 Dirichlet boundary condition을 갖는 2D elliptic PDE이다.
repo의 coefficient convention에 맞추면 diffusion, convection, reaction 항을 포함한
일반 형식은 다음과 같이 설명할 수 있다.

$$
\mathcal{L}u(x,y) = f(x,y), \qquad (x,y) \in \Omega = [0,1]^2
$$

$$
u(x,y) = 0, \qquad (x,y) \in \partial \Omega
$$

여기서 operator는 diffusion coefficient \(a(x,y)\), convection coefficient
\((b_x(x,y), b_y(x,y))\), reaction coefficient \(c(x,y)\)를 사용한다.
구현에서는 coefficient file이 다음 callable을 제공한다.

| 함수 | 의미 | 발표에서의 해석 |
|---|---|---|
| `a_fun(x, y)` | diffusion coefficient | 공간적으로 변하는 확산 강도 |
| `apx_fun(x, y)` | \(\partial_x a(x,y)\) | x-direction axial operator에 필요한 diffusion derivative |
| `apy_fun(x, y)` | \(\partial_y a(x,y)\) | y-direction axial operator에 필요한 diffusion derivative |
| `bx_fun(x, y)` | x-direction convection | x-line operator의 convection coefficient |
| `by_fun(x, y)` | y-direction convection | y-line operator의 convection coefficient |
| `c_fun(x, y)` | reaction coefficient | local reaction/source damping 항 |

내부 tensor convention은 `b_vals[0] = b_x`, `b_vals[1] = b_y`이다.
즉 x-direction line에서는 `bx_fun`, y-direction line에서는 `by_fun`을 사용한다.

### 2.2 발표 기준 coefficient family

발표 준비 문서에서는 사용자가 지정한 6개 coefficient family만 기준으로 삼는다.
첨부 config의 `Divergence_Free_Convection_Diffusion2.py`는 이번 발표 준비 문서의
기준 family에서 제외한다.

| 파일 | diffusion \(a(x,y)\) | convection | reaction | 발표에서의 역할 |
|---|---|---|---|---|
| `Pure_Poisson.py` | \(1\) | 없음 | 없음 | 가장 단순한 Poisson baseline |
| `Sinusoidal_Diffusion_Only.py` | \(1 + 0.5\sin(2\pi x)\sin(4\pi y)\) | 없음 | 없음 | 비등방적 주파수의 variable diffusion-only 문제 |
| `Sinusoidal_Diffusion_Only_Ver2.py` | \(1 + 0.5\sin(2\pi x)\sin(2\pi y)\) | 없음 | 없음 | 더 대칭적인 variable diffusion-only 문제 |
| `Smooth_Variable_Diffusion_Reaction.py` | \(1 + 0.5\sin(2\pi x)\sin(2\pi y)\) | 없음 | 있음 | smooth diffusion + reaction 문제 |
| `Diffusion_Reaction_Ver2.py` | \(1 + 0.5\sin(2\pi x)\sin(4\pi y)\) | 없음 | 있음 | 비등방적 diffusion + reaction 문제 |
| `Convection_Diffusion_Reaction.py` | \(1 + 0.5\sin(2\pi x)\sin(2\pi y)\) | 있음 | 있음 | diffusion, convection, reaction을 모두 포함하는 가장 복합적인 문제 |

### 2.3 coefficient 수식

#### Pure Poisson

```text
a(x,y) = 1
apx(x,y) = 0
apy(x,y) = 0
bx(x,y) = 0
by(x,y) = 0
c(x,y) = 0
```

이 문제는 GreenNet/CouplingNet의 기본 sanity check 역할을 한다. Coefficient가 상수이기
때문에 line별 operator 변화가 없고, axial Green function 구조를 가장 단순하게 설명할 수 있다.

#### Sinusoidal Diffusion Only

```text
a(x,y) = 1 + 0.5 sin(2*pi*x) sin(4*pi*y)
apx(x,y) = pi cos(2*pi*x) sin(4*pi*y)
apy(x,y) = 2*pi sin(2*pi*x) cos(4*pi*y)
bx(x,y) = 0
by(x,y) = 0
c(x,y) = 0
```

이 문제는 reaction과 convection 없이 diffusion coefficient만 공간적으로 변하는 경우이다.
y 방향에 \(4\pi y\)가 들어가므로 y-coordinate에 대한 variation이 더 빠르다.
발표에서는 “variable coefficient Green kernel learning”의 대표 예시로 사용할 수 있다.

#### Sinusoidal Diffusion Only Ver2

```text
a(x,y) = 1 + 0.5 sin(2*pi*x) sin(2*pi*y)
apx(x,y) = pi cos(2*pi*x) sin(2*pi*y)
apy(x,y) = pi sin(2*pi*x) cos(2*pi*y)
bx(x,y) = 0
by(x,y) = 0
c(x,y) = 0
```

이 문제는 x/y 방향에서 같은 frequency를 갖는 variable diffusion-only 문제이다.
첫 번째 diffusion-only 문제와 비교하면 coefficient variation의 symmetry가 더 강하다.

#### Smooth Variable Diffusion Reaction

```text
a(x,y) = 1 + 0.5 sin(2*pi*x) sin(2*pi*y)
apx(x,y) = pi cos(2*pi*x) sin(2*pi*y)
apy(x,y) = pi sin(2*pi*x) cos(2*pi*y)
bx(x,y) = 0
by(x,y) = 0
c(x,y) = 0.5 * (1 + 0.5 cos(2*pi*x) cos(2*pi*y))
```

이 문제는 variable diffusion에 smooth reaction 항을 추가한다. GreenNet의 exact/reference
`rel_green` 해석은 diffusion-only 문제보다 제한적이므로, 발표에서는 reconstruction 중심으로
해석하는 것이 자연스럽다.

#### Diffusion Reaction Ver2

```text
a(x,y) = 1 + 0.5 sin(2*pi*x) sin(4*pi*y)
apx(x,y) = pi cos(2*pi*x) sin(4*pi*y)
apy(x,y) = 2*pi sin(2*pi*x) cos(4*pi*y)
bx(x,y) = 0
by(x,y) = 0
c(x,y) = 0.5 * (1 + 0.5 cos(2*pi*x) cos(2*pi*y))
```

이 문제는 `Sinusoidal_Diffusion_Only.py`의 비등방적 diffusion variation에 reaction을
추가한 경우이다. Diffusion-only와 reaction 포함 문제의 차이를 비교하기 좋다.

#### Convection Diffusion Reaction

```text
a(x,y) = 1 + 0.5 sin(2*pi*x) sin(2*pi*y)
apx(x,y) = pi cos(2*pi*x) sin(2*pi*y)
apy(x,y) = pi sin(2*pi*x) cos(2*pi*y)
bx(x,y) = 0.25 sin(pi*x) sin(pi*y)
by(x,y) = -0.25 sin(pi*x) sin(pi*y)
c(x,y) = 0.5 * (1 + 0.5 cos(2*pi*x) cos(2*pi*y))
```

이 문제는 diffusion, directional convection, reaction을 모두 포함한다. 발표에서는
모델이 coefficient branch를 통해 세 종류의 operator 정보를 모두 받아야 하는 가장 복합적인
problem family로 설명할 수 있다.

## 3. Axial Green Function Method 관점

### 3.1 왜 axial decomposition인가

2D PDE를 직접 full 2D Green function으로 다루면 Green kernel은 \((x,y,\xi,\eta)\)에
의존하는 고차원 object가 된다. Axial Green function 관점은 이 문제를 x-line과 y-line의
1D operator 문제로 나누어, 각 line에서 1D Green kernel을 학습하거나 계산하는 방식이다.

예를 들어 y를 고정한 x-line에서는 다음과 같은 line operator를 생각한다.

$$
\mathcal{L}_x u(x; y_0) = f(x; y_0)
$$

반대로 x를 고정한 y-line에서는 다음과 같은 line operator를 생각한다.

$$
\mathcal{L}_y u(y; x_0) = f(y; x_0)
$$

coefficient가 2D field로 고정되어 있더라도, 각 line에서 보이는 coefficient slice는
line 위치에 따라 달라진다.

```text
2D coefficient field a(x,y)
        |
        +-- x-line at y=y0: a(x, y0)
        +-- x-line at y=y1: a(x, y1)
        +-- y-line at x=x0: a(x0, y)
        +-- y-line at x=x1: a(x1, y)
```

따라서 “한 문제에 하나의 Green function”이라기보다, “한 coefficient problem 안에 여러
axial line Green kernels가 존재한다”고 설명하는 것이 정확하다.

### 3.2 Green kernel convention

repo에서 사용하는 Green kernel matrix convention은 다음과 같다.

```text
G[row = evaluation coordinate x, col = source coordinate xi]
```

solution reconstruction은 source coordinate \(\xi\) 방향으로 적분한다.

$$
u(x) \approx \int_0^1 G(x,\xi) f(\xi)\,d\xi
$$

discrete grid에서는 마지막 dimension인 source coordinate 방향으로 trapezoid rule을 적용한다.
이 convention은 GreenNet artifact와 CouplingNet reconstruction 설명에서 일관되게 유지해야 한다.

### 3.3 발표 메시지

- Axial decomposition은 full 2D Green function 학습의 차원을 낮춘다.
- Coefficient가 line별로 달라지기 때문에 operator learning이 필요하다.
- GreenNet은 line coefficient와 source/evaluation coordinate를 이용해 line Green kernel을
  학습한다.
- CouplingNet은 x-line response와 y-line response를 결합해 2D consistency를 맞춘다.

## 4. GreenNet 설정과 역할

### 4.1 GreenNet의 역할

GreenNet은 각 axial line에서 source-to-solution map을 제공하는 1D Green kernel을 학습한다.
입력은 line-local coefficient 정보와 trunk coordinate \((x,\xi)\)이며, 출력은 line Green
kernel 값이다.

발표에서는 GreenNet을 다음처럼 설명할 수 있다.

```text
Line coefficient slice + coordinate pair (x, xi)
        |
        v
GreenNet
        |
        v
Predicted axial Green kernel G(x, xi)
        |
        v
u(x) = integral G(x, xi) f(xi) dxi
```

### 4.2 GreenNet baseline 설정

| 항목 | 값 | 발표에서의 의미 |
|---|---:|---|
| `n_points_per_line` | 129 | 각 1D axial line의 grid resolution |
| `step_size` | 0.0078125 | \(1/(129-1)\), uniform grid spacing |
| `samples_per_line` | 25 | training source samples per line |
| `validation_samples_per_line` | 5 | validation source samples per line |
| `sampler_mode` | `backward` | source/solution sample generation mode |
| `validation_sampler_mode` | `backward` | validation도 같은 sampler family 사용 |
| `scale_length` | `[0.05, 0.25]` | source field의 length-scale 범위 |
| `use_operator_learning` | true | coefficient-dependent operator learning |
| `dtype` | `float64` | 수치 정확도 우선 설정 |
| `integration_rule` | `trapezoid` | Green response 적분 규칙 |

### 4.3 GreenNet architecture 설정

| 항목 | 값 | 설명 |
|---|---:|---|
| `input_dim` | 2 | trunk input coordinate \((x,\xi)\) |
| `hidden_dim` | 128 | MLP hidden width |
| `depth` | 4 | MLP depth |
| `activation` | `rational` | rational activation 사용 |
| `use_green` | true | analytic Green wrapping/correction 구조 사용 |
| `branch_input_dim` | 129 | line coefficient/source branch input length |
| `use_fourier` | true | Fourier coordinate feature 사용 |
| `fourier_dim` | 32 | Fourier feature dimension |
| `dropout` | 0.0 | dropout 없음 |

### 4.4 GreenNet training 설정

| 항목 | 값 | 설명 |
|---|---:|---|
| `learning_rate` | 0.0005 | Adam 단계 learning rate |
| `epochs` | 4000 | Adam training epochs |
| `batch_size` | 1500 | Green kernel training batch size |
| `compute_validation_rel_sol` | true | validation reconstruction metric 계산 |
| `lbfgs_epochs` | 100 | Adam 이후 LBFGS fine-tuning epoch |
| `lbfgs_max_iter` | 1000 | LBFGS 내부 반복 |
| `lbfgs_history_size` | 200 | LBFGS history size |
| `compile.enabled` | true | `torch.compile` 사용 |
| `device` | `cuda:1` | GPU device |

### 4.5 GreenNet metric 해석

GreenNet 결과를 볼 때는 kernel 자체의 오차와 solution reconstruction 오차를 분리해야 한다.

| metric | 의미 | 발표에서의 주의점 |
|---|---|---|
| `loss` | Green kernel fitting loss | training convergence 확인 |
| `rel_green` | predicted Green kernel과 reference Green kernel의 relative error | exact/reference가 가능한 문제에서만 중심 metric으로 사용 |
| `train_rel_sol` | training-like source에 대한 solution reconstruction relative error | Green kernel이 source integration에 유용한지 확인 |
| `val_rel_sol` | validation-like source에 대한 reconstruction relative error | generalization 확인 |

Diffusion-only 문제에서는 exact/reference Green kernel 비교가 상대적으로 명확하다.
Reaction이 포함된 문제는 현재 exact/reference Green kernel 해석이 제한적이므로, 발표에서는
`rel_green`보다 reconstruction metric과 qualitative figure를 중심으로 보는 것이 안전하다.

### 4.6 GreenNet figure 후보

- `loss` curve: 학습 안정성과 수렴성 설명
- `rel_green` curve: reference kernel을 사용할 수 있는 문제에서 kernel accuracy 설명
- `train_rel_sol`, `val_rel_sol`: Green kernel을 이용한 source-to-solution reconstruction 성능
- selected axial Green heatmap: line별 \(G(x,\xi)\) 구조 시각화
- fixed-\(\xi\) 1D Green slice: boundary value와 diagonal/kink behavior 시각화
- coefficient line slice: line마다 operator가 달라진다는 점 설명
- source/reconstructed solution figure: Green kernel이 실제 source를 solution으로 재구성하는 과정 설명

## 5. CouplingNet 설정과 역할

### 5.1 CouplingNet의 역할

CouplingNet은 GreenNet이 제공하는 axial response를 이용해 2D solution reconstruction을
일관되게 만드는 모델이다. 핵심 출력은 axial flux-divergence decomposition으로 볼 수 있는
\(\phi\), \(\psi\)이다.

발표에서는 다음 흐름으로 설명한다.

```text
Source f + coefficient fields + GreenNet response
        |
        v
CouplingNet
        |
        +-- phi: x-direction flux-divergence component
        +-- psi: y-direction flux-divergence component
        |
        v
Axial Green integration
        |
        v
u_pred_x, u_pred_y, u_pred
```

여기서 \(u_{\text{pred}}\)는 x-axis reconstruction과 y-axis reconstruction을 결합한
최종 predicted solution으로 해석한다.

### 5.2 CouplingNet dataset 설정

| 항목 | 값 | 발표에서의 의미 |
|---|---:|---|
| `n_points_per_line` | 129 | GreenNet과 같은 grid resolution |
| `step_size` | 0.0078125 | uniform spacing |
| `samples_per_line` | 50 | CouplingNet source samples per line |
| `sampler_mode` | `backward` | source/solution data generation mode |
| `scale_length` | `[0.05, 0.25]` | source length-scale 범위 |
| `training_path` | `data/.../train` | saved train dataset |
| `validation_path` | `data/.../valid` | saved validation dataset |
| `test_path` | `data/.../test` | saved test dataset |
| `use_operator_learning` | true | coefficient-aware coupling learning |
| `dtype` | `float64` | 수치 정확도 우선 |

### 5.3 CouplingNet architecture 설정

| 항목 | 값 | 설명 |
|---|---:|---|
| `hidden_dim` | 128 | branch/trunk hidden width |
| `depth` | 4 | branch/trunk depth |
| `activation` | `rational` | rational activation |
| `branch_input_dim` | 129 | line source/coefficient input length |
| `trunk_input_dim` | 2 | config field는 유지되지만 axis-1D trunk가 활성화됨 |
| `coefficient_terms.diffusion` | true | diffusion coefficient 사용 |
| `coefficient_terms.convection` | true | directional convection 사용 |
| `coefficient_terms.reaction` | true | reaction coefficient 사용 |
| `branch_fusion.mode` | `product` | multiplicative feature fusion |
| `axis_1d_trunk.enabled` | true | shared 1D trunk 사용 |
| `axis_1d_trunk.boundary_aware_modes` | 4 | \(n=1,\dots,4\) sin/cos transverse encoding |

### 5.4 CouplingNet balance projection 설정

현재 baseline은 projection을 켜고 smooth-mask sine projection을 사용한다.

| 항목 | 값 | 의미 |
|---|---:|---|
| `balance_projection.enabled` | true | raw output을 balance-preserving output으로 변환 |
| `balance_projection.mode` | `smooth_mask` | boundary-aware smooth mask projection 사용 |
| `balance_projection.mask` | `sin` | \(m_\phi(y)=\sin(\pi y)\), \(m_\psi(x)=\sin(\pi x)\) |
| `smooth_mask_power` | 1.0 | mask exponent |
| `smooth_mask_diff_power` | 1.0 | raw difference mode exponent |
| `smooth_mask_diff_power_trainable` | false | exponent는 학습하지 않음 |
| `smooth_mask_eps` | \(10^{-12}\) | division 안정화 |

smooth-mask projection의 목적은 interior에서 \(\phi+\psi=f\) balance를 유지하면서,
transverse boundary 방향에서 raw difference mode가 boundary condition과 충돌하지 않도록
조절하는 것이다.

현재 sine mask는 다음 형태로 이해하면 된다.

$$
m_\phi(y) = \sin(\pi y), \qquad m_\psi(x) = \sin(\pi x)
$$

두 mask는 boundary에서 0이 되고 interior에서 양수이다. 따라서 boundary 근처에서 difference
mode contribution을 부드럽게 줄이는 효과가 있다.

### 5.5 CouplingNet training 설정

| 항목 | 값 | 설명 |
|---|---:|---|
| `batch_size` | 200 | Coupling training batch size |
| `epochs` | 3000 | training epochs |
| `learning_rate` | 0.002 | AdamW learning rate |
| `weight_decay` | 0.05 | AdamW weight decay |
| `gradient_clip_max_norm` | 1.0 | gradient clipping |
| `use_lr_schedule` | true | warmup + cosine schedule |
| `warmup_epochs` | 3 | warmup 기간 |
| `min_lr` | \(10^{-5}\) | schedule minimum learning rate |
| `periodic_checkpoint.enabled` | true | 주기적 checkpoint 저장 |
| `periodic_checkpoint.every_epochs` | 200 | 200 epoch마다 저장 |
| `best_rel_sol_checkpoint.enabled` | true | validation `rel_sol` 기준 best checkpoint 저장 |
| `compile.enabled` | true | `torch.compile` 사용 |

### 5.6 CouplingNet active loss

현재 발표 baseline의 active loss는 `energy_consistency`이다.

| loss | enabled | 발표에서의 취급 |
|---|---:|---|
| `energy_consistency` | true | 현재 CouplingNet baseline의 핵심 consistency loss |
| `l2_consistency` | false | baseline objective에서 제외 |
| `cross_consistency` | false | baseline objective에서 제외 |
| `balance_loss` | false | projection-off 실험용이므로 baseline에서 제외 |
| `symmetric_boundary_loss` | false | symmetric projection 실험용이므로 baseline에서 제외 |

## 6. CouplingNet 구조 설명

### 6.1 Branch input

CouplingNet branch는 source와 coefficient terms를 입력으로 사용한다. 현재 baseline에서는
diffusion, convection, reaction이 모두 켜져 있으므로 coefficient branch는 다음 정보를 모두
볼 수 있다.

```text
source branch: rhs / source line information
coefficient branch: a, b_axis, c
transverse branch: Enc_k(transverse coordinate)
```

여기서 `b_axis`는 x-line에서는 \(b_x\), y-line에서는 \(b_y\)이다.

### 6.2 Product branch fusion

현재 `branch_fusion.mode="product"`이다. 따라서 source feature, coefficient feature,
transverse feature는 learned fuser를 거치지 않고 component-wise product 방식으로 결합된다.

```text
branch_feat = source_feat * coefficient_feat * transverse_feat
```

이 설정은 기존 CouplingNet의 inductive bias를 유지한다. 발표에서는 “source, coefficient,
transverse coordinate가 서로 곱으로 상호작용하도록 강제하는 구조”로 설명할 수 있다.

### 6.3 Shared axis-1D trunk

현재 `axis_1d_trunk.enabled=true`이므로 trunk는 2D coordinate \((x,y)\)를 직접 받지 않는다.
대신 하나의 shared 1D trunk가 x-coordinate과 y-coordinate에 공통으로 사용된다.

```text
phi path: shared trunk input = x
psi path: shared trunk input = y
```

이 구조의 장점은 CouplingNet의 출력이 본질적으로 1D axial line 위의 scalar function이라는
구조와 더 잘 맞는다는 점이다. 즉 \(\phi\)는 x-line을 따라 변하는 function으로, \(\psi\)는
y-line을 따라 변하는 function으로 표현된다.

### 6.4 Boundary-aware transverse encoding

axis-1D trunk를 사용하면 trunk가 transverse coordinate를 직접 보지 않는다.
따라서 transverse coordinate 정보는 별도 branch로 전달한다.

현재 encoding은 raw coordinate \(t\)를 포함하지 않고 다음 fixed sin/cos feature만 사용한다.

$$
Enc_k(t) =
[\sin(\pi t), \cos(\pi t), \sin(2\pi t), \cos(2\pi t), \dots,
\sin(k\pi t), \cos(k\pi t)]
$$

현재 \(k=4\)이므로 feature는 \(2k=8\)차원이다.

```text
For phi: transverse t = fixed y-coordinate of the x-line
For psi: transverse t = fixed x-coordinate of the y-line
```

발표에서 중요한 메시지는 raw transverse coordinate를 직접 넣지 않고, boundary-aware periodic
feature만 넣는다는 점이다. 이는 boundary 근처의 구조를 표현하는 데 유리한 inductive bias로
설명할 수 있다.

### 6.5 Smooth-mask sine balance projection

CouplingNet raw output을 그대로 쓰면 \(\phi+\psi=f\) balance가 자동으로 보장되지 않는다.
현재 baseline은 projection을 사용해 interior balance를 맞춘다.

smooth-mask projection은 raw difference mode \(\phi_{\text{raw}}-\psi_{\text{raw}}\)를
사용하되, mask를 통해 boundary 방향에서 contribution을 조절한다.

Sine mask는 다음 성질을 갖는다.

```text
m_phi(y) = sin(pi*y): y=0, y=1에서 0
m_psi(x) = sin(pi*x): x=0, x=1에서 0
```

따라서 \(\phi\)는 y-boundary 방향, \(\psi\)는 x-boundary 방향에서 자연스럽게 boundary-aware
scaling을 받는다.

## 7. 학습 objective와 metric

### 7.1 GreenNet objective

GreenNet은 line Green kernel을 직접 학습한다. 학습 loss는 predicted Green kernel이 target
Green kernel 또는 generated training signal을 잘 맞추는지 본다. 하지만 발표에서는 kernel
loss만으로 충분하지 않다. 실제 목적은 Green kernel을 source와 적분해 solution을 잘 재구성하는
것이기 때문이다.

따라서 GreenNet 결과는 다음 두 축으로 해석한다.

```text
Kernel-level accuracy: rel_green
Reconstruction-level accuracy: train_rel_sol, val_rel_sol
```

`rel_green`은 exact/reference kernel이 의미 있는 문제에서 강하게 해석하고, reaction 포함
문제에서는 reconstruction metric과 qualitative figure를 더 중심에 둔다.

### 7.2 CouplingNet objective

CouplingNet baseline에서는 `energy_consistency`가 active loss이다. 이 loss는 x-axis solution
representation과 y-axis solution representation 사이의 physical energy consistency를 맞추는
역할을 한다.

발표에서는 다음처럼 설명하는 것이 좋다.

```text
GreenNet gives axial response.
CouplingNet learns phi and psi so that x-axis and y-axis reconstructions are physically consistent.
The active training signal is energy consistency.
```

### 7.3 CouplingNet metric 해석

| metric | 의미 | 발표에서의 해석 |
|---|---|---|
| `rel_sol` | predicted solution과 exact solution의 relative error | 최종 solution reconstruction 성능 |
| `rel_flux` | predicted flux-divergence와 exact flux-divergence의 relative error | \(\phi,\psi\) decomposition의 정확도 |
| `balance_l2` | \(f-\phi_{\text{pred}}-\psi_{\text{pred}}\) residual의 L2 scale | balance projection/field consistency 확인 |
| `balance_max_abs` | balance residual의 최대 절대값 | 국소적으로 큰 balance violation이 있는지 확인 |
| `energy_consistency` | x/y represented solution의 physical energy mismatch | active training objective의 수렴 확인 |

`balance_l2`와 `balance_max_abs`는 solution error와 같은 의미가 아니다. 이들은
\(\phi+\psi=f\)라는 decomposition balance가 얼마나 잘 맞는지 보는 diagnostic이다.

### 7.4 Error figure convention

발표용 error figure는 absolute error가 아니라 signed difference를 사용한다.

```text
solution signed error: u_pred - u
x-axis signed error: u_pred_x - u
y-axis signed error: u_pred_y - u
axis mismatch: u_pred_x - u_pred_y
flux signed error: phi_pred - phi, psi_pred - psi
balance residual: f - phi_pred - psi_pred
```

signed difference를 사용하면 error의 방향과 systematic bias를 볼 수 있다.

## 8. 발표자료로 옮길 figure/data 후보

### 8.1 GreenNet figure 후보

| figure | 목적 | 발표에서의 메시지 |
|---|---|---|
| training loss curve | 학습 수렴 확인 | Green kernel fitting이 안정적으로 진행됨 |
| `rel_green` curve | kernel-level accuracy | exact/reference가 가능한 문제에서 kernel 자체가 정확함 |
| `train_rel_sol`, `val_rel_sol` curve | reconstruction accuracy | Green kernel이 source-to-solution map으로 작동함 |
| Green heatmap \(G(x,\xi)\) | axial Green structure 시각화 | boundary, diagonal/kink behavior 확인 |
| fixed-\(\xi\) 1D slice | 특정 source 위치의 Green response | singular/diagonal 근처 구조와 boundary value 설명 |
| coefficient line slice | line별 operator variation | 같은 2D problem 안에서도 axial line problem이 달라짐 |
| reconstruction example | source-to-solution result | 학습된 Green kernel의 practical use 설명 |

### 8.2 CouplingNet figure 후보

| figure | 목적 | 발표에서의 메시지 |
|---|---|---|
| train/val loss | 학습 수렴 | Coupling training stability |
| `energy_consistency` | active objective | x/y represented solution consistency |
| `rel_sol` | 최종 solution accuracy | CouplingNet의 핵심 성능 |
| `rel_flux` | flux-divergence accuracy | \(\phi,\psi\) decomposition 품질 |
| source \(f\) | 입력 source | 문제 instance 설명 |
| exact solution \(u\) | reference | ground-truth solution |
| \(u_{\text{pred}}\) | prediction | 최종 reconstruction |
| \(u_{\text{pred}}-u\) | signed solution error | systematic bias 확인 |
| \(u_{\text{pred},x}-u_{\text{pred},y}\) | x/y mismatch | axial consistency 확인 |
| \(\phi,\psi\) | exact flux-divergence | decomposition target |
| \(\phi_{\text{pred}},\psi_{\text{pred}}\) | predicted flux-divergence | CouplingNet output 품질 |
| \(f-\phi-\psi\) | balance residual | balance projection/consistency 확인 |
| coefficient-family boxplot | family별 error distribution | 문제군별 성능 비교 |

### 8.3 figure 저장 기준

Plotly 기반 figure는 수정 가능성과 발표자료 재활용성을 위해 다음 형식을 유지하는 것이 좋다.

```text
.html: interactive inspection
.json: Plotly figure spec, later edit/rebuild
.png: slide 삽입용 raster image
.pdf: vector-like publication/presentation asset
```

## 9. 발표 흐름 제안

### 9.1 추천 발표 sequence

1. 문제 소개

   2D elliptic PDE, variable coefficient, Dirichlet boundary condition, source-to-solution
   reconstruction 문제를 소개한다.

2. 왜 Green function인가

   Green function은 source term을 solution으로 보내는 integral kernel이라는 점을 설명한다.
   하지만 full 2D Green function은 고차원이므로 직접 학습하기 어렵다는 점을 말한다.

3. Axial Green function idea

   2D 문제를 x-line/y-line의 1D 문제로 나누고, 각 line에서 Green kernel을 학습한다는
   아이디어를 제시한다.

4. GreenNet

   GreenNet이 line coefficient와 \((x,\xi)\) coordinate를 받아 axial Green kernel을
   학습하는 구조를 설명한다.

5. GreenNet result

   Green heatmap, fixed-\(\xi\) slice, reconstruction curve를 보여준다.

6. CouplingNet motivation

   x-line reconstruction과 y-line reconstruction을 단순히 따로 쓰는 것으로는 충분하지 않고,
   2D consistency와 flux-divergence decomposition이 필요하다는 점을 설명한다.

7. CouplingNet architecture

   source/coefficient branch, product fusion, shared axis-1D trunk, boundary-aware transverse
   encoding, smooth-mask sine projection을 순서대로 설명한다.

8. CouplingNet result

   `rel_sol`, `rel_flux`, selected sample heatmap, signed error, balance residual을 보여준다.

9. Coefficient family comparison

   6개 problem family에서 error distribution이나 representative figure를 비교한다.

10. Limitations and next steps

   Reaction 포함 문제의 reference Green 해석, more complex convection, ablation studies,
   projection/loss variants를 향후 과제로 제시한다.

### 9.2 슬라이드별 핵심 문장 후보

- “We learn line-wise Green kernels induced by fixed 2D coefficient fields.”
- “The model does not learn a single global 2D Green function; it learns axial Green responses.”
- “CouplingNet turns axial Green responses into a consistent 2D solution reconstruction.”
- “The shared 1D trunk reflects the line-wise nature of \(\phi\) and \(\psi\).”
- “Boundary-aware transverse encoding supplies the missing transverse information without feeding raw transverse coordinates.”
- “Sine smooth-mask projection enforces interior balance while respecting transverse boundary behavior.”

## 10. 현재 발표 기준에서 사용하지 않는 옵션

현재 발표 baseline 설명에서는 다음 옵션을 중심 구조로 설명하지 않는다.

| option | 현재 상태 | 문서에서의 취급 |
|---|---:|---|
| `source_stencil_lift` | disabled | 현재 baseline 설명에서 제외 |
| `green_response_feature` | disabled | 현재 baseline 설명에서 제외 |
| `trunk_positional_encoding` | disabled | axis-1D trunk와 동시에 사용하지 않는 옵션으로만 언급 가능 |
| `l2_consistency` | disabled | active objective에서 제외 |
| `cross_consistency` | disabled | active objective에서 제외 |
| `balance_loss` | disabled | projection-off 실험용으로만 구분 |
| `symmetric_boundary_loss` | disabled | symmetric projection 실험용으로만 구분 |
| `branch_fusion.product_fuser` | not active | ablation 후보로만 구분 |

이 옵션들은 repo에 구현되어 있더라도, 현재 발표 baseline의 핵심 설명에는 넣지 않는 것이
발표 메시지를 더 명확하게 만든다.

## 11. 한 페이지 요약

### 무엇을 학습하는가

- GreenNet은 axial 1D Green kernel \(G(x,\xi)\)을 학습한다.
- CouplingNet은 \(\phi,\psi\) flux-divergence decomposition을 학습해 2D solution을
  재구성한다.

### 왜 axial인가

- Full 2D Green function은 고차원 object이다.
- Axial decomposition은 문제를 x-line/y-line의 1D Green kernel 문제로 나눈다.
- Coefficient가 2D로 고정되어 있어도 line별 coefficient slice가 달라지므로, line-wise
  operator learning이 필요하다.

### GreenNet과 CouplingNet은 어떻게 연결되는가

- GreenNet은 source line을 solution line으로 보내는 axial response를 제공한다.
- CouplingNet은 source, coefficient, transverse coordinate 정보를 이용해 \(\phi,\psi\)를
  예측한다.
- 예측된 \(\phi,\psi\)는 GreenNet response와 결합되어 \(u_{\text{pred},x}\),
  \(u_{\text{pred},y}\), 최종 \(u_{\text{pred}}\)를 만든다.

### 현재 baseline에서 실제로 켜져 있는 핵심 option

- GreenNet:
  - backward sampler
  - operator learning
  - rational activation
  - Fourier feature
  - Adam + LBFGS
  - trapezoid integration

- CouplingNet:
  - diffusion/convection/reaction coefficient terms
  - branch fusion `product`
  - shared axis-1D trunk
  - boundary-aware sin/cos transverse encoding with 4 modes
  - smooth-mask balance projection
  - sine mask
  - energy consistency loss
  - AdamW + LR schedule

### 발표에서 보여줄 결과

- GreenNet:
  - Green kernel heatmap
  - fixed-\(\xi\) 1D Green slice
  - reconstruction metric
  - train/validation curves

- CouplingNet:
  - `rel_sol`, `rel_flux`
  - source/exact/predicted solution
  - signed solution and flux errors
  - balance residual
  - coefficient family별 boxplot

### 최종 발표 메시지

GreenNet은 fixed coefficient PDE family에서 line-wise Green kernel을 학습하고,
CouplingNet은 이 axial Green response를 이용해 2D solution과 flux-divergence를 일관되게
재구성한다. 현재 baseline은 shared 1D trunk, boundary-aware transverse encoding,
sine smooth-mask balance projection을 결합해 axial structure와 boundary behavior를 동시에
반영한다.
