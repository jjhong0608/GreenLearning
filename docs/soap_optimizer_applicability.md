# SOAP Optimizer Applicability for Complex CouplingNet

## Document Status

- 작성일: 2026-07-24
- 대상 프로젝트: `ComplexGeometry`
- 대상 학습 경로: complex geometry CouplingNet
- 목적: SOAP optimizer의 원리, 현재 프로젝트 적용 가능성, 기대 효과,
  위험, 그리고 향후 ablation 기준을 재사용 가능한 형태로 정리한다.
- 구현 상태: official SOAP commit
  `a1e553530fde97d0e6b307d7c82ac6d38b072340`을 vendoring하고 complex
  CouplingNet 전용 opt-in optimizer로 구현했다. AdamW는 기본값으로 유지한다.

## Executive Conclusion

SOAP은 현재 Complex CouplingNet에 기술적으로 적용할 수 있다. 현재 trainer가
표준 `torch.optim.Optimizer` 인터페이스를 사용하고 있고, CouplingNet
파라미터의 대부분이 2차원 weight matrix이므로 구조적인 적합성도 높다.

다만 SOAP을 AdamW의 즉각적인 대체재로 채택할 근거는 아직 부족하다.
SOAP 원 논문의 실험은 주로 360M 및 660M parameter language model의
large-batch pretraining을 대상으로 한다. 현재 프로젝트는 약 1.8M parameter의
physics-constrained operator-learning model이므로, iteration 감소가
wall-clock 감소로 이어지는지 별도로 확인해야 한다.

따라서 권장 결론은 다음과 같다.

> AdamW를 유지하면서 SOAP을 opt-in optimizer로 추가하고, 동일한 model,
> dataset, loss, seed를 사용하는 paired ablation으로 검증한다.

SOAP은 optimization dynamics를 개선할 수 있지만, 잘못된 loss, 실제 null
space, 누락된 admissibility 또는 geometry regularity 조건을 수정하지는
못한다.

## 1. SOAP의 정의

SOAP은 **ShampOo with Adam in the Preconditioner eigenbasis**의 약자다.
AdamW가 각 parameter coordinate를 독립적으로 조정하는 diagonal
preconditioner를 사용하는 반면, SOAP은 weight matrix의 행과 열 방향
gradient correlation을 추적한다.

Weight matrix

\[
W\in\mathbb R^{m\times n}
\]

의 step \(t\) gradient를 \(G_t\)라고 하자. SOAP은 Shampoo와 유사하게
다음 left/right statistics를 유지한다.

\[
L_t
=
\beta L_{t-1}
+
(1-\beta)G_tG_t^\top,
\]

\[
R_t
=
\beta R_{t-1}
+
(1-\beta)G_t^\top G_t.
\]

이 행렬들의 eigenbasis를

\[
L_t=Q_L\Lambda_LQ_L^\top,
\qquad
R_t=Q_R\Lambda_RQ_R^\top
\]

라고 하면 gradient를 다음과 같이 회전한다.

\[
\widetilde G_t
=
Q_L^\top G_tQ_R.
\]

SOAP은 이 rotated coordinate에서 Adam의 first/second moment를 갱신한다.

\[
\widetilde m_t
=
\beta_1\widetilde m_{t-1}
+
(1-\beta_1)\widetilde G_t,
\]

\[
\widetilde v_t
=
\beta_2\widetilde v_{t-1}
+
(1-\beta_2)\widetilde G_t^2.
\]

Rotated update는

\[
\widetilde U_t
=
\frac{\widetilde m_t}
{\sqrt{\widetilde v_t}+\epsilon}
\]

이고, 이를 원래 parameter coordinate로 되돌린다.

\[
U_t
=
Q_L\widetilde U_tQ_R^\top.
\]

SOAP은 Hessian을 직접 계산하는 Newton optimizer가 아니다. 또한 매 step마다
큰 matrix equation을 푸는 방식도 아니다. Gradient covariance의 Kronecker
structure를 사용하는 non-diagonal adaptive preconditioner로 이해하는 것이
정확하다.

Eigenbasis는 매 step 새로 계산하지 않고 `precondition_frequency`에 따라
주기적으로 갱신한다. 그 사이에는 현재 basis에서 Adam moment를 계속
갱신한다.

## 2. AdamW와의 차이

| 항목 | AdamW | SOAP |
|---|---|---|
| Preconditioner | Parameter-wise diagonal | Layer-wise non-diagonal eigenbasis |
| Gradient correlation | 좌표 간 correlation을 직접 사용하지 않음 | Matrix 행/열 correlation 반영 |
| Moment update | Original parameter basis | Shampoo eigenbasis |
| 1D parameter | AdamW | 공식 구현도 기본적으로 AdamW |
| 추가 주요 설정 | 없음 | `precondition_frequency` |
| Per-step cost | 낮음 | Matrix projection 및 factor update 비용 |
| Periodic cost | 없음 | QR 또는 eigendecomposition |
| State memory | \(O(mn)\) | \(O(mn+m^2+n^2)\) |

원 논문은 large-batch language-model experiment에서 SOAP이 AdamW보다
40% 이상 적은 iteration과 35% 이상 짧은 wall-clock time으로 같은 수준의
목표에 도달했다고 보고한다. 이 결과는 현재 프로젝트에 직접 적용되는
보장은 아니며, task-specific ablation이 필요하다.

## 3. 현재 프로젝트 상태

2026-07-24 기준으로 complex trainer는
`ComplexCouplingOptimizerFactory`를 통해 optimizer를 생성한다. Config에
`optimizer` block이 없으면 기존과 같은 AdamW를 사용하고,
`optimizer.name="soap"`일 때만 vendored SOAP을 사용한다.

```python
optimizer = ComplexCouplingOptimizerFactory(config).build(model.parameters())
```

학습 step은 다음 표준 순서를 따른다.

```text
optimizer.zero_grad(set_to_none=True)
loss.backward()
clip_grad_norm_()
optimizer.step()
scheduler.step()  # epoch 종료 후
```

현재 `configs/complex_coupling.json`의 주요 optimizer 관련 값은 다음과
같다.

```text
learning_rate         = 0.002
weight_decay          = 0.05
batch_size            = 400
epochs                = 3000
gradient_clip_max_norm= 1.0
use_lr_schedule       = true
warmup_epochs         = 3
min_lr                = 1e-5
dtype                 = float64
device                = cuda:1
```

현재 training split에는 800개 sample이 있으므로 batch size 400에서는
epoch당 optimizer step이 2회다.

```text
optimizer steps per epoch = 2
total optimizer steps      = about 6000
```

따라서 SOAP의 `precondition_frequency=10`은 약 5 epoch마다 한 번
eigenbasis를 갱신하는 것과 같다. 반면 현재 3-epoch warmup은 optimizer
step으로는 6회뿐이므로 SOAP에서는 warmup duration을 별도로 검토할 필요가
있다.

## 4. Model Structure Suitability

현재 `ComplexCouplingNet`을 실제 config로 생성하여 조사한 결과는 다음과
같다.

| 항목 | 값 |
|---|---:|
| 전체 trainable parameters | 1,814,416 |
| 2D matrix parameters | 1,807,360 |
| Matrix parameter 비율 | 약 99.6% |
| Hidden dimension | 256 |
| 가장 큰 matrix | `branch_fuser.weight: (256, 1024)` |
| 두 번째로 큰 matrix | `trunk_fuser.weight: (256, 768)` |

SOAP은 2D weight matrix에 주로 적용되고 1D bias/activation parameter에는
AdamW를 사용하는 것이 기본 정책이다. 현재 모델 parameter의 약 99.6%가
2D matrix이므로 SOAP이 실제로 작동할 parameter 비율은 높다.

다음 구성 요소들은 모두 matrix parameter를 포함한다.

- Source branch MLP
- Coefficient branch MLP
- Transverse branch MLP
- Geometry branch MLP
- Primary trunk MLP
- Pointwise transverse trunk MLP
- Branch product fuser
- Trunk product fuser

Branch/trunk feature가 product 또는 learned fuser를 통해 결합되므로 layer별
gradient scale과 correlation이 크게 달라질 가능성이 있다. 이 경우 SOAP의
non-diagonal preconditioning이 AdamW보다 유리할 수 있다. 다만 이는
현재 모델에 대한 이론적 가능성이지 아직 측정된 결과는 아니다.

## 5. Memory Estimate

공식 two-sided SOAP implementation이 각 matrix parameter에 대해 다음
state를 유지한다고 가정한다.

- Adam first moment
- Adam second moment
- Left/right gradient covariance factors
- Left/right eigenbasis matrices

현재 `float64` model에 대한 대략적인 optimizer-state estimate는 다음과
같다.

| Optimizer state | Estimated memory |
|---|---:|
| AdamW moments | 약 27.7 MiB |
| SOAP moments 및 factors | 약 99.9 MiB |
| SOAP additional factor/eigenbasis state | 약 72.3 MiB |

이 계산에는 model parameter, gradient, activation, CUDA allocator
fragmentation, QR workspace가 포함되지 않는다. 현재 model size와 A40
GPU memory를 고려하면 persistent optimizer state 자체는 주된 제약이
아닐 가능성이 높다.

더 중요한 비용은 다음 matrix의 periodic QR/eigendecomposition이다.

- \(1024\times1024\) factor
- \(768\times768\) factor
- 다수의 \(256\times256\) factor

작은 model에서는 forward/backward보다 optimizer matrix operation이
상대적으로 크게 보일 수 있으므로 wall-clock benchmark가 필수다.

## 6. 현재 코드와의 기술적 호환성

### 6.1 Optimizer API

공식 SOAP implementation은 `torch.optim.Optimizer`를 상속한다. 따라서
현재 trainer의 다음 동작과 호환된다.

- `zero_grad(set_to_none=True)`
- `step()`
- `param_groups`
- Gradient clipping before `step()`
- PyTorch learning-rate scheduler

### 6.2 Learning-Rate Scheduler

현재 warmup+cosine scheduler는 `optimizer.param_groups[*]["lr"]`에
multiplier를 적용한다. SOAP도 같은 optimizer interface를 제공하므로
scheduler 수식 자체는 재사용할 수 있다.

다만 다음 차이가 있다.

- SOAP frequency는 epoch가 아니라 optimizer step 기준이다.
- 공식 preliminary implementation은 첫 `step()`에서 preconditioner를
  초기화하고 parameter update를 생략한다.
- 현재 6-step warmup은 SOAP basis update 주기보다 짧을 수 있다.

따라서 scheduler compatibility와 scheduler suitability를 구분해야 한다.
코드는 호환되지만 현재 warmup 값이 적절하다는 뜻은 아니다.

### 6.3 `torch.compile`

현재 compile 대상은 model이고 optimizer는 일반 Python optimizer로
실행된다. 따라서 SOAP을 사용하기 위해 compiled model contract를
변경할 필요는 없다.

Optimizer 자체를 `torch.compile` 대상으로 넣는 것은 이번 적용 범위에
포함하지 않는 것이 안전하다.

### 6.4 Checkpoint

현재 CouplingNet checkpoint는 model-only safetensors이다. Optimizer 및
scheduler state를 저장하지 않으므로 AdamW를 SOAP으로 바꾸더라도 기존
checkpoint file format은 바뀌지 않는다.

반면 interrupted training resume를 지원하려면 SOAP의 다음 state까지
별도로 저장해야 한다.

- Adam moments
- Covariance factors
- Eigenbasis
- Optimizer step
- Scheduler state

현재 trainer는 완전한 optimizer resume를 제공하지 않으므로 이 문제는
초기 SOAP ablation의 blocker는 아니다.

## 7. Expected Advantages

### 7.1 Better Matrix-Wise Conditioning

AdamW는 각 coordinate의 second moment만 사용하지만 SOAP은 weight matrix의
row/column correlation을 반영한다. Branch/trunk/fuser의 gradient geometry가
anisotropic하면 더 적절한 update direction을 제공할 수 있다.

### 7.2 Potentially Faster Canonical-Energy Optimization

Current canonical energy objective가 parameter space에서 stiff하지만
수학적으로 올바른 objective라면, SOAP은 ill-conditioned direction을
precondition하여 같은 energy level에 더 적은 step으로 도달할 가능성이
있다.

### 7.3 Suitability for the Current Large Batch

현재 batch size 400은 800-sample training split의 절반이다. Second-order
preconditioner는 stochastic gradient noise가 작은 large-batch setting에서
상대적으로 유리하다는 것이 SOAP 원 논문과 공식 implementation의
권장 사항이다.

### 7.4 No Violation of Reference-Free Training

SOAP은 현재 scalar objective의 gradient만 사용한다. 따라서 reference
`sol`, target `phi`, target `psi`를 training objective에 추가할 필요가 없다.
현재 reference-free training 및 checkpoint-selection 원칙을 유지할 수
있다.

### 7.5 Natural Fit for Matrix-Dominated Parameters

현재 trainable parameter의 약 99.6%가 2D matrix다. SOAP의 주요 기능이
대부분의 model parameter에 적용되고, bias와 Rational activation의 일부
1D parameter만 AdamW fallback을 사용한다.

## 8. Disadvantages and Risks

### 8.1 It Cannot Repair the Mathematical Objective

SOAP은 optimization trajectory를 변경할 뿐 다음 문제를 해결하지 않는다.

- Loss의 true null space
- 잘못된 PDE balance/projection
- 누락된 boundary condition
- \(H_0^1(\Omega)\) admissibility 부족
- Geometry graph의 잘못된 physical adjacency
- Reconstruction formula 오류

Objective가 non-coercive하면 SOAP은 그 objective의 좋지 않은 minimum이나
flat direction에 더 빨리 도달할 수도 있다.

### 8.2 Wall-Clock Improvement Is Not Guaranteed

원 논문의 주 실험은 360M 및 660M language models이다. 현재 CouplingNet은
약 1.8M parameters이므로 optimizer overhead를 대규모 forward/backward에
amortize하기 어렵다.

SOAP이 iteration을 줄이더라도 다음 비용 때문에 전체 학습 시간이 늘어날
수 있다.

- Gradient projection
- Update back-projection
- Covariance-factor update
- Periodic QR/eigendecomposition

### 8.3 Hyperparameters Need Retuning

현재 AdamW는 PyTorch default betas `(0.9, 0.999)`를 사용한다. 공식 SOAP
example은 다음 값을 사용한다.

```text
lr                     = 3e-3
betas                  = (0.95, 0.95)
weight_decay           = 0.01
precondition_frequency = 10
```

현재 project 값은 `lr=0.002`, `weight_decay=0.05`다. 기존 값을 그대로
사용하거나 공식 값을 그대로 복사하는 것 모두 최적성을 보장하지 않는다.
Learning rate, betas, weight decay, frequency를 paired sweep해야 한다.

### 8.4 Float64 Cost

현재 model은 `float64`다. Full float64 factor accumulation과 decomposition은
float32보다 느리고 더 많은 memory를 사용한다.

Original preliminary implementation은 QR/eigendecomposition 중 일부를
float32로 계산한다. Meta Distributed Shampoo implementation은 factor
dtype을 선택할 수 있다. 프로젝트의 numerical-precision policy와 optimizer
implementation policy를 함께 결정해야 한다.

### 8.5 Preliminary Official Implementation

원 논문 저자의 공식 repository는 자신을 preliminary implementation이라고
표현하며 package release보다 단일 `soap.py` 복사를 안내한다. Lower-precision
및 distributed support도 향후 기능으로 안내한다.

따라서 production-quality integration에는 다음이 필요하다.

- Pinned implementation source/version
- Config validation
- State initialization test
- Dtype/device test
- Scheduler test
- Gradient clipping test
- Checkpoint/reproducibility metadata

### 8.6 Evidence Gap

SOAP 원 논문은 language-model pretraining에 집중한다. Physics-constrained
operator learning, Green reconstruction, canonical-energy objective,
disconnected complex geometry에서의 직접 근거는 없다.

따라서 현재 프로젝트에서의 효과에 대한 불확실성은 규칙의 모호성보다
domain-specific empirical evidence 부족에서 온다.

## 9. Candidate Implementations

### 9.1 Original SOAP Repository

장점:

- 원 논문의 알고리즘과 직접 대응한다.
- 단일 file implementation으로 초기 ablation이 쉽다.
- PyTorch optimizer interface를 따른다.

단점:

- Preliminary implementation이다.
- 공식 package release와 versioning이 약하다.
- Lower-precision 및 distributed support가 제한적이다.
- Repository code quality를 그대로 project core에 복사하면 유지보수 부담이
  생긴다.

용도:

- 빠른 research pilot
- 논문 알고리즘 재현

### 9.2 Meta Distributed Shampoo SOAP Mode

Meta의 `facebookresearch/optimizers`에는 eigenvalue-corrected Shampoo,
즉 SOAP mode가 포함되어 있다.

장점:

- Precision policy 지원
- QR/eigendecomposition 선택
- Preconditioner-size/frequency control
- Serial, DDP, FSDP 관련 지원
- 더 체계적인 optimizer-state 처리

단점:

- Dependency와 config surface가 더 크다.
- 현재 단일-GPU 소형 model에는 implementation complexity가 과할 수 있다.
- 현재 Python 3.14 environment와의 실제 import/install smoke가 필요하다.

현재 environment의 PyTorch 2.11 및 CUDA 12.6은 Meta documentation에
기재된 최소 범위 PyTorch 2.8 이상 및 CUDA 12.2 이상을 만족한다. 다만
이 사실만으로 Python 3.14까지 완전한 조합 호환성이 보장되지는 않는다.

## 10. Recommended Integration Policy

SOAP을 구현한다면 기존 AdamW를 제거하지 않고 optimizer factory를
도입하는 것이 적절하다.

구현된 config surface는 다음과 같다.

```json
{
  "coupling_training": {
    "optimizer": {
      "name": "soap",
      "betas": [0.95, 0.95],
      "eps": 1e-8,
      "profile_step_time": true,
      "soap": {
        "shampoo_beta": -1.0,
        "precondition_frequency": 10,
        "max_precondition_dim": 1024,
        "merge_dims": false,
        "precondition_1d": false,
        "normalize_grads": false,
        "correct_bias": true
      }
    }
  }
}
```

현재 구현은 다음 정책을 따른다.

1. `optimizer.name="adamw"`를 backward-compatible default로 유지한다.
2. SOAP은 complex CouplingNet의 opt-in experiment로 시작한다.
3. GreenNet 및 unit-square CouplingNet 적용은 별도 실험으로 둔다.
4. Scheduler, clipping, loss, projection, reconstruction은 바꾸지 않는다.
5. Complex CouplingNet의 `config_used.json`에는 resolved optimizer block과
   top-level provenance를 materialize한다. 같은 metadata를
   `optimizer_provenance.json`, training log, complex artifact summary에도
   기록한다.
6. Preconditioner frequency는 optimizer step 단위라고 문서화한다.
7. `profile_step_time=true`인 경우 AdamW와 SOAP의 optimizer time 및 peak
   allocated CUDA memory를 같은 metric schema로 기록한다.
8. Official first-step preconditioner initialization/no-update 동작을 그대로
   유지한다.
9. Model-only safetensors를 유지하며 optimizer-state resume는 지원하지 않는다.

구현 파일은 `src/greenonet/optimizers/soap.py`,
`src/greenonet/coupling_optimizer.py`, 그리고
`src/greenonet/complex_coupling_trainer.py`이다. Upstream attribution과 MIT
license는 `THIRD_PARTY_NOTICES.md`에 기록한다. Float64 model 호환을 위해
stored Shampoo factor/eigenbasis는 upstream의 float32 정책을 유지하되,
covariance update와 tensor contraction 경계에서 dtype을 명시적으로
변환한다.

## 11. Recommended Ablation

전체 3000-epoch run을 시작하기 전에 300--500 epoch pilot을 권장한다.
`configs/complex_coupling_soap.json`은 canonical complex config와 model,
dataset, loss, projection, clipping, scheduler 설정을 맞춘 SOAP template이다.
Pilot에서는 AdamW baseline과 SOAP config의 `epochs`를 같은 300--500 값으로
설정하고 장기 run 전에 optimizer telemetry를 먼저 비교한다.

### 11.1 Experiment Matrix

| Run | Optimizer | Betas | LR | Frequency |
|---|---|---|---:|---:|
| Baseline | AdamW | `(0.9, 0.999)` | `0.002` | N/A |
| SOAP-A | SOAP | `(0.9, 0.999)` | `0.002` | `10` |
| SOAP-B | SOAP | `(0.95, 0.95)` | LR sweep | `10` |
| SOAP-C | SOAP | best betas | best LR | `20` |
| SOAP-D | SOAP | best betas | best LR | `50` |

초기 LR sweep 예시는 다음과 같다.

```text
1e-3
2e-3
3e-3
```

Preconditioner dimension은 우선 full current-model coverage인 1024를
사용하고, optimizer overhead가 크면 512 이하의 one-sided/partial
preconditioning을 후속 ablation으로 검토한다.

### 11.2 Fair Comparison Requirements

다음 조건은 optimizer 외에는 같아야 한다.

- Dataset split
- Batch order
- Model initialization seed
- Loss definition
- Projection and reconstruction
- Gradient clipping
- Scheduler family
- Evaluation sample set
- GreenNet checkpoint

적어도 3개의 paired seed를 권장한다.

### 11.3 Two Budgets

SOAP은 step efficiency와 step cost가 모두 달라지므로 두 기준을 모두
비교해야 한다.

1. Equal optimizer-step budget
2. Equal wall-clock budget

Equal epoch만 비교하면 optimizer overhead와 convergence speed를 구분할 수
없다.

### 11.4 Metrics

필수 비교 지표:

- Train canonical energy
- Validation canonical energy
- Best validation energy
- Evaluation-only `rel_sol`
- Evaluation-only `rel_flux`
- Best checkpoint epoch/step
- Mean and percentile optimizer-step time
- Basis-update step time spike
- Total wall-clock time
- Peak GPU memory
- Gradient norm before clipping
- Gradient clipping frequency

현재 reference-free 원칙에 따라 `rel_sol`과 `rel_flux`는 reporting에만
사용하며 optimizer selection이나 checkpoint selection에 사용하지 않는다.

## 12. Decision Criteria

SOAP을 유지할 기준:

- 여러 paired seed에서 validation canonical energy가 일관되게 감소한다.
- 동일 energy 수준까지의 wall-clock이 AdamW보다 짧다.
- Evaluation-only solution/flux metrics가 악화되지 않는다.
- QR step의 time spike와 memory overhead가 허용 가능하다.
- Hyperparameter sensitivity가 지나치게 높지 않다.

SOAP을 기본값으로 채택하지 않을 기준:

- Step 수는 줄지만 wall-clock이 증가한다.
- Best result가 특정 seed나 특정 LR에만 나타난다.
- Canonical energy는 감소하지만 evaluation field quality가 악화된다.
- Numerical instability 또는 factor-decomposition failure가 발생한다.
- 기존 AdamW 대비 유지보수 비용이 효과보다 크다.

## 13. Final Assessment

| 평가 항목 | 판단 |
|---|---|
| 기술적 적용 가능성 | 높음 |
| 현재 matrix architecture 적합성 | 높음 |
| Persistent memory 부담 | 낮음에서 중간 |
| Optimizer compute 부담 | 중간에서 높음 |
| Canonical-energy 수렴 개선 가능성 | 중간 |
| Wall-clock 개선 가능성 | 불확실 |
| Geometry/admissibility 문제 해결 가능성 | 낮음 |
| 기본 optimizer 즉시 교체 권장 | 아니오 |
| Opt-in paired ablation 권장 | 예 |

기술적 적용성 판단의 확신도는 약 0.95다. 실제 Annulus CouplingNet 학습에서
AdamW보다 좋은 결과를 낼 가능성은 현재 근거로 약 0.55--0.65로 본다.

후자의 불확실성은 규칙이 모호해서가 아니라, SOAP의 원 실험 영역과 현재
physics-constrained complex-geometry operator-learning 영역 사이에 직접적인
검증 결과가 부족하기 때문이다.

## References

1. Nikhil Vyas et al.,
   [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321),
   ICLR 2025.
2. Nikhil Vyas et al.,
   [Official preliminary SOAP implementation](https://github.com/nikhilvyas/SOAP).
3. Meta Research,
   [PyTorch Distributed Shampoo and SOAP mode](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md).
