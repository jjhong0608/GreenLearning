# AMUSE Optimizer Applicability for Complex CouplingNet

## Document Status

- 작성일: 2026-07-27
- 대상 프로젝트: `ComplexGeometry`
- 대상 학습 경로: complex geometry CouplingNet
- 조사 대상: AMUSE arXiv v2 (`2605.22432`, 2026-07-14)와 official repository
  commit `48922743b32f33f919ab54edde3dbad0d0ce2dc7`
- 구현 상태: 분석만 완료했다. AMUSE는 현재 project code/config에 구현되어
  있지 않다.

## Executive Conclusion

AMUSE는 현재 Complex CouplingNet에 **기술적으로 적용 가능하고 구조적
적합성도 높다**. 현재 SOAP pilot config로 생성한 model은 1,814,586개의
trainable parameter를 가지며, 그중 1,807,506개, 즉 99.61%가 2D matrix
parameter이다. 따라서 Muon의 matrix update를 실제로 적용할 parameter가
충분하다.

그러나 AMUSE는 기존 optimizer factory에 이름 하나를 추가하는 정도의
drop-in replacement가 아니다. Schedule-Free optimizer 특성상 training에서는
gradient-evaluation iterate \(Y_t\), validation/inference/checkpoint에서는
averaged iterate \(X_t\)를 사용해야 한다. 현재 trainer는 model의
`train()`/`eval()`만 전환하고 optimizer에는 대응 전환을 하지 않으므로,
trainer lifecycle과 checkpoint 저장 순서를 함께 수정해야 한다.

공식 구현은 Newton-Schulz orthogonalization을 강제로 `bfloat16`에서 수행한다.
현재 Complex CouplingNet은 `float64`이므로, 이 mixed-precision optimizer
geometry를 명시적인 정책과 test 없이 도입해서는 안 된다.

따라서 권장 결론은 다음과 같다.

> AdamW와 SOAP을 유지하고, AMUSE를 complex CouplingNet 전용 opt-in
> research optimizer로 구현한 뒤 짧은 paired pilot으로만 검증한다.

AMUSE를 현재 default optimizer로 바꾸는 것은 권장하지 않는다. AMUSE 논문은
vision task와 LLM pretraining에서 강한 결과를 보였지만, elliptic PDE
operator learning, `float64` rational network, Green reconstruction,
canonical-energy objective에는 직접적인 검증이 없다.

## 1. AMUSE의 정의

AMUSE는 **Anytime MUon with Stable gradient Evaluation**의 약자다. 핵심은
다음 두 방법을 결합하는 것이다.

1. Muon은 matrix-valued parameter의 momentum update를 Newton-Schulz
   iteration으로 approximately orthogonalize한다.
2. Schedule-Free optimization은 fast/base sequence와 averaged sequence 사이의
   interpolation point에서 gradient를 평가하고, averaged sequence를 inference에
   사용한다.

AMUSE는 세 parameter sequence를 개념적으로 사용한다.

\[
Z_t:
\text{fast Muon base sequence},
\]

\[
X_t:
\text{averaged inference sequence},
\]

\[
Y_t
=
(1-\beta_t)Z_t+\beta_tX_t:
\text{gradient-evaluation sequence}.
\]

Matrix parameter의 gradient를

\[
G_t=\nabla\mathcal L(Y_t)
\]

라고 하면 Muon momentum과 orthogonalized update는

\[
M_t
=
\mu M_{t-1}+(1-\mu)G_t,
\]

\[
O_t
=
s_{\mathrm{NS}}\operatorname{NewtonSchulz}(M_t)
\]

이고, base sequence는

\[
Z_{t+1}
=
(1-\eta_t\lambda)Z_t-\eta_tO_t
\]

로 갱신된다. Averaged sequence는

\[
X_{t+1}
=
(1-c_{t+1})X_t+c_{t+1}Z_{t+1}
\]

로 갱신된다.

AMUSE는 warmup 동안

\[
\beta_t=\beta_{\mathrm{init}}
\]

을 유지하고, 그 이후 gradient-evaluation point를 점차 \(X_t\) 쪽으로
이동시킨다.

\[
\beta_t
=
1-
\left(
\frac{
c_{t+1}(1-c_{T_0+1})
}{
c_{T_0+1}(1-c_{t+1})
}
\right)^\rho
(1-\beta_{\mathrm{init}}).
\]

\(c_{t+1}=1/(t+1)\)인 단순한 경우에는

\[
\beta_t
=
1-
\left(\frac{T_0}{t}\right)^\rho
(1-\beta_{\mathrm{init}})
\]

로 정리된다. \(\rho\)가 클수록 gradient evaluation이 averaged trajectory
\(X_t\) 쪽으로 더 빨리 이동한다.

## 2. 왜 Muon에 Schedule-Free Averaging을 결합하는가

AMUSE 논문은 loss landscape를 다음 두 subspace로 해석한다.

- dominant subspace: 낮은 차원이지만 curvature가 큰 valley-wall direction
- bulk subspace: 높은 차원이고 curvature가 낮은 river direction

Muon의 matrix orthogonalization은 몇 개의 큰 singular component가 update를
지배하지 못하게 하여 bulk direction의 진행을 빠르게 할 수 있다. 반면 작은
noise도 normalization 과정에서 증폭되어 high-curvature direction의 oscillation을
만들 수 있다.

AMUSE는 training 후반으로 갈수록 gradient를 stable averaged sequence에 가까운
\(Y_t\)에서 평가한다. 논문의 주장은 orthogonalization 이전 gradient의
dominant component를 줄여 Muon의 빠른 bulk progress는 유지하면서 valley-wall
oscillation을 완화한다는 것이다.

`Anytime`은 total training horizon을 미리 정한 decay schedule 없이도 averaged
iterate \(X_t\)를 중간에 평가하고 사용할 수 있다는 의미다. 다만 linear warmup은
계속 사용한다. 즉, `schedule-free`는 **warmup까지 제거한다는 뜻이 아니라
post-warmup learning-rate decay가 필요 없다는 뜻**이다.

## 3. 논문 결과와 증거의 범위

논문은 image classification, image segmentation, ViT fine-tuning, Llama-style
LLM pretraining에서 AMUSE를 평가했다. 저자들은 Muon의 final performance에
도달하는 데 필요한 step이 다음과 같이 감소했다고 보고한다.

| Experiment | Reported step reduction |
|---|---:|
| 720M Llama pretraining | \(1.51\times\) fewer steps |
| ImageNet ResNet-50 | \(1.34\times\) fewer steps |
| ImageNet ViT fine-tuning | \(2.57\times\) fewer steps |

이 결과는 AMUSE의 가능성을 보여주지만 현재 프로젝트에 직접 외삽할 수는 없다.

- 논문 모델은 주로 Transformer, ResNet, ViT, U-Net이다.
- 현재 모델은 약 1.8M parameter의 DeepONet-style branch/trunk network다.
- 현재 objective는 likelihood나 classification loss가 아니라 reconstructed
  solution의 canonical energy consistency다.
- 현재 model과 loss는 `float64`이며 rational activation parameter를 포함한다.
- 현재 batch sample은 PDE source realization이고 token/image batch와 통계적
  성질이 다르다.

따라서 AMUSE가 현재 objective의 iteration count 또는 wall-clock을 줄인다는
결론은 반드시 project-specific ablation으로 검증해야 한다.

## 4. 현재 Complex CouplingNet의 구조적 적합성

`configs/complex_coupling_soap.json`으로 실제 model을 생성해 조사한 결과는
다음과 같다.

| Parameter category | Count | Fraction |
|---|---:|---:|
| Total trainable parameters | 1,814,586 | 100% |
| 2D matrix parameters | 1,807,506 | 99.6098% |
| 1D vector parameters | 7,079 | 0.3901% |
| Scalar parameters | 1 | 0.0001% |

주요 2D matrix는 source branch, transverse branch, geometry branch, primary
trunk, pointwise transverse trunk, branch fuser, trunk fuser에 분포한다.
가장 큰 matrix는 다음과 같다.

- `branch_fuser.weight`: `(256, 1024)`
- `trunk_fuser.weight`: `(256, 768)`
- 다수의 hidden weight: `(256, 256)`

이는 Muon-style update를 적용하기에 구조적으로 유리하다. 특히 branch/trunk와
learned fuser는 서로 다른 scale과 correlation을 가진 gradient를 만들 수 있어,
matrix update geometry가 AdamW의 coordinate-wise diagonal adaptation보다
유리할 가능성이 있다.

다만 `pre_projection_fusion`의 `(1,2)`, `(16,8)`, `(1,16)` weight는 small
correction/output head에 가깝다. 공식 AMUSE가 output head를 fallback optimizer로
처리하는 것과 맞추려면 이 작은 correction block은 SF-AdamW fallback group으로
두는 것이 보수적이다.

권장 parameter grouping은 다음과 같다.

| Group | Parameters | AMUSE base update |
|---|---|---|
| Muon matrix group | Main branch/trunk/fuser `ndim >= 2` weights | Muon |
| Fallback group | Bias, rational coefficients, scalar gate | SF-AdamW |
| Fallback head group | `pre_projection_fusion` correction weights/bias | SF-AdamW |

이 구분은 parameter name을 명시적으로 검사해야 하며, 단순한
`parameter.ndim >= 2`만으로 결정해서는 안 된다.

## 5. SOAP과의 차이

SOAP과 AMUSE는 모두 matrix structure를 사용하지만 같은 optimizer가 아니다.

| Item | SOAP | AMUSE |
|---|---|---|
| Main idea | Shampoo eigenbasis에서 Adam | Muon + Schedule-Free averaging |
| Matrix statistics | Row/column covariance factors | Matrix momentum |
| Matrix operation | Basis projection, periodic QR | Newton-Schulz every step |
| Adaptive second moment | Matrix weight에도 사용 | Fallback parameter에만 사용 |
| LR policy | 현재 project는 warmup+cosine | Linear warmup 후 constant LR |
| Inference parameter | Current parameter | Averaged sequence \(X_t\) |
| Trainer lifecycle | Standard optimizer | Optimizer `train()`/`eval()` 필수 |

SOAP은 matrix covariance를 축적해 adaptive coordinate system을 만들고, AMUSE는
momentum matrix의 singular geometry를 매 step normalize한다. 따라서 SOAP에서
좋았던 learning rate, betas, scheduler를 AMUSE에 그대로 복사해서는 안 된다.

AMUSE state는 matrix parameter당 \(Z_t\) copy와 momentum buffer를 보관하고,
fallback parameter에는 \(Z_t\)와 second moment를 보관한다. 현재 `float64`
1.814M model에서 persistent optimizer state는 약 27.7 MiB로 AdamW의 두 moment
state와 유사하다. SOAP의 covariance/eigenbasis state보다는 작을 가능성이 높지만,
AMUSE는 Newton-Schulz matrix multiplication을 매 optimizer step 수행하므로
per-step compute cost는 별도로 측정해야 한다.

## 6. 현재 Trainer와의 통합 차이

### 6.1 Optimizer Lifecycle

현재 trainer는 다음만 수행한다.

```text
model.train()
optimizer.step()
model.eval()
save model state
```

AMUSE는 다음 lifecycle이 필요하다.

```text
optimizer.train()   # parameters expose Y_t
model.train()
optimizer.step()

optimizer.eval()    # parameters expose averaged X_t
model.eval()
validation
save averaged model state

optimizer.train()   # restore Y_t before the next training step
```

이 전환을 누락하면 다음 두 오류가 생긴다.

1. `optimizer.step()`이 train mode가 아니라는 runtime error를 발생시킨다.
2. validation/checkpoint가 stable averaged \(X_t\)가 아니라
   gradient-evaluation \(Y_t\)를 사용한다.

따라서 trainer에 optimizer-aware context manager 또는 lifecycle adapter가
필요하다. 단순히 `isinstance(optimizer, AMUSE)`를 여러 위치에 분산시키는
방식은 피해야 한다.

### 6.2 Learning-Rate Schedule

현재 complex trainer는 epoch 단위 linear warmup + cosine decay를 외부 scheduler로
적용한다. 공식 AMUSE는 optimizer step 단위 linear warmup을 내부에서 적용하고,
그 이후 constant base learning rate를 사용한다.

AMUSE를 구현한다면 다음 contract를 권장한다.

- `optimizer.name="amuse"`이면 external cosine scheduler를 금지한다.
- `use_lr_schedule`의 의미를 재사용하지 말고 AMUSE 전용 `warmup_steps` 또는
  `warmup_fraction`을 명시한다.
- `warmup_epochs`를 재사용한다면 DataLoader가 생성된 뒤
  `warmup_steps=warmup_epochs*len(train_loader)`로 resolve한다.
- CSV의 `learning_rate`는 epoch 시작값이 아니라 실제 optimizer step의
  mean/last effective learning rate를 기록한다.

현재 SOAP config는 4,800 train sample과 batch size 300을 사용하므로 epoch당
16 optimizer step이다. 100 epoch run은 1,600 step이고 10 warmup epoch는
160 step이다. 첫 pilot에서는 이 160-step warmup이 합리적인 시작점이다.

### 6.3 Checkpoint Semantics

현재 checkpoint는 model-only safetensors이고 optimizer state를 저장하지 않는다.
AMUSE에서도 averaged \(X_t\) 상태에서 model을 저장하면 inference checkpoint로
사용할 수 있다.

그러나 다음 state가 없으므로 interrupted-training resume는 지원할 수 없다.

- base sequence \(Z_t\)
- matrix momentum
- fallback second moment
- averaging weights와 step counter
- current \(\beta_t\)

이는 현재 project의 model-only policy와 일치하므로 initial ablation blocker는
아니다. 다만 checkpoint metadata에 다음을 반드시 기록해야 한다.

- `inference_parameter_space="amuse_averaged_x"`
- official repository와 pinned commit
- parameter grouping policy
- resolved warmup steps
- \(\beta_{\mathrm{init}},\rho,\mu,\beta_2,\epsilon\)
- Newton-Schulz dtype
- `optimizer_resume_supported=false`

### 6.4 Validation and Best Checkpoint

Best-energy checkpoint selection은 validation canonical energy를 사용하므로
reference-free 원칙과 충돌하지 않는다. 다만 validation과 best checkpoint는
반드시 `optimizer.eval()` 이후 averaged \(X_t\)로 계산하고 저장해야 한다.

AMUSE를 사용하면서 \(Y_t\) validation과 \(X_t\) validation을 섞으면 AMUSE의
핵심 안정화 효과를 잘못 평가하게 된다.

### 6.5 `torch.compile`

현재 compile 대상은 model이고 optimizer는 compile하지 않는다. 이 구조는
원칙적으로 AMUSE와 양립할 수 있다. 그러나 optimizer `train()`/`eval()`이
parameter storage를 in-place interpolation하므로 compiled model의 guard와
validation 전환을 실제 smoke test로 확인해야 한다.

## 7. Precision Risk

공식 implementation은 Newton-Schulz 시작 시 update matrix를 다음처럼 변환한다.

```python
X = G.bfloat16()
```

즉, model parameter, gradient, momentum, \(Z_t\)는 `float64`로 유지되더라도
matrix orthogonalization 자체는 `bfloat16`에서 수행된다.

로컬 CPU smoke에서는 `float64` matrix/vector parameter에 대해 세 optimizer
step, `eval()`/`train()` round trip, finite update가 성공했다. State의
`z`, matrix momentum, fallback second moment도 `float64`로 유지됐다. 이것은
API가 동작한다는 확인일 뿐, PDE objective에서 필요한 optimizer precision이
충분하다는 증거는 아니다.

첫 구현에서는 official behavior를 보존해 재현성을 우선하고,
`newton_schulz_dtype="bfloat16"`을 provenance에 명시하는 것이 타당하다.
다음 상황이 관측될 때만 `float32` orthogonalization ablation을 추가한다.

- canonical energy가 BF16 quantization floor에서 정체함
- update norm 또는 gradient/update cosine이 비정상적으로 양자화됨
- AdamW/SOAP 대비 final energy가 일관되게 악화함
- CPU/GPU backend별 결과 차이가 큼

`float64` Newton-Schulz를 기본으로 바꾸는 것은 official algorithm과
compute profile을 크게 바꾸므로 권장하지 않는다.

## 8. 기대 장점

### 8.1 Matrix-Dominant Architecture에 적합

현재 parameter의 99.61%가 2D matrix이므로 AMUSE의 주요 update가 model의
대부분에 실제로 적용된다.

### 8.2 Anisotropic Objective에 대한 가능성

Branch/trunk/fuser의 multiplicative interaction과 Green reconstruction을 거친
canonical energy는 parameter-space curvature가 anisotropic할 가능성이 높다.
Muon의 orthogonalized update가 useful bulk direction의 진행을 빠르게 할 수 있다.
다만 현재 project의 Hessian spectrum을 측정한 결과는 아니므로 가설로 취급한다.

### 8.3 Anytime Validation

현재 run은 validation best-energy epoch가 final epoch보다 이른 경우가 있다.
AMUSE의 averaged \(X_t\)가 안정적으로 작동한다면, predefined cosine horizon에
덜 민감한 best-energy trajectory를 만들 가능성이 있다.

### 8.4 SOAP보다 작은 Persistent State 가능성

AMUSE는 Shampoo covariance/eigenbasis를 저장하지 않으므로 SOAP보다 persistent
optimizer state가 작을 가능성이 높다. GPU peak memory와 wall-clock은 실제
profiler로 확인해야 한다.

## 9. 한계와 위험

### 9.1 PDE 구조 문제를 해결하지 않음

AMUSE는 다음을 수정하지 않는다.

- canonical energy의 null space 또는 coercivity
- boundary admissibility
- axial-line adjacency와 topology transition
- projection 또는 Green reconstruction 수식
- training/test source distribution mismatch
- Green kernel approximation error

Optimizer는 현재 scalar objective를 더 효율적으로 줄일 수 있을 뿐,
objective가 누락한 physics를 추가하지 않는다.

### 9.2 Small Model에서의 Wall-Clock Overhead

약 1.8M parameter model에서는 forward/backward가 LLM보다 작다. 다수의
`256x256` matrix와 큰 fuser matrix에 Newton-Schulz를 매 step 적용하는 비용이
전체 wall-clock을 지배할 수 있다. AMUSE의 step 감소가 곧 wall-clock 감소를
뜻하지 않는다.

### 9.3 Freshness and Implementation Maturity

AMUSE v2는 2026-07-14에 공개된 매우 새로운 preprint이고 official repository도
package release가 아닌 research code다. 따라서 pinned vendoring, attribution,
upstream numerical regression test가 필요하다.

### 9.4 Hyperparameter Transfer Risk

논문의 LLM experiment는 AMUSE learning rate `0.01`을 자주 사용하지만,
현재 project에 이 값을 그대로 적용하는 것은 위험하다. Muon scaling과 current
canonical energy gradient scale이 다르므로 별도 stability screen이 필요하다.

## 10. 권장 Pilot

AMUSE를 구현한다면 첫 실험은 default 변경이 아니라 다음 paired pilot이어야 한다.

### Fixed Conditions

- 같은 geometry, fixed indexed-GP source samples, validation samples
- 같은 model initialization seed
- 같은 GreenNet checkpoint
- 같은 canonical energy objective
- 같은 pre-projection fuser setting
- 같은 batch order와 gradient clipping
- 같은 optimizer-step budget

### Initial AMUSE Screen

| Hyperparameter | Initial values |
|---|---|
| Learning rate | `1e-3`, `2e-3`, `5e-3` |
| Warmup | 160 optimizer steps |
| `beta_init` | `0.4`, `0.6` |
| `rho` | `0.8` |
| Muon momentum | `0.95` |
| Fallback `beta2` | `0.999` |
| Fallback epsilon | `1e-10` |
| Weight decay | `0.05` |
| Gradient clipping | `1.0` |
| Post-warmup decay | none |
| Newton-Schulz dtype | official `bfloat16` |

`5e-3`까지 stable하고 under-updating이 관측될 때만 `1e-2`를 추가한다. 논문의
`0.01`을 첫 단일 run으로 사용하지 않는다.

### Comparison

1. AdamW, current tuned SOAP, AMUSE를 equal optimizer-step budget으로 비교한다.
2. 같은 세 optimizer를 equal wall-clock budget으로 비교한다.
3. initial screen은 짧게 수행하고, 안정적인 후보만 3개 paired seed로 확장한다.
4. best validation canonical-energy checkpoint를 비교한다.
5. `rel_sol`과 `rel_flux`는 test/evaluation diagnostic으로만 사용한다.

### Required Telemetry

- effective step learning rate
- \(\beta_t\), \(c_t\), averaging weight
- \(\|X_t-Z_t\|\), \(\|Y_t-X_t\|\)
- gradient norm before clipping
- matrix update norm and gradient/update cosine
- optimizer step mean/p95/max time
- peak CUDA memory
- nonfinite update count
- validation canonical energy

## 11. Suggested Implementation Boundary

향후 구현 범위는 다음으로 제한하는 것이 타당하다.

- `optimizer.name="amuse"` complex CouplingNet-only opt-in
- pinned official source와 Apache-2.0 attribution
- strict AMUSE nested config
- explicit parameter-group classifier
- optimizer lifecycle adapter
- optimizer-step warmup and external scheduler rejection
- averaged-\(X_t\) validation/checkpoint
- provenance와 telemetry
- focused numerical and trainer integration tests

GreenNet, unit-square CouplingNet, model architecture, loss, projection,
reconstruction, dataset schema는 변경하지 않는다.

## Final Assessment

| Question | Assessment |
|---|---|
| 기술적으로 적용 가능한가 | 예 |
| 현재 model structure와 맞는가 | 높음, matrix parameter 99.61% |
| 현재 trainer에 drop-in 가능한가 | 아니오 |
| SOAP/AdamW를 즉시 대체해야 하는가 | 아니오 |
| short paired pilot 가치가 있는가 | 예 |
| PDE structural issue를 해결하는가 | 아니오 |
| current default로 추천하는가 | 아니오 |

- 기술적 적용 가능성에 대한 확신도: **0.94**
- 올바른 lifecycle/config를 구현할 수 있다는 확신도: **0.93**
- AMUSE가 tuned SOAP보다 좋은 wall-clock과 field quality를 낼 가능성:
  **0.55**

마지막 값의 불확실성은 규칙 모호성이 아니라 정보 부족이다. AMUSE의 공개
실험에는 현재와 같은 physics-constrained operator-learning setting이 없기
때문이다.

## Primary Sources

1. [AMUSE: Anytime MUon with Stable Gradient Evaluation](https://arxiv.org/abs/2605.22432)
2. [Official AMUSE repository](https://github.com/kjeiun/amuse)
3. [Official AMUSE optimizer implementation](https://github.com/kjeiun/amuse/blob/main/src/optim/AMUSE.py)
4. [Official language-model integration notes](https://github.com/kjeiun/amuse/blob/main/src/lm/README.md)
