# SOAP Optimizer Optional Integration Plan

## Summary

- SOAP은 **ShampOo with Adam in the Preconditioner eigenbasis**의 약자다.
  Shampoo 방식으로 weight matrix의 row/column gradient covariance를 추적하고,
  그 eigenbasis로 gradient를 회전한 뒤 Adam update를 수행한다.
- SOAP은 Hessian을 계산하는 Newton optimizer가 아니며 매 step마다 대규모
  matrix equation을 푸는 방식도 아니다. Covariance factor는 매 step
  갱신하지만 eigenbasis는 `precondition_frequency` optimizer step마다
  갱신한다.
- 현재 Complex CouplingNet은 약 1.8M parameters이고 약 99.6%가 2D
  matrix이므로 SOAP의 구조적 적용 가능성은 높다. 표준
  `torch.optim.Optimizer` API, gradient clipping, warmup/cosine scheduler와
  호환된다.
- AdamW를 대체하지 않고 backward-compatible 기본값으로 유지한다. SOAP은
  `coupling_training.optimizer.name="soap"`일 때만 선택되는 complex
  CouplingNet 전용 opt-in 기능으로 추가한다.
- GreenNet과 unit-square CouplingNet은 v1 범위에서 변경하지 않는다.
  Unit-square에서 SOAP을 요청하면 명확한 오류를 발생시킨다.
- 원 논문은 360M/660M language-model pretraining에서 AdamW보다 40% 이상
  적은 iteration과 35% 이상 짧은 wall-clock을 보고했지만, 현재
  physics-constrained operator-learning 문제에 같은 효과가 보장되지는 않는다.

## SOAP Algorithm

Weight gradient \(G_t\in\mathbb R^{m\times n}\)에 대해 다음 통계를 유지한다.

\[
L_t=\beta L_{t-1}+(1-\beta)G_tG_t^\top,
\qquad
R_t=\beta R_{t-1}+(1-\beta)G_t^\top G_t.
\]

\[
L_t=Q_L\Lambda_LQ_L^\top,
\qquad
R_t=Q_R\Lambda_RQ_R^\top,
\qquad
\widetilde G_t=Q_L^\top G_tQ_R.
\]

회전된 basis에서 Adam moments를 갱신한다.

\[
\widetilde m_t=\beta_1\widetilde m_{t-1}+(1-\beta_1)\widetilde G_t,
\qquad
\widetilde v_t=\beta_2\widetilde v_{t-1}+(1-\beta_2)\widetilde G_t^2.
\]

\[
U_t
=
Q_L
\frac{\widetilde m_t}{\sqrt{\widetilde v_t}+\epsilon}
Q_R^\top.
\]

2D matrix에는 two-sided SOAP preconditioning을 적용하고,
`precondition_1d=false`인 1D bias와 activation parameters에는
original-basis Adam-style update를 적용한다.

## Applicability

- **적합한 부분:** Complex CouplingNet의 source, coefficient, geometry,
  transverse branches와 branch/trunk fuser 대부분이 2D matrix다.
- **기존 학습 경로:** `zero_grad(set_to_none=True)`, backward, gradient
  clipping, `optimizer.step()`, epoch 단위 scheduler 순서를 유지한다.
- **Scheduler:** 기존 `CouplingLearningRateSchedule`은
  `Optimizer.param_groups`만 사용하므로 재사용한다. SOAP frequency는 epoch가
  아닌 optimizer step 단위라는 점을 로그와 문서에 명시한다.
- **Compile:** 현재 `torch.compile` 대상은 model뿐이므로 optimizer compile은
  추가하지 않는다.
- **Checkpoint:** model-only safetensors 형식을 유지한다. SOAP optimizer
  state를 포함한 interrupted-training resume는 v1에서 지원하지 않는다.
- **Reference-free 원칙:** SOAP은 현재 scalar objective의 gradient만 사용하므로
  reference `sol/phi/psi`를 loss나 checkpoint selection에 추가하지 않는다.

## Advantages And Disadvantages

- **장점:** Matrix 방향 간 gradient correlation 반영, anisotropic
  branch/fuser conditioning 개선 가능성, large-batch 환경 적합성, 더 적은
  optimizer step으로 canonical energy에 도달할 가능성.
- **단점:** AdamW보다 큰 optimizer state, matrix projection과 periodic
  QR/eigendecomposition 비용, float64 모델에서의 추가 비용,
  LR/betas/weight-decay/frequency retuning 필요.
- **중요한 한계:** SOAP은 loss null space, 누락된 boundary condition,
  admissibility, geometry adjacency 또는 reconstruction 수식 오류를 해결하지
  않는다.
- **실험 리스크:** 현재 모델은 원 논문의 모델보다 작으므로 iteration 감소가
  wall-clock 감소로 이어지지 않을 수 있다.
- **구현 리스크:** 공식 repository는 preliminary implementation이며 package
  release 대신 `soap.py` 복사를 안내한다.
- Meta Distributed Shampoo의 SOAP mode는 precision 및 distributed 기능이
  풍부하지만 v1에는 과도하므로 후속 대안으로 남긴다.

## Public Configuration

기존 `learning_rate`와 `weight_decay`는 optimizer 공통 설정으로 유지한다.

```json
"coupling_training": {
  "learning_rate": 0.002,
  "weight_decay": 0.05,
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
```

- `optimizer` block이 없으면 기존과 동일한 AdamW,
  `betas=[0.9,0.999]`, `eps=1e-8`을 사용한다.
- `shampoo_beta=-1`은 covariance EMA에 `betas[1]`을 사용한다.
- `max_precondition_dim`은 upstream의 `max_precond_dim`으로 전달한다.
- Betas는 `[0,1)`, epsilon은 양수, frequency와 dimension은 양의 정수로
  검증한다.
- Unknown keys, non-finite numeric values와 잘못된 boolean 타입은 fail
  fast한다.
- 공식 구현의 첫 `step()`은 preconditioner 초기화만 하고 parameter update를
  생략하므로 이를 유지하고 로그와 테스트로 고정한다.

## Implementation Plan

1. 공식 SOAP repository의 commit
   `a1e553530fde97d0e6b307d7c82ac6d38b072340`을 기준으로 source를
   고정한다.
2. `src/greenonet/optimizers/soap.py`에 algorithm을 vendoring하고 upstream
   URL, commit, MIT attribution을 source header와
   `THIRD_PARTY_NOTICES.md`에 기록한다.
3. Upstream의 update 수식과 첫-step 동작은 변경하지 않는다. Type
   annotation, explicit validation, dense-gradient 검사와 project naming만
   보강한다.
4. Upstream과의 재현성을 위해 v1 basis/factor dtype 동작은 공식 구현을
   따른다. Float64 model에서 finite update가 가능한지는 테스트하되 별도
   `factor_dtype` option은 추가하지 않는다.
5. `src/greenonet/config.py`에 `CouplingOptimizerConfig`와
   `SoapOptimizerConfig`를 추가하고 `CouplingTrainingConfig.optimizer`에서
   strict `from_raw(...)` parsing을 수행한다.
6. `src/greenonet/coupling_optimizer.py`에
   `ComplexCouplingOptimizerFactory`를 추가해 AdamW와 SOAP 생성을
   캡슐화한다.
7. `ComplexCouplingTrainer`의 직접 `optim.AdamW(...)` 생성을 factory
   호출로 교체한다. Loss, projection, reconstruction, gradient clipping과
   scheduler 순서는 변경하지 않는다.
8. Unit-square validation은 `optimizer.name="soap"`을 거부한다. Existing
   unit-square AdamW parameter-group 코드는 수정하지 않는다.
9. `profile_step_time=true`일 때 optimizer-step mean/p95/max,
   preconditioner update count와 peak device memory를 `training.log`와
   `complex_training_metrics.csv`에 기록한다.
10. CUDA timing은 profiling이 활성화된 경우에만 synchronization을 사용한다.
    비활성화된 기본 AdamW 실행에는 timing overhead를 추가하지 않는다.
11. `config_used.json`과 artifact summary에 optimizer name, resolved
    hyperparameters, upstream commit과 model-only checkpoint 정책을
    기록한다.
12. Canonical `configs/complex_coupling.json`은 AdamW 기본값으로 유지하고,
    별도의 `configs/complex_coupling_soap.json`을 paired pilot 설정으로
    추가한다.
13. README, `docs/memory.md`, `docs/soap_optimizer_applicability.md`를 실제
    config와 구현 상태에 맞게 갱신한다.
14. SOAP을 기본 optimizer로 승격하거나 장기 ablation을 실행하는 작업은 이
    구현 범위에 포함하지 않는다.

## Affected Files

- Core config/factory: `src/greenonet/config.py`, 새
  `src/greenonet/coupling_optimizer.py`.
- Optimizer source: 새 `src/greenonet/optimizers/soap.py`, package init와 MIT
  attribution.
- Runtime: `src/greenonet/complex_coupling_trainer.py`.
- Config provenance: train/eval/artifact config loaders와 complex artifact
  summary.
- Config examples: `configs/complex_coupling.json`, 새
  `configs/complex_coupling_soap.json`.
- Tests: 새 optimizer tests와 기존 config, scheduler, complex trainer tests.
- Documentation: `README.md`, `docs/memory.md`,
  `docs/soap_optimizer_applicability.md`.

## Test Plan

- **Config:** Default AdamW, SOAP round-trip, invalid
  beta/epsilon/frequency/dimension, unknown key, unit-square SOAP rejection.
- **Math:** Small deterministic matrix의 SOAP update를 pinned upstream output과
  비교한다.
- **State:** Matrix와 1D parameter state shape, dtype/device,
  `state_dict()` round-trip, `None` gradient와 sparse-gradient rejection을
  검증한다.
- **Frequency:** 첫 step no-op, basis-update frequency와 counter를 검증한다.
- **Optimizer behavior:** 작은 quadratic objective가 감소하고 update가
  finite인지 확인한다.
- **Trainer integration:** SOAP one-epoch smoke, clipping-before-step,
  scheduler compatibility, checkpoint 생성과 optimizer provenance logging을
  확인한다.
- **Telemetry:** Profiling on/off, CPU/CUDA timing, basis-update count와
  peak-memory fields를 검증한다.
- **Regression:** Optimizer block이 없는 config가 기존 AdamW와 동일한 LR
  sequence, parameter groups와 checkpoint keys를 유지하는지 확인한다.
- **Static/full checks:** Focused pytest, 전체 `pytest test`,
  `ruff check src cli test`, `ruff format src cli test`, `mypy src`,
  `git diff --check`.

## Pilot And Acceptance Criteria

- 구현 검증에는 작은 smoke test만 실행하고 실제 장기 training은 실행하지
  않는다.
- 후속 실험은 300-500 epoch pilot부터 시작한다.
- AdamW와 SOAP에서 dataset split, batch order, model seed, GreenNet
  checkpoint, loss, projection, clipping과 scheduler를 동일하게 유지한다.
- AdamW, SOAP `(0.9,0.999)`, SOAP `(0.95,0.95)`을 먼저 비교하고 LR
  `{1e-3,2e-3,3e-3}`과 frequency `{10,20,50}`은 단계적으로 탐색한다.
- 적어도 3개의 paired seed에서 equal optimizer-step budget과 equal
  wall-clock budget을 모두 비교한다.
- Validation canonical energy, detached `rel_sol/rel_flux`, optimizer time,
  basis-update spike, wall-clock과 peak memory를 기록한다.
- 여러 seed에서 동일 energy까지 wall-clock이 줄고 evaluation metric이
  악화되지 않을 때만 SOAP을 유지한다.
- 조건을 만족해도 이번 변경에서 SOAP을 기본값으로 바꾸지는 않는다.

## Rollback Strategy

- Runtime rollback은 `optimizer.name="adamw"`로 변경하거나 `optimizer`
  block을 제거하는 것이다.
- Optimizer는 model architecture와 safetensors key를 변경하지 않으므로
  checkpoint migration은 필요하지 않다.
- Code rollback은 factory의 SOAP branch, vendored implementation, SOAP
  config와 관련 tests만 제거한다.
- Scheduler, objective, projection, reconstruction, dataset 및 GreenNet
  경로에는 rollback 변경이 없어야 한다.
- AdamW regression이 발생하면 SOAP integration을 중단하고 기존 direct
  AdamW construction을 복원한다.

## Confidence

- SOAP 기술적 적용 가능성과 구현 계획에 대한 확신도: **0.96**.
- SOAP이 현재 Complex CouplingNet에서 AdamW보다 좋은 wall-clock과 field
  quality를 낼 가능성에 대한 경험적 확신도: **0.60**.
- 구현 규칙은 명확하다. 남은 불확실성은 규칙 모호성이 아니라,
  physics-constrained 약 1.8M-parameter model에서 SOAP을 직접 검증한 자료가
  부족하다는 **정보 부족**이다.

## References

1. Nikhil Vyas et al.,
   [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321),
   ICLR 2025.
2. Nikhil Vyas et al.,
   [Official preliminary SOAP implementation](https://github.com/nikhilvyas/SOAP).
3. Meta Research,
   [PyTorch Distributed Shampoo and SOAP mode](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md).

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/plan.md`의
"SOAP Optimizer Optional Integration Plan"을 기준 문서로 참고하여 SOAP
optimizer의 optional integration을 끝까지 구현한다.

완료는 다음 조건으로 검증한다.

- optimizer block이 없는 기존 config가 AdamW 동작을 그대로 유지할 것,
- complex CouplingNet에서 `optimizer.name="soap"` 학습 smoke가 성공할 것,
- SOAP config validation 및 pinned-upstream numerical tests가 통과할 것,
- gradient clipping과 warmup/cosine scheduler가 두 optimizer에서 동작할 것,
- model checkpoint key와 safetensors 형식이 변경되지 않을 것,
- optimizer provenance와 optional timing/memory telemetry가 로그에 기록될 것,
- unit-square CouplingNet과 GreenNet 동작이 변경되지 않을 것,
- focused tests와 전체 regression suite, Ruff, mypy, git diff check가
  통과할 것.

수정 범위는 CouplingNet optimizer config, pinned SOAP implementation,
complex optimizer factory, complex trainer integration, logging/telemetry,
paired pilot config, tests와 관련 문서로 제한한다.

SOAP을 기본 optimizer로 전환하거나 장기 ablation training을 실행하지 않는다.
Model backbone, loss, projection, reconstruction, dataset 및
sample/geometry NPZ schema를 변경하지 않는다.

각 구현 시도 후 가장 작은 관련 optimizer/config/trainer tests를 먼저 실행하고,
통과한 뒤 전체 regression suite를 실행한다.

AdamW backward compatibility 또는 checkpoint compatibility를 유지할 수
없다면 작업을 중단하고 다음을 보고한다.

1. 정확히 호환되지 않는 config, optimizer-state 또는 tensor contract,
2. 영향을 받는 기존 config와 checkpoint,
3. AdamW 기본 동작을 보존하는 가장 작은 rollback 또는 migration 전략.
```
