# GreenNet AdamW, SOAP 및 Warmup-Cosine Scheduler 통합 계획

## Summary

- Unit-square 및 complex GreenNet의 기본 optimizer를 `torch.optim.Adam`에서 `torch.optim.AdamW`로 변경하고, GreenNet 경로에서 기존 Adam 구현과 `"adam"` config option을 완전히 제거한다.
- SOAP은 GreenNet 전용 `training.optimizer.name="soap"` opt-in option으로 추가하며, 이미 vendoring된 공식 SOAP 구현과 provenance를 재사용한다.
- Linear warmup + cosine annealing scheduler는 AdamW와 SOAP의 1차 학습 단계에만 적용한다.
- AdamW/SOAP 단계가 끝난 뒤 실행되는 LBFGS는 현재 optimizer 생성, closure, line search, epoch 및 tolerance 설정을 그대로 유지한다.
- GreenONet model, reconstruction loss, `rel_sol`, `rel_green`, quadrature, dataset, geometry, checkpoint tensor key는 변경하지 않는다.
- 이 계획은 사용자가 project root의 `PLAN.md`에 저장하며, 이번 단계에서는 코드를 수정하지 않는다.

## Public Configuration

`TrainingConfig`에 다음 GreenNet optimizer/scheduler 설정을 추가한다.

```json
"training": {
  "learning_rate": 0.0005,
  "weight_decay": 0.0,
  "epochs": 4000,
  "use_lr_schedule": true,
  "warmup_epochs": 100,
  "min_lr": 1e-5,
  "optimizer": {
    "name": "adamw",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "profile_step_time": false,
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

- `optimizer.name`은 `"adamw"` 또는 `"soap"`만 허용한다.
- `optimizer` block이 없으면 `AdamW`, `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0.0`을 사용한다.
- `"adam"`은 지원하지 않으며 명시되면 “GreenNet Adam has been removed; use adamw” 오류로 fail fast한다.
- SOAP 예시 config에서는 `betas=(0.95,0.95)`를 명시하되, optimizer 종류에 따라 숨겨진 dynamic default를 적용하지 않는다.
- `use_lr_schedule=false`이면 AdamW 또는 SOAP가 전체 1차 학습 동안 고정 learning rate를 사용한다.
- `use_lr_schedule=true`이면 `warmup_epochs >= 0`, `0 <= min_lr <= learning_rate`를 검증한다.
- `weight_decay`는 finite nonnegative 값으로 검증하며 기본값 `0.0`으로 기존 GreenNet의 no-weight-decay 의미를 보존한다.
- 기존 Green config에 새 field가 없어도 정상적으로 parse되지만 optimizer는 의도적으로 AdamW로 변경된다.

## Implementation Changes

### 1. Shared Optimizer Configuration

- `GreenOptimizerConfig`를 추가해 AdamW/SOAP 선택, betas, epsilon, profiling 및 SOAP nested config를 strict parsing한다.
- 기존 `SoapOptimizerConfig`는 GreenNet과 Complex CouplingNet이 공유할 수 있도록 설명과 validation error prefix를 일반화한다.
- 기존 vendored `SOAP` source, upstream commit, MIT attribution 및 float64 bridge는 수정하지 않는다.
- `GreenOptimizerFactory`를 추가한다.
  - `adamw`: `torch.optim.AdamW`.
  - `soap`: 기존 vendored `SOAP`.
  - 두 optimizer 모두 `learning_rate`, `weight_decay`, `betas`, `eps`를 resolved Green config에서 받는다.
- 기존 Green trainer의 직접 `optim.Adam(...)` 호출과 Adam 관련 import를 모두 제거한다.
- SOAP 공식 첫-step preconditioner initialization과 parameter-update skip 동작은 유지한다.

### 2. Shared Warmup-Cosine Scheduler

- CouplingNet의 검증된 scheduler 수식을 generic `LinearWarmupCosineSchedule` helper로 추출한다.
- 기존 `CouplingLearningRateSchedule`은 wrapper를 유지해 현재 CouplingNet LR sequence가 bitwise-equivalent하게 보존되도록 한다.
- GreenNet은 같은 core helper를 `TrainingConfig`와 `training.*` 오류 prefix로 사용한다.
- 총 epoch \(T\), effective warmup \(W=\min(W_{\mathrm{configured}},T-1)\)에 대해 다음 LR을 사용한다.

\[
\eta_e=\eta_{\max}\frac{e}{W},
\qquad 1\le e\le W,
\]

\[
\eta_e=
\eta_{\min}
+\frac12(\eta_{\max}-\eta_{\min})
\left[
1+\cos\left(
\pi\frac{e-W-1}{T-W-1}
\right)
\right].
\]

- Epoch 순서는 `현재 LR 기록 -> AdamW/SOAP batch updates -> diagnostics/output -> scheduler.step()`으로 고정한다.
- Scheduler는 epoch마다 정확히 한 번만 step한다.
- 여러 optimizer parameter group이 있으면 기존 group LR 비율을 보존한다.
- SOAP의 `precondition_frequency`는 scheduler epoch가 아니라 optimizer step 기준임을 로그와 문서에 명시한다.

### 3. Green Trainer Integration

- Unit-square `Trainer`와 `ComplexGreenTrainer`가 동일한 Green optimizer factory, scheduler 및 optimizer profiler를 사용한다.
- Current Green configs처럼 dataset sample 수가 batch size보다 작으면 epoch당 optimizer step이 한 번이라는 사실을 telemetry에 정확히 반영한다.
- SOAP profiling이 활성화되면 step mean/p95/max time, step count, basis refresh count 및 CUDA peak memory를 기록한다.
- `green_training_metrics.csv`를 추가해 다음을 기록한다.
  - `phase`: `adamw`, `soap`, `lbfgs`
  - `epoch`, `learning_rate`, `loss`
  - `rel_sol`, optional `val_rel_sol`, optional `rel_green`
  - optional optimizer telemetry
- `training.log` 시작부에 resolved optimizer, weight decay, betas, SOAP frequency와 scheduler 설정을 기록하고 epoch log에 실제 사용 LR을 포함한다.
- `config_used.json`에는 resolved `training.optimizer`를 저장하고 `green_optimizer_provenance`에 implementation, SOAP upstream commit 및 model-only checkpoint policy를 기록한다.
- Optimizer/scheduler state는 safetensors에 넣지 않는다. 중단 지점부터의 exact optimizer resume는 이번 범위에 포함하지 않는다.

### 4. LBFGS Preservation

- AdamW/SOAP epoch가 모두 끝난 후 scheduler와 1차 optimizer를 더 이상 사용하지 않는다.
- 기존 `torch.optim.LBFGS` 생성 인자, `lbfgs_lr`, `max_iter`, `history_size`, `tolerance_grad`, `tolerance_change`, `strong_wolfe` line search 및 closure를 변경하지 않는다.
- LBFGS epoch 중에는 warmup/cosine scheduler를 step하지 않는다.
- LBFGS가 disabled이면 AdamW/SOAP 종료 결과를 바로 최종 checkpoint로 저장한다.
- 비교 가능성을 위해 AdamW/SOAP 종료 시점에 `model_pre_lbfgs.safetensors`를 저장하고, 기존 `model.safetensors`는 전체 LBFGS 종료 후 최종 모델로 유지한다.
- Existing Green model checkpoint tensor key와 load contract는 유지한다.

### 5. Configs, Artifacts, and Documentation

- `configs/default_green.json`과 `configs/complex_green.json`은 AdamW 및 scheduler를 명시적으로 사용하도록 갱신한다.
  - `learning_rate=5e-4`
  - `weight_decay=0.0`
  - `use_lr_schedule=true`
  - `warmup_epochs=100`
  - `min_lr=1e-5`
- Paired optimizer 실험용 `configs/complex_green_soap.json`을 추가하고 AdamW config와 model/data/LBFGS 설정을 동일하게 유지한다.
- SOAP config는 `betas=(0.95,0.95)`, `precondition_frequency=10`, `max_precondition_dim=1024`, `precondition_1d=false`, `normalize_grads=false`, `correct_bias=true`를 명시한다.
- Green artifact summary에는 optimizer/scheduler provenance를 추가하지만 artifact reconstruction 및 `rel_green` 의미는 변경하지 않는다.
- README와 `docs/memory.md`에 Adam 제거, AdamW default, SOAP opt-in, scheduler 적용 범위, LBFGS 분리 및 model-only resume 정책을 기록한다.

## Test Plan

- **Config:** optimizer block 누락 시 AdamW default, Adam 명시 거부, SOAP round-trip, unknown key와 invalid beta/epsilon/weight decay/frequency/dimension 검증.
- **Scheduler:** warmup LR sequence, cosine 시작점, 마지막 epoch의 정확한 `min_lr`, zero warmup, disabled fixed LR, parameter-group ratio 보존을 검증한다.
- **Factory:** AdamW 생성 인자, SOAP 생성 인자와 provenance, SOAP 첫-step no-op 및 float64 finite update를 검증한다.
- **Unit-square trainer:** AdamW fixed/scheduled smoke와 SOAP 2-step 이상 smoke를 수행하고 실제 LR 및 telemetry 기록을 확인한다.
- **Complex trainer:** `forward_pairs(...)`, uniform/split-quadrature reconstruction 모두에서 AdamW와 SOAP smoke를 실행한다.
- **LBFGS:** AdamW/SOAP 종료 뒤 LBFGS가 기존 인자와 closure로 실행되고 scheduler가 LBFGS 중 호출되지 않는지 확인한다.
- **Checkpoint:** 기존 Green safetensors를 새 코드에서 로드할 수 있고 model key가 바뀌지 않으며 pre-LBFGS/final checkpoint가 구분되는지 확인한다.
- **Regression:** CouplingNet scheduler와 SOAP factory의 기존 결과, complex Green scaling, `rel_green`, artifact export 및 unit-square Green reconstruction을 유지한다.

검증 순서는 다음과 같다.

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_green_optimizer.py \
  test/test_green_lr_scheduler.py \
  test/test_runner.py \
  test/test_complex_green_trainer.py \
  test/test_io_config.py \
  test/test_cli_train.py \
  test/test_export_green_artifacts.py

PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_coupling_optimizer.py \
  test/test_coupling_lr_scheduler.py \
  test/test_complex_coupling_trainer.py

PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest test

~/.conda/envs/green_net/bin/python -m ruff check src cli test
~/.conda/envs/green_net/bin/python -m ruff format src cli test
~/.conda/envs/green_net/bin/python -m mypy src
git diff --check
```

## Rollback Strategy

- SOAP runtime rollback은 `training.optimizer.name="adamw"`로 변경한다.
- Scheduler runtime rollback은 `training.use_lr_schedule=false`로 변경한다.
- SOAP 또는 scheduler integration에 문제가 생기면 Green model/loss/data/LBFGS를 건드리지 않고 factory 또는 scheduler branch만 비활성화할 수 있어야 한다.
- Adam은 의도적으로 제거하므로 rollback option으로 유지하지 않는다.
- Optimizer와 scheduler가 model state key를 변경하지 않으므로 checkpoint migration은 없어야 한다.
- Existing checkpoint compatibility가 깨지면 구현을 중단하고 정확한 incompatible key, 영향받는 checkpoint와 최소 migration을 보고한다.

## Assumptions and Confidence

- AdamW는 unit-square와 complex GreenNet 모두의 기본 optimizer다.
- SOAP은 두 GreenNet geometry mode에서 opt-in으로 사용할 수 있다.
- Scheduler는 AdamW/SOAP 단계에만 적용하고 LBFGS에는 적용하지 않는다.
- `weight_decay=0.0`을 기본으로 두어 optimizer class 변경 외의 regularization 변화는 만들지 않는다.
- Canonical Green configs는 scheduler를 활성화하지만, 외부 legacy config가 scheduler field를 생략하면 fixed LR을 사용한다.
- Existing model checkpoint는 호환되며 optimizer/scheduler resume는 지원하지 않는다.
- 구현 계획 확신도는 **0.98**다.
- SOAP의 GreenNet 수렴 개선 가능성에 대한 경험적 확신도는 **0.65**다.
- 남은 불확실성은 규칙 모호성이나 구현 정보 부족이 아니라, effective source batch 25인 GreenNet에서 SOAP이 AdamW보다 좋은 convergence/wall-clock을 제공할지에 대한 실험적 불확실성이다.

