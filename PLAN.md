# Cumulative Optimizer-Step Scheduler and Fixed-Step Validation Plan

## Summary

GreenNet과 CouplingNet의 AdamW/SOAP 학습에서 learning-rate scheduler를 epoch가 아닌 **누적 `optimizer.step()` 호출 수**로 갱신한다. Validation도 epoch 경계와 분리하여 `validation_every_steps`마다 실행하고, 전체 first-stage 학습의 마지막 step에서는 interval 배수가 아니더라도 한 번 더 실행한다.

적용 범위는 다음으로 고정한다.

- Complex 및 legacy unit-square CouplingNet 모두 적용한다.
- Complex 및 unit-square GreenNet의 AdamW/SOAP first stage에 적용한다.
- GreenNet LBFGS의 optimizer, scheduler 부재, epoch-level diagnostic 동작은 그대로 유지한다.
- CouplingNet periodic checkpoint의 `every_epochs`와 파일명은 그대로 유지한다.
- Model, loss, projection, reconstruction, optimizer 수식, checkpoint tensor key는 변경하지 않는다.

## Public Config

`TrainingConfig`와 `CouplingTrainingConfig`에 동일한 두 옵션을 추가한다.

```json
{
  "use_lr_schedule": true,
  "warmup_steps": 240,
  "validation_every_steps": 24
}
```

- `warmup_steps`: optional nonnegative integer다.
- `validation_every_steps`: active validation dataset을 사용하는 run에서는 반드시 명시해야 하는 positive integer다.
- Validation이 비활성인 GreenNet run에서는 `validation_every_steps`를 생략한다.
- 기존 `warmup_epochs`는 parse compatibility를 위해 유지한다.
- `warmup_steps`가 없으면 `warmup_epochs * steps_per_epoch`로 변환한다.
- `warmup_steps`와 양수 `warmup_epochs`를 동시에 지정하면 ambiguous config로 fail fast한다.
- 기존 config는 실행 가능하지만 scheduler가 epoch staircase가 아니라 stepwise curve로 바뀌므로 numerical trajectory까지 backward-compatible한 것은 아니다.
- `epochs`는 계속 학습 종료 조건이며 별도의 `total_steps` config는 추가하지 않는다.

## Step-Based Schedule

Epoch당 optimizer step 수와 전체 step 수는 runtime DataLoader에서 계산한다.

\[
B=\operatorname{len}(\text{train loader}),
\qquad
T=E B,
\]

여기서 \(E\)는 `epochs`, \(B\)는 `steps_per_epoch`, \(T\)는 `total_optimizer_steps`다. 현재 `drop_last=false`이므로 map-style dataset에서는 사실상 \(B=\lceil N/\text{batch size}\rceil\)이다.

1-based optimizer step \(s=1,\ldots,T\), effective warmup \(W\)에 대해:

\[
\eta_s
=
\eta_{\max}\frac{s}{W},
\qquad 1\le s\le W,
\]

\[
\eta_s
=
\eta_{\min}
+
\frac12(\eta_{\max}-\eta_{\min})
\left[
1+\cos\left(
\pi\frac{s-W-1}{T-W-1}
\right)
\right],
\qquad W+1\le s\le T.
\]

- \(W=0\)이면 step 1에서 base LR로 시작해 step \(T\)에서 `min_lr`에 도달한다.
- \(W\ge T\)이면 기존 규칙처럼 effective warmup을 \(T-1\)로 제한한다.
- Warmup 마지막 step과 cosine 첫 step은 모두 base LR를 사용한다.
- 여러 parameter group에는 동일 multiplier를 적용해 LR 비율을 보존한다.
- `use_lr_schedule=false`이면 모든 step에서 고정 LR를 사용한다.
- PyTorch 권고 순서대로 매 batch에서 `optimizer.step()` 후 `scheduler.step()`을 정확히 한 번 호출한다. [PyTorch optimizer scheduling documentation](https://docs.pytorch.org/docs/stable/optim)
- SOAP의 첫 preconditioner-initialization no-op도 하나의 optimizer 호출로 계산한다. 이는 기존 equal-step budget 및 SOAP frequency 의미와 일치한다.
- Optimizer 호출이 예외로 실패하면 global step과 scheduler를 진행하지 않는다.

논문용 dataset-size 비교는 다음과 같이 동일한 schedule을 갖는다.

| Train samples | Batch | Epochs | Steps/epoch | Total steps |
|---:|---:|---:|---:|---:|
| 600 | 200 | 800 | 3 | 2400 |
| 1200 | 200 | 400 | 6 | 2400 |
| 2400 | 200 | 200 | 12 | 2400 |
| 4800 | 200 | 100 | 24 | 2400 |

모든 run은 `warmup_steps=240`, `validation_every_steps=24`를 사용하므로 동일한 LR trajectory와 정확히 100회의 scheduled validation을 갖는다.

## Validation Contract

- `global_step`은 first-stage 시작 시 0이며 epoch 경계에서 초기화하지 않는다.
- Validation은 `global_step % validation_every_steps == 0`일 때 실행한다.
- 마지막 optimizer step이 interval 배수가 아니면 final validation을 한 번 추가한다.
- 마지막 step이 이미 interval 배수이면 중복 validation하지 않는다.
- Step 0 baseline validation은 추가하지 않는다.
- Validation은 optimizer update와 scheduler advancement 이후 실행하되, 기록되는 `learning_rate`는 직전 optimizer update에 실제 사용한 LR다.
- Validation 중 `model.eval()`과 `torch.no_grad()`를 사용하고 이후 기존 train mode를 복원한다.
- Complex CouplingNet best-energy 및 best-physics checkpoint는 step validation event에서만 갱신한다.
- Legacy CouplingNet best-`rel_sol` checkpoint도 같은 step validation event에서 갱신한다.
- GreenNet은 기존처럼 validation metric만 계산하며 새로운 best-validation checkpoint 정책은 추가하지 않는다.
- `log_interval`은 epoch-aggregate training log 주기로 유지되고 validation cadence에는 관여하지 않는다.

## Implementation Steps

1. `src/greenonet/config.py`에 `warmup_steps`와 `validation_every_steps`를 추가하고 type, finite/range, conflicting warmup config를 strict하게 검증한다.
2. `src/greenonet/learning_rate_scheduler.py`의 공통 schedule을 optimizer-step 단위로 일반화하고 Green/Coupling wrapper가 DataLoader 생성 후 `steps_per_epoch`로 resolve하도록 변경한다.
3. Config parsing 시에는 unresolved fields만 검증하고, runtime에 `steps_per_epoch`, `total_optimizer_steps`, warmup source와 effective warmup을 확정한다.
4. Complex/legacy Coupling trainer를 batch-level scheduler hook과 validation trigger를 갖도록 리팩터링한다. Epoch train aggregation과 periodic checkpoint는 유지한다.
5. Complex/unit-square Green trainer의 AdamW/SOAP loop도 같은 step controller를 사용한다. `compute_validation_rel_sol=true`일 때만 fixed-step validation을 실행한다.
6. GreenNet LBFGS는 별도 step counter에 포함하지 않고 기존 epoch validation과 scheduler-disabled 동작을 보존한다.
7. \(K=1\) tangent eta cap schedule을 `cap_for_step_index(...)`로 전환해 LR warmup step과 공유한다. \(K\ge2\)의 schedule-not-applicable 경로는 유지한다.
8. Train epoch row와 step validation row를 분리한다. 공통 필드는 `epoch`, `global_step`, `step_in_epoch`, `split`, `learning_rate`로 둔다.
9. Epoch train row에는 `learning_rate_first`, `learning_rate_last`를 추가하고 기존 `learning_rate`는 마지막 optimizer update LR의 alias로 유지한다.
10. Green recorder에는 기존 `phase`를 유지하면서 `split=train|val`을 추가한다. LBFGS row의 first-stage step 필드는 비워 두어 다른 step budget과 혼동하지 않게 한다.
11. Startup log와 provenance에 `steps_per_epoch`, `total_optimizer_steps`, warmup source, configured/effective warmup steps, validation interval과 expected validation count를 기록한다.
12. `config_used.json`에는 configured fields를 저장하고, DataLoader 이후 resolve된 값은 Green/Coupling training-schedule provenance와 `training.log`에 저장한다.
13. Artifact loader는 runtime DataLoader 없이 scheduler를 resolve하지 않는다. 저장된 resolved provenance가 있으면 사용하고, 없으면 configured-only schedule로 명시한다.
14. Existing log plotter는 `global_step`이 있으면 이를 x-axis로 사용하고, 과거 log에서는 epoch parser로 fallback하도록 유지한다.
15. Active validation을 사용하는 shipped Green/Coupling config에 `validation_every_steps=24`를 명시하고, 논문용 equal-step configs는 `warmup_steps=240`으로 통일한다.
16. README와 `docs/memory.md`에 step definition, legacy conversion, mandatory validation interval, final validation, SOAP first-step counting 및 LBFGS 제외 규칙을 기록한다.

## Affected Files

- Config/schedule core: `src/greenonet/config.py`, `src/greenonet/learning_rate_scheduler.py`, Green/Coupling scheduler wrappers와 tangent eta schedule.
- Runtime: `src/greenonet/complex_coupling_trainer.py`, `src/greenonet/coupling_trainer.py`, `src/greenonet/trainer.py`, `src/greenonet/complex_green_trainer.py`.
- Provenance/export: `cli/train.py`, Green/Coupling artifact exporters, `src/greenonet/green_optimizer.py`.
- Analysis/config/docs: Green/Coupling log plotters, `configs/`, `numerical_examples/`, `README.md`, `docs/memory.md`.
- Tests: scheduler, config/CLI, 네 trainer, SOAP telemetry, tangent schedule, artifacts와 log parser tests.

## Test Plan

- Exact LR sequence: warmup, cosine start, final `min_lr`, zero warmup, single-step edge case와 disabled schedule을 검증한다.
- Equal-budget regression: `3x800`, `6x400`, `12x200`, `24x100`이 동일한 2400-step LR sequence를 생성하는지 확인한다.
- Legacy config: `warmup_epochs`가 runtime step으로 정확히 변환되고 `warmup_steps`와의 충돌이 거부되는지 확인한다.
- Validation trigger: interval 배수, non-divisible final step, divisible final deduplication, interval이 total steps보다 큰 경우를 검증한다.
- Trainer integration: 네 trainer에서 scheduler 호출 수가 optimizer 호출 수와 같고 validation이 epoch 수가 아닌 global step에서 발생하는지 확인한다.
- Checkpoint behavior: best checkpoint는 validation event에서만 갱신되고 periodic checkpoint는 기존 epoch cadence와 filename을 유지하는지 확인한다.
- SOAP: 첫 no-op 호출도 global step, LR schedule, validation cadence 및 telemetry count에 포함되는지 확인한다.
- Tangent: \(K=1\) eta cap이 stepwise warmup을 따르고 \(K=2,3,4\)에는 계속 적용되지 않는지 검증한다.
- Mode restoration: mid-epoch validation 후 dropout/train mode가 정확히 복원되는지 확인한다.
- Logging/provenance: CSV/log의 global step, LR start/end, validation index와 resolved schedule metadata를 검증한다.
- LBFGS regression: scheduler가 적용되지 않고 기존 epoch-level diagnostic 및 checkpoint behavior가 유지되는지 확인한다.
- Plot/artifact regression: 새 step log와 과거 epoch log를 모두 읽고 checkpoint tensor key 및 artifact numerical field가 변하지 않는지 확인한다.

검증 순서는 scheduler/config tests, 각 trainer focused tests, artifact/log parser tests, 전체 regression과 정적 검사로 고정한다.

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_coupling_lr_scheduler.py \
  test/test_green_lr_scheduler.py \
  test/test_complex_coupling_trainer.py \
  test/test_coupling.py \
  test/test_green_optimizer.py \
  test/test_complex_green_trainer.py \
  test/test_cli_train.py

PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest test
ruff check src cli test
ruff format src cli test
~/.conda/envs/green_net/bin/python -m mypy src
git diff --check
```

## Rollback Strategy

- Scheduler, validation trigger, recorder schema를 공통 helper와 trainer wiring으로 분리하여 모델 및 objective 코드와 독립적으로 되돌릴 수 있게 한다.
- Rollback 시 epoch-end `scheduler.step()`과 epoch validation을 복원하고 새 step fields를 config에서 제거한다.
- `warmup_epochs`, periodic checkpoint와 과거 log parser를 유지하므로 이전 config와 artifact를 복구할 수 있다.
- Checkpoint는 model-only safetensors이므로 optimizer/scheduler state migration이나 model key migration이 필요 없다.
- 기존 checkpoint, log, artifact 파일은 수정하거나 덮어쓰지 않는다.
- Step 전환 때문에 numerical regression이 발생해도 model, Green reconstruction, projection, loss 또는 optimizer implementation을 되돌리지 않는다.

## Confidence

- 구현 계획 확신도: **0.96**.
- Equal-step 실험에서 LR 및 validation opportunity confound를 제거할 확신도: **0.99**.
- 규칙 모호성은 사용자 선택으로 해소되었다.
- 남은 불확실성은 정보 부족에 해당한다. 특히 GreenNet full-dataset validation을 24 step마다 수행할 때의 wall-clock overhead는 실제 dataset 크기와 hardware에 따라 측정해야 한다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Cumulative Optimizer-Step Scheduler and Fixed-Step Validation Plan"을
기준 문서로 참고하여 optimizer-step scheduler와 fixed-step validation
integration을 끝까지 구현한다.

완료는 다음 조건으로 검증한다.

- GreenNet과 CouplingNet의 AdamW/SOAP first-stage scheduler가 cumulative
  optimizer step마다 정확히 한 번 갱신될 것,
- warmup_steps가 explicit step contract로 동작하고 기존 warmup_epochs는
  steps_per_epoch를 이용한 compatibility fallback으로 동작할 것,
- warmup_steps와 양수 warmup_epochs의 동시 지정은 fail fast할 것,
- validation dataset을 사용하는 run은 positive validation_every_steps를
  반드시 명시할 것,
- 모든 논문용 실험 config가 validation_every_steps=24를 사용할 것,
- validation이 global step의 24배수와 non-divisible final step에서 실행되고
  final duplicate는 생성되지 않을 것,
- train metric은 epoch aggregate, validation metric은 independent step event로
  기록될 것,
- best checkpoint는 step validation에서 갱신되고 periodic checkpoint는 기존
  epoch cadence를 유지할 것,
- SOAP의 first-step initialization call이 global optimizer step에 포함될 것,
- K=1 tangent eta cap은 LR warmup step을 공유하고 K>=2 동작은 바뀌지 않을 것,
- GreenNet LBFGS는 scheduler 없이 기존 epoch-level 동작을 유지할 것,
- config_used, logs, CSV와 provenance에서 configured/resolved step contract를
  구분해 추적할 수 있을 것,
- model architecture, objective, checkpoint tensor key, projection,
  reconstruction 및 dataset schema가 변경되지 않을 것,
- focused tests와 전체 pytest, Ruff, mypy, git diff check가 통과할 것.

수정 범위는 training config, shared step scheduler/validation policy,
GreenNet/CouplingNet trainer wiring, tangent eta schedule, logging/provenance,
artifact/log parsers, active experiment configs, tests와 문서로 제한한다.

Loss, optimizer 수식, model backbone, Green reconstruction, projection,
geometry/sample NPZ schema, LBFGS algorithm과 periodic checkpoint cadence는
변경하지 않는다. 실제 장기 numerical training은 실행하지 않는다.

각 구현 단계 후 scheduler/config focused tests를 먼저 실행하고, 각 trainer와
artifact/log parser tests를 거친 뒤 전체 regression suite를 실행한다.

기존 model checkpoint compatibility 또는 first-stage/LBFGS 경계를 유지할 수
없다면 작업을 중단하고 다음을 보고한다.

1. 정확히 충돌하는 schedule, validation, metric 또는 tensor contract,
2. 영향을 받는 config, checkpoint, log, artifact와 trainer,
3. model과 objective를 보존하는 가장 작은 rollback 또는 migration 전략.
```
