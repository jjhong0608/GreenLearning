# Explicit Global Training Seed 구현 계획

## Summary

GreenNet과 CouplingNet에 독립적인 base training seed를 추가하고, model initialization, synthetic Green data, DataLoader shuffle, dropout 및 CPU/CUDA RNG를 재현 가능하게 만든다. 공식 training CLI의 활성 학습 단계에는 seed를 필수로 요구하고, 논문용 config에서는 PyTorch deterministic algorithms도 활성화한다.

현재 `dataset.coupling_source.indexed_gp.seed`는 CouplingNet source realization만 결정하며 새 training seed와 독립적으로 유지한다. Model architecture, optimizer 수식, checkpoint tensor key와 dataset schema는 변경하지 않는다.

구현 확신도는 **0.98**이다. 규칙이나 정보 부족은 없다. 다만 서로 다른 GPU, CUDA, cuBLAS, PyTorch 또는 compiler version 사이의 bitwise 동일성은 seed만으로 보장할 수 없으며, 동일 software/hardware 환경에서의 재현성을 보장 대상으로 둔다.

## Public Configuration

GreenNet과 CouplingNet에 각각 독립적인 seed를 둔다.

```json
"training": {
  "seed": 0,
  "deterministic_algorithms": true
},
"coupling_training": {
  "seed": 0,
  "deterministic_algorithms": true
}
```

- `training.seed`는 GreenNet data generation, model initialization, shuffle 및 runtime RNG의 base seed다.
- `coupling_training.seed`는 CouplingNet model initialization, shuffle 및 runtime RNG의 base seed다.
- Seed는 boolean이 아닌 정수이며 범위는 `[0, 2^32-1]`로 검증한다.
- `deterministic_algorithms`는 strict boolean이며 기본값은 `false`로 둔다.
- `pipeline.run_green=true`인데 `training.seed`가 누락되면 fail fast한다.
- `pipeline.run_coupling=true`인데 `coupling_training.seed`가 누락되면 fail fast한다.
- 실행하지 않는 stage의 seed는 생략할 수 있어 pretrained GreenNet 기반 CouplingNet config를 간결하게 유지한다.
- 기존 checkpoint의 evaluation/export는 seed가 없는 과거 `config_used.json`도 계속 허용한다. 명시적 seed 요구는 새 training 시작에만 적용한다.
- 논문 실험용 complex config에는 해당 활성 stage의 seed와 `deterministic_algorithms=true`를 명시한다.

## Seed Semantics

Base seed 하나에서 SHA-256 기반의 stable namespace derivation으로 다음 uint32 sub-seed를 만든다.

```text
green:data_train
green:data_valid
green:model
green:runtime
green:loader_train
green:loader_lbfgs

coupling:model
coupling:runtime
coupling:loader_train
```

- Python의 process-randomized `hash()`는 사용하지 않는다.
- Green train/validation data seed를 분리해 training sample 수가 달라져도 validation data가 변하지 않게 한다.
- DataLoader는 전용 CPU `torch.Generator`를 사용해 global RNG 소비와 shuffle 순서를 분리한다.
- Model 생성 직전에 model seed를 적용하고, 생성 직후 runtime seed를 다시 적용해 parameter 수 차이가 dropout/runtime RNG를 이동시키지 않게 한다.
- Coupling indexed-GP source는 계속 `dataset.coupling_source.indexed_gp.seed`로 결정한다.
- 같은 Coupling training seed에서 `num_train`이 달라도 model initialization은 동일해야 한다.
- 현재 constructor 순서상 `product`와 `product_fuser`의 공통 parameter는 같은 seed에서 동일하게 초기화하고, `product_fuser`의 추가 layer만 별도 parameter로 존재하는 것을 테스트로 고정한다.

## Implementation Changes

1. `src/greenonet/config.py`의 `TrainingConfig`와 `CouplingTrainingConfig`에 `seed: int | None`과 `deterministic_algorithms: bool`을 추가한다. 공통 validator로 type/range를 검증하고, pipeline-aware validator가 활성 stage의 missing seed를 거부한다.

2. 새 `src/greenonet/reproducibility.py`에 seed dataclass와 적용 helper를 둔다. Helper는 `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`을 적용하고 named sub-seed 및 DataLoader generator를 생성한다.

3. Strict mode에서는 `torch.use_deterministic_algorithms(True, warn_only=False)`, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`를 적용한다. CUDA가 활성화되기 전에 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 설정하고 unsupported nondeterministic operation은 조용히 fallback하지 않고 오류로 노출한다.

4. `cli/train.py`는 config parsing 직후 활성 stage seed를 검증한다. GreenNet과 CouplingNet을 순차 실행하더라도 각 stage 시작 시 자기 seed context를 다시 적용해 GreenNet RNG 소비가 CouplingNet initialization에 영향을 주지 않게 한다.

5. GreenNet runner에서 현재 `seed=training_cfg.epochs` 전달을 제거하고 `training.seed`를 source of truth로 사용한다. Train data, validation data, model 및 runtime seed를 각각 적용하고, complex 및 legacy Green trainer의 shuffled DataLoader와 LBFGS loader에 명시적 generator를 전달한다.

6. CouplingNet은 model 생성 전에 coupling model seed를 적용하고 생성 후 runtime seed를 재설정한다. Complex 및 legacy Coupling trainer의 shuffled DataLoader에 coupling loader generator를 전달한다. Validation/evaluation loader는 계속 `shuffle=false`로 유지한다.

7. Direct trainer에 이미 생성된 model을 전달하는 API는 유지한다. 공식 reproducibility guarantee는 `cli/train.py`와 runner path에 적용하며, direct caller가 model initialization까지 재현하려면 공개 seed helper를 model 생성 전에 호출하도록 문서화한다.

8. `config_used.json`에는 입력한 base seed와 deterministic flag를 materialize하고, Green/Coupling별 resolved sub-seed, 적용 범위, device, strict mode 및 source-seed 분리 정책을 provenance block으로 기록한다.

9. `training.log`에는 stage 시작 시 base/model/runtime/loader/data seed, deterministic mode, CUDA/cuDNN 상태를 한 번 기록한다. Artifact summary에도 base seed와 deterministic provenance를 전달한다.

10. Canonical GreenNet 및 ComplexCouplingNet config에 명시적 seed를 추가한다. Archived checkpoint의 `config_used.json`은 수정하지 않으며 model-only safetensors 형식과 state-dict key는 유지한다.

11. README와 `docs/memory.md`에 source seed와 training seed의 차이, paired-ablation 절차, strict determinism의 환경 한계, model-only checkpoint가 RNG/optimizer state resume를 지원하지 않는다는 점을 기록한다.

## Test Plan

- Config tests: valid boundary seed, negative/overflow/bool/float rejection, non-boolean deterministic flag, active-stage missing seed rejection, inactive-stage omission 허용, save/load round-trip.
- Seed helper tests: 같은 base/namespace는 같은 sub-seed, 다른 stage/namespace는 다른 sub-seed, Python/NumPy/Torch RNG 반복 일치.
- GreenNet tests: 같은 seed에서 generated train/validation data, initial state, shuffle order와 one-step result 일치; 다른 seed에서 차이 발생; training sample 수가 달라도 model 및 validation seed가 불변.
- CouplingNet tests: 같은 seed에서 state dict, minibatch order와 one-step parameter update 일치; 다른 seed에서 차이 발생; indexed-GP source seed는 training seed 변경에 영향받지 않음.
- Paired fusion tests: 같은 seed의 `product`와 `product_fuser`에서 공통 state-dict tensor가 정확히 같고 추가 fuser tensor만 다름.
- Pipeline tests: Green과 Coupling을 같이 실행하거나 pretrained Green을 사용할 때 Coupling initialization이 동일함.
- Determinism tests: CPU strict mode의 반복 결과가 exact match하고, CUDA strict test는 가용 환경에서 실행하며 unsupported op는 명시적 오류로 검증한다.
- Provenance tests: `config_used.json`, logs, artifacts에 base/resolved seed와 deterministic mode가 기록됨.
- Regression tests: 기존 checkpoint load/export, indexed-GP identity, optimizer, scheduler, GreenNet LBFGS, unit-square 및 complex paths의 기존 behavior 유지.
- 검증 순서: focused config/reproducibility tests, Green/Coupling trainer tests, 전체 `pytest test`, `ruff check src cli test`, `ruff format src cli test`, `mypy src`, `git diff --check`.

## Rollback Strategy

- Runtime rollback은 `deterministic_algorithms=false`로 strict kernel enforcement만 끄되 seed 기반 initialization/shuffle 재현성은 유지한다.
- Code rollback은 reproducibility helper, 두 config field, CLI stage seeding, DataLoader generator 및 provenance만 제거한다.
- Model architecture와 checkpoint key가 바뀌지 않으므로 checkpoint migration은 필요하지 않다.
- Strict CUDA mode가 특정 SOAP 또는 compiled operation과 충돌하면 seed 기능은 유지하고, 정확한 unsupported operation과 환경을 보고한 뒤 해당 experiment config에서만 strict mode를 끈다.
- Model-only checkpoint는 중간 epoch RNG/optimizer state를 저장하지 않으므로 exact resume는 이번 범위에 포함하지 않는다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Explicit Global Training Seed 구현 계획"을 기준 문서로 참고하여 GreenNet과
CouplingNet의 명시적 training seed 및 deterministic reproducibility integration을
끝까지 구현한다.

완료는 다음 조건으로 검증한다.

- `training.seed`와 `coupling_training.seed`가 독립적인 base seed로 동작할 것,
- 활성 training stage에서 seed가 누락되면 fail fast할 것,
- source-generation seed와 training seed의 의미가 분리되어 유지될 것,
- 같은 seed에서 model initialization, DataLoader order 및 one-step update가
  반복 실행 간 일치할 것,
- 다른 seed에서는 initialization과 shuffle order가 달라질 것,
- GreenNet 실행 여부가 CouplingNet initialization을 변경하지 않을 것,
- training sample 수가 달라도 같은 seed의 model initialization이 유지될 것,
- `product`와 `product_fuser`의 공통 parameter가 paired seed에서 동일할 것,
- configurable strict deterministic mode와 CUDA fail-fast 동작이 구현될 것,
- config_used, training log 및 artifact provenance에 base/resolved seed가 기록될 것,
- 기존 model checkpoint tensor key와 safetensors 형식이 변경되지 않을 것,
- indexed-GP source identity, optimizer, scheduler, LBFGS, projection,
  reconstruction 및 loss semantics가 변경되지 않을 것,
- focused tests와 전체 pytest, Ruff, mypy, git diff check가 통과할 것.

수정 범위는 training config, reproducibility helper, Green/Coupling runner와
DataLoader wiring, seed provenance, canonical configs, 관련 tests와 문서로 제한한다.

Model backbone, objective, projection, reconstruction, source-generation formula,
geometry/sample NPZ schema, optimizer 수식 및 exact-resume 기능은 변경하지 않는다.

각 구현 단계 후 가장 작은 config/seed tests를 먼저 실행하고, Green/Coupling
trainer integration tests를 거쳐 전체 regression suite를 실행한다.

동일 software/hardware 환경에서도 deterministic execution을 유지할 수 없다면
작업을 중단하고 다음을 보고한다.

1. nondeterministic한 정확한 operation과 실행 환경,
2. 영향을 받는 GreenNet 또는 CouplingNet stage와 test,
3. seed 기반 paired initialization/shuffle을 보존하는 가장 작은 fallback 전략.
```
