# Complex CouplingNet Source-Only Dataset 및 Fixed GP Source 구현 계획

## Summary

- Complex CouplingNet의 train/validation source는 모든 epoch에서 동일하게 유지하는 **fixed source policy**로 고정한다.
- Source backend는 config에서 다음 두 방식 중 하나를 선택한다.
  - `npz`: 전용 CLI가 미리 생성한 deterministic source-only `.npz`
  - `indexed_gp`: `(base_seed, split_id, sample_index)`로 매번 같은 source를 재현하는 runtime GP generator
- 두 backend는 동일한 GP sampler, seed derivation, geometry masking을 공유하며 같은 설정과 sample identity에 대해 같은 full-grid `rhs`를 생성해야 한다.
- Train/validation reference diagnostics는 별도 config로 on/off한다.
  - enabled: 현재처럼 `sol`을 요구하고 `rel_sol`, optional `rel_flux`를 계산한다.
  - disabled: `rhs`만 요구하고 `rel_sol`, `rel_flux`를 계산하거나 출력하지 않는다.
- Test evaluation은 기존 FEniCSx-generated `rhs/sol/phi/psi` 경로를 유지한다.
- GreenNet, Complex CouplingNet architecture, output contract, objective, projection, reconstruction, model checkpoint key는 변경하지 않는다.
- `plan.md`는 이번 작업에서 수정하지 않으며, 사용자가 이 계획을 project root의 `plan.md`에 작성한다.

## Public Configuration

`dataset`에 complex CouplingNet 전용 nested config를 추가한다.

### 기존 Full-Reference NPZ 동작

Config block이 없을 때도 아래와 동일하게 해석하여 기존 config를 보존한다.

```json
"dataset": {
  "coupling_source": {
    "mode": "npz"
  },
  "reference_diagnostics": {
    "training": true,
    "validation": true
  },
  "training_path": "data/complex_samples/example/train",
  "validation_path": "data/complex_samples/example/valid",
  "test_path": "data/complex_samples/example/test"
}
```

- Train/validation NPZ는 `rhs`, `sol`을 요구한다.
- `phi/psi` 또는 legacy `uxx/uyy`는 optional이다.
- `rel_sol`과 available `rel_flux`를 기존처럼 기록한다.

### Source-Only NPZ 동작

```json
"dataset": {
  "coupling_source": {
    "mode": "npz"
  },
  "reference_diagnostics": {
    "training": false,
    "validation": false
  },
  "training_path": "data/complex_sources/annulus_gp/train",
  "validation_path": "data/complex_sources/annulus_gp/valid",
  "test_path": "data/complex_samples/annulus_reference/test"
}
```

- Train/validation NPZ는 `rhs`만 요구한다.
- 파일에 `sol/phi/psi`가 있더라도 읽거나 diagnostic에 사용하지 않는다.
- Test dataset은 기존 full-reference contract로 별도 로드한다.

### Index-Seeded Fixed GP 동작

```json
"dataset": {
  "coupling_source": {
    "mode": "indexed_gp",
    "indexed_gp": {
      "num_train": 4000,
      "num_valid": 200,
      "seed": 0,
      "lengthscale": 0.2,
      "amplitude": 1.0,
      "mean": 0.0
    }
  },
  "reference_diagnostics": {
    "training": false,
    "validation": false
  },
  "test_path": "data/complex_samples/annulus_reference/test"
}
```

- `training_path`와 `validation_path`는 지정하지 않는다.
- `test_path`는 기존 full-reference test dataset을 계속 가리킨다.
- 같은 index를 다시 읽을 때 source를 다시 계산할 수는 있지만 값은 항상 동일하다.
- Runtime provider는 전체 source를 메모리에 cache하지 않는다. GP covariance factor만 provider 초기화 시 한 번 계산하고, sample은 index seed로 재생성하여 memory 사용량을 sample 수와 독립적으로 유지한다.

### Validation Rules

- `coupling_source.mode`는 `"npz"` 또는 `"indexed_gp"`만 허용한다.
- `npz` mode:
  - `training_path`는 필수다.
  - validation을 사용하면 `validation_path`도 필수다.
  - `indexed_gp` block이 있으면 fail fast한다.
- `indexed_gp` mode:
  - `num_train >= 1`, `num_valid >= 0`
  - `seed >= 0`
  - `lengthscale > 0`, `amplitude >= 0`, finite `mean`
  - `training_path` 또는 `validation_path`가 있으면 unused-path 오류로 fail fast한다.
  - train/validation reference diagnostics가 enabled이면 reference가 없으므로 fail fast한다.
- `best_energy_checkpoint` 또는 `best_physics_checkpoint`가 enabled이면 validation source가 반드시 존재해야 한다.
- Unit-square mode에서 non-default `coupling_source` 또는 `reference_diagnostics`를 사용하면 complex-only option 오류를 낸다.
- Unknown nested key와 잘못된 bool/numeric 타입은 fail fast한다.

## Implementation Changes

### 1. Shared Source Core

- FEniCSx package 아래의 순수 NumPy GP, indexed seed, raw geometry-grid masking을 generic `src/greenonet/complex_sources/` package로 이동한다.
- 다음 공통 contract를 제공한다.
  - separable squared-exponential GP sampler
  - `derive_indexed_seed(base_seed, split, index)`
  - geometry full-grid loader와 valid-mask 적용
  - `generate_fixed_rhs(...)`
- Seed는 기존 규칙을 유지한다.

\[
\mathrm{SeedSequence}
\left[
\mathrm{base\_seed},
\mathrm{split\_id},
\mathrm{sample\_index}
\right].
\]

- Split ID는 `train=0`, `valid=1`, `test=2`로 고정한다.
- GP는 full Cartesian grid에서 먼저 sampling하고, valid point만 full-grid output에 남기며 outside-domain은 `0.0`으로 채운다.
- Existing FEniCSx imports는 backward-compatible re-export 또는 새 shared module import로 전환하여 기존 FEniCSx sample 값과 seed 정책을 바꾸지 않는다.
- NumPy가 권장하는 deterministic ID와 root seed 조합 방식을 따른다: [NumPy parallel random generation](https://numpy.org/doc/stable/reference/random/parallel.html).

### 2. Source Providers

- 공통 provider contract를 추가한다.

```text
len(provider)
provider[index] -> rhs_full, sample_index, sample_name
```

- `NpzComplexSourceProvider`
  - split directory의 `.npz` 파일을 정렬해서 읽는다.
  - diagnostics disabled이면 `rhs`만 요구한다.
  - diagnostics enabled이면 `rhs`, `sol`을 요구하고 optional flux target을 읽는다.
- `IndexedGpComplexSourceProvider`
  - configured sample count를 dataset length로 사용한다.
  - split과 index로 seed를 파생한다.
  - shared source core를 이용해 deterministic masked full-grid `rhs`를 반환한다.
  - `sol/phi/psi`는 반환하지 않는다.
- `ComplexCouplingDataset`은 provider에서 받은 full-grid `rhs`를 기존 valid-point gather, source branch interpolation, amplitude normalization 경로로 전달한다.
- Existing `ComplexCouplingDataset(path, ...)` constructor는 NPZ + diagnostics-enabled default로 유지하여 직접 호출하는 기존 tests와 downstream code를 보존한다.

### 3. Optional Reference Tensors 및 Metrics

- Batch tensor shape와 compile stability를 유지하기 위해 reference가 없을 때는 shape-compatible zero tensor와 `has_solution=false`, `has_flux=false`를 사용한다.
- Diagnostics enabled split에서는 `sol` 누락을 오류로 처리한다.
- Diagnostics disabled split에서는 reference array가 파일에 있어도 읽지 않는다.
- Trainer와 evaluator metric 계산을 조건부로 변경한다.
  - `has_solution=true`: `rel_sol` 계산
  - flux target available: `rel_flux` 계산
  - 둘 다 false: metric key 자체를 생성하지 않음
- Diagnostics disabled이면 console log와 `complex_training_metrics.csv`에서 `rel_sol`, `rel_flux` field를 출력하지 않는다.
- Canonical energy, bulk/boundary energy, learning rate, optimizer telemetry, best-energy/best-physics checkpoint 동작은 변경하지 않는다.
- Reference-free objective 계산에는 계속 predicted `u_phi/u_psi`, projected `phi/psi`, `rhs`, coefficients, geometry를 사용한다.

### 4. Dataset/Runner Dispatch

- `DatasetConfig`에 strict `from_raw(...)` parsing을 추가해 path, dtype, scale fields와 새 nested config를 한 곳에서 복원한다.
- Train, eval, Green/Coupling artifact config loader의 중복 `DatasetConfig(**raw)` parsing을 공통 parser로 통일한다.
- Complex train dispatch:
  - `npz`: 기존 paths에서 NPZ provider 생성
  - `indexed_gp`: config count와 GP options로 generated provider 생성
- Test dispatch는 source backend와 무관하게 `test_path`의 full-reference NPZ를 diagnostics-enabled mode로 로드한다.
- `config_used.json`에는 resolved source backend, GP parameters, diagnostics policy, fixed-source convention을 기록한다.

### 5. Source-Only NPZ CLI

새 CLI `cli/make_complex_sources.py`를 추가한다.

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python \
  cli/make_complex_sources.py \
  --geometry data/geometry/annulus_02_05_1_128.npz \
  --out data/complex_sources/annulus_gp \
  --num-train 4000 \
  --num-valid 200 \
  --lengthscale 0.2 \
  --amplitude 1.0 \
  --mean 0.0 \
  --seed 0
```

CLI options:

```text
--geometry PATH
--out PATH
--num-train INT
--num-valid INT
--lengthscale FLOAT
--amplitude FLOAT
--mean FLOAT
--seed INT
--overwrite
--validate / --no-validate
```

- v1 CLI는 serial generation으로 고정한다. Source generation은 FEniCSx solve보다 충분히 저렴하며, parallel worker option은 이번 범위에 포함하지 않는다.
- OOP 구조로 `ComplexSourceGenerationConfig`, generator, atomic writer, CLI class를 둔다.
- Rich console logging과 `<out>/make_complex_sources.log`를 사용한다.
- Output:

```text
<out>/
  train/sample_000000.npz
  valid/sample_000000.npz
  generation_summary.json
  make_complex_sources.log
```

- 각 sample NPZ는 full-grid float64 `rhs` 하나만 저장한다.
- Existing file은 기본 fail fast, `--overwrite`일 때만 atomic replace한다.
- Summary에는 geometry path, grid shape, valid count, split counts, GP parameters, root seed, indexed seed policy, outside-domain policy를 기록한다.
- Validation enabled이면 저장 직후 key, shape, dtype, finite value, outside-domain zero를 검사한다.

### 6. Documentation

- `README.md`에 두 backend config와 source-only generation command를 추가한다.
- `docs/memory.md`에는 다음 durable convention을 기록한다.
  - train/validation source는 fixed
  - source backend는 NPZ 또는 indexed GP
  - 두 backend는 동일 sample identity를 공유
  - diagnostics-off train/validation은 `rhs` only
  - test reference는 FEniCSx 경로 유지
- FEniCSx sample docs에는 GP source core가 공유되지만 PDE reference solve contract는 변경되지 않는다고 명시한다.
- 사용자가 작성할 root `plan.md`는 구현 agent가 기준 문서로 사용한다.

## Affected Files

- **Config 및 dispatch:** `src/greenonet/config.py`, `cli/train.py`, eval/artifact config loaders.
- **Core data:** 새 `src/greenonet/complex_sources/`, `src/greenonet/complex_coupling_data.py`, trainer/evaluator의 conditional diagnostic path.
- **CLI:** 새 `cli/make_complex_sources.py`; 기존 FEniCSx GP/seed modules는 shared core를 import하거나 re-export한다.
- **Tests:** source core/CLI tests, complex dataset/trainer/config tests, FEniCSx regression tests.
- **Docs:** `README.md`, `docs/memory.md`, FEniCSx/source-generation documentation.
- Model architecture, GreenNet, projection, reconstruction, sample geometry schema는 영향받지 않는다.

## Test Plan

### Source Core 및 Backend Parity

- 같은 `(seed, split, index, geometry, GP config)`에서 repeated generation이 동일한 `rhs`를 반환한다.
- 다른 split 또는 index는 다른 source를 반환한다.
- DataLoader access order와 shuffle에 관계없이 sample identity가 유지된다.
- NPZ CLI의 `sample_000047.npz`와 indexed provider의 index 47이 같은 `rhs`를 생성한다.
- Full-grid shape, finite values, valid-point values, outside-domain zero를 검증한다.
- Existing FEniCSx generator가 shared core 전후 동일한 indexed source를 생성하는지 regression으로 확인한다.

### Config

- Block이 없는 기존 config가 `mode=npz`, diagnostics train/validation enabled로 해석된다.
- NPZ source-only 및 indexed GP config가 parse/save/load round-trip한다.
- Invalid mode, count, seed, GP numeric value, unknown key, conflicting path를 거부한다.
- Indexed GP와 enabled reference diagnostics 조합을 거부한다.
- Unit-square mode에서 complex source options를 거부한다.
- Validation checkpoint enabled인데 validation source가 없는 config를 거부한다.

### Dataset 및 Collate

- Source-only NPZ가 diagnostics-off mode에서 `rhs`만으로 load/collate된다.
- 같은 NPZ가 diagnostics-on mode에서는 missing `sol` 오류를 낸다.
- Full-reference NPZ의 기존 `rhs/sol/phi/psi` behavior가 유지된다.
- Indexed provider가 epoch 반복과 접근 순서에 관계없이 같은 sample을 반환한다.
- Both provider outputs가 동일한 source branch와 source amplitude를 만든다.
- Missing reference를 나타내는 mask와 zero placeholder가 batch/device 이동에서 유지된다.

### Trainer 및 Evaluation

- Source-only NPZ와 indexed GP 각각 one-epoch complex training smoke를 통과한다.
- Diagnostics disabled run은 `rel_sol`/`rel_flux`를 계산하거나 log/CSV에 기록하지 않는다.
- Validation canonical energy로 best-energy checkpoint가 정상 저장된다.
- Reference array를 변경해도 diagnostics-off loss와 checkpoint criterion이 변하지 않는다.
- Diagnostics-enabled 기존 run은 기존 metric을 유지한다.
- Test evaluator와 artifact exporter는 full-reference test dataset에서 `rel_sol`, optional `rel_flux`, solution/flux figures를 계속 생성한다.
- Model safetensors key와 output contract가 변경되지 않는지 확인한다.

### CLI 및 Regression

- Source CLI schema, overwrite, atomic write, validation, summary/log 생성을 검증한다.
- `green_net` environment에서 `dolfinx/gmsh/petsc4py` import 없이 CLI가 실행되는지 확인한다.
- Existing complex CouplingNet, FEniCSx sample, unit-square CouplingNet, complex GreenNet tests를 실행한다.

검증 순서:

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_complex_sources.py \
  test/test_make_complex_sources.py \
  test/test_complex_coupling_data.py \
  test/test_complex_coupling_trainer.py \
  test/test_io_config.py \
  test/test_cli_train.py

PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_fenicsx_sample_schema.py \
  test/test_complex_coupling_artifacts.py \
  test/test_coupling.py

PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest test

ruff check src cli test
ruff format src cli test
~/.conda/envs/green_net/bin/python -m mypy src
git diff --check
```

## Rollback Strategy

- Runtime rollback은 `coupling_source.mode="npz"`와 reference diagnostics train/validation enabled를 사용하거나 새 config block을 제거하는 것이다.
- Source-only NPZ와 CLI output은 additive artifact이므로 기존 FEniCSx sample을 수정하거나 삭제하지 않는다.
- Code rollback은 source providers, source-only CLI, nested config, conditional diagnostic path만 제거하고 existing NPZ dataset constructor를 복원한다.
- Shared GP extraction을 롤백할 때는 backward-compatible re-export를 제거하고 기존 FEniCSx import 위치로 되돌린다.
- Model architecture와 safetensors state dict를 변경하지 않으므로 checkpoint migration은 필요 없다.
- Backend parity를 유지할 수 없다면 indexed GP rollout을 중단하고 deterministic NPZ backend만 보존한 뒤 seed/kernel/masking 차이를 보고한다.
- Existing config의 current behavior가 깨지면 새 source integration을 merge하지 않고 기존 full-reference NPZ path를 우선 복구한다.

## Acceptance Criteria

- Train/validation source가 epoch마다 바뀌지 않는다.
- Config로 NPZ와 indexed GP backend를 선택할 수 있다.
- 같은 identity의 두 backend가 같은 `rhs`를 생성한다.
- Source-only train/validation은 FEniCSx 없이 실행된다.
- Diagnostics disabled run은 `sol/phi/psi`를 요구하지 않고 `rel_sol/rel_flux`를 출력하지 않는다.
- Test evaluation은 FEniCSx reference metrics와 artifacts를 유지한다.
- Best-energy checkpoint selection은 source-only validation에서 정상 동작한다.
- Existing config, model checkpoint, unit-square, GreenNet, FEniCSx sample behavior에 regression이 없다.

## Confidence

- 구현 계획 확신도: **0.97**
- Source backend parity와 reference-free train/validation 구현 가능성: **0.99**
- 남은 불확실성은 정보 부족이나 규칙 모호성이 아니라, 대규모 indexed GP dataset을 epoch마다 deterministic regeneration할 때의 실제 CPU 비용에 관한 일반적인 성능 리스크다.
- v1은 memory-bounded regeneration을 채택하고 cache와 parallel source generation은 후속 최적화로 남긴다.

