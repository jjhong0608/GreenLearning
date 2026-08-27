# Optional Post-Training Best-Energy CouplingNet Artifact Export Plan

## Summary

Complex CouplingNet 학습과 기존 final-model test diagnostic이 모두 끝난 뒤, 저장된 **best-energy checkpoint를 다시 로드하여** 기존 CouplingNet artifact exporter를 자동 실행하는 optional post-training 단계를 추가한다.

이 기능은 다음 원칙으로 고정한다.

- `coupling_artifacts` block이 없거나 `enabled=false`이면 현재 학습 동작이 완전히 유지된다.
- 자동 생성은 `dataset.geometry_mode="complex"`인 Complex CouplingNet에만 적용한다.
- artifact는 항상 `complex_coupling_model_best_energy.safetensors`에서 생성한다.
- 출력 위치는 config로 받지 않고 checkpoint와 같은 디렉터리의 `artifacts_best_energy/`로 고정한다.
- 학습 중 사용한 `config_used.json`, GreenNet checkpoint, tangent response context를 재사용한다.
- 기존 standalone `cli/export_coupling_artifacts.py`는 그대로 지원한다.
- model architecture, loss, checkpoint tensor key, dataset/geometry NPZ schema는 변경하지 않는다.

## Public Config

다음 top-level config를 추가한다.

```json
"coupling_artifacts": {
  "enabled": true,
  "checkpoint": "best_energy",
  "device": null,
  "theme": "plotly_white",
  "selected_samples": null,
  "save_generated_data": true,
  "plot_workers": 1,
  "coefficient_vector_max_points": 400,
  "show_domain_boundary": true,
  "visualization_mesh": "data/visualization_mesh/unit_square_h_1_128.npz",
  "directional_color_quantile": 0.99
}
```

`outdir`는 config option으로 제공하지 않는다. 다음 경로를 항상 사용한다.

```python
coupling_checkpoint = work_dir / "complex_coupling_model_best_energy.safetensors"
artifact_outdir = coupling_checkpoint.parent / "artifacts_best_energy"
```

경로는 비교와 존재 검사를 하기 전에 `Path.resolve()`로 정규화한다. 이는 상대 경로와 상위 디렉터리 성분을 일관되게 처리하기 위한 것이다. [Python pathlib documentation](https://docs.python.org/3/library/pathlib.html)

Config 기본값은 다음으로 고정한다.

- `enabled=false`
- `checkpoint="best_energy"`
- `device=null`: `coupling_training.device` 사용
- `theme="plotly_white"`
- `selected_samples=null`: 기존 min/q25/q50/q75/max `rel_sol` sample 선택
- `save_generated_data=true`
- `plot_workers=1`: 기존 exporter와 마찬가지로 provenance 값이며 새 병렬 plotting은 추가하지 않음
- `coefficient_vector_max_points=400`
- `show_domain_boundary=true`
- `visualization_mesh=null`
- `directional_color_quantile=0.99`

`checkpoint`는 이번 구현에서 `"best_energy"`만 허용한다. `outdir`를 포함한 unknown key, 잘못된 boolean, 빈 theme, 음수 sample index, nonpositive integer, `(0.5,1.0]` 밖의 quantile은 fail fast한다.

## Validation Contract

`coupling_artifacts.enabled=true`이면 학습 시작 전에 다음을 검증한다.

- `pipeline.run_coupling=true`
- `dataset.geometry_mode="complex"`
- `coupling_training.best_energy_checkpoint.enabled=true`
- best-energy checkpoint를 생성할 validation source가 존재함
- `dataset.test_path`가 지정되어 있고 full-reference test NPZ를 제공함
- test data에 `sol`과 directional target `phi/psi` 또는 기존 alias `uxx/uyy`가 있음
- `pipeline.run_green=false`이면 `pipeline.green_pretrained_path`가 존재함
- `visualization_mesh`가 설정되면 파일이 존재함
- unit-square legacy CouplingNet에서 `enabled=true`이면 complex-only option이라는 오류를 발생시킴

Artifact가 disabled이면 위 추가 조건을 적용하지 않는다. 따라서 source-only training/validation만 사용하는 기존 실행에는 영향이 없다.

## Path Resolution

자동 export request는 다음과 같이 구성한다.

- Config: `<work_dir>/config_used.json`
- CouplingNet: `<work_dir>/complex_coupling_model_best_energy.safetensors`
- Output: `<work_dir>/artifacts_best_energy/`
- GreenNet, `pipeline.run_green=false`: `pipeline.green_pretrained_path`
- GreenNet, `pipeline.run_green=true`: `<work_dir>/model.safetensors`
- Coefficients: `dataset.coefficient_functions_path`
- Test data: `dataset.test_path`
- Visualization mesh: `coupling_artifacts.visualization_mesh`
- Tangent context: training CLI의 `--tangent-context`가 있으면 그 경로를 우선하고, 없으면 `<work_dir>/tangent_response_context.safetensors`
- Directional color range와 figure option: `coupling_artifacts`의 resolved config

Exporter는 원본 config가 아니라 `config_used.json`을 읽는다. 따라서 geometry-only로 자동 결정된 tangent subspace dimension과 실제 optimizer/source provenance가 artifact에 반영된다.

## Implementation Steps

1. `src/greenonet/config.py`에 strict `CouplingArtifactsConfig` dataclass와 `from_raw(...)`/`to_raw(...)`를 추가한다.
2. cross-config validator를 추가해 complex mode, best-energy checkpoint, validation, test reference, Green checkpoint 및 visualization mesh 조건을 학습 전에 검사한다.
3. `cli/train.py`에서 top-level `coupling_artifacts`를 parse하고 resolved config를 `config_used.json`에 materialize한다.
4. 새 `src/greenonet/post_training_coupling_artifacts.py`에 `PostTrainingCouplingArtifactRunner`와 request resolver를 구현한다.
5. runner는 고정된 checkpoint/config/output 경로를 계산하고 기존 `CouplingArtifactRequest`를 생성한다.
6. `cli/export_coupling_artifacts.py`의 geometry dispatch와 logger 구성을 공용 helper로 분리하여 standalone CLI와 post-training runner가 같은 exporter entrypoint를 사용하게 한다.
7. Complex CouplingNet 실행 순서는 `training → 기존 final-model test evaluation → in-memory trainer/model 해제 → best-energy artifact export`로 고정한다.
8. artifact 생성 전에 training DataLoader, trainer, final CouplingNet reference를 해제한다. CUDA에서는 필요할 때 cache를 비운 뒤 best-energy model을 exporter가 다시 로드하도록 한다.
9. existing nonempty `artifacts_best_energy/`가 있으면 덮어쓰거나 섞지 않고 fail fast한다. 비어 있거나 존재하지 않을 때만 export를 시작한다.
10. exporter 오류는 catch하여 성공처럼 처리하지 않는다. 학습 checkpoint는 보존하지만 train process는 nonzero로 종료되어 queue의 `_SUCCESS`가 작성되지 않게 한다.
11. `src/greenonet/complex_coupling_artifacts.py` summary에 `generation_trigger`, `checkpoint_selector`, config/checkpoint/GreenNet/outdir 경로를 추가한다.
12. artifact export가 성공하면 기존 `summary.json`, per-sample CSV, raw NPZ, scatter/mesh/coefficient/projection figures와 export log가 `artifacts_best_energy/`에 존재해야 한다.
13. README와 `docs/memory.md`에 optional config, 고정 경로, 실행 순서, best-energy와 final-model diagnostic의 차이, 실패 정책을 기록한다.
14. 기존 numerical experiment config는 자동으로 활성화하지 않는다. 새 block이 없는 모든 config는 현재와 동일하게 artifacts를 자동 생성하지 않는다.

## Affected Files

- Config와 validation: `src/greenonet/config.py`
- Post-training orchestration: 새 `src/greenonet/post_training_coupling_artifacts.py`
- Training lifecycle: `cli/train.py`
- Shared exporter dispatch/logger: `cli/export_coupling_artifacts.py`, `src/greenonet/coupling_artifacts.py`
- Complex summary provenance: `src/greenonet/complex_coupling_artifacts.py`
- Tests: `test/test_io_config.py`, `test/test_cli_train.py`, `test/test_export_coupling_artifacts.py`, `test/test_complex_coupling_artifacts.py`
- Documentation: `README.md`, `docs/memory.md`

## Test Plan

- Config omission과 `enabled=false`가 기존 동작을 보존하는지 확인한다.
- valid config round-trip과 unknown `outdir`, 잘못된 checkpoint/boolean/integer/quantile/sample index 거부를 확인한다.
- enabled 상태에서 unit-square mode, `run_coupling=false`, best-energy disabled, validation/test/Green checkpoint 누락을 각각 거부한다.
- `run_green=false/true`에서 Green checkpoint가 각각 pretrained path와 `<work_dir>/model.safetensors`로 결정되는지 확인한다.
- coupling/config/tangent/output 경로가 모두 `<work_dir>` 기준으로 정확히 결정되는지 확인한다.
- disabled 상태에서는 exporter가 호출되지 않고 artifact directory도 생성되지 않는지 확인한다.
- enabled 상태에서는 trainer와 기존 final test evaluation 이후 exporter가 정확히 한 번 호출되는지 확인한다.
- exporter request가 final checkpoint가 아니라 best-energy checkpoint를 사용하는지 확인한다.
- nonempty output directory가 있으면 model load 전에 실패하고 기존 파일이 유지되는지 확인한다.
- exporter failure가 process failure로 전파되지만 training checkpoint와 metrics가 유지되는지 확인한다.
- generated summary가 post-training trigger와 resolved provenance를 기록하는지 확인한다.
- standalone export CLI의 기존 arguments와 결과가 유지되는지 regression test를 수행한다.
- 작은 complex fixture로 best-energy checkpoint부터 실제 artifact bundle까지 생성하는 smoke test를 수행한다.
- model state-dict key, GreenNet, unit-square CouplingNet, loss와 checkpoint selection의 regression이 없는지 확인한다.

검증 순서는 다음과 같이 고정한다.

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_io_config.py \
  test/test_cli_train.py \
  test/test_export_coupling_artifacts.py \
  test/test_complex_coupling_artifacts.py

PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest test
ruff check src cli test
ruff format src cli test
~/.conda/envs/green_net/bin/python -m mypy src
git diff --check
```

실제 장기 CouplingNet 재학습은 구현 검증 범위에 포함하지 않는다.

## Rollback Strategy

- 실행 단위 rollback은 `coupling_artifacts.enabled=false`로 바꾸거나 block을 삭제하는 것이다.
- 생성 기능을 제거해도 기존 model-only safetensors와 standalone exporter는 그대로 사용할 수 있다.
- Code rollback은 config dataclass, post-training runner와 train CLI hook만 제거한다.
- model, trainer objective, projection, reconstruction, GreenNet과 NPZ schema에는 rollback 변경이 없어야 한다.
- export 도중 실패해 부분적인 `artifacts_best_energy/`가 남으면 자동 덮어쓰지 않는다. 해당 디렉터리를 명시적으로 이동 또는 삭제한 뒤 standalone exporter나 학습 후 export를 다시 실행한다.
- artifact export 실패는 기존 학습 결과를 무효화하지 않으며 best-energy checkpoint를 standalone CLI로 다시 export할 수 있다.

## Acceptance Criteria

- config block이 없거나 disabled이면 현재 실행과 수치 결과가 바뀌지 않는다.
- enabled이면 Complex CouplingNet 학습 종료 후 정확히 한 번 artifact export가 실행된다.
- artifact는 final model이 아니라 best-energy checkpoint에서 생성된다.
- artifact output은 항상 checkpoint 디렉터리의 `artifacts_best_energy/`이다.
- `outdir`는 public config에 존재하지 않으며 입력하면 오류가 발생한다.
- `config_used.json`, Green checkpoint와 tangent context가 실제 학습 실행과 일치한다.
- 기존 standalone artifact CLI가 계속 동작한다.
- artifact 오류가 숨겨지지 않으며 학습 checkpoint는 보존된다.
- model architecture, checkpoint tensor contract와 dataset schema가 변경되지 않는다.

## Confidence

- 구현 계획 확신도: **0.98**
- 필요한 설정과 실패 정책에는 정보 부족이나 규칙 모호성이 없다.
- 남은 위험은 artifact export의 실행 시간과 GPU peak memory라는 운영 비용이며, 학습 객체를 먼저 해제하고 checkpoint를 다시 로드하는 순서로 완화한다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Optional Post-Training Best-Energy CouplingNet Artifact Export Plan"을 기준
문서로 참고하여 CouplingNet post-training artifact generation을 끝까지
구현한다.

완료는 다음 조건으로 검증한다.

- coupling_artifacts block이 없거나 enabled=false이면 기존 학습 동작이 유지될 것,
- enabled=true는 complex CouplingNet, run_coupling, validation source,
  best-energy checkpoint와 full-reference test data를 strict하게 요구할 것,
- artifact가 final model이 아닌
  complex_coupling_model_best_energy.safetensors에서 생성될 것,
- output directory가 checkpoint 위치의 artifacts_best_energy로 고정될 것,
- outdir config option은 제공되지 않고 입력하면 오류가 발생할 것,
- config_used.json, 실제 GreenNet checkpoint와 tangent response context를
  재사용할 것,
- 기존 final-model test diagnostic 이후 artifact export가 정확히 한 번 실행될 것,
- 기존 nonempty artifact directory를 덮어쓰거나 기존 결과와 섞지 않을 것,
- artifact export 실패는 nonzero로 전파하되 완료된 학습 checkpoint와 metrics를
  보존할 것,
- standalone export_coupling_artifacts.py 동작이 유지될 것,
- summary에 post-training trigger, best-energy selector와 resolved path
  provenance가 기록될 것,
- model architecture, checkpoint tensor key, loss, projection, reconstruction,
  GreenNet 및 geometry/sample NPZ schema가 변경되지 않을 것,
- focused tests와 전체 pytest, Ruff, mypy, git diff check가 통과할 것.

수정 범위는 artifact config/validation, post-training runner, train CLI lifecycle,
공유 exporter dispatch/logger, complex artifact provenance, 관련 tests,
README와 docs/memory.md로 제한한다.

기존 experiment config를 자동 활성화하거나 장기 retraining을 실행하지 않는다.
Unit-square legacy CouplingNet의 artifact lifecycle도 변경하지 않는다.

각 구현 단계 후 가장 작은 config/request/lifecycle tests를 먼저 실행하고,
통과한 뒤 실제 complex artifact smoke와 전체 regression suite를 실행한다.

best-energy checkpoint 또는 기존 standalone exporter compatibility를 유지할 수
없다면 작업을 중단하고 다음을 보고한다.

1. 정확히 충돌하는 config, checkpoint, path 또는 artifact contract,
2. 영향을 받는 training run, checkpoint와 artifact,
3. 기존 standalone export와 disabled 기본 동작을 보존하는 가장 작은 rollback 전략.
```
