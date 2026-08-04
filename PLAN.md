# Boundary-Off Canonical Energy Ablation 구현 계획

## Summary

- Complex CouplingNet의 canonical energy에 fixed scalar boundary weight \(\lambda_{\partial}\)를 추가한다.

\[
E_{\mathrm{canonical}}
=
E_{\mathrm{bulk}}+E_{\partial},
\qquad
E_{\mathrm{optimized}}
=
E_{\mathrm{bulk}}+\lambda_{\partial}E_{\partial}.
\]

- `boundary_weight=1.0`은 현재 동작을 정확히 보존하고, `boundary_weight=0.0`은 tangent correction을 유지하면서 bulk energy만 최적화하는 boundary-off ablation이다.
- Boundary-off에서도 \(E_{\partial}\), \(E_{\partial,x}\), \(E_{\partial,y}\)는 항상 계산하고 diagnostic으로 기록한다.
- 이 기능은 complex CouplingNet 전용이다. GreenNet, unit-square CouplingNet, model architecture, projection, reconstruction, geometry/sample NPZ schema는 변경하지 않는다.
- 기존 config와 checkpoint는 migration 없이 사용할 수 있다.

## Public Interface

`coupling_training`에 다음 complex-only 설정을 추가한다.

```json
"canonical_energy": {
  "boundary_weight": 0.0
}
```

- `boundary_weight` 기본값은 `1.0`이다.
- 허용 범위는 finite numeric `>=0.0`이다.
- `bool`, 음수, `NaN`, `Inf`, unknown key는 fail fast한다.
- `0.0`은 완전한 boundary-off, `0.1` 같은 값은 약한 boundary anchor ablation으로 해석한다.
- Fixed sample-independent scalar이며 learnable parameter나 scheduler를 추가하지 않는다.
- Unit-square CouplingNet에서 non-default boundary weight를 요청하면 complex-only option이라는 오류를 낸다.
- 기존 tangent baseline은 유지하고, `configs/complex_coupling_soap_tangent_boundary_off.json`을 별도 paired-experiment config로 추가한다.

## Implementation Changes

1. `ComplexCanonicalEnergyConfig`를 추가하고 `CouplingTrainingConfig.canonical_energy`에서 strict `from_raw(...)` parsing과 round-trip을 지원한다.
2. `cli/train.py`가 resolved canonical-energy 설정을 `config_used.json`에 기록하도록 연결한다.
3. 기존 `canonical_complex_energy_loss(...)`는 unweighted \(E_{\mathrm{bulk}}+E_{\partial}\) audit 계산으로 보존한다.
4. Shared complex objective에서 별도로
   \[
   E_{\mathrm{optimized},b}
   =
   E_{\mathrm{bulk},b}
   +
   \lambda_{\partial}E_{\partial,b}
   \]
   를 계산한다. `boundary_weight=0`이면 objective가 bulk tensor만 직접 사용하도록 분기해 boundary 항이 gradient에 들어가지 않게 한다.
5. Boundary-off에서도 동일한 reconstructed fields로 boundary diagnostic을 계산한다. Backward는 실제 scalar objective에 연결된 graph만 사용한다는 점을 gradient test로 고정한다. [PyTorch autograd mechanics](https://docs.pytorch.org/docs/main/notes/autograd.html)
6. Relative split consistency가 활성화된 경우에도 energy numerator는 unweighted canonical energy가 아니라 `E_optimized`를 사용한다. Mass term과 normalization은 변경하지 않는다.
7. Weak operator closure는 기존처럼 selected split objective에 더하며 변경하지 않는다.
8. Metric contract를 다음처럼 고정한다.
   - `loss`: 실제 total objective
   - `loss_energy_optimized`: \(E_{\mathrm{bulk}}+\lambda_{\partial}E_{\partial}\)
   - `loss_energy_consistency`: unweighted canonical \(E_{\mathrm{bulk}}+E_{\partial}\)
   - `loss_energy_bulk`: \(E_{\mathrm{bulk}}\)
   - `loss_energy_boundary`, `loss_energy_boundary_x`, `loss_energy_boundary_y`: 항상 계산되는 diagnostic
9. `best_energy_checkpoint`는 boundary-off를 checkpoint 선택까지 일관되게 반영하도록 validation `loss_energy_optimized`를 최소화한다. `best_physics_checkpoint`는 계속 total `loss`를 사용한다.
10. Trainer 시작 로그와 metric CSV에 `boundary_weight`, optimized formula, boundary diagnostic 유지 여부를 기록한다.
11. Evaluator와 artifact exporter는 trainer와 같은 objective helper를 재사용한다. Artifact summary에는 boundary weight, optimization inclusion, unweighted audit formula, optimized checkpoint metric을 기록하고 aggregate metrics에 `loss_energy_optimized`를 추가한다.
12. `README.md`와 `docs/memory.md`에 boundary-off가 canonical boundary condition을 삭제하는 production default가 아니라 tangent와의 역할 중복을 검증하는 ablation이라는 점을 기록한다.

## Affected Files

- Config와 provenance: `src/greenonet/config.py`, `cli/train.py`
- Loss와 objective: `src/greenonet/complex_losses.py`, `src/greenonet/complex_coupling_objective.py`
- Runtime와 export: complex trainer, evaluator, artifact exporter
- Experiment config: 새 boundary-off tangent config
- Tests: config, complex energy, trainer, artifact 및 CLI tests
- Documentation: `README.md`, `docs/memory.md`

## Test Plan

- **Config:** omitted config가 `boundary_weight=1.0`으로 parse되는지, `0`, `0.1`, `1` round-trip, invalid numeric/boolean/unknown key rejection, unit-square non-default rejection을 검증한다.
- **Energy math:** weight 1은 기존 결과와 exact하게 같고, weight 0은 bulk와 같으며, intermediate weight는 `bulk + weight*boundary`와 일치하는지 확인한다.
- **Null mode diagnostic:** constant residual에서 optimized energy는 weight 0일 때 0이지만 unweighted canonical 및 boundary diagnostic은 양수로 남는지 확인한다.
- **Gradient:** weight 0의 model gradient가 direct bulk-only gradient와 같고 boundary contribution이 들어가지 않는지 확인한다.
- **Relative split:** enabled 상태에서 weighted energy numerator를 사용하고 mass/normalization contract는 유지하는지 확인한다.
- **Trainer:** tangent + boundary-off one-step smoke, metric/log/CSV schema, best-energy checkpoint가 `loss_energy_optimized`를 사용하는지 검증한다.
- **Evaluator/artifact:** boundary-off summary, per-sample/aggregate optimized energy, boundary diagnostic 유지, reference-target-free contract를 확인한다.
- **Regression:** default weight 1에서 기존 metric 값, checkpoint key, unit-square CouplingNet, GreenNet, optimizer/scheduler, projection/reconstruction 동작이 유지되어야 한다.

## Verification

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_complex_energy.py \
  test/test_complex_coupling_trainer.py \
  test/test_complex_coupling_artifacts.py \
  test/test_io_config.py \
  test/test_cli_train.py

PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest test

ruff check src cli test
ruff format src cli test
~/.conda/envs/green_net/bin/python -m mypy src
git diff --check
```

## Rollback Strategy

- Runtime rollback은 `canonical_energy` block을 제거하거나 `boundary_weight=1.0`으로 되돌리는 것이다.
- 기존 tangent baseline config는 수정하지 않으므로 paired baseline을 그대로 재사용할 수 있다.
- Model architecture와 safetensors key가 변하지 않으므로 checkpoint migration은 필요하지 않다.
- Code rollback은 canonical-energy config, weighted objective metric, 별도 experiment config와 관련 tests/provenance만 제거한다.
- Boundary diagnostic, canonical energy helper, tangent projection 및 reconstruction은 rollback 과정에서도 변경하지 않는다.

## Assumptions And Confidence

- Boundary-off는 complex CouplingNet의 모든 projection mode에서 사용할 수 있지만 첫 실험은 symmetric tangent projection과 paired comparison한다.
- Boundary-off는 gradient와 best-energy checkpoint 선택 모두에서 boundary 항을 제외한다.
- Boundary 항은 loss에서 꺼져도 diagnostic과 artifact에서는 항상 유지한다.
- 기존 config를 생략한 실행은 `boundary_weight=1.0`으로 완전히 backward compatible하다.
- 실제 장기 ablation training은 구현 범위에 포함하지 않는다.
- 구현 계획 확신도는 **0.99**다.
- Boundary-off가 solution quality를 개선할 가능성에 대한 경험적 확신도는 **0.45**다. 불확실성은 규칙 모호성이 아니라 paired retraining 결과가 아직 없다는 정보 부족이다.
- 이 응답에서는 `PLAN.md`를 수정하지 않으며, 사용자가 이 계획을 project root의 `PLAN.md`에 작성한다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Boundary-Off Canonical Energy Ablation 구현 계획"을 기준 문서로 참고하여
boundary-weight integration을 끝까지 구현한다.

완료는 다음 조건으로 검증한다.

- config를 생략하면 `boundary_weight=1.0`으로 기존 canonical
  bulk-plus-boundary objective가 정확히 유지될 것,
- `boundary_weight=0.0`이면 training gradient와 best-energy checkpoint
  선택에서 boundary energy가 완전히 제외될 것,
- boundary-off에서도 unweighted canonical, boundary total, boundary x/y
  diagnostic이 로그, CSV, evaluator 및 artifact에 계속 기록될 것,
- `loss_energy_optimized`와 `loss_energy_consistency`의 의미가 명확히
  분리될 것,
- relative split consistency가 활성화되면 weighted optimized energy를
  numerator로 사용할 것,
- tangent projection, weak closure, optimizer, scheduler 및 reconstruction
  수식이 변경되지 않을 것,
- 별도 paired experiment config가 추가되고 기존 tangent baseline config는
  유지될 것,
- unit-square CouplingNet과 GreenNet이 변경되지 않을 것,
- model architecture와 safetensors checkpoint key가 변경되지 않을 것,
- focused tests와 전체 pytest, Ruff, mypy, git diff check가 통과할 것.

수정 범위는 complex canonical-energy config, weighted objective,
checkpoint metric, trainer/evaluator/artifact provenance, 별도 experiment
config, 관련 tests와 문서로 제한한다.

Boundary term 계산 자체, tangent correction, Green reconstruction,
model backbone, reference-target-free 원칙, geometry/sample NPZ schema는
변경하지 않는다.

각 구현 단계 후 config와 energy math tests를 먼저 실행하고, 이어서
trainer/artifact integration tests와 전체 regression suite를 실행한다.
실제 장기 boundary-off training은 실행하지 않는다.

기존 `boundary_weight=1.0` 수치 또는 checkpoint architecture compatibility를
유지할 수 없다면 작업을 중단하고 다음을 보고한다.

1. 정확히 달라지는 loss, metric 또는 tensor contract,
2. 영향을 받는 config, checkpoint와 artifact,
3. 기존 canonical objective를 보존하는 가장 작은 rollback 또는 migration 전략.
```
