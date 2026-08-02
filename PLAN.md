# Fixed Alpha Tempered Column-Diagonal Projection 구현 계획

## Summary

- 기존 `column_diagonal_green_response` projection에 **sample-independent fixed \(\alpha\)**를 optional config로 추가한다.
- Regularized Green-response gain을
  \[
  \bar\gamma_x^2=\gamma_x^2+\varepsilon,
  \qquad
  \bar\gamma_y^2=\gamma_y^2+\varepsilon
  \]
  로 두고 correction weight를
  \[
  w_\phi^{(\alpha)}
  =
  \frac{(\bar\gamma_y^2)^\alpha}
  {(\bar\gamma_x^2)^\alpha+(\bar\gamma_y^2)^\alpha},
  \qquad
  w_\psi^{(\alpha)}=1-w_\phi^{(\alpha)}
  \]
  로 계산한다.
- Projection은 기존대로
  \[
  r=f-p-q,
  \qquad
  \phi=p+w_\phi^{(\alpha)}r,
  \qquad
  \psi=q+w_\psi^{(\alpha)}r
  \]
  를 사용하므로 모든 \(\alpha\)에서 \(\phi+\psi=f\)를 정확히 보존한다.
- \(\alpha=0\)은 physical symmetric correction, \(\alpha=1\)은 현재 column-diagonal correction이며, \(0<\alpha<1\)은 Green-response anisotropy를 완화한 tempered correction이다.
- Learnable scalar, sample-dependent scalar, pointwise \(\alpha\), alpha-prediction network, alpha loss 또는 regularizer는 추가하지 않는다.
- CouplingNet/GreenNet architecture, model parameter, state-dict key, source contract, loss, reconstruction 및 optimizer는 변경하지 않는다.
- 구현 확신도는 **0.99**다. 규칙과 코드 연결 지점은 명확하며 정보 부족이나 규칙 모호성은 없다. `gain_exponent=0.25`의 실제 성능 개선 가능성에 대한 경험적 확신도는 **0.75**다.

## Public Configuration

기존 nested projection config에 다음 field를 추가한다.

```json
"balance_projection": {
  "enabled": true,
  "mode": "column_diagonal_green_response",
  "column_diagonal_green_response": {
    "gain_squared_eps": 1e-12,
    "gain_exponent": 0.25
  }
}
```

- Public 이름은 `gain_exponent`로 고정한다. 일반적인 temperature convention과 방향이 혼동될 수 있으므로 `temperature` 또는 `tau`라는 config 이름은 사용하지 않는다.
- Dataclass 기본값은 `gain_exponent=1.0`으로 둔다. 기존 config에 field가 없으면 현재 column-diagonal 동작을 그대로 유지한다.
- 허용 범위는 finite numeric \([0,1]\)이다.
- `bool`, 문자열, `NaN`, infinity, 음수, `1`보다 큰 값은 fail fast한다.
- Unknown nested key를 거부하는 현재 strict parsing을 유지한다.
- `physical_symmetric` mode에서는 nested exponent가 projection 결과에 영향을 주지 않는다.
- `configs/complex_coupling.json`의 `physical_symmetric` 기본 설정은 변경하지 않는다.
- 다음 tempered SOAP 실험을 위해 `configs/complex_coupling_soap.json`에는 `gain_exponent=0.25`를 명시한다.
- 여러 \(\alpha\)용 config 파일을 추가로 복제하지 않는다. 후속 paired experiment에서는 같은 config를 복사하거나 해당 field만 변경해 \(\alpha\in\{0,0.25,0.5,1\}\)을 비교한다.

## Implementation Steps

### 1. Config Contract

- `ColumnDiagonalGreenResponseProjectionConfig`에 `gain_exponent: float = 1.0`을 추가한다.
- `__post_init__`에서 타입, finite 여부 및 \([0,1]\) 범위를 검증한다.
- `from_raw(...)`의 허용 key를 `gain_squared_eps`, `gain_exponent`로 확장한다.
- JSON serialization/deserialization, train CLI, eval CLI 및 checkpoint embedded config가 동일한 nested parser를 사용하도록 기존 경로를 유지한다.
- Learnable 여부를 나타내는 별도 config는 만들지 않는다. Fixed-only가 유일한 contract다.

### 2. Stable Tempered Weight Computation

- `ColumnDiagonalGreenResponseContext.from_gain_squared(...)`가 config 전체 또는 `gain_exponent`를 받아 cached correction weight를 생성하도록 확장한다.
- Regularization을 exponent 적용 전에 수행한다.
  \[
  \bar\gamma_s^2=\gamma_s^2+\varepsilon.
  \]
- 수치 및 backward behavior를 다음처럼 고정한다.
  - `gain_exponent == 0.0`: `w_phi=w_psi=0.5`를 직접 생성한다.
  - `gain_exponent == 1.0`: 기존 ratio 수식
    \[
    w_\phi=\bar\gamma_y^2/(\bar\gamma_x^2+\bar\gamma_y^2)
    \]
    을 그대로 사용해 기존 floating-point 경로를 보존한다.
  - 그 외:
    \[
    \ell=\log\bar\gamma_y^2-\log\bar\gamma_x^2,
    \qquad
    w_\phi=\operatorname{sigmoid}(\alpha\ell)
    \]
    을 사용해 overflow/underflow를 방지한다.
- `w_psi=1-w_phi`로 계산해 weight sum을 직접 보존한다.
- Context에 fixed `gain_exponent`를 저장하되 tensor parameter나 autograd graph에는 넣지 않는다.
- Segment-wise Green gain 계산, cache lifetime, full-Gram 미생성 및 no-global-solve contract는 그대로 유지한다.

### 3. Projection, Trainer, Evaluator

- `apply_complex_balance_projection(...)`의 correction 수식과 exact-balance construction은 변경하지 않는다. Tempered context가 제공하는 fixed weights를 그대로 소비한다.
- `gain_exponent=0`에서 unequal gains를 사용하더라도 결과가 `physical_symmetric` mode와 일치하도록 검증한다.
- Trainer와 evaluator는 기존 context cache를 그대로 사용한다. Batch 또는 sample마다 weight를 다시 계산하지 않는다.
- Context 최초 생성 로그에 `gain_exponent`, weight min/max 및 기존 gain/floor/cache 정보를 추가한다.
- Training objective, best-energy checkpoint, detached `rel_sol/rel_flux`, local weak-residual reliability reconstruction 및 SOAP scheduler에는 변경을 가하지 않는다.
- Changed exponent checkpoint는 tensor architecture상 load 가능하지만 projection/network co-adaptation 때문에 다른 exponent로 post-hoc 비교하지 않으며, 각 exponent 실험은 처음부터 재학습해야 한다는 안내를 유지한다.

### 4. Artifact Provenance

- Artifact `summary.json`의 column-diagonal section에 다음을 기록한다.
  - `gain_exponent`
  - `fixed_exponent=true`
  - `learnable_exponent=false`
  - tempered weight formula
  - \(\alpha=0\)/\(\alpha=1\) endpoint 의미
- Top-level projection formula도 exponent를 포함한 식으로 갱신한다.
- `data/column_diagonal_green_response_fields.npz`에는 기존 gain/regularized gain/weight 배열과 함께 scalar `gain_exponent`를 저장한다.
- Selected-sample raw archive는 이미 실제 correction weights와 corrections를 저장하므로 schema를 추가로 확장하지 않는다.
- 기존 gain/weight Plotly figure set은 유지한다. Figure title 또는 hover metadata에 fixed exponent를 표시해 artifact만 보고도 projection 강도를 확인할 수 있게 한다.
- `gain_exponent=1` artifact는 기존 수치와 일치해야 한다.

### 5. Config And Documentation

- `configs/complex_coupling_soap.json`의 column-diagonal block에 `gain_exponent: 0.25`를 추가한다.
- README에 fixed exponent 수식, endpoint 의미, config 예시 및 paired retraining 요구사항을 추가한다.
- `docs/complex_column_diagonal_green_response_projection.md`에 full column metric과 tempering의 관계를 기록한다. Tempering은 row method나 full-Gram solve가 아니라 diagonal anisotropy strength 조절임을 명시한다.
- `docs/memory.md`에 다음 durable convention을 기록한다.
  - 기본 exponent는 `1.0`
  - `0`은 symmetric correction
  - fixed-only이며 learnable/sample-dependent alpha는 지원하지 않음
  - exponent 변경 비교는 같은 seed/data로 각각 재학습
- 현재 README/docs의 row-norm rejection 및 local weak-residual reconstruction 설명은 유지한다.

## Affected Files

- Config/runtime core:
  - `src/greenonet/config.py`
  - `src/greenonet/complex_green_response_projection.py`
  - `src/greenonet/complex_projection.py`
- Logging/artifacts:
  - `src/greenonet/complex_coupling_trainer.py`
  - `src/greenonet/complex_coupling_evaluator.py`
  - `src/greenonet/complex_coupling_artifacts.py`
- Experiment config:
  - `configs/complex_coupling_soap.json`
- Tests:
  - `test/test_complex_projection.py`
  - `test/test_complex_reconstruction.py`
  - `test/test_complex_coupling_trainer.py`
  - `test/test_complex_coupling_artifacts.py`
  - `test/test_io_config.py`
  - `test/test_cli_train.py`
  - `test/test_coupling.py`
- Documentation:
  - `README.md`
  - `docs/complex_column_diagonal_green_response_projection.md`
  - `docs/memory.md`

`PLAN.md`는 사용자가 이 계획 내용으로 project root에 직접 작성하며, 이번 계획 수립 단계에서는 파일을 수정하지 않는다.

## Test Plan

### Config Tests

- Field가 없을 때 `gain_exponent==1.0`인지 확인한다.
- `gain_exponent=0`, `0.25`, `0.5`, `1`이 parse되고 JSON/checkpoint config round-trip 되는지 확인한다.
- `bool`, 문자열, `NaN`, infinity, `-0.1`, `1.1`을 fail fast하는지 확인한다.
- Unknown nested key와 기존 row-norm alias rejection이 유지되는지 확인한다.
- Unit-square CouplingNet이 column-diagonal mode를 계속 거부하는지 확인한다.

### Weight Math Tests

- \(\alpha=1\)에서 기존 ratio formula와 결과가 일치하는지 확인한다.
- Unequal gains에서도 \(\alpha=0\)이 정확히 `0.5/0.5`를 반환하는지 확인한다.
- \(\alpha=0.25\)와 `0.5`가 analytic power/log-sigmoid reference와 일치하는지 확인한다.
- 모든 exponent에서 `w_phi+w_psi==1`, finite 및 \([0,1]\) 범위를 만족하는지 확인한다.
- 매우 작거나 큰 positive gain ratio에서도 intermediate exponent가 finite인지 확인한다.
- Gain tensors와 weights가 sample-independent, detached 및 cache-once 상태인지 확인한다.
- Existing \(L^4\) source-column gain scaling test는 변경 없이 유지한다.

### Projection Tests

- \(\alpha=0,0.25,0.5,1\) 모두에서
  \[
  \phi+\psi=f
  \]
  가 float64 exact-balance tolerance를 만족하는지 확인한다.
- \(\alpha=0\) column context와 `physical_symmetric` 결과가 projected physical/response 및 difference update에서 일치하는지 확인한다.
- \(\alpha=1\)이 기존 column projection fixture를 그대로 통과하는지 확인한다.
- Intermediate exponent에서 expected correction과 difference update를 검증한다.
- Raw response에 대한 gradient가 finite하고 alpha/gain context에는 gradient가 없는지 확인한다.

### Integration And Artifact Tests

- Trainer/evaluator가 같은 exponent와 cached context를 공유하고 context를 한 번만 생성하는지 확인한다.
- Training/evaluator log에 fixed exponent가 기록되는지 확인한다.
- Artifact summary가 exponent, fixed/learnable 상태 및 formula를 기록하는지 확인한다.
- Run-level NPZ에 scalar exponent와 tempered weights가 저장되는지 확인한다.
- `gain_exponent=1` regression과 `save_generated_data=false` path가 유지되는지 확인한다.
- Existing solution/flux/weak-residual figures와 cross-key 부재 contract가 유지되는지 확인한다.

### Verification Commands

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest \
  test/test_complex_projection.py \
  test/test_complex_reconstruction.py \
  test/test_complex_coupling_trainer.py \
  test/test_complex_coupling_artifacts.py \
  test/test_io_config.py \
  test/test_cli_train.py \
  test/test_coupling.py
```

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python -m pytest test
ruff check src cli test
ruff format src cli test
~/.conda/envs/green_net/bin/python -m mypy src
git diff --check
```

실제 장기 paired retraining은 구현 검증 범위에 포함하지 않는다.

## Acceptance Criteria

- 기존 config가 field 없이 `gain_exponent=1.0`으로 기존 column-diagonal 결과를 유지한다.
- `gain_exponent=0.25`가 fixed, sample-independent tempered weights를 생성한다.
- \(\alpha=0\)은 physical symmetric correction과 일치하고 \(\alpha=1\)은 기존 column correction과 일치한다.
- 모든 exponent에서 physical source balance가 정확히 보존된다.
- Learnable parameter, gate, sample-conditioned network 또는 alpha gradient가 생성되지 않는다.
- Model state-dict key와 GreenNet/CouplingNet architecture가 변경되지 않는다.
- Artifact와 로그만으로 사용된 exponent와 projection formula를 재현할 수 있다.
- `configs/complex_coupling_soap.json`은 다음 실험용 `gain_exponent=0.25`를 명시한다.
- Unit-square CouplingNet, GreenNet, optimizer, scheduler, loss 및 reconstruction regression이 없다.

## Rollback Strategy

- Runtime rollback은 `gain_exponent`를 생략하거나 `1.0`으로 설정하는 것이다.
- Code rollback은 config field, tempered weight helper, exponent provenance 및 관련 tests만 제거한다.
- Model architecture와 state dict가 변하지 않으므로 checkpoint migration은 필요하지 않다.
- `configs/complex_coupling_soap.json`에서 `gain_exponent`를 제거하면 기존 column-diagonal 실험 설정으로 돌아간다.
- Default \(\alpha=1\) 수치가 기존 fixture와 일치하지 않으면 intermediate stable formula 적용을 중단하고 `alpha==1` legacy ratio branch를 우선 복구한다.
- AdamW/SOAP, scheduler, objective, dataset, geometry/sample NPZ 및 Green reconstruction에는 rollback 변경이 없어야 한다.

## Confidence

- 구현 계획 및 기존 behavior 보존 확신도: **0.99**
- `gain_exponent=0.25`가 transition weight jump를 줄일 확신도: **0.99**
- 재학습 후 `rel_flux`와 `rel_sol`을 함께 개선할 경험적 확신도: **0.75**
- 규칙 모호성이나 필수 정보 부족은 없다. 남은 불확실성은 projection/network co-adaptation이 tempered setting에서 어떻게 다시 형성되는지에 관한 실험적 불확실성이다.

## Executable `/goal` Draft

```text
/goal

`/home/jjhong0608/Documents/GreenNetResearch/ComplexGeometry/PLAN.md`의
"Fixed Alpha Tempered Column-Diagonal Projection 구현 계획"을 기준 문서로
참고하여 fixed gain exponent integration을 끝까지 구현한다.

완료는 다음 조건으로 검증한다.

- 기존 column-diagonal config에서 exponent를 생략하면 `gain_exponent=1.0`으로
  현재 projection 결과가 유지될 것,
- `gain_exponent`가 finite numeric `[0,1]` 범위에서 strict하게 parse되고
  config/save/load round-trip 될 것,
- alpha 0은 physical symmetric correction과 일치하고 alpha 1은 기존
  column-diagonal correction과 일치할 것,
- alpha 0.25와 0.5가 stable log-ratio 수식에 따라 fixed tempered weight를
  생성할 것,
- 모든 exponent에서 `phi+psi=rhs`가 정확히 보존될 것,
- exponent는 sample-independent fixed scalar이며 model parameter, gradient,
  learned gate 또는 sample-dependent network가 추가되지 않을 것,
- Green-response context가 기존처럼 한 번만 생성되고 trainer/evaluator/export가
  같은 cached weight를 사용할 것,
- summary, raw NPZ, figures와 logs에 fixed exponent와 projection formula가
  기록될 것,
- `configs/complex_coupling_soap.json`이 다음 paired experiment용
  `gain_exponent=0.25`를 명시할 것,
- CouplingNet/GreenNet architecture와 model checkpoint key가 변경되지 않을 것,
- unit-square CouplingNet, optimizer, scheduler, loss, reconstruction 및 기존
  artifact behavior에 regression이 없을 것,
- focused tests와 전체 pytest, Ruff, mypy, git diff check가 통과할 것.

수정 범위는 column-diagonal projection config, fixed weight computation,
context logging/provenance, complex artifact export, SOAP experiment config,
관련 tests와 문서로 제한한다.

Learnable global alpha, sample-dependent alpha, pointwise alpha, alpha network,
alpha loss, row-norm projection, full-Gram solve, model backbone, training
objective, Green reconstruction 및 geometry/sample NPZ schema는 변경하지 않는다.

각 구현 단계 후 가장 작은 config/math/projection tests를 먼저 실행하고,
통과한 뒤 trainer/artifact integration tests와 전체 regression suite를 실행한다.
실제 장기 paired retraining은 실행하지 않는다.

기존 `gain_exponent=1.0` 수치 또는 checkpoint architecture compatibility를
유지할 수 없다면 작업을 중단하고 다음을 보고한다.

1. 정확히 달라지는 projection 수식, floating-point 결과 또는 tensor contract,
2. 영향을 받는 config, checkpoint, artifact와 tests,
3. alpha 1 legacy path를 보존하는 가장 작은 rollback 또는 migration 전략.
```
