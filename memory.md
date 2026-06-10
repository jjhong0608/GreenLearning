# Coupler Repo Memory

이 문서는 Coupler repo에서 사용자와 Codex가 합의한 연구 맥락, 모델 구조,
coefficient 의미, 실험 설계 기준, 논문용 데이터/figure 생성 기준을 기록한다.
`README.md`는 사용법, `AGENTS.md`는 작업 규칙, 이 문서는 연구/실험 의사결정
기억을 담당한다.

## Stable Project Context

- 이 repo의 중심 목표는 axial Green function 관점에서 2D PDE solution을
  재구성하는 GreenONet/CouplingNet 실험을 수행하는 것이다.
- GreenONet은 axial Green kernel 또는 그 근사를 학습/제공한다.
- CouplingNet은 axial decomposition을 이용해 solution/flux reconstruction을
  학습한다.
- 주요 PDE family는 Poisson, variable diffusion, diffusion-reaction,
  convection-diffusion이다.
- 논문용 결과 생성에서는 coefficient family, model variant, train/eval split,
  metric, figure 저장 규칙을 명확히 분리한다.

## Stable Modeling Decisions

- Convection coefficient는 scalar `b_fun`보다 vector form
  `bx_fun(x, y)`, `by_fun(x, y)`를 권장 API로 사용한다.
- Legacy coefficient file이 `b_fun(x, y)`만 제공하면
  `bx_fun = b_fun`, `by_fun = b_fun`으로 해석한다.
- 내부 tensor convention은 `b_vals[0] = b_x`, `b_vals[1] = b_y`를 유지한다.
  별도 `b_x`/`b_y` tensor field를 만들지 않는다.
- `branch_coefficient`는 diffusion/convection/reaction coefficient를 통합하는
  generic coefficient branch이다.
- `coupling_model.coefficient_terms` config가 diffusion, convection, reaction
  입력 여부를 제어한다.
- `source_stencil_lift.enabled=true` path와 standard coefficient branch path는
  의미가 다르므로 섞지 않는다.
- `terminal.width`는 top-level runtime/logging config이며 Rich terminal wrapping을
  제어한다. `training.log` file output과는 별도 surface로 취급한다.

## Coefficient Families

- `Pure_Poisson.py`: constant diffusion, zero convection, zero reaction.
- `Sinusoidal_Diffusion_Only.py`: variable diffusion, zero convection,
  zero reaction.
- `Sinusoidal_Diffusion_Only_Ver2.py`: alternate smooth variable diffusion,
  zero convection, zero reaction.
- `Diffusion_Reaction_Ver2.py`: variable diffusion with reaction,
  zero convection.
- `Smooth_Variable_Diffusion_Reaction.py`: smooth variable diffusion with reaction,
  zero convection.
- `Convection_Diffusion_Reaction.py`: variable diffusion with convection and reaction.
- `Divergence_Free_Convection_Diffusion.py`: variable diffusion,
  divergence-free convection with amplitude `2.0`, zero reaction.
- 새 coefficient file을 추가할 때는 `a_fun`, `apx_fun`, `apy_fun`,
  `bx_fun`, `by_fun`, `c_fun`을 정의한다.

## Important Mathematical Checks

- `apx_fun`, `apy_fun`은 반드시 `a_fun`의 해석적 편미분과 일치해야 한다.
- Divergence-free convection은 `d bx / dx + d by / dy = 0`을 만족해야 한다.
- Reaction-free 문제는 `c_fun(x, y) = 0`이어야 한다.
- Green accuracy가 좋아도 solution reconstruction이 나쁠 수 있다. 이런 경우
  kernel 사용 경로, coefficient alignment, axis convention, normalization,
  quadrature/integration path를 우선 의심한다.
- Convection-diffusion 문제에서 `bx_fun`은 x-direction line/operator에,
  `by_fun`은 y-direction line/operator에 들어간다.

## GreenNet / GreenONet Notes

- GreenNet은 특정 coefficient problem에 대한 Green's function을 학습한다.
- 이 repo의 논문용 문제 설정에서는 2D coefficient field를 문제별로 고정한다.
- 하지만 GreenNet이 다루는 것은 full 2D Green's function 하나가 아니라 axial
  decomposition에서 생기는 1D Green's function들이다.
- 따라서 같은 2D coefficient problem 안에서도 각 x-axis/y-axis line이 보는
  1D operator coefficient slice가 달라질 수 있고, 각 축선의 Green's function도
  서로 다르다고 해석해야 한다.
- 현재 GreenNet 학습의 training/validation dataset은 저장된 외부 dataset을
  읽는 방식이 아니라 config의 sampling 설정으로 `ForwardSampler` 또는
  `BackwardSampler`가 즉석 생성한 `TrainingData`를 `AxialDataset`으로 감싸는
  방식이다.

## Model And Loss Notes

- `balance_projection.enabled=false`이면 CouplingNet은 projection 없이 raw
  output을 반환하는 path를 사용한다.
- `balance_loss`는 projection off일 때만 허용한다.
- `symmetric_boundary_loss`는 symmetric projection on일 때 boundary condition을
  raw difference mode에 학습시키기 위한 loss이다.
- `smooth_mask` projection은 mask option을 갖는다. 기본은 `quadratic`,
  추가 option으로 `sin`을 사용할 수 있다.
- `axis_1d_trunk.enabled=true`이면 shared 1D trunk를 사용하고, transverse 정보는
  boundary-aware sin/cos branch로 넣는다.
- Axis-1D trunk의 boundary-aware encoding은 raw coordinate `t`를 포함하지 않고
  `sin(n*pi*t)`, `cos(n*pi*t)` for `n=1..k`만 사용한다.

## Experiment And Figure Planning

- 논문용 데이터 생성 전에 coefficient family, model variant, train/eval split,
  output directory, plot target을 먼저 확정한다.
- GreenNet 결과는 `rel_green`, `train_rel_sol`, `val_rel_sol`을 함께 본다.
- GreenNet 논문용 산출물은 run-level training metrics, per-line metrics,
  selected-line Green kernel data, selected-line coefficient slices,
  source-to-solution reconstruction data를 분리해서 저장하는 방향을 기본으로 한다.
- GreenNet `training.log`의 `rel_green`은 diffusion-only/no-convection/no-reaction
  문제에서만 유효한 metric으로 해석한다. Convection 또는 reaction이 포함된
  문제에서는 로그에 값이 있어도 논문용 Green accuracy metric으로 사용하지 않는다.
- GreenNet problem selection의 현재 논의 기준은 `Pure_Poisson.py`와 두
  diffusion-only 문제를 `rel_green` 중심 주력 문제로 두고, reaction/convection
  포함 문제는 reconstruction 중심 보조/확장 산출물로 다루는 것이다.
- GreenNet 논문용 산출물은 training loop에 묶기보다 checkpoint/config/dataset을
  다시 읽는 별도 재생성 script 또는 CLI로 만드는 방향을 선호한다.
- GreenNet이 주장하려는 내용이 fixed coefficient problem에서 학습한 Green
  function으로 같은 sampler/source distribution의 source term에 대한 solution을
  잘 재구성한다는 것이라면, 논문용 evaluation data는 별도 저장 dataset 없이도
  학습 때와 같은 config/sampler 설정으로 다시 생성해도 충분하다고 본다.
  다만 seed와 generation sequence는 metadata에 명시해서 training data 재사용과
  같은-distribution evaluation을 구분한다.
- GreenNet artifact exporter 구현은 새 CLI만 두기보다 dataset 재생성,
  Green kernel/reconstruction metric, fixed-`xi` slice 추출, Plotly multi-format
  저장을 작은 helper로 분리해 training loop와 독립시키는 방향이 좋다.
- GreenNet artifact exporter surface는 `cli/export_green_artifacts.py`이며,
  helper는 `src/greenonet/green_artifacts.py`와 `src/greenonet/plotly_io.py`에
  둔다. 기본 evaluation은 `validation_like`, 기본 seed는 `12345`, 기본 fixed
  `xi` fractions는 `0.25, 0.5, 0.75`이다. Device는 기본적으로 config의
  `training.device`를 따르며, 필요하면 CLI `--device`로 override한다.
- 논문용 산출물은 최소 산출물만 만드는 방향보다 가능한 한 많이 생성한 뒤,
  그중 논문에 적합한 결과와 figure를 선택하는 방향을 기본으로 한다.
- CouplingNet 논문용 run-level training curves는 `train loss`, `val loss`,
  `l2_consistency`, `energy_consistency`, `rel_flux`, `rel_sol`만 사용한다.
  Other auxiliary losses are ignored for paper-facing curves unless explicitly
  requested later.
- CouplingNet selected-sample figures are limited to source `f`, exact solution
  `u`, `u_pred`, `u_pred_x`, `u_pred_y`, signed solution errors
  `u_pred - u`, `u_pred_x - u`, `u_pred_y - u`, mismatch `u_pred_x - u_pred_y`,
  exact flux-divergences `phi`, `psi`, predicted flux-divergences `phi_pred`,
  `psi_pred`, signed flux-divergence errors `phi_pred - phi`, `psi_pred - psi`,
  and balance fields `phi + psi`, `f - phi - psi`. Error figures must be signed
  differences, not absolute values.
- CouplingNet null-space and closure diagnostics are not part of the selected
  paper-facing figure set unless explicitly re-enabled later.
- CouplingNet 논문용 산출물에는 test-set aggregate metrics, balance/projection
  diagnostics, boundary diagnostics, coefficient/source context figures,
  ablation comparison tables, and raw metric/data archive를 유지한다.
- CouplingNet artifact 구현 시 current `CouplingEvaluator`는 일부 tensor
  computation을 재사용할 수 있지만, paper-facing selected figures는 별도 path
  또는 option으로 분리해야 한다. Current evaluator uses absolute errors and
  null/closure diagnostics, while the paper-facing set requires signed errors and
  excludes null/closure figures.
- CouplingNet paper artifact exporter surface는 `cli/export_coupling_artifacts.py`
  이며 helper는 `src/greenonet/coupling_artifacts.py`에 둔다. Test data는
  `dataset.test_path`의 `.npz`를 `CouplingDataset`으로 읽고, coefficient는 기본적으로
  `dataset.coefficient_functions_path`를 사용하며 `--coefficients`로 override한다.
  Device는 기본적으로 `coupling_training.device`를 따르고 `--device`가 우선한다.
- `plot_coupling_logs.py`는 paper-facing run-level curve 전용으로 유지한다. 출력은
  `loss`, `l2_consistency`, `energy_consistency`, `rel_flux`, `rel_sol` 5개만 생성하며,
  optional auxiliary loss curve는 debug용 `plot_logs.py`에서 다룬다.
- Best epoch과 final epoch을 분리해서 해석한다.
- Plot은 비교 실험 단위로 같은 scale, theme, output naming을 유지한다.
- Plotly 기반 논문용 figure는 나중에 수정하기 쉽도록 `html`, `png`, `pdf`와
  함께 Plotly figure spec `json`도 저장하는 것을 기본으로 한다. Static export가
  실패해도 `html`과 `json`은 저장해야 한다.
- `plot_green_logs.py`도 GreenNet log comparison figures를 `html`, `json`으로
  항상 저장하고, 가능한 경우 `png`, `pdf`도 함께 저장한다.
- Figure 후보:
  - coefficient field visualization: `a`, `bx`, `by`, `c`
  - source/solution sample visualization
  - Green kernel heatmap per selected axial line
  - fixed-`xi` 1D Green function slices per selected axial line, with boundary
    values and diagonal/singularity behavior highlighted
  - training curves for GreenONet and CouplingNet
  - metric comparison table/bar chart
  - solution reconstruction and error heatmap
  - flux reconstruction and error heatmap

## Verification Defaults

- `.venv`가 없으면 `/home/jjhong0608/.conda/envs/green_net/bin/python`을 사용한다.
- Focused pytest를 먼저 실행하고, 이후 touched files에 대해 `ruff check`와
  필요한 `mypy`를 실행한다.
- Repo-wide `mypy src`는 2026-06 기준 통과 상태로 정리되었다. 새 변경 이후에는
  focused mypy뿐 아니라 가능한 경우 `mypy src` 전체 통과를 유지한다.
- Markdown 파일은 `ruff` 대상이 아니다.
- 이미 dirty worktree인 경우, 요청 범위와 무관한 변경은 건드리지 않는다.

## Open Planning Items

- 논문용 coefficient family 목록 확정.
- 각 family별 dataset 규모와 random seed 확정.
- GreenONet baseline config와 CouplingNet baseline config 확정.
- Ablation 목록 확정:
  - projection on/off
  - `balance_loss`
  - `symmetric_boundary_loss`
  - `smooth_mask` mask type
  - axis-1D trunk
  - coefficient term inputs
- Figure 목록과 paper section별 배치 확정.
- 결과 저장 directory naming convention 확정.
- Run log, metric CSV, generated figure를 어떤 형식으로 archive할지 확정.

## Update Policy

- Codex는 이 repo에서 답변하기 전에 이 `memory.md`를 먼저 참고한다.
- 답변 이후에는 새로 합의된 durable decision, 실험 기준, coefficient 의미,
  figure/data planning 기준이 생겼는지 판단하고 이 파일을 업데이트한다.
- 이 파일은 모든 작업 로그를 기록하는 곳이 아니다.
- 앞으로 반복해서 참조해야 하는 수학적 의미, API convention, 실험 설계 기준,
  figure planning 기준만 추가한다.
- 일회성 command output, transient error, 구현 세부 diff는 기록하지 않는다.
- 논문용 결과 생성 과정에서 확정된 dataset/figure/run naming은 이 파일에
  누적해서 갱신한다.
