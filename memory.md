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
- `coupling_model.branch_fusion.mode`는 CouplingNet branch feature 결합 방식을
  제어한다. 기본값 `product`는 기존 multiplicative fusion을 보존하고,
  실험 옵션 `product_fuser`는 active branch features와 그 component-wise product를
  concat한 뒤 learned fuser로 최종 branch representation을 만든다.
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
- 현재 numerical integration rule은 `trapezoid`, `simpson` 중심이며, 수치 적분
  오차를 줄이기 위한 새 `adaptive Gaussian quadrature` rule 추가를 검토한다.
- 고차 정확도 수치적분의 우선 적용 대상은 CouplingNet이 아니라 GreenNet이다.
  특히 Poisson GreenNet에서 exact/reference Green function과 predicted Green
  kernel 사이의 `rel_green` error가 적분 구간 폭을 줄일수록 함께 줄어드는 현상을
  보고, model error와 quadrature error를 분리하기 위해 Green kernel error 계산의
  수치적분 오차를 줄이는 것이 목표이다. GreenNet training loss도 Green kernel을
  source와 적분해 solution reconstruction error를 계산하므로 같은 quadrature error를
  포함할 수 있고, 새 고차 적분 rule은 GreenNet loss 계산에도 적용 대상으로 본다.
- GreenNet용 Romberg 후보는 Green kernel의 diagonal kink를 고려해 `xi = x`에서
  적분 구간을 나누는 `split Romberg`를 우선 논의한다. 단순 whole-interval
  Romberg는 Richardson extrapolation의 smoothness 가정이 diagonal kink에서 약해질
  수 있다.
- 전역 Romberg를 129 grid points(`2^7 + 1`)에서 쓰면 absolute quadrature error가
  반드시 크게 폭발한다고 보기는 어렵지만, coarse Romberg levels에서 diagonal kink가
  grid node와 정렬되지 않는 `x_i`가 많아 고차 수렴 이득이 약해질 수 있다. 따라서
  model error가 이미 작다면 129 points에서도 quadrature error floor가 관찰될 수 있다.
- `split_gauss_legendre`는 adaptive Gaussian보다 계산 비용과 batching 리스크가 낮은
  GreenNet loss용 고차 적분 후보로 검토한다. `xi = x`에서 구간을 나누고 각 smooth
  subinterval에 fixed Gauss-Legendre nodes/weights를 적용하는 deterministic rule이다.
- `split_gauss_legendre` 적용 방향은 먼저 source-free `rel_green` 고정밀 diagnostic에
  적용하고, 이후 GreenNet training loss에는 기존 GP sampled source grid를 Gaussian
  nodes로 보간하는 source interpolation 기반으로 시도한다. Source interpolation error는
  별도 diagnostic으로 분리해 확인해야 한다.
- GreenNet `split_gauss_legendre` 구현은 `training.green_quadrature` config로
  제어한다. 기본값은 `enabled=false`이며, `rule="split_gauss_legendre"`를 사용한다.
  `source_interpolation` 기본값은 재현성을 위해 `"linear"`이고, 명시적 실험 옵션으로
  natural cubic spline인 `"cubic"`도 지원한다. `apply_to_loss=true`이면 Gaussian
  node에서 GreenNet을 직접 평가하고 source grid를 선택된 interpolation method로
  보간해 reconstruction loss를 계산한다. `apply_to_rel_green=true`이면 Poisson
  constant-coefficient source-free diagnostic에서 Gaussian node의 predicted/exact
  Green kernel을 비교한다. `source_interp_rel_error`는 interpolation diagnostic
  logging key이다. Cubic source interpolation은 `rel_green` path가 아니라 GreenNet
  reconstruction loss/artifact reconstruction의 `f_grid -> f(xi_q)` 보간 path에만
  직접 영향을 준다.
- `split_gauss_legendre`의 `xi` 방향 적분 정확도는 Gaussian node에서 integrand를
  직접 평가할 수 있는 source-free `rel_green`에서는 기존 `xi` grid 간격에 직접
  묶이지 않고, quadrature order와 split 구간 내 smoothness가 지배한다. 하지만
  GreenNet reconstruction loss에서는 source `f`가 grid sample로만 주어져 Gaussian
  node source 값을 보간하므로 source interpolation error는 여전히 grid 간격과
  interpolation method에 의존한다. 또한 현재 v1은 바깥 `x` 방향 residual/norm 적분은
  기존 sampled grid와 `training.integration_rule`을 사용하므로 그 부분도 grid 간격의
  영향을 받는다.
- `split_gauss_legendre` node 위치는 같은 physical evaluation point `x_i`와 같은
  quadrature `order`에 대해서는 grid spacing과 무관하게 동일하다. 표준
  Gauss-Legendre nodes를 `[0, x_i]`, `[x_i, 1]`로 affine mapping하기 때문이다.
  단, `x_i = xi` diagonal point 자체는 Gaussian node가 아니라 split interval boundary이며,
  Gauss-Legendre rule은 endpoint를 평가하지 않는다. Grid를 refine하면 새 `x_i`가
  추가되어 그 새 point들에 대한 node set이 추가되는 것이고, 기존과 같은 physical
  `x_i`의 node set은 바뀌지 않는다.
- GreenNet reconstruction에서 source interpolation error를 더 줄이려면 GP/source
  sampling grid만 더 fine하게 두는 방향도 가능하다. 다만 현재 `ForwardSampler`는
  source, solution, coefficient grid가 같은 line grid에 묶여 있고 RBF centers/alpha도
  `x.size(0)`에 의존하므로, 단순히 source grid만 키우면 같은 realization의 더 촘촘한
  평가가 아니라 다른 realization이 될 수 있다. 올바른 구현은 latent RBF/GP realization을
  먼저 고정하고 coarse solution grid와 fine source/quad grid에서 같은 realization을
  평가하거나, quadrature node의 source 값을 직접 생성/저장하는 방식이어야 한다.
- Fine source grid 방식은 `TrainingData`/`AxialDataset`에 coarse `F`와 별도 fine
  source tensor/grid를 추가하고, GreenNet `split_gauss_legendre` reconstruction path가
  fine source에서 Gaussian node 값을 보간하게 하는 방향이 안전하다. `ForwardSampler`는
  같은 RBF mixture와 같은 normalization scale을 coarse/fine grid에서 평가해야 한다.
  `BackwardSampler`는 target solution과 RHS 일관성을 위해 fine RHS representation으로
  BVP를 solve하거나, coarse RHS target과 fine RHS loss가 서로 다른 문제가 되지 않도록
  정책을 고정해야 한다.
- GreenNet fine source grid는 `training.green_quadrature.source_sampling`으로
  제어한다. 기본값은 `{"enabled": false, "factor": 1}`이고, 켜는 경우
  `factor > 1`이어야 하며 `m_fine = factor * (m - 1) + 1`이다. `TrainingData`는
  optional `F_FINE`, `F_FINE_GRID`를 갖고, `AxialDataset`은 fine source가 없으면
  legacy 7-field batch, 있으면
  `coords, solution, source, source_fine, source_fine_grid, a, ap, b, c`의 9-field
  batch를 반환한다. `ForwardSampler`와 `BackwardSampler`는 같은 latent RBF
  realization을 coarse/fine grid에서 평가하고 같은 coarse-solution normalization
  scale을 사용한다. `BackwardSampler`는 fine source가 켜지면 fine RHS/coefficient
  grid로 BVP를 solve하고 coarse grid에서 target solution을 저장한다. GreenNet
  `split_gauss_legendre` reconstruction loss와 artifact reconstruction은 fine
  source가 있으면 그 grid에서 `linear` 또는 `cubic` interpolation을 수행하고,
  source-free `rel_green` path는 fine source를 사용하지 않는다.
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
- `symmetric_boundary_loss` 계산은 `CouplingTrainer._symmetric_boundary_loss`에
  있으며, projection output이 아니라 `CouplingNet.raw_flux_at_coords`로 얻은
  raw flux를 사용한다. Phi boundary residual은
  `(phi_raw_boundary - psi_raw_endpoint) + f`, psi boundary residual은
  `(phi_raw_endpoint - psi_raw_boundary) - f`이고, 각 edge 방향으로 적분한 뒤
  `0.5 * (phi_loss + psi_loss)`로 합친다.
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
- GreenNet `training.log`와 artifact summary의 `rel_green`은 exact/reference
  1D line Green kernel을 구성할 수 있는 경우에 유효하다. `c=0, b=0`이면
  diffusion reference를 쓰고, `c=0, b!=0`이면 convection-diffusion reference를
  쓴다. Reaction `c`가 nonzero이면 현재 exact/reference가 없으므로 invalid/skip
  처리한다.
- `ExactGreenFunction`의 기본 `forward()`, `__call__()`, `error()`는 diffusion-only
  reference로 유지한다. Reaction-free convection-diffusion line reference가
  필요하면 `convection_diffusion(b)`를 명시적으로 호출하고, x-lines에는 `bx`,
  y-lines에는 `by`의 line slice를 넘긴다. GreenNet `rel_green` 경로는 같은
  reference policy를 사용한다.
- Exact/reference Green kernel matrix convention은 `G[row=x, col=xi]`이다.
  Reconstruction은 마지막 dimension인 `xi` 방향으로 적분한다. Convection-diffusion
  kernel은 비대칭이므로 이 orientation을 해석식 테스트로 고정해야 한다.
- GreenNet problem selection의 현재 논의 기준은 `Pure_Poisson.py`와 두
  diffusion-only 문제를 `rel_green` 중심 주력 문제로 두고, reaction-free
  convection-diffusion 문제도 `rel_green_reference="convection_diffusion"`로
  비교할 수 있다. Reaction 포함 문제는 reconstruction 중심 보조/확장 산출물로
  다루는 것이다.
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
- CouplingNet selected-sample flux-divergence figures should exclude boundary
  grid values. CouplingNet predictions use zero-padding at boundaries only for
  trapezoid-rule integration compatibility with boundary-zero Green functions;
  those padded boundary values are not meaningful flux-divergence predictions
  and should not be shown in paper-facing flux-divergence heatmaps.
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
- `plot_coupling_logs.py`는 `--show-annotations` 옵션을 켜면 각 train/val trace의
  마지막 값과 최소값을 표시한다. 기본값은 꺼짐으로 유지한다. 값 표시는
  실제 curve point에 연결된 Plotly annotation으로 두고, last/min 위치에는 작은
  marker도 함께 찍는다. y-axis가 log-scale이므로 plotted marker 위치는
  `1e-16` floor로 맞추고, annotation y coordinate는 Plotly log-axis 규칙에 맞게
  `log10(plotted_y)`를 사용한다. Annotation이 켜진 figure는 마지막 epoch label이
  오른쪽에 놓일 수 있도록 x-axis 오른쪽 padding을 둔다.
- `plot_coupling_logs_aux_temp.py`는
  `checkpoints/Pure_Poisson/expreiments/raw_output_with_balance_loss/training.log`
  확인용 root-level 임시 script이다. Paper-facing default에는 넣지 않고,
  기존 5개 curve에 `balance_loss`, `symmetric_boundary_loss`만 추가해 본다.
- Best epoch과 final epoch을 분리해서 해석한다.
- Plot은 비교 실험 단위로 같은 scale, theme, output naming을 유지한다.
- Plotly 기반 논문용 figure는 나중에 수정하기 쉽도록 `html`, `png`, `pdf`와
  함께 Plotly figure spec `json`도 저장하는 것을 기본으로 한다. Static export가
  실패해도 `html`과 `json`은 저장해야 한다.
- `plot_green_logs.py`도 GreenNet log comparison figures를 `html`, `json`으로
  항상 저장하고, 가능한 경우 `png`, `pdf`도 함께 저장한다.
- `plot_coupling_rel_sol_boxplots.py`는 workshop용 CouplingNet test result CSV
  비교 script이다. `checkpoints/For_Workshop/CouplingNetResults`의 네
  `*_per_sample_metrics.csv` 파일에서 `rel_sol`을 읽고 문제별 Plotly boxplot을
  `html/json/png/pdf` 네 형식으로 저장한다. 이 script는 점 표시 없이 박스 플롯만
  그리며 `rel_sol`은 `%` 단위(100배)로 표시한다.
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
- 학회 발표 준비 문서는 `docs/conference_presentation_preparation.md`에 작성한다.
  GreenNet/CouplingNet 설명은 사용 중인 설정만 반영한다. Disabled option인
  `source_stencil_lift`, `green_response_feature`, disabled losses 등은 발표 준비
  문서의 핵심 설정 설명에서 제외한다.
- 학회 발표 준비 문서에서 우선 다룰 coefficient family는 `Pure_Poisson.py`,
  `Sinusoidal_Diffusion_Only.py`, `Smooth_Variable_Diffusion_Reaction.py`,
  `Convection_Diffusion_Reaction.py`, `Sinusoidal_Diffusion_Only_Ver2.py`,
  `Diffusion_Reaction_Ver2.py`이다.
- 다음 단계의 학회 발표용 Markdown 문서는 영어로 작성한다. 최종 목적은 PPT 슬라이드
  작성이지만, Markdown 자체는 슬라이드 단위로 나누지 않고 전체 흐름을 가진 발표 원고형
  문서로 작성한다. 기준 자료는 `docs/conference_presentation_preparation.md`이다.
- 영어 발표용 continuous narrative Markdown은
  `docs/conference_presentation_narrative.md`에 둔다. 이 문서는 PPT slide deck이
  아니라 PPT 제작 전 단계의 영어 발표 원고형 자료이다.
- `docs/conference_presentation_narrative.md`는 코드/학습 설정 중심이 아니라 모델
  구조와 수학적 아이디어 중심으로 작성한다. GreenNet은 analytic Green form과 neural
  correction으로 1D Green function을 근사하는 모델로 설명하고, CouplingNet은
  MIONet-style source decomposition과 smooth-mask projection으로 `phi + psi = f`
  balance를 만족시키는 모델로 설명한다.
- 발표 narrative의 PDE operator는
  `L u = -div(a grad u) + b dot grad u + c u = f`로 설명한다. CouplingNet의 exact
  decomposition은 `phi = -partial_x(a partial_x u) + b_x partial_x u + 1/2 c u`,
  `psi = -partial_y(a partial_y u) + b_y partial_y u + 1/2 c u`로 두어
  `phi + psi = f`가 되게 설명한다.
- GreenNet analytic form 설명에는 Green function의 Dirac-delta property와 boundary
  zero condition을 먼저 제시한다. Poisson Green kernel은 boundary behavior를 주고,
  polynomial/envelope factors는 boundary compatibility를, integrated Green-type term과
  coefficients `A(x)`, `B(x)`는 Dirac/Heaviside singular terms를 구조적으로 처리하기
  위한 장치로 설명한다.
- GreenNet objective는 predicted Green kernel 자체를 exact/reference Green과 직접
  MSE로 맞추는 것이 아니다. 코드 기준 training loss는 learned Green kernel을 source와
  xi 방향으로 적분해 reconstructed solution을 만든 뒤, 이것을 exact solution과 비교하는
  reconstruction loss이다. `rel_green`/exact Green 비교는 별도 metric/diagnostic으로
  설명한다.
- `docs/conference_presentation_narrative.md`는 GreenNet 입력/출력을 coefficient line
  profiles와 `(x, xi)`에서 complete learned kernel `G_hat(x, xi)`로 설명하고, objective는
  `G_hat`을 source와 적분한 reconstructed solution loss로 설명하도록 수정했다. CouplingNet
  설명에는 `phi + psi = f`, `u_x approx u_y`, energy consistency
  `int a |grad(u_x-u_y)|^2`, alpha-only sine smooth-mask projection을 반영했다.
- CouplingNet projection 설명은 balance relation뿐 아니라 fiberwise Green reconstruction의
  transverse boundary compatibility를 위한 구조로 설명한다. `G_x`는 x-endpoint boundary,
  `G_y`는 y-endpoint boundary를 자연스럽게 만족하므로, source split/projection에는
  transverse boundary behavior를 반영하는 smooth masks가 필요하다고 설명한다.
- `docs/conference_presentation_narrative.md`에는 위 operator, GreenNet analytic
  property, exact `phi`/`psi` split, projection boundary-compatibility 설명을 반영했다.

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
