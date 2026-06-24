# Modified GreenONet

Axial-inspired neural solver for the 2D Poisson equation with Dirichlet boundaries. The project follows the AGENTS guidelines: class-first design, rich logging mirrored to disk, Plotly visualizations, and TDD.

## Setup
- Activate the virtual environment: `source .venv/bin/activate`
- Install runtime deps: `pip install -e .`
- On Linux `x86_64` with CPython `3.14`, the editable install now pins Torch to the official PyTorch `2.11.0+cu126` wheel from `download.pytorch.org`. Other environments fall back to `torch>=2.11.0`, so install a different PyTorch CUDA build manually if you need one.
- See dev tools (ruff/mypy/pytest): `pip install -e .[dev]`
- Ensure `PYTHONPATH` includes `src` when running commands in this repo.
- FEniCSx sample generation is intentionally isolated from the main `green_net` training environment. Create the optional solver environment with `conda env create -f environment-fenicsx.yml`, then verify it with `conda run -n green_fenicsx python -c "import dolfinx, gmsh, petsc4py, torch"`. Do not add FEniCSx to the main `pyproject.toml` dependencies.

## Usage
- Train with the sample config: `PYTHONPATH=src python cli/train.py --config configs/default.json --work-dir checkpoints/run`
- Logs: Rich console output plus `training.log` in the chosen `work-dir`.
- Artifacts: `loss_curve.html`, `green_heatmap.html`, and weights `model.safetensors` in `work-dir`.
- Config: the CLI copies the input JSON to `config_used.json` in the `work-dir`.
- Checkpoints: `*.safetensors` include JSON-encoded model config metadata; use `greenonet.io.load_model_with_config` to restore model+config.
- Loss: Green's-function reconstruction — integrates the learned Green kernel against the source and matches the recovered solution (no direct output MSE).
- GreenNet structure: the analytic Green wrapping remains active when `model.use_green=true`, while the learned correction now uses fused line encoders for `a`, `a'`, `b`, and `c` plus a hybrid trunk with smooth `(x, xi)` features.
- Green analytic coefficients: under the conservative operator `-d_x(a(x) d_x u) + b(x) d_x u + c(x) u = f`, the implemented wrapping uses `A(x, xi) = 1 / a(x)` and `B(x, xi) = (a'(x) + b(x)) / a(x)^2`, so both coefficients depend on the evaluation-side `x` values.
- Exact Green references: `greenonet.greens.ExactGreenFunction` keeps its default diffusion-only reference for `forward()`, `__call__()`, and `error()`. GreenNet `rel_green` uses the exact/reference line kernel selected from sampled coefficients: diffusion reference when `b=0, c=0`, convection-diffusion reference via `convection_diffusion(b)` when `b!=0, c=0`, and skip/invalid when reaction `c` is nonzero. Exact kernels follow the reconstruction convention `G[row=x, col=xi]`; for convection-diffusion lines, pass the axis-local convection slice (`bx` for x-lines, `by` for y-lines).
- Integration rule: set `training.integration_rule` and `coupling_training.integration_rule` to `"simpson"` or `"trapezoid"` to control sampled-data quadrature in Green/Coupling training, evaluation metrics, and coupling RHS normalization. Green synthetic samplers reuse `training.integration_rule` for sample normalization.
- Optimizers: GreenONet uses Adam by default with optional multi-epoch LBFGS fine-tuning (see `TrainingConfig.lbfgs_*`). CouplingNet training uses AdamW.
- Torch compile: set `training.compile.enabled=true` to wrap GreenONet with `torch.compile`, and set `coupling_training.compile.enabled=true` to do the same for CouplingNet. The flags are independent, optional, and checkpoint saving still unwraps compiled models to keep load/save compatibility.
- Terminal logging: set top-level `terminal.width` to a positive integer such as `250` to fix Rich console log width for all project loggers. Omit `terminal` or set `width=null` to keep Rich's automatic terminal-width detection; disk `training.log` output is unchanged.
- CouplingNet: a shared branch/trunk MIONet consumes axial-line inputs `(a, b, c, f)` together with interior coordinates and predicts axial flux-divergences `(phi_x, psi_y)` through a single shared DeepONet-style readout followed by optional balance projection.
- Complex geometry mode: set `dataset.geometry_mode="complex"` to train/evaluate CouplingNet on a precomputed non-rectangular valid-point geometry. The default remains `unit_square`, so existing configs keep the original `CouplingDataset`, `CouplingNet`, `CouplingTrainer`, evaluator, and artifact exporter. Complex mode requires `dataset.geometry_path` and interprets `dataset.training_path`, `dataset.validation_path`, and `dataset.test_path` as directories of full-grid sample `.npz` files. `coupling_model.branch_input_dim` is reused as the number of fixed unit-interval branch samples per segment, while `hidden_dim`, `depth`, `activation`, `dropout`, `use_bias`, and `dtype` are reused by `ComplexCouplingNet`.
- Unit-square vs complex architecture notes: see `docs/unit_square_vs_complex_geometry.md` for the implementation-aware comparison of the preserved unit-square GreenNet/CouplingNet core path and the separate complex geometry path. See `docs/unit_square_vs_complex_geometry_math.md` for the code-surface-free mathematical comparison focused on branch, trunk, projection, and Green reconstruction.
- Complex geometry schema: the geometry `.npz` must contain `coords_valid`, `valid_grid_y_index`, `valid_grid_x_index`, `x_segment_id`, `y_segment_id`, `x_local_t`, `y_local_t`, `x_segment_left`, `x_segment_right`, `x_segment_y`, `x_segment_length`, `y_segment_bottom`, `y_segment_top`, `y_segment_x`, `y_segment_length`, `x_recon_ptr`, `x_recon_t`, `x_recon_weight`, `x_recon_valid_index`, `y_recon_ptr`, `y_recon_t`, `y_recon_weight`, `y_recon_valid_index`, `x_edges`, `y_edges`, `hx`, and `hy`. Reconstruction arrays use `valid_index == -1` for hard-zero segment endpoints; valid points must be strictly interior in segment-local coordinates.
- Circular geometry generation: use `PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_circular_geometry.py --step-size 0.05 --radius 1.0 --out data/geometry/unit_circle_h005.npz` to generate a centered circular complex geometry. The default radius is `1.0`; non-unit circles use `--radius R` and a filename such as `circle_r2_h005.npz`. The generator requires `2 * radius / step_size` to be an integer, builds the full grid on `[-radius, radius]`, excludes boundary grid points and degenerate boundary lines, stores only axial chord segments with valid interior points, writes unit-local nonuniform trapezoid reconstruction weights, and validates the saved `.npz` with `load_complex_geometry` by default. Pass `--overwrite` to replace an existing output and `--no-validate` to skip the post-write schema check.
- Complex sample schema: each sample `.npz` must contain full-grid `rhs` and `sol` arrays indexed as `[row=y, col=x]`. Optional flux targets use preferred keys `phi` and `psi`; legacy keys `uxx` and `uyy` are accepted as fallback. Full-grid values are gathered into valid-point order using `valid_grid_y_index` and `valid_grid_x_index`.
- FEniCSx complex sample generation: use `cli/make_fenicsx_samples.py` from the optional `green_fenicsx` environment to generate CouplingNet sample `.npz` files for complex geometry. The geometry `.npz` must include `grid_x` and `grid_y` because the generator writes full-grid arrays, while the existing `coords_valid` and valid index arrays define which grid values belong to the domain. Domain input must be exactly one of `--gmsh-script <path>` or `--msh <path>`. A Gmsh script must define `build_domain(gmsh, context)` and return `{"surface_tags": [...]}`; disconnected multi-surface domains must also return `point_surface_tags`, one surface tag per valid geometry point, so the generator can embed each valid point in the correct surface. Script mode embeds valid grid points as internal Gmsh points by default and requires vertex coverage by default; `.msh` mode evaluates at valid points and does not require vertex coverage unless `--require-valid-points-in-mesh` is passed. `examples/unit_circle_gmsh.py` is unit-radius by default but reads `radius` from the geometry `.npz`, so non-unit circular geometry and the FEniCSx disk mesh stay aligned.
- FEniCSx sample command example:
  ```
  PYTHONPATH=src conda run -n green_fenicsx python cli/make_fenicsx_samples.py \
    --geometry data/geometry/unit_circle_h005.npz \
    --out data/complex_samples/unit_circle_h005 \
    --gmsh-script examples/unit_circle_gmsh.py \
    --num-train 1000 \
    --num-valid 100 \
    --num-test 100 \
    --lengthscale 0.2 \
    --amplitude 1.0 \
    --seed 0 \
    --mesh-size 0.03
  ```
  The output layout is `<out>/train/sample_000000.npz`, `<out>/valid/sample_000000.npz`, `<out>/test/sample_000000.npz`, plus `<out>/make_fenicsx_samples.log` and `<out>/generation_summary.json`. Every sample stores full-grid `rhs`, `sol`, `phi`, and `psi` arrays with shape `(len(grid_y), len(grid_x))`; values outside `coords_valid` are zero-filled. `rhs` is sampled from a separable squared-exponential GP on the Cartesian grid, then zero-filled outside the valid domain points before the FEM solve. `sol`, `phi`, and `psi` are evaluated from FEniCSx functions at valid grid points and written back to full-grid arrays.
- FEniCSx sample-level parallelism: add `--num-workers N --sample-seed-policy indexed` to split independent samples across spawned Python worker processes. Each worker performs single-process FEniCSx solves; this is not MPI domain decomposition. `num_workers=1` keeps the legacy sequential RNG path. Parallel mode requires indexed seeds, so each `(split, index)` sample is reproducible regardless of worker scheduling. By default existing sample files fail fast; pass `--overwrite` to replace them or `--skip-existing` to resume without rewriting existing samples. Only the parent process writes `generation_summary.json`.
  ```
  PYTHONPATH=src conda run -n green_fenicsx python cli/make_fenicsx_samples.py \
    --geometry data/geometry/unit_circle_h01.npz \
    --out data/complex_samples/unit_circle_h01_small_parallel \
    --gmsh-script examples/unit_circle_gmsh.py \
    --num-train 32 \
    --num-valid 8 \
    --num-test 8 \
    --lengthscale 0.2 \
    --amplitude 1.0 \
    --mean 0.0 \
    --seed 0 \
    --mesh-size 0.025 \
    --solution-degree 3 \
    --target-degree 2 \
    --coefficients coefficients/Pure_Poisson.py \
    --num-workers 4 \
    --sample-seed-policy indexed
  ```
- FEniCSx `phi`/`psi` convention: the generator solves the weak form `int a grad(u).grad(v) + (b.grad(u)) v + c u v = int f v` with homogeneous Dirichlet boundary conditions, then projects the direction-split targets `phi=-d_x(a d_x u)+b_x d_x u+0.5 c u` and `psi=-d_y(a d_y u)+b_y d_y u+0.5 c u`. `generation_summary.json` records the valid-point relative balance residual for `phi + psi ~= rhs`.
- Circular-domain smoke and small dataset workflow:
  ```
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_circular_geometry.py \
    --step-size 0.25 \
    --out data/geometry/unit_circle_h025.npz \
    --overwrite

  PYTHONPATH=src conda run -n green_fenicsx python cli/make_fenicsx_samples.py \
    --geometry data/geometry/unit_circle_h025.npz \
    --out data/complex_samples/unit_circle_h025_smoke \
    --gmsh-script examples/unit_circle_gmsh.py \
    --num-train 1 \
    --num-valid 0 \
    --num-test 0 \
    --lengthscale 0.2 \
    --amplitude 1.0 \
    --mean 0.0 \
    --seed 0 \
    --mesh-size 0.035 \
    --solution-degree 3 \
    --target-degree 2 \
    --coefficients coefficients/Pure_Poisson.py

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/validate_complex_samples.py \
    --geometry data/geometry/unit_circle_h025.npz \
    --sample-root data/complex_samples/unit_circle_h025_smoke \
    --splits train \
    --coefficients coefficients/Pure_Poisson.py \
    --branch-input-dim 4 \
    --max-balance-residual 1e-2

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_circular_geometry.py \
    --step-size 0.1 \
    --out data/geometry/unit_circle_h01.npz \
    --overwrite

  PYTHONPATH=src conda run -n green_fenicsx python cli/make_fenicsx_samples.py \
    --geometry data/geometry/unit_circle_h01.npz \
    --out data/complex_samples/unit_circle_h01_small \
    --gmsh-script examples/unit_circle_gmsh.py \
    --num-train 32 \
    --num-valid 8 \
    --num-test 8 \
    --lengthscale 0.2 \
    --amplitude 1.0 \
    --mean 0.0 \
    --seed 0 \
    --mesh-size 0.025 \
    --solution-degree 3 \
    --target-degree 2 \
    --coefficients coefficients/Pure_Poisson.py

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/validate_complex_samples.py \
    --geometry data/geometry/unit_circle_h01.npz \
    --sample-root data/complex_samples/unit_circle_h01_small \
    --splits train valid test \
    --coefficients coefficients/Pure_Poisson.py \
    --branch-input-dim 4 \
    --max-balance-residual 5e-2
  ```
  The circular smoke path uses `h=0.25`, `mesh_size=0.035`, `solution_degree=3`, `target_degree=2`, and one train sample. The small dataset path uses `h=0.1`, `mesh_size=0.025`, the same FEM degrees, and `32/8/8` train/valid/test samples. Start with `coefficients/Pure_Poisson.py`; move to diffusion, reaction, and convection families only after the circular balance residual is stable.
  With these defaults, the refined `h=0.25` smoke sample has been observed to pass `1e-2` (`max ~= 7.8e-3`), while the `h=0.1` small dataset is a schema/loadability and residual-distribution check (`max ~= 3.7e-2`). Refine `mesh_size`, `solution_degree`, or `target_degree` before treating the small dataset as a strict `1e-2` quality dataset.
- Complex GreenNet training: when `dataset.geometry_mode="complex"` and `pipeline.run_green=true`, GreenNet trains on the connected x/y segments from `dataset.geometry_path` as a flat interval list `N = Sx + Sy`. Every connected interval is an independent 1D domain, so disconnected components on the same axial line remain separate rows. `dataset.samples_per_line` and `dataset.validation_samples_per_line` mean samples per connected interval. `dataset.training_path`, `dataset.validation_path`, and `dataset.test_path` are ignored by GreenNet in complex mode and remain CouplingNet full-grid sample directories. During L-BFGS fine-tuning, new complex GreenNet runs log `train_rel_sol`, validation `val_rel_sol` when enabled, and `rel_green` when the exact/reference policy supports it.
- Complex coefficient normalization: each physical segment is mapped to the unit interval before GreenONet/CouplingNet use. The unit coefficients are `a_unit=a_phys`, `ap_unit=L*ap_phys`, `b_unit=L*b_phys`, `c_unit=L^2*c_phys`, and `f_unit=L^2*f_phys`. GreenNet trunk coordinates are always unit `(t, eta) in [0,1]^2`; unit reconstruction integrates `G_unit(t, eta) f_unit(eta)` over `eta` without multiplying by an additional segment length.
- Complex CouplingNet behavior: the model is source-conditioned like the unit-square CouplingNet. Full-grid `rhs` is gathered to valid points, lifted into segment-local unit source branches with endpoint hard-zero values, scaled as `f_unit=L^2*f_phys`, normalized by segment unit-interval L2 norm, and then used as the mandatory source branch. The coefficient branch is controlled by `coupling_model.coefficient_terms` in `[a,b,c]` order; `a'` is kept only for GreenONet reconstruction queries. The geometry branch consumes `[s_left, s_right, s_mid, L, L^2, 1/L]`, the transverse branch consumes globally normalized transverse Fourier features, and the trunk consumes segment-local `t`. Raw unit outputs are source-norm scaled, converted to physical `phi`/`psi`, projected with hard symmetric balance in physical variables, converted back to unit outputs, and reconstructed segment by segment with the precomputed nonuniform trapezoid weights.
- Complex disabled features: `cross_consistency`, `smooth_mask`, `balance_loss`, `source_stencil_lift`, and `green_response_feature` are unit-square-only surfaces. Complex trainer/evaluator/artifact paths do not compute, log, serialize, or export cross-related metric keys or placeholder fields.
- CouplingNet coefficient terms: in the standard branch path (`source_stencil_lift.enabled=false`), `coupling_model.coefficient_terms` controls which operator coefficients enter the generic `branch_coefficient`. Enabled terms are concatenated in `[a, b, c]` order from `diffusion`, `convection`, and `reaction`; the default `diffusion=true, convection=false, reaction=false` preserves the previous diffusion-only coefficient branch. If all three are false, CouplingNet uses a source-only pure Poisson branch path and skips `branch_coefficient`.
- CouplingNet branch fusion: set `coupling_model.branch_fusion.mode` to choose how branch features are combined before the trunk readout. The default `product` keeps the existing multiplicative fusion of source, coefficient, and optional transverse branch features. The experimental `product_fuser` mode concatenates the active branch features with their component-wise product and passes them through a learned fuser, preserving the product bias while allowing a learned final branch representation.
- CouplingNet optional source stencil lift: set `coupling_model.source_stencil_lift.enabled=true` to add input-side learned source and coefficient 5-point stencil encoders. It reconstructs canonical full grids from `rhs_raw` and `a_vals`, feeds normalized source `f` stencils to a source encoder and raw coefficient `a` stencils to a separate coefficient encoder, optionally normalizes the source lifted scalar field by interior RMS, and sends the two lifted fields through separate source and coefficient branch networks before multiplying their branch features. Set `coupling_model.source_stencil_lift.encoder_type` to `"mlp"` for nonlinear five-stencil encoders or `"linear"` for direct affine maps from each five-stencil to its scalar field. The coefficient lift keeps RMS output normalization by default with `coefficient_normalization="rms"`; opt into the bounded coefficient output `beta * tanh(r_coef)` with `coefficient_normalization="tanh"` and `coefficient_tanh_beta`. `b_vals` and `c_vals` are not part of this first coefficient encoder version, and the physical `rhs_raw`, `rhs_norm`, balance projection, losses, and evaluation targets remain unchanged.
- CouplingNet optional Green response feature: set `coupling_model.green_response_feature.enabled=true` to append the frozen axial Green response `G(rhs_tilde)` to the existing normalized source branch input, so the branch sees `[rhs_tilde, G(rhs_tilde)]`. The trainer and evaluator compute this feature from the current `green_kernel`; `CouplingNet` does not own the Green kernel and still uses `rhs_raw` for balance projection. This first version is axis-local only, has no separate normalization option, and cannot be enabled together with `source_stencil_lift`.
- CouplingNet optional trunk positional encoding: set `coupling_model.trunk_positional_encoding.enabled=true` to replace raw unit-square trunk coordinates with deterministic coordinate features. The default `mode="fourier"` appends axis-aligned Fourier features `[sin(2*pi*f*x), cos(2*pi*f*x), sin(2*pi*f*y), cos(2*pi*f*y)]` with log-spaced frequencies from `1` to `max_frequency`; the defaults `num_frequencies=4` and `max_frequency=8.0` give `[1, 2, 4, 8]`. Set `mode="boundary_algebraic"` to append Dirichlet/domain-aware algebraic features `[x(1-x), y(1-y), x*y, x^2, y^2, x(1-x)y(1-y)]`. Set `include_input=false` to drop raw `(x, y)` from either encoded trunk input. This feature is unit-square-only and cannot be enabled in complex geometry mode.
- CouplingNet optional shared axis-1D trunk: set `coupling_model.axis_1d_trunk={"enabled": true, "boundary_aware_modes": k}` to use one shared 1D trunk for both axes in unit-square mode. In this mode `phi` evaluates the shared trunk on `x`, `psi` evaluates the same trunk on `y`, and a separate transverse branch receives only boundary-aware features `Enc_k(t)=[sin(n*pi*t), cos(n*pi*t)]` for `n=1..k`; raw transverse coordinate `t` is not included. For `phi`, `t` is the fixed line coordinate `y`; for `psi`, `t` is the fixed line coordinate `x`. In complex mode, the trunk is always segment-local 1D `t`, and `axis_1d_trunk.num_frequencies` / `max_frequency` control Fourier features of the globally normalized transverse coordinate `r_hat`. The default values are `boundary_aware_modes=4`, `num_frequencies=4`, and `max_frequency=8.0`.
- Balance projection: CouplingNet defaults to `coupling_model.balance_projection={"enabled": true, "mode": "symmetric", "mask": "quadratic"}`, the fixed interior `0.5/0.5` residual split. Set `mode="smooth_mask"` to use smooth transverse masks that preserve exact interior `phi + psi = f` while damping the raw difference mode near transverse boundaries. `balance_projection.mask` selects `"quadratic"` masks `m_phi=y(1-y)`, `m_psi=x(1-x)` or `"sin"` masks `m_phi=sin(pi*y)`, `m_psi=sin(pi*x)`; omitted masks and legacy string configs such as `balance_projection="smooth_mask"` still load as `"quadratic"`. For `"quadratic"`, `smooth_mask_normalize=true` scales the masks by `4`; `"sin"` is already normalized and ignores that scaling. For either mask, `coupling_model.smooth_mask_power` applies the mask exponent `p`, and `coupling_model.smooth_mask_diff_power` applies the difference-mode exponent `q` in `beta = 0.5 * (2 * alpha_soft)^q`; both default to `1.0`, which recovers the previous projection. Set `coupling_model.smooth_mask_diff_power_trainable=true` to learn only `q` as a bounded scalar with `smooth_mask_diff_power_min <= q <= smooth_mask_diff_power_max` (defaults `0.25` and `2.0`); this requires projection `enabled=true` and `mode="smooth_mask"`. Set `enabled=false` to return the raw CouplingNet `(phi, psi)` output without projection; `mode` and `mask` remain config metadata but are not used in the forward pass.
- Sine smooth-mask projection example: use `coupling_model.balance_projection={"enabled": true, "mode": "smooth_mask", "mask": "sin"}` to select `sin(pi*y)` / `sin(pi*x)` masks while keeping the same balance projection and `smooth_mask_power` / `smooth_mask_diff_power` controls.
- Coupling losses: Coupling training uses independently controlled losses under `coupling_training.losses`: represented-solution L2 consistency, energy consistency, cross consistency, `balance_loss`, and `symmetric_boundary_loss`. The nested loss config is the only supported schema. `balance_loss` penalizes the raw common-grid residual `f - phi - psi` and is allowed only when `coupling_model.balance_projection.enabled=false`; enabling it with projection on fails fast instead of being ignored. `symmetric_boundary_loss` trains the symmetric projection's raw difference mode at transverse boundaries by penalizing `(phi_raw - psi_raw) + f` on `phi` boundaries `y=0,1` and `(phi_raw - psi_raw) - f` on `psi` boundaries `x=0,1`. It is allowed only with `coupling_model.balance_projection.enabled=true`, `mode="symmetric"`, and without `source_stencil_lift` or `green_response_feature`.
- Coupling source-lift diagnostics: when the source stencil lift is enabled, training and validation log `source_lift_corr_g_f`, `source_lift_rel_diff_g_f`, and `source_lift_g_rms` to track how the learned interior source branch input compares with the normalized physical source.
- Coupling optimizer config: `coupling_training.learning_rate` and `coupling_training.weight_decay` control the main CouplingNet AdamW group. Set `coupling_training.source_stencil_lift_learning_rate` or `coupling_training.source_stencil_lift_weight_decay` to create a separate optimizer group for the input-side 5-stencil source encoder; omitted source values fall back to the main group values. If trainable smooth-mask `q` is enabled, it is excluded from the main group and placed in a separate `smooth_mask_diff_power` group with the main learning rate and zero weight decay. The shared LR schedule applies the same multiplicative factor to all groups.
- Coupling gradient clipping: CouplingNet training clips gradients with `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` by default. Set `coupling_training.gradient_clip_max_norm` to a positive value to change the threshold, or `null` to disable clipping.
- Coupling validation mode: validation and standalone evaluation run with `model.eval()` and `torch.no_grad()`, then restore the model's previous train/eval mode. Dropout remains active for training batches but is inactive for validation metrics and best-validation checkpoint selection.
- Coupling periodic checkpoints: set `coupling_training.periodic_checkpoint.enabled=true` and `coupling_training.periodic_checkpoint.every_epochs=<int>` to save additional AdamW-phase checkpoints as `coupling_model_epoch_XXXX.safetensors`.
- Coupling best-validation checkpoint: set `coupling_training.best_rel_sol_checkpoint.enabled=true` to save `coupling_model_adam_best_rel_sol.safetensors` whenever CouplingNet AdamW training achieves a new best validation `rel_sol`. This requires a validation dataset.
- Coupling loss config: use `coupling_training.losses.l2_consistency`, `coupling_training.losses.energy_consistency`, `coupling_training.losses.cross_consistency`, `coupling_training.losses.balance_loss`, and `coupling_training.losses.symmetric_boundary_loss`, each with `enabled` and `weight`. The sample config keeps projection on with `mode="smooth_mask"` and leaves `balance_loss.enabled=false` and `symmetric_boundary_loss.enabled=false`; a raw-output run should set `coupling_model.balance_projection.enabled=false` before enabling `coupling_training.losses.balance_loss`. A symmetric-boundary run should set `coupling_model.balance_projection.enabled=true`, `mode="symmetric"`, and then enable `coupling_training.losses.symmetric_boundary_loss`.
- Energy consistency auxiliary loss: `energy_consistency` now computes the face-based physical energy of the represented-solution residual `r = u_phi^(x) - u_psi^(y)`, using forward edge differences, arithmetic face coefficients, and the density `a_face * |D_face r|^2`.
- Cross consistency auxiliary loss: `cross_consistency` penalizes the cross-operator terms `L_x(u_psi^(y)) <-> phi` and `L_y(u_phi^(x)) <-> psi`, reusing the same represented solutions, conservative stencil, common-grid slicing, and quadrature rule as the rest of the Coupling trainer.
- Green hybrid trunk: the learned Green correction always sees smooth handcrafted features `x`, `xi`, `x*xi`, `x^2`, `xi^2`, `x-xi`, `(x-xi)^2`, and `sqrt((x-xi)^2 + eps)`. If `model.use_fourier=true`, the Fourier embedding is appended to that structured trunk basis instead of replacing it.
- Coupling LR schedule: set `coupling_training.use_lr_schedule=true` with `warmup_epochs` and `min_lr` to enable linear warmup + cosine annealing during CouplingNet AdamW training.
- Coefficients: the training CLI resolves `a_fun`, `apx_fun`, `apy_fun`, directional convection coefficients `bx_fun`/`by_fun`, and `c_fun`, then forwards the same function set into the axial sampler and Coupling datasets. Internally `b_vals[0]` stores x-direction convection and `b_vals[1]` stores y-direction convection.
- Coefficient functions: set `dataset.coefficient_functions_path` to a Python file that defines callable `a_fun`, `apx_fun`, `apy_fun`, `bx_fun`, `by_fun`, and `c_fun` with signature `(x, y) -> Tensor`; `configs/sinusoidal_coefficients.py` is a default-equivalent example. Legacy files that define only `b_fun` are interpreted as `bx_fun = b_fun` and `by_fun = b_fun`; mixing `b_fun` with `bx_fun`/`by_fun` or defining only one directional convection function fails fast.
- Sampler mode: set `dataset.sampler_mode` to `"forward"` (sample `u`, derive `f`) or `"backward"` (sample `f`, solve `-d/dx(a u') + b u' + c u = f` with `scipy.integrate.solve_bvp` to recover `u`).
- Green validation reconstruction: set `training.compute_validation_rel_sol=true` to generate a separate synthetic validation dataset for GreenONet and log `val_rel_sol` alongside training `rel_sol`.
- Green validation dataset controls: use `dataset.validation_samples_per_line` to choose how many validation samples are generated per axial line, `dataset.validation_scale_length` to override the validation sampler length scale, and `dataset.validation_sampler_mode` to override the validation sampler type. If either validation override is omitted, Green validation reuses the training-side `dataset.scale_length` and `dataset.sampler_mode`.
- Green per-line export: `per_line_metrics.csv` now includes validation reconstruction columns `val_rel_sol_line`, `val_rel_sol_line_mean`, `val_rel_sol_line_min`, `val_rel_sol_line_max`, and `val_rel_sol_line_std` when Green validation is enabled; `per_line_metrics_summary.json` also includes validation summary entries.
- Dataset domain: the CLI ignores `dataset.domain` (sampling defaults to the unit square), so configs can omit it safely.
- Dataset split symlinks: use `python make_splits.py --source <npz_dir> --out <split_dir> --train-ratio 0.8 --valid-ratio 0.1 --test-ratio 0.1 --seed 42` to create `train/`, `valid/`, and `test/` symlink views. Ratios must be non-negative and sum to `1.0`; zero-ratio splits are created as empty directories. Keep the source `.npz` files in place because outputs are symlinks, not copies.
- For closer parity with the original `/home/jjhong0608/Documents/GreenONet`, the entrypoint `greenonet.runner.run_green_o_net` mirrors the classic `run_green_o_net.py` API and now includes `sampler_mode` for forward/backward synthetic data generation.

## Evaluation

- Exporting paper-oriented GreenNet checkpoint artifacts:
  ```
  PYTHONPATH=src python cli/export_green_artifacts.py \
    --checkpoint checkpoints/run_green_net/model.safetensors \
    --config checkpoints/run_green_net/config_used.json \
    --outdir checkpoints/run_green_net/green_artifacts \
    --eval-seed 12345 \
    --eval-split validation_like \
    --eval-samples-per-line 10 \
    --line-indices 0 32 64 96 128 \
    --xi-fractions 0.25 0.5 0.75 \
    --device cpu \
    --theme plotly_white
  ```
  The exporter reloads the GreenONet checkpoint, regenerates evaluation source/solution samples from the config's Green sampler distribution, and writes `summary.json`, metric CSVs, selected raw arrays, Green-kernel heatmaps, fixed-`xi` one-dimensional Green slices, coefficient slices, and axial reconstruction figures. It uses `training.device` from the config unless `--device` is provided. Every Plotly figure is saved as `.html`, editable Plotly `.json`, `.png`, and `.pdf`; if static export fails because Kaleido/Chrome is unavailable, `.html` and `.json` are still written. Use `--eval-seed` and the saved metadata to distinguish same-distribution evaluation from reusing the exact training samples. The paper-facing `rel_green` metric is exported when an exact/reference line Green kernel is supported: diffusion-only runs use `rel_green_reference="diffusion"`, reaction-free convection-diffusion runs use `rel_green_reference="convection_diffusion"`, and reaction runs record a skip reason.
- Plotting GreenONet logs: `python plot_green_logs.py --logs checkpoints/run_green_net/training.log --outdir plots_green --theme plotly_white` (supports multiple logs, `--labels`, and Plotly templates via `--theme`; outputs `loss`, `train_rel_sol`, `val_rel_sol`, and `rel_green` figures as HTML/JSON plus PNG/PDF if available).
- Plotting paper-facing CouplingNet logs: `python plot_coupling_logs.py --logs checkpoints/coupling_run/training.log checkpoints/coupling_run_2/training.log --labels run1 run2 --outdir plots_coupling --theme plotly_white`. Add `--show-annotations` to label each trace's last value and minimum value near the corresponding curve points. This intentionally writes only `loss`, `l2_consistency`, `energy_consistency`, `rel_flux`, and `rel_sol` training curves, each with train/validation traces when present. Every figure is saved as `.html` and editable Plotly `.json`, plus `.png`/`.pdf` when static export is available.
- Plotting recent Coupling logs from the current `_run_training_phase - epoch ...` format: `python plot_logs.py --logs checkpoints/test_diffusion/coupling/single_unknown/backward/training.log --outdir plots_coupling_recent`. This plots total loss, L2 consistency, energy consistency, cross consistency, optional `balance_loss`, optional `symmetric_boundary_loss`, `rel_flux`, and `rel_sol` from the current Coupling trainer log lines and ignores compile/checkpoint noise lines.
- Complex GreenNet training example:
  ```
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_circular_geometry.py \
    --step-size 0.05 \
    --out data/geometry/unit_circle_h005.npz

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/train.py \
    --config configs/complex_green.json \
    --work-dir checkpoints/complex_green

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/export_green_artifacts.py \
    --checkpoint checkpoints/complex_green/model.safetensors \
    --config checkpoints/complex_green/config_used.json \
    --outdir checkpoints/complex_green/green_artifacts \
    --device cpu
  ```
  The complex GreenNet path writes the standard training curves and `model.safetensors`, plus `per_interval_metrics.csv`, `per_interval_metrics_summary.json`, and a first-interval `green_heatmap.html`. Artifact export records `geometry_mode="complex"`, the geometry path, interval counts, per-interval metrics, selected flat-interval kernels, coefficient slices, and unit reconstruction figures.
- Plotting complex GreenNet per-interval metrics:
  ```
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python plot_complex_green_interval_metrics.py \
    --csv checkpoints/complex_domain_poisson/green/artifacts/metrics/per_interval_metrics.csv \
    --outdir checkpoints/complex_domain_poisson/green/artifacts/figures/per_interval_metrics \
    --overwrite
  ```
  This reads `per_interval_metrics.csv` with pandas and writes Plotly figures for solution reconstruction error versus transverse coordinate and segment length, axis-wise distributions, Green-kernel error summaries, and x/y chord maps colored by `rel_sol_interval_mean`. Outputs are saved as `.html` and editable `.json`, plus `.png`/`.pdf` when static Plotly export is available. `metrics_summary.json` records row counts, axis counts, metric summaries, source CSV, and generated figure names.
- Plotting complex CouplingNet per-sample test metrics:
  ```
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python plot_complex_coupling_sample_metrics.py \
    --csv checkpoints/complex_domain_poisson/coupling/metrics/test_per_sample_metrics.csv \
    --outdir checkpoints/complex_domain_poisson/coupling/figures/test_per_sample_metrics \
    --overwrite
  ```
  This reads `test_per_sample_metrics.csv` with pandas and writes Plotly figures for `rel_sol`, `rel_flux`, `loss`, log-scaled loss, relative-metric distributions, `rel_sol` versus `rel_flux`, and best/worst sample rankings by `rel_sol`, `rel_flux`, and `loss`. Relative errors are stored as raw fractions in the CSV and summary, but figures display them as percentages. Outputs are saved as `.html` and editable `.json`, plus `.png`/`.pdf` when static Plotly export is available. `metrics_summary.json` records raw and percent summaries, metric correlations, best/worst sample records, source CSV, and generated figure names.
- Exporting paper-oriented CouplingNet checkpoint artifacts:
  ```
  PYTHONPATH=src python cli/export_coupling_artifacts.py \
    --config checkpoints/run_coupling/config_used.json \
    --coupling-checkpoint checkpoints/run_coupling/coupling_model.safetensors \
    --green-checkpoint checkpoints/run_green_net/model.safetensors \
    --outdir checkpoints/run_coupling/coupling_artifacts \
    --device cpu \
    --theme plotly_white
  ```
  The exporter reads `dataset.test_path` as a `CouplingDataset`, uses `dataset.coefficient_functions_path` unless `--coefficients` overrides it, and uses `coupling_training.device` unless `--device` is provided. If `--selected-samples` is omitted, selected heatmaps default to five `rel_sol` representatives: min, q25, q50, q75, and max; pass `--selected-samples 0 5 12` to override this policy explicitly. It writes `summary.json`, per-sample and aggregate metric CSVs, selected raw `.npz` archives, coefficient/source context figures, solution figures, flux-divergence figures, and balance figures. Paper-facing error heatmaps are signed differences only: `u_pred - u`, `u_pred_x - u`, `u_pred_y - u`, `u_pred_x - u_pred_y`, `phi_pred - phi`, and `psi_pred - psi`. The paper exporter does not create null-space or closure diagnostic figures; those remain in the debug evaluator path.
- CouplingNet debug evaluation on test data (per-sample metrics and diagnostic plots):
  1. Set `dataset.test_path` in your config.
  2. Run:
     ```
     python cli/eval_coupling.py \
       --config configs/default.json \
       --coupling-checkpoint checkpoints/your_coupling.safetensors \
       --green-checkpoint checkpoints/run_green_net/model.safetensors \
       --work-dir checkpoints/eval_run
     ```
  3. Outputs: per-file metrics CSV (relative L2 for solution and flux-divergence via the configured `coupling_training.integration_rule`) and Plotly heatmaps (Times New Roman, bold) for exact/pred/error of solution and flux-divergence saved as pdf/png/html.
- Complex geometry training/evaluation example:
  ```
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_circular_geometry.py \
    --step-size 0.05 \
    --out data/geometry/unit_circle_h005.npz

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/train.py \
    --config configs/complex_geometry.json \
    --work-dir checkpoints/complex_geometry_run

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/eval_coupling.py \
    --config checkpoints/complex_geometry_run/config_used.json \
    --coupling-checkpoint checkpoints/complex_geometry_run/complex_coupling_model.safetensors \
    --green-checkpoint checkpoints/run_green_net/model.safetensors \
    --work-dir checkpoints/complex_geometry_eval

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/export_coupling_artifacts.py \
    --config checkpoints/complex_geometry_run/config_used.json \
    --coupling-checkpoint checkpoints/complex_geometry_run/complex_coupling_model.safetensors \
    --green-checkpoint checkpoints/run_green_net/model.safetensors \
    --outdir checkpoints/complex_geometry_run/complex_artifacts \
    --device cpu
  ```
  The complex artifact exporter writes `summary.json`, `metrics/per_sample_metrics.csv`, `data/selected_raw_arrays.npz`, and Plotly valid-point scatter figures on `coords_valid`. Primary fields include `rhs`, `sol`, `u_pred=0.5*(u_phi+u_psi)`, `u_phi`, `u_psi`, projected physical `phi`/`psi`, signed solution errors `u_pred_error`, `u_phi_error`, `u_psi_error`, split mismatch `u_split_mismatch`, and optional target flux fields plus signed `phi_error`/`psi_error` when sample flux targets are available. Error and mismatch figures are signed differences with zero-centered diverging colors. Raw unit outputs are archived in the `.npz` only, not used as primary figures.
- Coupling null-space diagnostics: evaluation also exports `null_sol_x`, `null_sol_y`, and `null_sol_residual` heatmaps, where `q` is inferred from the flux errors and integrated with the pretrained Green kernels to visualize the hidden null-space contribution in solution space.
- Coupling closure diagnostics: evaluation also exports `closure_phi_residual` and `closure_psi_residual` heatmaps for the exact-flux baseline `L_fd(G(phi_exact)) - phi_exact` and `L_fd(G(psi_exact)) - psi_exact`, where `L_fd` is the conservative stencil for `-d_s(a d_s u) + b d_s u + c u`.
- Evaluation batching: CouplingNet evaluation uses `coupling_training.batch_size` to batch computations while still saving plots per sample.
- Plot export parallelism: CouplingNet evaluation exports all per-sample heatmaps with a process pool (default `plot_workers=4`).
- Per-line CSV bar comparison: use `python plot_per_line_bars.py --csv-a <run_a/per_line_metrics.csv> --csv-b <run_b/per_line_metrics.csv> --label-a run_a --label-b run_b --outdir <output_dir>` to generate grouped-bar figures for `(x, rel_sol_line_mean)`, `(x, val_rel_sol_line_mean)`, `(x, rel_green_line_mean)`, `(y, rel_sol_line_mean)`, `(y, val_rel_sol_line_mean)`, `(y, rel_green_line_mean)`. Training and validation solution figures include error bars from `rel_sol_line_std` and `val_rel_sol_line_std`; Green-function figures use `rel_green_line_mean` only. Outputs are saved as `.png` and `.pdf`.

## Development

- Tests first: `PYTHONPATH=src pytest test`
- Lint/format/type-check: `ruff check src`, `ruff format src`, `mypy src`
- Key dirs: `src/` core code (including axial-line sampler and cleaned runner), `cli/` CLIs, `configs/` JSON configs, `checkpoints/` outputs, `test/` tests.

## References
- Axial Green's Function Method (see `refenreces/` PDFs) as conceptual inspiration for the axial decomposition in `GreenONetModel`.
