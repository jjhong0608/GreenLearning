# Modified GreenONet

Axial-inspired neural solver for the 2D Poisson equation with Dirichlet boundaries. The project follows the AGENTS guidelines: class-first design, rich logging mirrored to disk, Plotly visualizations, and TDD.

## Setup
- Activate the virtual environment: `source .venv/bin/activate`
- Install runtime deps: `pip install -e .`
- On Linux `x86_64` with CPython `3.14`, the editable install now pins Torch to the official PyTorch `2.11.0+cu126` wheel from `download.pytorch.org`. Other environments fall back to `torch>=2.11.0`, so install a different PyTorch CUDA build manually if you need one.
- See dev tools (ruff/mypy/pytest): `pip install -e .[dev]`
- Ensure `PYTHONPATH` includes `src` when running commands in this repo.

## Usage
- Train with the sample config: `PYTHONPATH=src python cli/train.py --config configs/default.json --work-dir checkpoints/run`
- Logs: Rich console output plus `training.log` in the chosen `work-dir`.
- Artifacts: `loss_curve.html`, `green_heatmap.html`, and weights `model.safetensors` in `work-dir`.
- Config: the CLI copies the input JSON to `config_used.json` in the `work-dir`.
- Checkpoints: `*.safetensors` include JSON-encoded model config metadata; use `greenonet.io.load_model_with_config` to restore model+config.
- Loss: Green's-function reconstruction — integrates the learned Green kernel against the source and matches the recovered solution (no direct output MSE).
- GreenNet structure: the analytic Green wrapping remains active when `model.use_green=true`, while the learned correction now uses fused line encoders for `a`, `a'`, `b`, and `c` plus a hybrid trunk with smooth `(x, xi)` features.
- Green analytic coefficients: under the conservative operator `-d_x(a(x) d_x u) + b(x) d_x u + c(x) u = f`, the implemented wrapping uses `A(x, xi) = 1 / a(x)` and `B(x, xi) = (a'(x) + b(x)) / a(x)^2`, so both coefficients depend on the evaluation-side `x` values.
- Integration rule: set `training.integration_rule` and `coupling_training.integration_rule` to `"simpson"` or `"trapezoid"` to control sampled-data quadrature in Green/Coupling training, evaluation metrics, and coupling RHS normalization. Green synthetic samplers reuse `training.integration_rule` for sample normalization.
- Optimizers: GreenONet uses Adam by default with optional multi-epoch LBFGS fine-tuning (see `TrainingConfig.lbfgs_*`). CouplingNet training is Adam-only.
- Torch compile: set `training.compile.enabled=true` to wrap GreenONet with `torch.compile`, and set `coupling_training.compile.enabled=true` to do the same for CouplingNet. The flags are independent, optional, and checkpoint saving still unwraps compiled models to keep load/save compatibility.
- CouplingNet: a shared branch/trunk MIONet consumes axial-line inputs `(a, b, c, f)` together with interior coordinates and predicts axial flux-divergences `(phi_x, psi_y)` through a single shared DeepONet-style readout followed by balance projection.
- CouplingNet optional source stencil lift: set `coupling_model.source_stencil_lift.enabled=true` to add an input-side learned 5-point source preprocessing module. It reconstructs the canonical full source from `rhs_raw`, normalizes by physical interior RMS, maps each interior source stencil to a scalar lifted source `g`, optionally normalizes `g` by interior RMS, and uses interior-only `g` as the source branch input. The coefficient branch still consumes boundary-including coefficient lines, and the physical `rhs_raw`, `rhs_norm`, balance projection, losses, and evaluation targets remain unchanged.
- Balance projection: CouplingNet applies a fixed symmetric interior balance projection, splitting the residual evenly with a `0.5/0.5` correction between the x- and y-axis flux components.
- Coupling losses: the model returns final axial flux-divergences after the fixed balance projection. Coupling training uses three independently controlled losses under `coupling_training.losses`: represented-solution L2 consistency, energy consistency, and cross consistency. The nested loss config is the only supported schema.
- Coupling source-lift diagnostics: when the source stencil lift is enabled, training and validation log `source_lift_corr_g_f`, `source_lift_rel_diff_g_f`, and `source_lift_g_rms` to track how the learned interior source branch input compares with the normalized physical source.
- Coupling optimizer config: use the flat shared optimizer fields in `coupling_training` (`learning_rate`, `epochs`, `use_lr_schedule`, `warmup_epochs`, `min_lr`) for the single-stage Coupling trainer.
- Coupling periodic checkpoints: set `coupling_training.periodic_checkpoint.enabled=true` and `coupling_training.periodic_checkpoint.every_epochs=<int>` to save additional Adam-phase checkpoints as `coupling_model_epoch_XXXX.safetensors`.
- Coupling best-validation checkpoint: set `coupling_training.best_rel_sol_checkpoint.enabled=true` to save `coupling_model_adam_best_rel_sol.safetensors` whenever Adam training achieves a new best validation `rel_sol`. This requires a validation dataset.
- Coupling loss config: use `coupling_training.losses.l2_consistency`, `coupling_training.losses.energy_consistency`, and `coupling_training.losses.cross_consistency`, each with `enabled` and `weight`. The sample config defaults all three to `enabled=true` and `weight=1.0`.
- Energy consistency auxiliary loss: `energy_consistency` now computes the face-based physical energy of the represented-solution residual `r = u_phi^(x) - u_psi^(y)`, using forward edge differences, arithmetic face coefficients, and the density `a_face * |D_face r|^2`.
- Cross consistency auxiliary loss: `cross_consistency` penalizes the cross-operator terms `L_x(u_psi^(y)) <-> phi` and `L_y(u_phi^(x)) <-> psi`, reusing the same represented solutions, conservative stencil, common-grid slicing, and quadrature rule as the rest of the Coupling trainer.
- Green hybrid trunk: the learned Green correction always sees smooth handcrafted features `x`, `xi`, `x*xi`, `x^2`, `xi^2`, `x-xi`, `(x-xi)^2`, and `sqrt((x-xi)^2 + eps)`. If `model.use_fourier=true`, the Fourier embedding is appended to that structured trunk basis instead of replacing it.
- Coupling LR schedule: set `coupling_training.use_lr_schedule=true` with `warmup_epochs` and `min_lr` to enable linear warmup + cosine annealing during Adam only.
- Coefficients: the training CLI defines `a_fun`, `apx_fun`, `apy_fun`, plus `b_fun` and `c_fun`, and forwards them into the axial sampler.
- Sampler mode: set `dataset.sampler_mode` to `"forward"` (sample `u`, derive `f`) or `"backward"` (sample `f`, solve `-d/dx(a u') + b u' + c u = f` with `scipy.integrate.solve_bvp` to recover `u`).
- Green validation reconstruction: set `training.compute_validation_rel_sol=true` to generate a separate synthetic validation dataset for GreenONet and log `val_rel_sol` alongside training `rel_sol`.
- Green validation dataset controls: use `dataset.validation_samples_per_line` to choose how many validation samples are generated per axial line, `dataset.validation_scale_length` to override the validation sampler length scale, and `dataset.validation_sampler_mode` to override the validation sampler type. If either validation override is omitted, Green validation reuses the training-side `dataset.scale_length` and `dataset.sampler_mode`.
- Green per-line export: `per_line_metrics.csv` now includes validation reconstruction columns `val_rel_sol_line`, `val_rel_sol_line_mean`, `val_rel_sol_line_min`, `val_rel_sol_line_max`, and `val_rel_sol_line_std` when Green validation is enabled; `per_line_metrics_summary.json` also includes validation summary entries.
- Dataset domain: the CLI ignores `dataset.domain` (sampling defaults to the unit square), so configs can omit it safely.
- For closer parity with the original `/home/jjhong0608/Documents/GreenONet`, the entrypoint `greenonet.runner.run_green_o_net` mirrors the classic `run_green_o_net.py` API and now includes `sampler_mode` for forward/backward synthetic data generation.

## Evaluation

- Plotting GreenONet logs: `python plot_green_logs.py --logs checkpoints/run_green_net/training.log --outdir plots_green` (supports multiple logs and `--labels`; outputs HTML plus PNG/PDF if available).
- Plotting CouplingNet logs with separate error figures: `python plot_coupling_logs.py --logs checkpoints/coupling_run/training.log checkpoints/coupling_run_2/training.log --labels run1 run2 --outdir plots_coupling`.
- Plotting recent Adam-only Coupling logs from the current `_run_training_phase - epoch ...` format: `python plot_logs.py --logs checkpoints/test_diffusion/coupling/single_unknown/backward/training.log --outdir plots_coupling_recent`. This plots total loss, L2 consistency, energy consistency, cross consistency, `rel_flux`, and `rel_sol` from the current Coupling trainer log lines and ignores compile/checkpoint noise lines.
- CouplingNet evaluation on test data (per-sample metrics and plots):
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
