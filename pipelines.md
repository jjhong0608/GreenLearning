# Training Pipeline (Axial GreenONet)

1) **Axial grid setup**  
   - `make_square_axial_lines(step_size)` builds uniform x-/y-aligned lines over the square domain (float64 coordinates).

2) **Forward sampling**  
   - `ForwardSampler` draws RBF mixtures along each line (random or deterministic centers; scale length can be fixed or sampled from a range).  
   - Enforces Dirichlet via a linear boundary interpolant.  
   - Computes PDE pieces analytically on the line: `f = -∂x(a∂x u) + b∂x u + c u`, boundary correction `g`, then normalizes energy via Simpson’s rule.

3) **TrainingData assembly**  
   - Line-wise tensors `(U, F, A, AP, B, C, COORDS)` are stacked across all x- and y-lines for the requested per-line sample count.

4) **Dataset flattening**  
   - `AxialDataset` flattens `TrainingData` into pointwise samples `(coords, solution, source)` ready for PyTorch DataLoader.

5) **Model construction**  
   - `GreenONetModel` combines two axis MLPs (x/y) plus a coupling MLP over both coordinates. Activations include `tanh`, `relu`, `gelu`, `rational`.

6) **Training loop**  
   - `Trainer` builds a DataLoader, computes data MSE + PDE residual loss via autograd Laplacian, optimizes with Adam, and logs via Rich + file handler.

7) **Artifacts & logging**  
   - `training.log` and Plotly `loss_curve.html` saved in the chosen work dir. Optional checkpoints can be added later.

8) **Entrypoints**  
   - Programmatic: `run_green_o_net(...)` in `greenonet.runner`.  
   - CLI: `PYTHONPATH=src python cli/train.py --config configs/default.json --work-dir checkpoints/run`.
