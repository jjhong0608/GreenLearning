# Codex Instruction: Add Stage 2 Coupler-Only Training Without Delta Regularization

## Scope

You are modifying the existing local `Coupler` branch of `jjhong0608/GreenLearning`.

Assume the local repository already contains the latest post-projection coupler implementation. In particular, assume the current local code already does the following:

1. Computes the raw interior CouplingNet output.
2. Computes the pre-projection residual:
   \[
   r^0 = f - (\phi^0 + \psi^0).
   \]
3. Applies the fixed balance projection.
4. Applies `FiveStencilStencilMLPCoupler` **after** the projection.
5. Passes the pre-projection residual to the coupler as `auxiliary_residual`.
6. Builds coupler features as:
   \[
   [\widehat{d^p}, \widehat{r^0}, \hat f, a^x, a^y, b^x, b^y, c^x, c^y].
   \]
7. Applies the scalar null-space update:
   \[
   \phi \leftarrow \phi+\delta,\qquad \psi\leftarrow\psi-\delta.
   \]

This task is **not** to redesign the coupler.  
This task is to add a **Stage 2 coupler-only training mode**.

Do **not** run or suggest git operations such as:

```text
git checkout
git pull
git merge
git rebase
git branch
git switch
```

Work only on the current local working tree.

---

## High-Level Objective

Add a Stage 2 training mode that:

1. Loads a Stage 1 checkpoint.
2. Freezes the main CouplingNet.
3. Trains only the post-projection five-stencil coupler.
4. Uses only the coupled energy-consistency loss for optimization.
5. Logs projected/coupled diagnostic metrics.
6. Does **not** use delta regularization.
7. Does **not** use early stopping.
8. Keeps the existing best-checkpoint policy based on validation relative solution error.

Mathematically, Stage 2 should solve:

\[
\theta_{\mathrm{main}} \ \text{fixed},
\qquad
\theta_c \ \text{trainable},
\]

\[
\mathcal{L}_{\mathrm{stage2}}
=
w_c\,\mathcal{E}(\phi^c,\psi^c).
\]

Definitions:

- \((\phi^0,\psi^0)\): raw CouplingNet interior output before projection.
- \((\phi^p,\psi^p)\): projected split after the fixed balance projection.
- \((\phi^c,\psi^c)\): post-coupler split.
- \(\mathcal{E}(\cdot,\cdot)\): existing energy-consistency loss.
- \(w_c\): `training.stage2.coupled_energy_weight`.

The projected energy:

\[
\mathcal{E}(\phi^p,\psi^p)
\]

must be computed only as a diagnostic metric, not as part of the Stage 2 loss.

---

## Important Exclusions

Do **not** add delta regularization in this implementation.

Do not add any of the following fields, unless they already exist for another reason and are unrelated to this task:

```text
delta_reg_weight
delta_regularization
delta_smoothness_weight
loss_stage2_delta_reg
```

Do not add early stopping for Stage 2.

Stage 2 should run for the configured number of epochs.

However, keep the existing best-checkpoint mechanism. The best checkpoint should still be selected using validation relative solution error:

```text
val_rel_sol
lower is better
```

Do not switch best-checkpoint selection to coupled energy, projected energy, or relative improvement.

---

## Files to Inspect and Modify

Primary files:

```text
src/greenonet/config.py
configs/default_coupling.json
src/greenonet/coupling_model.py
src/greenonet/coupling_trainer.py
cli/train.py
test/test_coupling.py
```

Additional files if relevant in the local codebase:

```text
src/greenonet/io.py
test/test_io_config.py
README.md
CouplingNet.md
coupling_net.md
model_structure.md
```

Only modify documentation files that actually exist.

---

# Part 1 — Add Stage 2 Config

## Add a nested Stage 2 config

In `src/greenonet/config.py`, add a nested dataclass using the project’s existing config style.

Suggested structure:

```python
@dataclass
class CouplingStage2Config:
    enabled: bool = False
    checkpoint_path: str | None = None

    freeze_main: bool = True
    train_coupler_only: bool = True

    coupled_energy_weight: float = 1.0

    lr: float = 1.0e-3
    weight_decay: float = 0.0
    epochs: int | None = None

    early_stopping: bool = False

    log_relative_improvement: bool = True
    log_delta_norm_ratio: bool = True
```

Then add it to `CouplingTrainingConfig`:

```python
stage2: CouplingStage2Config = field(default_factory=CouplingStage2Config)
```

If the repository uses a different config initialization pattern, adapt to the local style while preserving the same nested keys and defaults.

---

## Required default values

Defaults must preserve existing behavior:

```text
stage2.enabled = False
stage2.checkpoint_path = None
stage2.freeze_main = True
stage2.train_coupler_only = True
stage2.coupled_energy_weight = 1.0
stage2.lr = 1.0e-3
stage2.weight_decay = 0.0
stage2.epochs = None
stage2.early_stopping = False
stage2.log_relative_improvement = True
stage2.log_delta_norm_ratio = True
```

Do not add delta regularization fields.

---

## Update `configs/default_coupling.json`

Add the following block inside the training config section, not inside `coupling_model`:

```json
"stage2": {
  "enabled": false,
  "checkpoint_path": null,
  "freeze_main": true,
  "train_coupler_only": true,
  "coupled_energy_weight": 1.0,
  "lr": 0.001,
  "weight_decay": 0.0,
  "epochs": null,
  "early_stopping": false,
  "log_relative_improvement": true,
  "log_delta_norm_ratio": true
}
```

The default must be disabled.

---

# Part 2 — Ensure Model Intermediates Are Available

Stage 2 needs both projected and coupled interior fluxes.

If `CouplingNet.forward()` already supports:

```python
return_intermediates: bool = False
detach_coupler_input: bool = False
```

then keep that behavior.

If not, add it.

---

## Required `CouplingNet.forward()` behavior

Default behavior must remain unchanged:

```python
out = model(...)
assert isinstance(out, torch.Tensor)
```

When:

```python
out, aux = model(..., return_intermediates=True)
```

the returned `aux` dictionary must include:

```python
{
    "raw_int": raw_int,
    "pre_projection_residual": pre_projection_residual,
    "projected_int": projected_int_before_coupler,
    "coupled_int": coupled_int_after_coupler,
}
```

Expected shapes:

```text
raw_int:                 (B, 2, N, N)
pre_projection_residual: (B, N, N)
projected_int:           (B, 2, N, N)
coupled_int:             (B, 2, N, N)
out:                     (B, 2, N, N+2)
```

The final output flux must be padded from `coupled_int`.

---

## Required forward structure

The model should follow this structure:

```python
raw_int = flux_tilde * rhs_norm.unsqueeze(-1)

pre_projection_residual = self._compute_balance_residual(
    flux_int=raw_int,
    rhs_raw=rhs_raw,
)

projected_int = self._apply_balance_projection(raw_int, rhs_raw)
projected_int_before_coupler = projected_int

if self.coupler is not None:
    coupler_input = projected_int
    coupler_residual = pre_projection_residual

    if detach_coupler_input:
        coupler_input = coupler_input.detach()
        coupler_residual = coupler_residual.detach()

    coupled_int = self.coupler(
        raw_int=coupler_input,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        auxiliary_residual=coupler_residual,
    )
else:
    coupled_int = projected_int

output_flux = torch.zeros(
    b,
    axis,
    n_lines,
    m_points,
    dtype=coupled_int.dtype,
    device=coupled_int.device,
)
output_flux[:, :, :, 1:-1] = coupled_int

if return_intermediates:
    return output_flux, {
        "raw_int": raw_int,
        "pre_projection_residual": pre_projection_residual,
        "projected_int": projected_int_before_coupler,
        "coupled_int": coupled_int,
    }

return output_flux
```

Important:

- Do not move the coupler before projection.
- Do not overwrite the projected state in a way that loses the pre-coupler projected state.
- In Stage 2, call the model with:
  ```python
  return_intermediates=True
  detach_coupler_input=True
  ```
- Even though the main network is frozen in Stage 2, `detach_coupler_input=True` is still recommended to simplify the computation graph and prevent gradient leakage.

---

# Part 3 — Add Main-Freeze / Coupler-Only Utility

Stage 2 should freeze all main model parameters and unfreeze only the coupler.

Add a utility in the most appropriate local location, for example `coupling_trainer.py`, a model utility module, or `cli/train.py`.

Recommended implementation:

```python
def freeze_main_train_coupler_only(model: nn.Module) -> None:
    """Freeze all parameters except the post-projection coupler."""
    if getattr(model, "coupler", None) is None:
        raise RuntimeError(
            "Stage 2 requires model.coupler, but model.coupler is None."
        )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.coupler.parameters():
        param.requires_grad = True
```

Prefer direct access to `model.coupler.parameters()` rather than searching parameter names for the substring `"coupler"`.

---

## Defensive checks

When `training.stage2.enabled == true`, validate:

```text
coupling_model.coupler.enabled == true
model.coupler is not None
training.stage2.checkpoint_path is not None
```

If any condition fails, raise a clear `ValueError` or `RuntimeError`.

---

# Part 4 — Checkpoint Loading for Stage 2

Stage 2 must start from a Stage 1 checkpoint.

In `cli/train.py` or the existing checkpoint setup path:

```python
if config.training.stage2.enabled:
    if config.training.stage2.checkpoint_path is None:
        raise ValueError(
            "training.stage2.checkpoint_path must be provided when "
            "training.stage2.enabled=true."
        )

    # Load model weights from the Stage 1 checkpoint.
    # Do not restore the Stage 1 optimizer state.
    load_model_weights_only(
        model=model,
        checkpoint_path=config.training.stage2.checkpoint_path,
        strict=True,
    )

    freeze_main_train_coupler_only(model)
```

Use the repository’s existing checkpoint loading helpers if they exist.

Important:

- Load model weights.
- Do not restore Stage 1 optimizer state.
- Create a new optimizer for Stage 2.
- Freeze main before creating the Stage 2 optimizer.

If the existing checkpoint loader always restores optimizer state, add an option or separate path for model-only loading.

---

# Part 5 — Stage 2 Optimizer

When `training.stage2.enabled == true`, the optimizer must include only trainable coupler parameters.

Suggested pattern:

```python
trainable_params = [p for p in model.parameters() if p.requires_grad]

if not trainable_params:
    raise RuntimeError("No trainable parameters found for Stage 2.")

optimizer = torch.optim.Adam(
    trainable_params,
    lr=config.training.stage2.lr,
    weight_decay=config.training.stage2.weight_decay,
)
```

Use the project’s optimizer construction style if it already exists, but preserve the requirement:

```text
Stage 2 optimizer must not include frozen main CouplingNet parameters.
```

---

# Part 6 — Add Stage 2 Loss Path in `CouplingTrainer`

## Goal

When `training.stage2.enabled == true`, compute:

\[
\mathcal{L}_{stage2}
=
w_c \mathcal{E}(\phi^c,\psi^c).
\]

The projected energy must be computed only as a diagnostic metric:

\[
\mathcal{E}_{proj}
=
\mathcal{E}(\phi^p,\psi^p).
\]

Do not include projected energy in the Stage 2 loss.

---

## Add or reuse full-flux padding helper

The projected flux in `aux["projected_int"]` is interior-only. Add a helper if none exists:

```python
@staticmethod
def _pad_interior_flux_like(
    flux_int: torch.Tensor,
    template_flux: torch.Tensor,
) -> torch.Tensor:
    """Pad an interior flux tensor into the full axial flux layout.

    Args:
        flux_int:      (B, 2, N, N)
        template_flux: (B, 2, N, N+2)

    Returns:
        full_flux:     (B, 2, N, N+2)
    """
    if flux_int.dim() != 4:
        raise ValueError(f"flux_int must be 4D, got {tuple(flux_int.shape)}.")
    if template_flux.dim() != 4:
        raise ValueError(
            f"template_flux must be 4D, got {tuple(template_flux.shape)}."
        )
    if flux_int.shape[:3] != template_flux.shape[:3]:
        raise ValueError(
            "flux_int and template_flux have incompatible leading shapes: "
            f"{tuple(flux_int.shape)} vs {tuple(template_flux.shape)}."
        )
    if template_flux.shape[-1] != flux_int.shape[-1] + 2:
        raise ValueError(
            "template_flux last dimension must be flux_int last dimension + 2."
        )

    full_flux = torch.zeros_like(template_flux)
    full_flux[:, :, :, 1:-1] = flux_int
    return full_flux
```

---

## Refactor energy loss if needed

Avoid duplicating the entire energy-consistency implementation.

If possible, factor out a helper such as:

```python
def _energy_loss_from_flux(
    self,
    flux: torch.Tensor,
    batch: CouplingBatch,
) -> torch.Tensor:
    ...
```

or use an equivalent local helper that computes the existing energy-consistency loss for a given full flux.

The key requirement is:

```text
The same energy-consistency definition must be used for projected_flux and coupled_flux.
```

---

## Stage 2 loss pseudo code

Add a separate Stage 2 branch in the training step:

```python
if self.config.stage2.enabled:
    return self._step_loss_stage2(batch)
```

Suggested implementation:

```python
def _step_loss_stage2(self, batch):
    pred_flux, aux = self.model(
        coords=batch.coords,
        a_vals=batch.a_vals,
        b_vals=batch.b_vals,
        c_vals=batch.c_vals,
        rhs_raw=batch.rhs_raw,
        rhs_tilde=batch.rhs_tilde,
        rhs_norm=batch.rhs_norm,
        return_intermediates=True,
        detach_coupler_input=True,
    )

    projected_flux = self._pad_interior_flux_like(
        aux["projected_int"],
        template_flux=pred_flux,
    )

    coupled_flux = pred_flux

    loss_energy_projected = self._energy_loss_from_flux(
        projected_flux,
        batch,
    )

    loss_energy_coupled = self._energy_loss_from_flux(
        coupled_flux,
        batch,
    )

    loss = (
        self.config.stage2.coupled_energy_weight
        * loss_energy_coupled
    )

    delta = self._delta_from_intermediates(
        projected_int=aux["projected_int"],
        coupled_int=aux["coupled_int"],
    )

    relative_improvement = (
        (loss_energy_projected - loss_energy_coupled)
        / (loss_energy_projected.abs() + 1.0e-12)
    )

    delta_norm_ratio = self._delta_norm_ratio(
        delta=delta,
        projected_int=aux["projected_int"],
    )

    metrics = {
        "loss": float(loss.detach()),
        "loss_stage2_projected_energy": float(loss_energy_projected.detach()),
        "loss_stage2_coupled_energy": float(loss_energy_coupled.detach()),
        "stage2_energy_improvement": float(
            (loss_energy_projected - loss_energy_coupled).detach()
        ),
        "stage2_relative_improvement": float(relative_improvement.detach()),
        "stage2_delta_norm_ratio": float(delta_norm_ratio.detach()),
    }

    return loss, metrics
```

Use the actual local batch field names.

Important:

- `loss_energy_projected` is monitor-only.
- `loss_energy_projected` must not be added to `loss`.
- No delta regularization should be added.
- No early stopping should be added.

---

# Part 7 — Logging Helpers

## Delta extraction helper

Add this helper to the trainer or a local utility module:

```python
@staticmethod
def _delta_from_intermediates(
    projected_int: torch.Tensor,
    coupled_int: torch.Tensor,
) -> torch.Tensor:
    """Recover scalar coupler correction delta from projected/coupled states."""
    phi_p = projected_int[:, 0]
    psi_p = projected_int[:, 1].transpose(-1, -2)

    phi_c = coupled_int[:, 0]
    psi_c = coupled_int[:, 1].transpose(-1, -2)

    delta_phi = phi_c - phi_p
    delta_psi = psi_p - psi_c

    return 0.5 * (delta_phi + delta_psi)
```

This is for logging only. Do not use it for regularization.

---

## Delta norm ratio helper

Add:

```python
@staticmethod
def _delta_norm_ratio(
    delta: torch.Tensor,
    projected_int: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Compute ||delta|| / (||phi_projected|| + ||psi_projected||)."""
    phi_p = projected_int[:, 0]
    psi_p = projected_int[:, 1].transpose(-1, -2)

    num = torch.linalg.vector_norm(
        delta.reshape(delta.shape[0], -1),
        dim=1,
    )

    den = (
        torch.linalg.vector_norm(
            phi_p.reshape(phi_p.shape[0], -1),
            dim=1,
        )
        + torch.linalg.vector_norm(
            psi_p.reshape(psi_p.shape[0], -1),
            dim=1,
        )
        + eps
    )

    return torch.mean(num / den)
```

---

## Required Stage 2 metrics

Log all of the following:

```text
loss
loss_stage2_projected_energy
loss_stage2_coupled_energy
stage2_energy_improvement
stage2_relative_improvement
stage2_delta_norm_ratio
```

Definitions:

\[
\text{stage2\_energy\_improvement}
=
\mathcal{E}_{proj}
-
\mathcal{E}_{coupled}.
\]

\[
\text{stage2\_relative\_improvement}
=
\frac{
\mathcal{E}_{proj}
-
\mathcal{E}_{coupled}
}{
|\mathcal{E}_{proj}|+\epsilon
}.
\]

\[
\text{stage2\_delta\_norm\_ratio}
=
\frac{
\|\delta\|_2
}{
\|\phi^p\|_2+\|\psi^p\|_2+\epsilon
}.
\]

---

# Part 8 — Best Checkpoint and Early Stopping

## Early stopping

Do not implement Stage 2 early stopping.

Even though the config includes:

```json
"early_stopping": false
```

it should remain false and should not stop training early.

Stage 2 should run for the configured number of epochs.

---

## Best checkpoint

Keep the existing best-checkpoint policy unchanged.

The best checkpoint should still be selected according to validation relative solution error:

```text
val_rel_sol
lower is better
```

Do not switch the Stage 2 best checkpoint metric to:

```text
loss_stage2_coupled_energy
loss_stage2_projected_energy
stage2_relative_improvement
stage2_energy_improvement
```

These metrics are diagnostic only.

Add explicit comments if helpful:

```text
Stage 2 uses coupled energy for optimization, but best-checkpoint selection remains
based on validation relative solution error for consistency with the existing training
workflow.
```

---

# Part 9 — Tests

Add or update tests in `test/test_coupling.py`, `test/test_io_config.py`, or the most appropriate existing files.

Do not remove existing tests.

---

## Test 1 — Stage 2 config parsing

Verify that a config block like:

```json
"stage2": {
  "enabled": true,
  "checkpoint_path": "dummy_stage1.pt",
  "freeze_main": true,
  "train_coupler_only": true,
  "coupled_energy_weight": 1.0,
  "lr": 0.001,
  "weight_decay": 0.0,
  "epochs": 500,
  "early_stopping": false,
  "log_relative_improvement": true,
  "log_delta_norm_ratio": true
}
```

is parsed into the expected dataclass values.

Also verify that no delta regularization field is required.

---

## Test 2 — Freeze main and train coupler only

Create a model with coupler enabled.

Call:

```python
freeze_main_train_coupler_only(model)
```

Verify:

```python
for p in model.coupler.parameters():
    assert p.requires_grad

for name, p in model.named_parameters():
    if p not in set(model.coupler.parameters()):
        assert not p.requires_grad
```

Use `id(p)` sets rather than direct tensor comparison when checking membership:

```python
coupler_param_ids = {id(p) for p in model.coupler.parameters()}
for p in model.parameters():
    if id(p) in coupler_param_ids:
        assert p.requires_grad
    else:
        assert not p.requires_grad
```

---

## Test 3 — Stage 2 optimizer contains only coupler parameters

After freeze and optimizer creation, verify:

```python
opt_param_ids = {
    id(p)
    for group in optimizer.param_groups
    for p in group["params"]
}
coupler_param_ids = {id(p) for p in model.coupler.parameters()}

assert opt_param_ids == coupler_param_ids
```

or at least:

```python
assert opt_param_ids.issubset(coupler_param_ids)
```

depending on whether all coupler parameters are trainable.

---

## Test 4 — Stage 2 loss uses only coupled energy

With delta regularization absent and projected energy monitor-only, verify:

```python
loss == coupled_energy_weight * loss_stage2_coupled_energy
```

within numerical tolerance.

Projected energy must appear in metrics but must not be included in `loss`.

---

## Test 5 — Required Stage 2 metrics are logged

Verify metrics include:

```text
loss_stage2_projected_energy
loss_stage2_coupled_energy
stage2_energy_improvement
stage2_relative_improvement
stage2_delta_norm_ratio
```

---

## Test 6 — Delta norm ratio helper

Test `_delta_from_intermediates()` and `_delta_norm_ratio()` on simple synthetic tensors.

Example:

\[
\phi^c=\phi^p+\delta,
\qquad
\psi^c=\psi^p-\delta.
\]

Verify recovered `delta` equals the known synthetic delta.

---

## Test 7 — No early stopping behavior is introduced

If there is a trainer config or trainer setup test, verify:

```python
config.training.stage2.early_stopping is False
```

and that Stage 2 does not change the existing best-checkpoint metric away from validation relative solution error.

Do not add a test that expects Stage 2 to stop early.

---

## Test 8 — Model forward intermediates

If not already covered by existing tests, verify:

```python
out, aux = model(..., return_intermediates=True, detach_coupler_input=True)
```

and check:

```text
aux["projected_int"]
aux["coupled_int"]
aux["pre_projection_residual"]
aux["raw_int"]
```

exist and have expected shapes.

---

# Part 10 — Documentation Updates

Update documentation only where relevant.

Add a short Stage 2 description:

```text
Stage 2 coupler-only training loads a Stage 1 checkpoint, freezes the main CouplingNet,
and optimizes only the post-projection five-stencil coupler using coupled energy
consistency. Projected energy is logged as a diagnostic baseline, and the logs also
include relative energy improvement and delta norm ratio. Delta regularization is not
used in the current Stage 2 implementation.
```

Also mention:

```text
Stage 2 does not use early stopping. Best checkpoint selection remains based on
validation relative solution error.
```

---

# Part 11 — Validation Commands

Run at least:

```bash
PYTHONPATH=src pytest test/test_coupling.py -q
PYTHONPATH=src pytest test/test_io_config.py -q
```

If feasible, also run:

```bash
PYTHONPATH=src pytest test -q
ruff check src test
mypy src
```

If `ruff` or `mypy` fails due to pre-existing unrelated issues, report that explicitly with the relevant failure summary.

---

# Acceptance Criteria

The implementation is correct if all of the following are true:

1. `training.stage2` config exists and defaults to disabled.
2. `training.stage2` does not include delta regularization fields.
3. Stage 2 requires `checkpoint_path` when enabled.
4. Stage 2 loads model weights from the Stage 1 checkpoint.
5. Stage 2 does not restore the Stage 1 optimizer state.
6. Stage 2 freezes all main CouplingNet parameters.
7. Stage 2 leaves only `model.coupler.parameters()` trainable.
8. Stage 2 optimizer includes only trainable coupler parameters.
9. Stage 2 model forward uses `return_intermediates=True`.
10. Stage 2 model forward uses `detach_coupler_input=True`.
11. Stage 2 loss is:
    \[
    \texttt{coupled\_energy\_weight} \times \mathcal{E}_{coupled}.
    \]
12. Projected energy is computed only as a metric.
13. `stage2_relative_improvement` is logged.
14. `stage2_delta_norm_ratio` is logged.
15. No delta regularization loss is added.
16. No early stopping is added.
17. Best checkpoint selection remains based on validation relative solution error.
18. Existing non-Stage-2 training behavior remains unchanged when `stage2.enabled=false`.

---

# Things Not to Do

Do not:

- run or suggest git branch operations
- redesign the coupler
- move the coupler before projection
- remove `auxiliary_residual`
- change `point_features = 9`
- add diagonal or corner stencil points
- change the fixed balance projection formula
- add delta regularization
- add `delta_reg_weight`
- add `loss_stage2_delta_reg`
- add early stopping for Stage 2
- change best-checkpoint selection away from validation relative solution error
- include projected energy in the Stage 2 optimization loss
- include frozen main parameters in the Stage 2 optimizer
- restore Stage 1 optimizer state for Stage 2

---

# Instruction Review from Codex Perspective

Before completing the patch, review the implementation against this checklist.

## Stage 2 Config

- [ ] Did I add `training.stage2` as a nested config?
- [ ] Does it default to disabled?
- [ ] Did I avoid adding delta regularization fields?
- [ ] Is `early_stopping` false by default?

## Checkpoint and Freeze

- [ ] Does Stage 2 require `checkpoint_path`?
- [ ] Does Stage 2 load model weights only?
- [ ] Does Stage 2 avoid restoring the Stage 1 optimizer state?
- [ ] Does Stage 2 freeze the main CouplingNet?
- [ ] Are only `model.coupler.parameters()` trainable?

## Optimizer

- [ ] Is the Stage 2 optimizer newly created?
- [ ] Does it include only trainable coupler parameters?
- [ ] Does it use `stage2.lr` and `stage2.weight_decay`?

## Loss

- [ ] Is Stage 2 loss based only on coupled energy?
- [ ] Is projected energy monitor-only?
- [ ] Is there no delta regularization term?
- [ ] Is there no projected energy term in the optimization loss?

## Logging

- [ ] Are `loss_stage2_projected_energy` and `loss_stage2_coupled_energy` logged?
- [ ] Is `stage2_energy_improvement` logged?
- [ ] Is `stage2_relative_improvement` logged?
- [ ] Is `stage2_delta_norm_ratio` logged?

## Model Intermediates

- [ ] Does Stage 2 call the model with `return_intermediates=True`?
- [ ] Does Stage 2 call the model with `detach_coupler_input=True`?
- [ ] Are `projected_int` and `coupled_int` available?

## Checkpoint Policy

- [ ] Did I avoid implementing early stopping?
- [ ] Did I keep best checkpoint selection based on validation relative solution error?
- [ ] Did I avoid changing best checkpoint metric to coupled energy?

## Final Expected Summary

When reporting completion, summarize the change like this:

```text
Added Stage 2 coupler-only training. Stage 2 loads a Stage 1 checkpoint, freezes the
main CouplingNet, trains only the post-projection five-stencil coupler, and optimizes
coupled energy consistency. Projected energy, relative improvement, and delta norm
ratio are logged for diagnostics. Delta regularization and early stopping are not used.
Best checkpoint selection remains based on validation relative solution error.
```
