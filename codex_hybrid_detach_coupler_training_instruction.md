# Codex Instruction: Add Hybrid-Detach Training for the Post-Projection Five-Stencil Coupler

## Scope

You are modifying the existing local `Coupler` branch of `jjhong0608/GreenLearning`.

Assume the local repository already contains the latest post-projection coupler implementation. In particular, assume the current local code already does the following:

1. Computes the raw interior CouplingNet output.
2. Computes the pre-projection residual
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

This task is **not** to redesign the coupler. This task is to add a **hybrid-detach training mode**.

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

Add support for the following hybrid-detach energy training objective:

\[
\mathcal{L}_{\mathrm{hybrid\_detach}}
=
w_p \, \mathcal{E}(\phi^p,\psi^p)
+
w_c \,
\mathcal{E}
\left(
C_\theta(
\operatorname{detach}(\phi^p,\psi^p),
\operatorname{detach}(r^0)
)
\right).
\]

Definitions:

- \((\phi^0,\psi^0)\): raw interior CouplingNet output before projection.
- \(r^0=f-(\phi^0+\psi^0)\): pre-projection residual.
- \((\phi^p,\psi^p)\): projected split after fixed balance projection.
- \(C_\theta\): post-projection five-stencil coupler.
- \((\phi^c,\psi^c)=C_\theta(\phi^p,\psi^p;r^0)\): post-coupler final split.
- \(\mathcal{E}(\cdot,\cdot)\): existing energy-consistency loss.

The main CouplingNet should be trained by the projected energy term:

\[
w_p \, \mathcal{E}(\phi^p,\psi^p).
\]

The coupler should be trained by the coupled energy term:

\[
w_c \, \mathcal{E}(\phi^c,\psi^c),
\]

where both the projected input and the pre-projection residual passed into the coupler are detached.

This is crucial:

```python
coupler_input = projected_int.detach()
coupler_residual = pre_projection_residual.detach()
```

Do **not** detach only `projected_int`. If `pre_projection_residual` is not detached, gradients from the coupled loss may leak back into the main CouplingNet through the residual feature.

---

## Required Design Principles

### Keep the post-projection coupler

Do not move the coupler back before projection.

The model structure should remain:

\[
\text{raw CouplingNet output}
\rightarrow
\text{balance projection}
\rightarrow
\text{five-stencil coupler}.
\]

### Keep the coupler architecture

Do not change:

- `FiveStencilStencilMLPCoupler` class name.
- explicit five-stencil gather.
- `point_features = 9`.
- `in_channels = 45`.
- scalar `delta` output.
- null-space update:
  ```python
  phi1 = phi0 + delta
  psi1 = psi0 - delta
  ```
- final-layer zero initialization.
- nested `coupling_model.coupler` config schema.

### Add hybrid-detach training support without breaking the default path

Default behavior must remain backward compatible.

If `hybrid_detach.enabled == False`, existing training behavior should remain unchanged.

If `return_intermediates == False`, `CouplingNet.forward()` should still return only the final full flux tensor, as before.

---

## Files to Inspect and Modify

Primary files:

```text
src/greenonet/coupling_model.py
src/greenonet/coupling_trainer.py
src/greenonet/config.py
configs/default_coupling.json
test/test_coupling.py
```

Likely documentation updates:

```text
README.md
CouplingNet.md
coupling_net.md
model_structure.md
```

Only update documentation files that actually exist.

---

# Part 1 — Modify `CouplingNet.forward()`

## Goal

Expose both:

1. the projected interior flux before the coupler, and
2. the coupled interior flux after the coupler.

Also allow the coupler path to receive detached inputs when requested.

---

## Required API Change

Extend `CouplingNet.forward()` with two keyword-only or trailing optional arguments:

```python
return_intermediates: bool = False
detach_coupler_input: bool = False
```

The default must preserve current behavior:

```python
out = model(...)
assert isinstance(out, torch.Tensor)
```

When `return_intermediates=True`, return:

```python
out, aux = model(..., return_intermediates=True)
```

where:

```python
aux = {
    "raw_int": raw_int,
    "pre_projection_residual": pre_projection_residual,
    "projected_int": projected_int_before_coupler,
    "coupled_int": coupled_int_after_coupler,
}
```

Expected shapes:

```text
raw_int:                  (B, 2, N, N)
pre_projection_residual:  (B, N, N)
projected_int:            (B, 2, N, N)
coupled_int:              (B, 2, N, N)
out:                      (B, 2, N, N+2)
```

---

## Required Forward Logic

Use this logic as the target structure.

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

Important details:

- `projected_int_before_coupler` must store the projection output **before** any coupler correction.
- `coupled_int` must store the post-coupler output.
- Do not overwrite `projected_int` in a way that loses the pre-coupler projected state.
- When `detach_coupler_input=True`, both `coupler_input` and `coupler_residual` must be detached.
- The final `output_flux` must be built from `coupled_int`.
- If no coupler is configured, `coupled_int = projected_int`.

---

## Balance Residual Helper

If the local code already has `_compute_balance_residual()`, reuse it. If not, add it.

It must use the same canonical orientation as `_apply_balance_projection()`.

```python
def _compute_balance_residual(
    self,
    flux_int: torch.Tensor,
    rhs_raw: torch.Tensor,
) -> torch.Tensor:
    """Compute canonical residual f - (phi + psi) on interior points.

    Args:
        flux_int: (B, 2, N, N)
        rhs_raw:  (B, 2, N, N+2)

    Returns:
        residual: (B, N, N)
    """
    rhs_x_int = rhs_raw[:, 0, :, 1:-1]
    phi = flux_int[:, 0]
    psi_t = flux_int[:, 1].transpose(-1, -2)
    return rhs_x_int - (phi + psi_t)
```

---

# Part 2 — Preserve and Validate Coupler Auxiliary Residual Behavior

If the local code already supports:

```python
auxiliary_residual: torch.Tensor | None = None
```

in `FiveStencilStencilMLPCoupler.forward()` and `_build_canonical_point_features()`, keep that behavior.

If not, add it.

The point-feature construction must be:

```python
phi0, psi0 = self._canonicalize_flux(raw_int)
f = rhs_raw[:, 0, :, 1:-1]

scale = torch.sqrt(
    torch.mean(f * f, dim=(-1, -2), keepdim=True) + self.eps
)

diff_field = 0.5 * (phi0 - psi0)

if auxiliary_residual is None:
    balance_residual = f - (phi0 + psi0)
else:
    if tuple(auxiliary_residual.shape) != tuple(f.shape):
        raise ValueError(
            "auxiliary_residual must have shape matching canonical source "
            f"{tuple(f.shape)}, got {tuple(auxiliary_residual.shape)}."
        )
    balance_residual = auxiliary_residual

diff_hat = diff_field / scale
res_hat = balance_residual / scale
f_hat = f / scale

q = torch.stack(
    [
        diff_hat,
        res_hat,
        f_hat,
        ax,
        ay,
        bx,
        by,
        cx,
        cy,
    ],
    dim=1,
)
```

Interpretation in hybrid-detach mode:

- `raw_int` passed to the coupler is the **projected** interior flux.
- `diff_field` is therefore:
  \[
  d^p=\frac12(\phi^p-\psi^p).
  \]
- `auxiliary_residual` is detached pre-projection residual:
  \[
  \operatorname{detach}(r^0).
  \]

---

# Part 3 — Add Hybrid-Detach Config

## Add a nested config dataclass

In `src/greenonet/config.py`, add a nested training config. Use the project’s existing config/dataclass style.

Suggested dataclass:

```python
@dataclass
class CouplingHybridDetachConfig:
    enabled: bool = False
    projected_energy_weight: float = 1.0
    coupled_energy_weight: float = 0.1
    detach_coupler_input: bool = True
```

Then add this to `CouplingTrainingConfig`:

```python
hybrid_detach: CouplingHybridDetachConfig = field(
    default_factory=CouplingHybridDetachConfig
)
```

If the repository uses a different config initialization pattern, adapt to the local style while preserving the same nested keys and defaults.

---

## Default JSON config

In `configs/default_coupling.json`, add:

```json
"hybrid_detach": {
  "enabled": false,
  "projected_energy_weight": 1.0,
  "coupled_energy_weight": 0.1,
  "detach_coupler_input": true
}
```

Place it inside the training config section, not inside `coupling_model`.

The default must be `enabled=false` to preserve existing behavior.

---

# Part 4 — Modify `CouplingTrainer`

## Goal

When `training.hybrid_detach.enabled == true`, compute:

\[
\mathcal{L}
=
w_p \, \mathcal{E}(\phi^p,\psi^p)
+
w_c \, \mathcal{E}(\phi^c,\psi^c),
\]

where:

- \((\phi^p,\psi^p)\) comes from `aux["projected_int"]`;
- \((\phi^c,\psi^c)\) comes from the final model output;
- the final model output should have been produced using detached coupler input if `detach_coupler_input=True`.

---

## Add a helper to pad interior flux

The trainer needs to compute energy loss on `aux["projected_int"]`, which is an interior tensor. Add a helper if none exists:

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

## Avoid duplicating energy-loss logic

Do not copy-paste the full energy-loss computation twice if avoidable.

Prefer to refactor the existing code so that energy loss for a given full flux can be computed by a helper.

For example, if `_step_loss()` currently does:

```python
pred_flux = self.model(...)
phi = pred_flux[:, 0]
psi = pred_flux[:, 1]
u_phi_x = ...
u_psi_y = ...
loss_energy = self._energy_consistency_loss(...)
```

factor out a helper:

```python
def _represented_solutions_from_flux(
    self,
    flux: torch.Tensor,
    ...
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    return u_phi_x, u_psi_y
```

or:

```python
def _energy_loss_from_flux(
    self,
    flux: torch.Tensor,
    batch: CouplingBatch,
) -> torch.Tensor:
    ...
```

Use the project’s local style. The key requirement is that hybrid-detach mode computes the same energy-consistency loss on both:

1. projected full flux, and
2. coupled final full flux.

---

## Hybrid-detach training path

Inside `_step_loss()` or the equivalent training-step function, add a branch like:

```python
if self.config.hybrid_detach.enabled:
    pred_flux, aux = self.model(
        coords=coords,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        rhs_tilde=rhs_tilde,
        rhs_norm=rhs_norm,
        return_intermediates=True,
        detach_coupler_input=self.config.hybrid_detach.detach_coupler_input,
    )

    projected_flux = self._pad_interior_flux_like(
        aux["projected_int"],
        template_flux=pred_flux,
    )

    coupled_flux = pred_flux

    loss_energy_projected = self._energy_loss_from_flux(
        projected_flux,
        batch_or_needed_inputs,
    )

    loss_energy_coupled = self._energy_loss_from_flux(
        coupled_flux,
        batch_or_needed_inputs,
    )

    loss = (
        self.config.hybrid_detach.projected_energy_weight
        * loss_energy_projected
        + self.config.hybrid_detach.coupled_energy_weight
        * loss_energy_coupled
    )

    metrics["loss_energy_projected"] = float(loss_energy_projected.detach())
    metrics["loss_energy_coupled"] = float(loss_energy_coupled.detach())
    metrics["loss_hybrid_detach"] = float(loss.detach())
```

Use the actual local batch variable names.

Important:

- In hybrid-detach mode, do not also add the original `loss_energy_consistency` on the final flux unless this is intentionally configured.
- The clean first implementation should treat hybrid-detach energy as replacing the existing energy-consistency term.
- If `l2_consistency` or `cross_consistency` are enabled, you may keep them on the final output only if this is consistent with the existing trainer structure. Do not silently double-count the energy loss.

A safe policy:

```text
If hybrid_detach.enabled == true:
    use hybrid-detach energy as the energy component.
    do not also compute the old single-output energy component.
    keep existing non-energy losses only if they are already enabled and easy to preserve.
```

---

## Evaluation behavior

During validation/test, the final model output should remain the post-coupler flux.

`detach_coupler_input` only affects gradient flow. It does not change the numerical forward output except for autograd graph connectivity.

For metric reporting, use the final output as before.

If useful, also log projected/coupled energy separately during training and validation when `hybrid_detach.enabled=true`.

---

# Part 5 — Metrics

Add the following metrics when hybrid-detach is enabled:

```text
loss_energy_projected
loss_energy_coupled
loss_hybrid_detach
```

If practical, also add:

```text
energy_improvement = loss_energy_projected - loss_energy_coupled
```

This value shows whether the coupler immediately reduces the energy-consistency loss relative to the projected split.

Optional but useful:

```text
delta_norm_ratio
```

Only add `delta_norm_ratio` if it can be done cleanly without complicating the model API. It is not required for the first implementation.

---

# Part 6 — Tests

Add or update tests in `test/test_coupling.py` or the most appropriate existing test file.

Do not remove existing compatibility tests.

---

## Test 1 — Default forward remains backward compatible

Verify:

```python
out = model(...)
assert isinstance(out, torch.Tensor)
```

and shape:

```text
(B, 2, N, N+2)
```

This should work without passing `return_intermediates`.

---

## Test 2 — `return_intermediates=True`

Verify:

```python
out, aux = model(..., return_intermediates=True)
```

Expected keys:

```python
{
    "raw_int",
    "pre_projection_residual",
    "projected_int",
    "coupled_int",
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

Also verify that `out[:, :, :, 1:-1]` matches `aux["coupled_int"]`.

---

## Test 3 — projected and coupled states are distinct when coupler is nonzero

With the default zero-initialized final coupler layer, projected and coupled states may be equal. To test the separation:

1. instantiate model with coupler enabled;
2. manually make the coupler produce nonzero correction, or replace the coupler with a deterministic mock coupler;
3. run:

```python
out, aux = model(..., return_intermediates=True)
```

Verify:

```python
not torch.allclose(aux["projected_int"], aux["coupled_int"])
```

when nonzero correction is forced.

Also verify both states preserve balance:

```python
phi_p = aux["projected_int"][:, 0]
psi_p = aux["projected_int"][:, 1].transpose(-1, -2)

phi_c = aux["coupled_int"][:, 0]
psi_c = aux["coupled_int"][:, 1].transpose(-1, -2)

f = rhs_raw[:, 0, :, 1:-1]

torch.testing.assert_close(phi_p + psi_p, f)
torch.testing.assert_close(phi_c + psi_c, f)
```

---

## Test 4 — detach behavior

Use a mock coupler to verify that when:

```python
detach_coupler_input=True
```

the coupler receives detached tensors.

Mock coupler signature:

```python
class InspectingCoupler(nn.Module):
    def forward(
        self,
        raw_int,
        a_vals,
        b_vals,
        c_vals,
        rhs_raw,
        auxiliary_residual=None,
    ):
        assert raw_int.requires_grad is False
        assert auxiliary_residual is not None
        assert auxiliary_residual.requires_grad is False
        return raw_int
```

Then call:

```python
out, aux = model(..., return_intermediates=True, detach_coupler_input=True)
```

Make sure the input tensors require grad where appropriate so the test is meaningful.

Also add the opposite test if easy:

```python
detach_coupler_input=False
```

and verify the coupler input is not detached.

---

## Test 5 — pre-projection residual is still non-detached in `aux`

When `return_intermediates=True`, `aux["pre_projection_residual"]` should be the original non-detached tensor unless the implementation intentionally documents otherwise.

The tensor passed to the coupler may be detached. The tensor stored in `aux` should remain useful for diagnostics and projected loss analysis.

---

## Test 6 — config parsing

Test that the new nested config parses correctly.

Example JSON block:

```json
"hybrid_detach": {
  "enabled": true,
  "projected_energy_weight": 1.0,
  "coupled_energy_weight": 0.1,
  "detach_coupler_input": true
}
```

Verify the dataclass fields match.

If there are existing config round-trip tests, update them.

---

## Test 7 — trainer hybrid-detach path

Add a small trainer-level test if feasible.

It should verify that when `hybrid_detach.enabled=true`, the step loss contains/logs:

```text
loss_energy_projected
loss_energy_coupled
loss_hybrid_detach
```

and that the old single-output energy term is not double-counted.

If constructing a full trainer test is too heavy, add a smaller unit test around the helper that computes energy loss from a full flux and around `_pad_interior_flux_like()`.

---

# Part 7 — Documentation Updates

Update documentation that still describes the coupler as pre-projection.

Preferred wording:

```text
The five-stencil StencilMLP coupler is applied after the fixed balance projection.
The projection first enforces phi + psi = f. The coupler then performs a null-space
redistribution by predicting a scalar delta and applying
(phi, psi) -> (phi + delta, psi - delta), so phi + psi = f remains preserved.
```

Add a short description of hybrid-detach training:

```text
Hybrid-detach training supervises both the projected split and the coupled split.
The main CouplingNet is trained through the projected energy-consistency loss.
The coupler is trained through a coupled energy-consistency loss computed from
detached projected fields and detached pre-projection residuals, preventing the
coupled loss from backpropagating into the main CouplingNet.
```

Also update any `CouplerConfig` docstring that still says the coupler is inserted before projection.

---

# Part 8 — Validation Commands

Run at least:

```bash
PYTHONPATH=src pytest test/test_coupling.py -q
```

Also run if feasible:

```bash
PYTHONPATH=src pytest test/test_io_config.py -q
PYTHONPATH=src pytest test -q
ruff check src test
mypy src
```

If `ruff` or `mypy` fails due to pre-existing unrelated issues, report that explicitly and include the relevant failure summary.

---

# Acceptance Criteria

The implementation is correct if all of the following are true:

1. Default `CouplingNet.forward()` still returns only the final full flux tensor.
2. `CouplingNet.forward(..., return_intermediates=True)` returns `(output_flux, aux)`.
3. `aux` contains:
   - `raw_int`
   - `pre_projection_residual`
   - `projected_int`
   - `coupled_int`
4. `projected_int` is the post-projection, pre-coupler interior flux.
5. `coupled_int` is the post-coupler interior flux.
6. The final output flux is padded from `coupled_int`.
7. When `detach_coupler_input=True`, both projected coupler input and pre-projection residual passed to the coupler are detached.
8. The `aux["projected_int"]` value remains usable for projected energy loss.
9. Hybrid-detach config exists with defaults:
   - `enabled = False`
   - `projected_energy_weight = 1.0`
   - `coupled_energy_weight = 0.1`
   - `detach_coupler_input = True`
10. When `hybrid_detach.enabled=false`, existing training behavior is unchanged.
11. When `hybrid_detach.enabled=true`, trainer computes:
    \[
    w_p \mathcal{E}(\phi^p,\psi^p)
    +
    w_c \mathcal{E}(\phi^c,\psi^c)
    \]
    without double-counting the old single-output energy loss.
12. Tests verify default behavior, intermediate return, detach behavior, config parsing, and hybrid-detach training metrics.
13. Documentation no longer describes the coupler as pre-projection.

---

# Things Not to Do

Do not:

- run or suggest git branch operations
- move the coupler back before projection
- redesign the coupler architecture
- rename `FiveStencilStencilMLPCoupler`
- remove `auxiliary_residual`
- remove the scalar null-space update
- change `point_features = 9`
- add diagonal/corner stencil points
- change the fixed balance projection formula
- make hybrid detach enabled by default
- silently double-count the old energy loss and the new hybrid-detach energy loss
- detach only `projected_int` while leaving `pre_projection_residual` attached in the coupler path

---

# Instruction Review from Codex Perspective

Before completing the patch, review the implementation against these questions.

## Can I identify exactly what to change?

- [ ] Do I know where to add `return_intermediates`?
- [ ] Do I know where to add `detach_coupler_input`?
- [ ] Do I know which tensors should be stored in `aux`?
- [ ] Do I know how to pad `projected_int` back to full flux shape?

## Is the gradient flow correct?

- [ ] Does projected energy train the main CouplingNet?
- [ ] Does coupled energy train the coupler?
- [ ] Does coupled energy avoid backpropagating into the main CouplingNet when detach is enabled?
- [ ] Are both `projected_int` and `pre_projection_residual` detached before being passed to the coupler?

## Is the post-projection structure preserved?

- [ ] Is projection applied before the coupler?
- [ ] Is the coupler still using post-projection difference `0.5 * (phi - psi)`?
- [ ] Is the coupler still receiving pre-projection residual as auxiliary feature?
- [ ] Does the final post-coupler field still satisfy `phi + psi = f`?

## Is the trainer loss unambiguous?

- [ ] Is hybrid-detach energy a replacement for the old single-output energy component when enabled?
- [ ] Are projected and coupled energy losses logged separately?
- [ ] Is `loss_hybrid_detach` logged?
- [ ] Are non-energy losses handled without double-counting energy?

## Are tests sufficient?

- [ ] Does default forward remain backward compatible?
- [ ] Is intermediate return tested?
- [ ] Is detach behavior tested with a mock coupler?
- [ ] Is config parsing tested?
- [ ] Is trainer hybrid-detach behavior tested or at least covered by helper-level tests?

## Final expected summary

When reporting completion, summarize the change as:

```text
Added hybrid-detach training support for the post-projection five-stencil coupler.
CouplingNet.forward can now return raw/projected/coupled intermediates and can detach
the projected coupler input plus pre-projection residual before calling the coupler.
The trainer can compute projected energy for the main CouplingNet and coupled energy
for the coupler with configurable weights. Default behavior remains unchanged because
hybrid_detach.enabled defaults to false.
```
