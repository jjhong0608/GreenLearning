# Codex Instructions: Add Smooth-Mask Balance Projection to CouplingNet

## Goal

Add an optional smooth-mask balance projection mode to `CouplingNet`.

The current CouplingNet uses the symmetric balance projection

\[
\phi_{\mathrm{proj}}
=
\phi_0+\frac12\{f-(\phi_0+\psi_0)\},
\]

\[
\psi_{\mathrm{proj}}
=
\psi_0+\frac12\{f-(\phi_0+\psi_0)\}.
\]

Keep this existing behavior as the default.

Add a new optional projection mode using smooth masks

\[
m_\phi(y)=4y(1-y),
\qquad
m_\psi(x)=4x(1-x),
\]

and define, on the interior common grid,

\[
w_\phi=\frac{m_\phi}{m_\phi+m_\psi},
\qquad
w_\psi=\frac{m_\psi}{m_\phi+m_\psi},
\]

\[
\alpha=\frac{m_\phi m_\psi}{m_\phi+m_\psi}.
\]

Then project raw flux-divergence outputs by

\[
\phi_{\mathrm{proj}}
=
w_\phi f+\alpha(\phi_0-\psi_0),
\]

\[
\psi_{\mathrm{proj}}
=
w_\psi f-\alpha(\phi_0-\psi_0).
\]

This must satisfy the balance identity exactly on the interior common grid:

\[
\phi_{\mathrm{proj}}+\psi_{\mathrm{proj}}=f.
\]

The purpose is to preserve the current branch-trunk CouplingNet architecture while adding a boundary-admissible smooth-mask projection option.

---

## Files to Inspect and Modify

Primary files:

```text
src/greenonet/coupling_model.py
src/greenonet/config.py
```

Also inspect, and modify only if needed:

```text
tests/
configs/
README.md
```

Do not make broad architectural changes.

---

## Current Code Context

In `src/greenonet/coupling_model.py`, `CouplingNet.forward()` currently computes interior flux values and then applies `_apply_balance_projection()`:

```python
coords_int = coords[:, :, 1:-1, :]
...
raw_int = flux_tilde * norm_exp
projected_int = self._apply_balance_projection(raw_int, rhs_raw)
```

The current projection is:

```python
rhs_x_int = rhs_raw[:, 0, :, 1:-1]
phi = flux_int[:, 0]
psi_t = flux_int[:, 1].transpose(-1, -2)
res = rhs_x_int - (phi + psi_t)

phi = phi + 0.5 * res
psi_t = psi_t + 0.5 * res
projected = flux_int.clone()
projected[:, 0] = phi
projected[:, 1] = psi_t.transpose(-1, -2)
return projected
```

Refactor this so the existing symmetric projection remains available and is still the default.

---

## Configuration Requirements

Extend `CouplingModelConfig` in `src/greenonet/config.py`.

Add fields similar to:

```python
balance_projection: Literal["symmetric", "smooth_mask"] = "symmetric"
smooth_mask_normalize: bool = True
smooth_mask_eps: float = 1e-12
```

Notes:

1. `balance_projection="symmetric"` must preserve the existing behavior.
2. `balance_projection="smooth_mask"` must activate the new smooth-mask projection.
3. `smooth_mask_normalize=True` should use \(4y(1-y)\) and \(4x(1-x)\).
4. If `smooth_mask_normalize=False`, use \(y(1-y)\) and \(x(1-x)\).
5. Existing configs that do not specify these fields must still work.

Before editing config parsing code, inspect the existing config loader. If it constructs dataclasses from JSON dictionaries, ensure these new fields are accepted without breaking existing configs.

---

## CouplingNet Initialization Requirements

In `CouplingNet.__init__()`, store the projection options.

Either store the whole config:

```python
self.config = config
```

or preferably store explicit attributes:

```python
self.balance_projection = config.balance_projection
self.smooth_mask_normalize = config.smooth_mask_normalize
self.smooth_mask_eps = config.smooth_mask_eps
```

Use the style most consistent with the existing codebase.

---

## Projection Function Refactor

Refactor the projection code into separate methods.

Recommended structure:

```python
def _apply_symmetric_balance_projection(
    self,
    flux_int: torch.Tensor,
    rhs_raw: torch.Tensor,
) -> torch.Tensor:
    ...
```

```python
def _apply_smooth_mask_balance_projection(
    self,
    flux_int: torch.Tensor,
    rhs_raw: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    ...
```

```python
def _apply_balance_projection(
    self,
    flux_int: torch.Tensor,
    rhs_raw: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    if self.balance_projection == "symmetric":
        return self._apply_symmetric_balance_projection(flux_int, rhs_raw)
    if self.balance_projection == "smooth_mask":
        return self._apply_smooth_mask_balance_projection(flux_int, rhs_raw, coords)
    raise ValueError(f"Unsupported balance_projection: {self.balance_projection}")
```

Update the `forward()` call site:

```python
projected_int = self._apply_balance_projection(raw_int, rhs_raw, coords)
```

The symmetric projection does not need `coords`, but the dispatcher should accept it so the forward code is simple.

---

## Tensor Layout Requirements

Be careful with tensor layout. This is the most important implementation detail.

Inside the projection code, use the common grid layout:

```python
phi0 = flux_int[:, 0]                    # (B, n, n)
psi0 = flux_int[:, 1].transpose(-1, -2)  # (B, n, n)
f = rhs_raw[:, 0, :, 1:-1]               # (B, n, n)
```

Here:

- `phi0` is the x-line flux-divergence component in common `(y, x)` layout.
- `psi0` must be transposed to the same common `(y, x)` layout.
- `f` should be the RHS on the same interior common grid.
- The returned tensor must have the same shape and layout as `flux_int`.

After computing `phi` and `psi_t` in common layout:

```python
projected = flux_int.clone()
projected[:, 0] = phi
projected[:, 1] = psi_t.transpose(-1, -2)
return projected
```

---

## Coordinate Extraction for Smooth Masks

The smooth masks should be evaluated on the interior common grid.

Recommended coordinate extraction:

```python
x_axis = coords[0, 0, :, 0]      # (m_points,)
y_lines = coords[0, :, 0, 1]     # (n_lines,)

x_inner = x_axis[1:-1]           # (n_lines,)
y_inner = y_lines                # (n_lines,)
```

Then build masks:

```python
m_phi = y_inner * (1.0 - y_inner)   # depends on y
m_psi = x_inner * (1.0 - x_inner)   # depends on x

if self.smooth_mask_normalize:
    m_phi = 4.0 * m_phi
    m_psi = 4.0 * m_psi
```

Broadcast to common layout:

```python
m_phi = m_phi.view(1, -1, 1)  # (1, n, 1)
m_psi = m_psi.view(1, 1, -1)  # (1, 1, n)
```

Make sure masks are moved to the same dtype and device as `flux_int`:

```python
x_axis = x_axis.to(device=flux_int.device, dtype=flux_int.dtype)
y_lines = y_lines.to(device=flux_int.device, dtype=flux_int.dtype)
```

If the existing dataset coordinate layout suggests a better extraction method, use it, but preserve the semantic requirement:

- `m_phi` depends only on the transverse \(y\)-coordinate for \(\phi\).
- `m_psi` depends only on the transverse \(x\)-coordinate for \(\psi\).
- Both are evaluated on the interior common grid.

---

## Smooth-Mask Projection Formula

Implement the new projection preferably in the difference form:

```python
denom = (m_phi + m_psi).clamp_min(self.smooth_mask_eps)

w_phi = m_phi / denom
w_psi = m_psi / denom
alpha = (m_phi * m_psi) / denom

diff = phi0 - psi0

phi = w_phi * f + alpha * diff
psi_t = w_psi * f - alpha * diff
```

This is equivalent to the residual form:

```python
phi_hat = m_phi * phi0
psi_hat = m_psi * psi0
res = f - (phi_hat + psi_hat)
phi = phi_hat + w_phi * res
psi_t = psi_hat + w_psi * res
```

Use the difference form unless there is a strong code-style reason not to.

Important: on the interior grid, `denom` should be positive. The clamp is only a numerical safeguard. It should not normally affect the result.

---

## Balance Identity Requirement

For `balance_projection="smooth_mask"`, the following must hold on the interior common grid:

```python
phi = projected[:, 0]
psi_t = projected[:, 1].transpose(-1, -2)
f = rhs_raw[:, 0, :, 1:-1]

torch.testing.assert_close(phi + psi_t, f, ...)
```

Use tolerances appropriate to dtype:

- float64: approximately `atol=1e-10`, `rtol=1e-10`
- float32: approximately `atol=1e-5`, `rtol=1e-5`

---

## Backward Compatibility Requirements

1. Existing behavior must remain unchanged when `balance_projection="symmetric"`.
2. The default value must be `"symmetric"`.
3. Existing training scripts and config files must still run without specifying the new fields.
4. Do not change the CouplingNet branch/trunk architecture.
5. Do not change GreenONet or GreenNet code.
6. Do not change trainer losses.
7. Do not switch to full-grid flux prediction.
8. Do not implement boundary lifting.
9. Do not replace the two-output split by a one-sided definition such as `psi = f - phi`.

---

## Tests to Add or Update

Inspect the existing test structure first and follow its style.

Add tests for the projection if no appropriate tests already exist.

### Test 1: Symmetric projection preserves old formula

Create a small synthetic tensor setup and verify that `balance_projection="symmetric"` produces:

\[
\phi_{\rm sym}
=
\phi_0+\frac12(f-\phi_0-\psi_0),
\]

\[
\psi_{\rm sym}
=
\psi_0+\frac12(f-\phi_0-\psi_0).
\]

Use the common layout for comparison.

### Test 2: Smooth-mask projection satisfies exact balance

For `balance_projection="smooth_mask"`, verify:

\[
\phi_{\rm proj}+\psi_{\rm proj}=f.
\]

In code:

```python
phi = projected[:, 0]
psi_t = projected[:, 1].transpose(-1, -2)
f = rhs_raw[:, 0, :, 1:-1]
torch.testing.assert_close(phi + psi_t, f, ...)
```

### Test 3: Shape preservation

Verify:

```python
projected.shape == flux_int.shape
```

### Test 4: dtype and device preservation

Verify that output dtype and device match input dtype and device.

### Test 5: smooth-mask special behavior

Check at least one simple grid that:

- `m_phi` varies only with the interior y-coordinate.
- `m_psi` varies only with the interior x-coordinate.
- `m_phi + m_psi` is positive on the interior grid.
- if normalized, the maximum mask value is approximately 1 on grids containing or near the midpoint.

### Optional Test 6: all-one masks reduce to symmetric projection

If you factor out a common helper that accepts masks directly, test that `m_phi = m_psi = 1` gives the same result as the symmetric projection.

Do not over-engineer this if it requires unnatural exposure of private helpers.

---

## Recommended Implementation Order

1. Inspect `src/greenonet/config.py` and config loading code.
2. Add the new fields to `CouplingModelConfig`.
3. Update `CouplingNet.__init__()` to store projection options.
4. Split the existing projection into `_apply_symmetric_balance_projection()`.
5. Add `_apply_smooth_mask_balance_projection()`.
6. Add the dispatcher `_apply_balance_projection(..., coords)`.
7. Update `forward()` to pass `coords` into `_apply_balance_projection`.
8. Add tests for symmetric compatibility and smooth-mask balance.
9. Run the relevant tests.
10. Run formatting/linting if the repo uses it.

---

## Validation Commands

Use the repository's existing test commands if available. Based on the README style, likely commands include:

```bash
PYTHONPATH=src pytest tests
```

Also run any project-specific formatting or linting commands if they are already used by the repo, for example:

```bash
ruff check src tests
ruff format src tests
mypy src
```

Do not introduce new dependencies.

---

## Self-Check Checklist Before Finishing

Before finalizing the patch, verify:

- [ ] Existing default projection remains symmetric.
- [ ] `balance_projection="smooth_mask"` activates the new formula.
- [ ] The smooth-mask formula uses the difference form:
  \[
  \phi=w_\phi f+\alpha(\phi_0-\psi_0),
  \quad
  \psi=w_\psi f-\alpha(\phi_0-\psi_0).
  \]
- [ ] `phi0`, `psi0`, and `f` are all in the same common `(B, n, n)` layout before projection.
- [ ] `psi` is transposed back before writing into `projected[:, 1]`.
- [ ] `m_phi` depends on interior y only.
- [ ] `m_psi` depends on interior x only.
- [ ] The output shape equals `flux_int.shape`.
- [ ] The output dtype/device match `flux_int`.
- [ ] `phi + psi == f` holds on the interior common grid in smooth-mask mode.
- [ ] Existing tests still pass.
- [ ] No GreenNet, trainer loss, or branch/trunk architecture changes were introduced.

---

## Short Rationale for Future Readers

The symmetric projection removes the raw sum mode and keeps the raw difference mode:

\[
\phi_{\rm proj}
=
\frac12 f+\frac12(\phi_0-\psi_0),
\qquad
\psi_{\rm proj}
=
\frac12 f-\frac12(\phi_0-\psi_0).
\]

The smooth-mask projection is a spatially weighted generalization:

\[
\phi_{\rm proj}
=
w_\phi f+\alpha(\phi_0-\psi_0),
\qquad
\psi_{\rm proj}
=
w_\psi f-\alpha(\phi_0-\psi_0),
\]

where

\[
w_\phi+w_\psi=1.
\]

Therefore, it preserves exact interior balance while reducing the raw difference-mode contribution near the transverse boundaries through the smooth masks.
