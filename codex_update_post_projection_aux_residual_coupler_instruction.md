# Codex Update Instruction: Move the Five-Stencil Coupler After Projection and Use Pre-Projection Residual as an Auxiliary Feature

## Scope

You are modifying the existing implementation on the `Coupler` branch of `jjhong0608/GreenLearning`.

Assume the local repository is already checked out on the intended working branch. **Do not run, suggest, or perform any git operations**, including:

```text
git checkout
git pull
git merge
git rebase
git branch
git switch
```

This is **not** a request to reimplement the coupler from scratch. Update the current `Coupler` branch implementation.

---

## Current Implementation Summary

The current implementation already contains:

- `CouplerConfig`
- nested `CouplingModelConfig.coupler`
- `FiveStencilStencilMLPCoupler`
- explicit five-stencil gather
- pointwise StencilMLP
- scalar null-space correction
- zero-initialized final coupler layer
- tests for shape, identity, null-space preservation, gather behavior, and enabled/disabled compatibility

The current `CouplingNet.forward()` applies the coupler **before** the balance projection:

```python
raw_int = flux_tilde * norm_exp

if self.coupler is not None:
    raw_int = self.coupler(
        raw_int=raw_int,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
    )

projected_int = self._apply_balance_projection(raw_int, rhs_raw)
```

The current `FiveStencilStencilMLPCoupler` already uses projection-aware point features:

```python
diff_field = 0.5 * (phi0 - psi0)
balance_residual = f - (phi0 + psi0)

diff_hat = diff_field / scale
res_hat = balance_residual / scale
f_hat = f / scale

q = torch.stack(
    [diff_hat, res_hat, f_hat, ax, ay, bx, by, cx, cy],
    dim=1,
)
```

The current coupler update is:

```python
phi1 = phi0 + delta
psi1 = psi0 - delta
```

Keep this null-space update.

---

## Target Implementation

Change the structure from:

\[
\text{raw 1D CouplingNet output}
\rightarrow
\text{FiveStencilStencilMLPCoupler}
\rightarrow
\text{balance projection}
\]

to:

\[
\text{raw 1D CouplingNet output}
\rightarrow
\text{balance projection}
\rightarrow
\text{FiveStencilStencilMLPCoupler}
\]

However, the coupler should still receive the **pre-projection balance residual** as an auxiliary diagnostic feature.

In mathematical terms:

\[
r^0 = f - (\phi^0 + \psi^0)
\]

\[
(\phi^p, \psi^p) = \mathrm{Projection}(\phi^0, \psi^0)
\]

\[
d^p = \frac{1}{2}(\phi^p - \psi^p)
\]

The coupler point feature should be:

\[
q_{ij}
=
[
\widehat{d^p}_{ij},
\widehat{r^0}_{ij},
\hat f_{ij},
a^x_{ij},
a^y_{ij},
b^x_{ij},
b^y_{ij},
c^x_{ij},
c^y_{ij}
]
\]

where:

\[
\widehat{d^p}_{ij} = d^p_{ij}/s,
\qquad
\widehat{r^0}_{ij} = r^0_{ij}/s,
\qquad
\hat f_{ij} = f_{ij}/s
\]

and:

\[
s = \sqrt{\operatorname{mean}_{ij}(f_{ij}^2) + \epsilon}.
\]

The coupler still predicts one scalar correction:

\[
\delta_{ij}
\]

and applies:

\[
\phi^c_{ij} = \phi^p_{ij} + \delta_{ij}
\]

\[
\psi^c_{ij} = \psi^p_{ij} - \delta_{ij}.
\]

Therefore:

\[
\phi^c + \psi^c
=
\phi^p + \psi^p
=
f.
\]

So the post-projection coupler must preserve the balance condition.

---

## Main Design Intent

The fixed balance projection controls the **sum coordinate**:

\[
\phi + \psi.
\]

The five-stencil coupler should control only the **difference coordinate**:

\[
d = \frac{1}{2}(\phi - \psi).
\]

By applying the coupler after projection, the model first enforces:

\[
\phi + \psi = f
\]

and then performs a local five-stencil redistribution of the balanced split. The pre-projection residual \(r^0\) is passed only as an auxiliary diagnostic feature, not as a quantity that the coupler directly corrects.

---

## Files to Inspect and Likely Modify

Primary files:

```text
src/greenonet/coupling_model.py
test/test_coupling.py
```

Documentation files, if they mention the previous pre-projection coupler order:

```text
README.md
CouplingNet.md
coupling_net.md
model_structure.md
```

Do not change the nested config schema unless strictly necessary.

---

## Required Code Changes

### 1. Add a helper for balance residual

Add a helper to `CouplingNet` that computes the balance residual using the same canonical orientation convention as `_apply_balance_projection()`.

Suggested implementation:

```python
def _compute_balance_residual(
    self,
    flux_int: torch.Tensor,
    rhs_raw: torch.Tensor,
) -> torch.Tensor:
    """Compute canonical pre-projection balance residual f - (phi + psi).

    Args:
        flux_int: (B, 2, N, N), interior axial flux-divergence.
        rhs_raw:  (B, 2, N, N+2), raw source.

    Returns:
        residual: (B, N, N), canonical orientation.
    """
    rhs_x_int = rhs_raw[:, 0, :, 1:-1]
    phi = flux_int[:, 0]
    psi_t = flux_int[:, 1].transpose(-1, -2)
    return rhs_x_int - (phi + psi_t)
```

This must use the same orientation as `_apply_balance_projection()`.

---

### 2. Change `CouplingNet.forward()` insertion order

Replace the current order:

```python
raw_int = flux_tilde * norm_exp

if self.coupler is not None:
    raw_int = self.coupler(
        raw_int=raw_int,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
    )

projected_int = self._apply_balance_projection(raw_int, rhs_raw)
```

with:

```python
raw_int = flux_tilde * norm_exp

pre_projection_residual = self._compute_balance_residual(
    flux_int=raw_int,
    rhs_raw=rhs_raw,
)

projected_int = self._apply_balance_projection(raw_int, rhs_raw)

if self.coupler is not None:
    projected_int = self.coupler(
        raw_int=projected_int,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        auxiliary_residual=pre_projection_residual,
    )
```

After this, the existing zero-padding logic should remain unchanged:

```python
projected_flux = torch.zeros(...)
projected_flux[:, :, :, 1:-1] = projected_int
return projected_flux
```

The name `projected_int` may still be used even after the coupler, but understand that after the coupler it is the **post-projection, post-coupler interior flux**.

---

### 3. Extend `FiveStencilStencilMLPCoupler.forward()`

Update the coupler signature from:

```python
def forward(
    self,
    raw_int: torch.Tensor,
    a_vals: torch.Tensor,
    b_vals: torch.Tensor,
    c_vals: torch.Tensor,
    rhs_raw: torch.Tensor,
) -> torch.Tensor:
```

to:

```python
def forward(
    self,
    raw_int: torch.Tensor,
    a_vals: torch.Tensor,
    b_vals: torch.Tensor,
    c_vals: torch.Tensor,
    rhs_raw: torch.Tensor,
    auxiliary_residual: torch.Tensor | None = None,
) -> torch.Tensor:
```

The `raw_int` argument now usually means the **projected interior flux** when called from `CouplingNet.forward()`.

Keep backward compatibility by allowing `auxiliary_residual=None`.

---

### 4. Extend `_build_canonical_point_features()`

Update the helper signature from:

```python
def _build_canonical_point_features(
    self,
    raw_int: torch.Tensor,
    a_vals: torch.Tensor,
    b_vals: torch.Tensor,
    c_vals: torch.Tensor,
    rhs_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
```

to:

```python
def _build_canonical_point_features(
    self,
    raw_int: torch.Tensor,
    a_vals: torch.Tensor,
    b_vals: torch.Tensor,
    c_vals: torch.Tensor,
    rhs_raw: torch.Tensor,
    auxiliary_residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
```

Inside this helper:

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
    balance_residual = auxiliary_residual

diff_hat = diff_field / scale
res_hat = balance_residual / scale
f_hat = f / scale
```

Then build the same 9-channel feature tensor:

```python
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

Important:

- `diff_field` must be computed from the current `raw_int` given to the coupler.
- In the new call path, this is the **post-projection** field.
- `balance_residual` must be the supplied **pre-projection residual** when `auxiliary_residual` is not `None`.
- Do not replace `auxiliary_residual` with the post-projection residual.

---

### 5. Validate `auxiliary_residual` shape

If `auxiliary_residual` is provided, validate its shape.

Suggested logic:

```python
if auxiliary_residual is not None:
    if tuple(auxiliary_residual.shape) != tuple(f.shape):
        raise ValueError(
            "auxiliary_residual must have shape matching canonical source "
            f"{tuple(f.shape)}, got {tuple(auxiliary_residual.shape)}."
        )
    balance_residual = auxiliary_residual
else:
    balance_residual = f - (phi0 + psi0)
```

The expected shape is:

```text
(B, N, N)
```

---

## Full Pseudo Code

### `CouplingNet`

```python
class CouplingNet(nn.Module, ActivationFactoryMixin):
    ...

    def _compute_balance_residual(
        self,
        flux_int: torch.Tensor,
        rhs_raw: torch.Tensor,
    ) -> torch.Tensor:
        rhs_x_int = rhs_raw[:, 0, :, 1:-1]
        phi = flux_int[:, 0]
        psi_t = flux_int[:, 1].transpose(-1, -2)
        return rhs_x_int - (phi + psi_t)

    def forward(...):
        ...
        flux_tilde = combined.reshape(b, axis, n_lines, m_points - 2)

        norm_exp = rhs_norm.unsqueeze(-1)
        raw_int = flux_tilde * norm_exp

        pre_projection_residual = self._compute_balance_residual(
            flux_int=raw_int,
            rhs_raw=rhs_raw,
        )

        projected_int = self._apply_balance_projection(raw_int, rhs_raw)

        if self.coupler is not None:
            projected_int = self.coupler(
                raw_int=projected_int,
                a_vals=a_vals,
                b_vals=b_vals,
                c_vals=c_vals,
                rhs_raw=rhs_raw,
                auxiliary_residual=pre_projection_residual,
            )

        projected_flux = torch.zeros(
            b,
            axis,
            n_lines,
            m_points,
            dtype=projected_int.dtype,
            device=projected_int.device,
        )
        projected_flux[:, :, :, 1:-1] = projected_int
        return projected_flux
```

Use `projected_int.dtype` and `projected_int.device` after the coupler to avoid accidental dtype/device mismatches.

---

### `FiveStencilStencilMLPCoupler`

```python
class FiveStencilStencilMLPCoupler(nn.Module, ActivationFactoryMixin):
    point_features: int = 9

    def _build_canonical_point_features(
        self,
        raw_int: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        auxiliary_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(raw_int, a_vals, b_vals, c_vals, rhs_raw)

        phi0, psi0 = self._canonicalize_flux(raw_int)
        f = rhs_raw[:, 0, :, 1:-1]

        ax = a_vals[:, 0, :, 1:-1]
        ay = a_vals[:, 1, :, 1:-1].transpose(-1, -2)

        bx = b_vals[:, 0, :, 1:-1]
        by = b_vals[:, 1, :, 1:-1].transpose(-1, -2)

        cx = c_vals[:, 0, :, 1:-1]
        cy = c_vals[:, 1, :, 1:-1].transpose(-1, -2)

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

        return q, scale, phi0, psi0

    def forward(
        self,
        raw_int: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        auxiliary_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, scale, phi0, psi0 = self._build_canonical_point_features(
            raw_int=raw_int,
            a_vals=a_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            rhs_raw=rhs_raw,
            auxiliary_residual=auxiliary_residual,
        )

        q5 = self._gather_5_stencil(q)
        delta_hat = cast(torch.Tensor, self.local_mlp(q5)).squeeze(1)
        delta = self.residual_scale * scale * delta_hat

        phi1 = phi0 + delta
        psi1 = psi0 - delta

        return self._decanonicalize_flux(phi1, psi1, raw_int)
```

---

## Tests to Add or Update

Keep all existing tests that are still conceptually valid. Do not remove compatibility tests.

### 1. Existing tests that should still pass

The following categories should remain valid:

```text
shape preservation
initial identity
null-space preservation
five-stencil cardinal-neighbor gather
disabled-coupler compatibility
enabled-coupler initially identity behavior
config parsing / round-trip tests
```

If a test fails because it assumes the old `raw -> coupler -> projection` order, update the test to the new `raw -> projection -> coupler` order.

---

### 2. Add a direct auxiliary-residual feature test

Add or update a test verifying that `_build_canonical_point_features()` uses:

```text
channel 0: diff_hat from the projected field
channel 1: res_hat from the supplied pre-projection auxiliary residual
channel 2: f_hat
```

Suggested test structure:

```python
def test_five_stencil_coupler_uses_projected_diff_and_auxiliary_residual():
    torch.set_default_dtype(torch.float64)

    bsz = 1
    n = 3
    m = n + 2

    phi_projected = torch.tensor(
        [[[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0]]],
        dtype=torch.float64,
    )

    psi_projected = torch.tensor(
        [[[0.5, 1.5, 2.5],
          [3.5, 4.5, 5.5],
          [6.5, 7.5, 8.5]]],
        dtype=torch.float64,
    )

    f = phi_projected + psi_projected

    # This residual intentionally differs from f - (phi_projected + psi_projected),
    # which is zero. It represents the pre-projection residual.
    auxiliary_residual = torch.tensor(
        [[[0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9]]],
        dtype=torch.float64,
    )

    raw_int = torch.zeros((bsz, 2, n, n), dtype=torch.float64)
    raw_int[:, 0] = phi_projected
    raw_int[:, 1] = psi_projected.transpose(-1, -2)

    a_vals = torch.zeros((bsz, 2, n, m), dtype=torch.float64)
    b_vals = torch.zeros((bsz, 2, n, m), dtype=torch.float64)
    c_vals = torch.zeros((bsz, 2, n, m), dtype=torch.float64)
    rhs_raw = torch.zeros((bsz, 2, n, m), dtype=torch.float64)

    rhs_raw[:, 0, :, 1:-1] = f

    ax = torch.full((bsz, n, n), 1.0, dtype=torch.float64)
    ay = torch.full((bsz, n, n), 2.0, dtype=torch.float64)
    bx = torch.full((bsz, n, n), 3.0, dtype=torch.float64)
    by = torch.full((bsz, n, n), 4.0, dtype=torch.float64)
    cx = torch.full((bsz, n, n), 5.0, dtype=torch.float64)
    cy = torch.full((bsz, n, n), 6.0, dtype=torch.float64)

    a_vals[:, 0, :, 1:-1] = ax
    a_vals[:, 1, :, 1:-1] = ay.transpose(-1, -2)

    b_vals[:, 0, :, 1:-1] = bx
    b_vals[:, 1, :, 1:-1] = by.transpose(-1, -2)

    c_vals[:, 0, :, 1:-1] = cx
    c_vals[:, 1, :, 1:-1] = cy.transpose(-1, -2)

    coupler = FiveStencilStencilMLPCoupler(
        CouplerConfig(enabled=True, padding="zero")
    ).to(dtype=torch.float64)

    q, scale, phi0, psi0 = coupler._build_canonical_point_features(
        raw_int=raw_int,
        a_vals=a_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        rhs_raw=rhs_raw,
        auxiliary_residual=auxiliary_residual,
    )

    expected_scale = torch.sqrt(
        torch.mean(f * f, dim=(-1, -2), keepdim=True) + coupler.eps
    )
    expected_diff_hat = 0.5 * (phi_projected - psi_projected) / expected_scale
    expected_res_hat = auxiliary_residual / expected_scale
    expected_f_hat = f / expected_scale

    assert q.shape == (bsz, 9, n, n)

    torch.testing.assert_close(scale, expected_scale)
    torch.testing.assert_close(phi0, phi_projected)
    torch.testing.assert_close(psi0, psi_projected)

    torch.testing.assert_close(q[:, 0], expected_diff_hat)
    torch.testing.assert_close(q[:, 1], expected_res_hat)
    torch.testing.assert_close(q[:, 2], expected_f_hat)

    torch.testing.assert_close(q[:, 3], ax)
    torch.testing.assert_close(q[:, 4], ay)
    torch.testing.assert_close(q[:, 5], bx)
    torch.testing.assert_close(q[:, 6], by)
    torch.testing.assert_close(q[:, 7], cx)
    torch.testing.assert_close(q[:, 8], cy)
```

This test is important because it distinguishes:

```text
post-projection residual = f - (phi_projected + psi_projected) = 0
```

from:

```text
pre-projection auxiliary residual = r^0
```

The test must verify that channel 1 uses the supplied auxiliary residual, not the post-projection residual.

---

### 3. Add an auxiliary-residual shape validation test

Add a test that passes an incorrectly shaped `auxiliary_residual` and expects `ValueError`.

Example:

```python
def test_five_stencil_coupler_rejects_bad_auxiliary_residual_shape():
    ...
    bad_auxiliary_residual = torch.zeros((bsz, n, n + 1), dtype=torch.float64)

    with pytest.raises(ValueError, match="auxiliary_residual"):
        coupler._build_canonical_point_features(
            raw_int=raw_int,
            a_vals=a_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            rhs_raw=rhs_raw,
            auxiliary_residual=bad_auxiliary_residual,
        )
```

---

### 4. Add a post-coupler balance preservation test

Verify that, after `CouplingNet.forward()` with `coupler.enabled=True`, the final interior flux still satisfies:

\[
\phi + \psi = f
\]

up to numerical precision.

Test idea:

1. Instantiate a small `CouplingNet` with coupler enabled.
2. Force the coupler final layer or `local_mlp` to produce nonzero `delta`.
3. Run forward.
4. Check:

```python
phi = out[:, 0, :, 1:-1]
psi = out[:, 1, :, 1:-1].transpose(-1, -2)
f = rhs_raw[:, 0, :, 1:-1]

torch.testing.assert_close(phi + psi, f, rtol=..., atol=...)
```

The existing final-layer zero initialization means a normal forward pass may not prove nonzero coupler behavior. Therefore, manually perturb the final layer weights or replace `local_mlp` with a simple deterministic module if needed.

---

### 5. Update disabled/enabled compatibility tests

For `coupler.enabled=false`, the output should be exactly the projection output.

For `coupler.enabled=true` at initialization, the output should also match the projection output because the final coupler layer is zero-initialized.

These tests should be interpreted under the new order:

```text
raw -> projection -> initially-identity coupler
```

not the old order:

```text
raw -> initially-identity coupler -> projection
```

In both cases the initial output may be numerically identical, but the test description should match the new semantics.

---

### 6. Add or update insertion-order test if practical

If practical, add a test that verifies the coupler receives a projected input. For example, monkeypatch the coupler with a module that asserts:

```python
phi = raw_int[:, 0]
psi = raw_int[:, 1].transpose(-1, -2)
f = rhs_raw[:, 0, :, 1:-1]
torch.testing.assert_close(phi + psi, f)
```

inside its `forward()`.

The mock coupler signature should include `auxiliary_residual`.

---

## Documentation Updates

Update documentation that describes the old pre-projection coupler order.

Preferred wording:

```text
The optional five-stencil StencilMLP coupler is applied after the fixed balance
projection. The projection first enforces phi + psi = f. The coupler then performs
a null-space redistribution of the balanced split by predicting a scalar delta and
applying (phi, psi) -> (phi + delta, psi - delta). Therefore phi + psi = f remains
preserved after the coupler.

The coupler uses projection-aware local features: the post-projection difference
diff = 0.5 * (phi - psi), the pre-projection balance residual r0 = f - (phi_raw + psi_raw)
as an auxiliary diagnostic feature, the normalized source f, and canonical coefficient
channels ax, ay, bx, by, cx, cy.
```

Do not describe the coupler as being inserted before projection.

---

## Config Requirements

Do not change the nested config schema.

The existing structure should remain valid:

```json
"coupling_model": {
  "coupler": {
    "enabled": true,
    "type": "five_stencil_stencil_mlp",
    "hidden_channels": 64,
    "depth": 2,
    "activation": "gelu",
    "dropout": 0.0,
    "residual_scale_init": 0.05,
    "padding": "replicate",
    "eps": 1e-12
  }
}
```

No new config field is required for this update.

---

## Validation Commands

Run at least:

```bash
PYTHONPATH=src pytest test/test_coupling.py -q
```

If feasible, also run:

```bash
PYTHONPATH=src pytest test/test_io_config.py -q
PYTHONPATH=src pytest test -q
ruff check src test
mypy src
```

If `ruff` or `mypy` fails due to pre-existing unrelated issues, report that explicitly with the relevant failure summary.

---

## Acceptance Criteria

The update is correct if all of the following are true:

1. The coupler is applied after `_apply_balance_projection()`.
2. The coupler receives `raw_int=projected_int`.
3. The coupler receives `auxiliary_residual=pre_projection_residual`.
4. `pre_projection_residual` is computed from the raw pre-projection interior flux.
5. `pre_projection_residual` uses the same canonical orientation as `_apply_balance_projection()`.
6. The coupler point features are:
   - channel 0: `diff_hat` from post-projection `raw_int`
   - channel 1: `res_hat` from pre-projection auxiliary residual
   - channel 2: `f_hat`
   - channels 3-8: `ax, ay, bx, by, cx, cy`
7. `point_features = 9` and `in_channels = 45` remain unchanged.
8. The coupler still predicts one scalar `delta`.
9. The coupler still applies:
   - `phi1 = phi0 + delta`
   - `psi1 = psi0 - delta`
10. The final output still satisfies \(\phi+\psi=f\) after the coupler.
11. `coupler.enabled=false` behavior remains projection-only.
12. `coupler.enabled=true` with zero-initialized final layer remains initially identical to projection-only output.
13. The nested config schema is unchanged.
14. Tests verify the auxiliary residual feature behavior.

---

## Things Not to Do

Do not:

- run or suggest git checkout/pull/merge/rebase/branch operations
- reimplement `FiveStencilStencilMLPCoupler` from scratch
- rename `FiveStencilStencilMLPCoupler`
- change the nested `coupling_model.coupler` config schema
- change `point_features` from 9
- remove the auxiliary residual channel
- replace the five-stencil gather with a plain CNN
- include diagonal or corner stencil points
- make the coupler output independent `delta_phi` and `delta_psi`
- change the fixed balance projection formula
- remove final-layer zero initialization
- remove existing compatibility tests

---

# Codex Self-Review Checklist

Before finishing, review your patch against this checklist.

## Implementation Order

- [ ] Is `pre_projection_residual` computed before `_apply_balance_projection()`?
- [ ] Is `_apply_balance_projection()` called before `self.coupler(...)`?
- [ ] Does `self.coupler(...)` receive `raw_int=projected_int`?
- [ ] Does `self.coupler(...)` receive `auxiliary_residual=pre_projection_residual`?

## Residual Semantics

- [ ] Is `pre_projection_residual = f - (phi_raw + psi_raw)`?
- [ ] Is `psi_raw` transposed into canonical orientation before computing the residual?
- [ ] Does the coupler use `auxiliary_residual` when provided?
- [ ] Does the coupler avoid replacing the auxiliary residual with the post-projection residual?

## Feature Construction

- [ ] Is `diff_field = 0.5 * (phi_projected - psi_projected)`?
- [ ] Is `res_hat = auxiliary_residual / scale`?
- [ ] Is `f_hat = f / scale`?
- [ ] Are the feature channels ordered as `[diff_hat, res_hat, f_hat, ax, ay, bx, by, cx, cy]`?
- [ ] Are `point_features = 9` and `in_channels = 45` unchanged?

## Null-Space Correction

- [ ] Does the coupler still predict a scalar `delta`?
- [ ] Does it still apply `phi1 = phi0 + delta` and `psi1 = psi0 - delta`?
- [ ] Does the final post-coupler field preserve `phi + psi = f`?

## Tests

- [ ] Do existing coupler tests still pass?
- [ ] Is there a test proving channel 1 uses the supplied pre-projection auxiliary residual?
- [ ] Is there a test rejecting badly shaped auxiliary residuals?
- [ ] Is there a test or mock verifying the coupler receives projected input?
- [ ] Is there a test verifying post-coupler balance preservation?

## Documentation

- [ ] Did I update documentation from "pre-projection coupler" to "post-projection coupler"?
- [ ] Did I explain that the pre-projection residual is passed as an auxiliary feature?

## Final Expected Summary

When reporting completion, summarize the change like this:

```text
Updated the existing five-stencil StencilMLP coupler to run after the fixed balance
projection. The model now computes the pre-projection residual r0 = f - (phi_raw + psi_raw),
applies the balance projection, and then calls the coupler on the projected balanced split
while passing r0 as an auxiliary feature. The coupler still predicts a scalar null-space
correction delta and applies (phi, psi) -> (phi + delta, psi - delta), so phi + psi = f is
preserved after the coupler. The nested config schema and coupler hyperparameters remain
unchanged.
```
