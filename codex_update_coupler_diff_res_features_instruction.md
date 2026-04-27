# Codex Update Instruction: Replace Independent `phi`/`psi` Coupler Inputs with Null-Space `diff`/`res` Features

## Scope

You are modifying the existing implementation on the `Coupler` branch of `jjhong0608/GreenLearning`.

Assume the repository is already checked out on the intended working branch. **Do not run or suggest any git branch operations** such as `checkout`, `pull`, `merge`, `rebase`, or branch creation.

This is an update to the existing five-stencil coupler implementation. Do **not** reimplement the coupler from scratch, and do **not** replace it with a plain CNN, global MLP, or 9-stencil convolution.

---

## Current Situation

The current `Coupler` branch already contains:

- `CouplerConfig`
- `CouplingModelConfig.coupler`
- `FiveStencilStencilMLPCoupler`
- nested JSON config under `coupling_model.coupler`
- tests for:
  - shape preservation
  - initial identity
  - null-space preservation
  - five-stencil cardinal-neighbor gather
  - disabled-coupler compatibility
  - initially-identity enabled-coupler behavior

The current `FiveStencilStencilMLPCoupler.forward()` builds local point features using independent canonical flux channels:

```python
phi_hat = phi0 / scale
psi_hat = psi0 / scale
f_hat = f / scale

q = torch.stack(
    [phi_hat, psi_hat, f_hat, ax, ay, bx, by, cx, cy],
    dim=1,
)
```

This update changes only the **local point feature construction**.

---

## Main Goal

Replace the independent `phi_hat` and `psi_hat` input channels with projection-aware null-space coordinates:

\[
d_{ij} = \frac{1}{2}(\phi_{ij} - \psi_{ij})
\]

\[
r_{ij} = f_{ij} - (\phi_{ij} + \psi_{ij})
\]

The new point feature vector must be:

\[
q_{ij}
=
[
\hat d_{ij},
\hat r_{ij},
\hat f_{ij},
a^x_{ij},
a^y_{ij},
b^x_{ij},
b^y_{ij},
c^x_{ij},
c^y_{ij}
]
\]

where

\[
\hat d_{ij} = d_{ij} / s,
\qquad
\hat r_{ij} = r_{ij} / s,
\qquad
\hat f_{ij} = f_{ij} / s
\]

and

\[
s = \sqrt{\operatorname{mean}_{ij}(f_{ij}^2) + \epsilon}.
\]

The coupler still predicts a scalar correction \(\delta_{ij}\) and applies the same null-space update:

\[
\phi^1_{ij} = \phi^0_{ij} + \delta_{ij}
\]

\[
\psi^1_{ij} = \psi^0_{ij} - \delta_{ij}
\]

Therefore, the coupler must still preserve:

\[
\phi^1 + \psi^1 = \phi^0 + \psi^0
\]

before the existing balance projection.

---

## Reason for the Change

`phi` and `psi` are not two unrelated local states. In canonical orientation, they are two components of the axial decomposition. The balance projection acts on their sum:

\[
\phi + \psi
\]

while the projection-invariant degree of freedom is their difference:

\[
d = \frac{1}{2}(\phi - \psi).
\]

The five-stencil coupler updates only this projection-invariant null-space coordinate. Therefore, feeding `diff` and `balance_residual` is more consistent than feeding `phi` and `psi` independently.

---

## Files to Inspect and Likely Modify

Primary:

```text
src/greenonet/coupling_model.py
test/test_coupling.py
```

Documentation, if it currently describes independent `phi`/`psi` coupler inputs:

```text
README.md
CouplingNet.md
coupling_net.md
model_structure.md
codex_five_stencil_coupler_instruction.md
```

Do not modify config schema unless necessary. The feature count remains 9, so the existing nested config should remain valid.

---

## Required Implementation Details

### 1. Keep the existing class name

Keep:

```python
class FiveStencilStencilMLPCoupler(nn.Module, ActivationFactoryMixin):
```

Do not rename the class.

---

### 2. Keep `point_features = 9`

The point feature count remains 9:

```python
point_features: int = 9
```

The five-stencil gathered input size therefore remains:

```python
in_channels = 5 * self.point_features  # 45
```

Do not change default hidden width, depth, activation, residual scale, or padding semantics.

---

### 3. Preserve canonical orientation logic

Continue to define canonical fields as:

```python
phi0 = raw_int[:, 0]
psi0 = raw_int[:, 1].transpose(-1, -2)
```

Continue to define canonical coefficients as:

```python
f = rhs_raw[:, 0, :, 1:-1]

ax = a_vals[:, 0, :, 1:-1]
ay = a_vals[:, 1, :, 1:-1].transpose(-1, -2)

bx = b_vals[:, 0, :, 1:-1]
by = b_vals[:, 1, :, 1:-1].transpose(-1, -2)

cx = c_vals[:, 0, :, 1:-1]
cy = c_vals[:, 1, :, 1:-1].transpose(-1, -2)
```

---

### 4. Replace independent `phi`/`psi` features

Replace this pattern:

```python
phi_hat = phi0 / scale
psi_hat = psi0 / scale
f_hat = f / scale

q = torch.stack(
    [phi_hat, psi_hat, f_hat, ax, ay, bx, by, cx, cy],
    dim=1,
)
```

with this pattern:

```python
sum_field = phi0 + psi0
diff_field = 0.5 * (phi0 - psi0)
balance_residual = f - sum_field

diff_hat = diff_field / scale
res_hat = balance_residual / scale
f_hat = f / scale

q = torch.stack(
    [diff_hat, res_hat, f_hat, ax, ay, bx, by, cx, cy],
    dim=1,
)
```

Do not feed `phi_hat` and `psi_hat` as independent state channels.

---

### 5. Keep the same scale definition

Use the same per-sample scalar source scale:

```python
scale = torch.sqrt(
    torch.mean(f * f, dim=(-1, -2), keepdim=True) + self.eps
)
```

Shape should be:

```python
scale.shape == (B, 1, 1)
```

---

### 6. Preserve the five-stencil gather

The gather must remain a strict five-point stencil:

```text
center
i+1
i-1
j+1
j-1
```

No corner features may be included.

The existing `_gather_5_stencil()` behavior should remain valid.

---

### 7. Preserve the null-space output update

After predicting `delta_hat`, keep the same correction form:

```python
delta = self.residual_scale * scale * delta_hat

phi1 = phi0 + delta
psi1 = psi0 - delta

return self._decanonicalize_flux(phi1, psi1, raw_int)
```

Do not change this to independent `delta_phi` and `delta_psi`.

---

## Recommended Refactor for Testability

Please factor point-feature construction into a private helper method so tests can verify the new channels directly.

Suggested signature:

```python
def _build_canonical_point_features(
    self,
    raw_int: torch.Tensor,
    a_vals: torch.Tensor,
    b_vals: torch.Tensor,
    c_vals: torch.Tensor,
    rhs_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return:
        q:     (B, 9, N, N)
        scale: (B, 1, 1)
        phi0:  (B, N, N), canonical orientation
        psi0:  (B, N, N), canonical orientation
    """
```

Then `forward()` should call this helper:

```python
q, scale, phi0, psi0 = self._build_canonical_point_features(
    raw_int=raw_int,
    a_vals=a_vals,
    b_vals=b_vals,
    c_vals=c_vals,
    rhs_raw=rhs_raw,
)

q5 = self._gather_5_stencil(q)
delta_hat = cast(torch.Tensor, self.local_mlp(q5)).squeeze(1)
delta = self.residual_scale * scale * delta_hat

phi1 = phi0 + delta
psi1 = psi0 - delta

return self._decanonicalize_flux(phi1, psi1, raw_int)
```

This helper is optional from a public API perspective but strongly preferred because it makes the intended feature order testable without relying on fragile source-code inspection.

---

## Full Pseudo Code

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Canonical flux orientation.
        phi0 = raw_int[:, 0]                        # (B, N, N)
        psi0 = raw_int[:, 1].transpose(-1, -2)      # (B, N, N)

        # Canonical source.
        f = rhs_raw[:, 0, :, 1:-1]                  # (B, N, N)

        # Canonical coefficient fields.
        ax = a_vals[:, 0, :, 1:-1]
        ay = a_vals[:, 1, :, 1:-1].transpose(-1, -2)

        bx = b_vals[:, 0, :, 1:-1]
        by = b_vals[:, 1, :, 1:-1].transpose(-1, -2)

        cx = c_vals[:, 0, :, 1:-1]
        cy = c_vals[:, 1, :, 1:-1].transpose(-1, -2)

        # Per-sample source scale.
        scale = torch.sqrt(
            torch.mean(f * f, dim=(-1, -2), keepdim=True) + self.eps
        )

        # Projection-aware coordinates.
        sum_field = phi0 + psi0
        diff_field = 0.5 * (phi0 - psi0)
        balance_residual = f - sum_field

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
    ) -> torch.Tensor:
        self._validate_inputs(raw_int, a_vals, b_vals, c_vals, rhs_raw)

        q, scale, phi0, psi0 = self._build_canonical_point_features(
            raw_int=raw_int,
            a_vals=a_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            rhs_raw=rhs_raw,
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

### 1. Keep existing tests

The following existing tests should still pass without conceptual changes:

```text
test_five_stencil_coupler_shape_preservation
test_five_stencil_coupler_initial_identity
test_five_stencil_coupler_preserves_phi_plus_psi_for_nonzero_delta
test_five_stencil_coupler_gather_uses_only_cardinal_neighbors
test_coupling_net_with_disabled_coupler_matches_baseline
test_coupling_net_enabled_coupler_is_initially_identity
```

If any of these break, fix the implementation rather than weakening the test.

---

### 2. Add a direct feature-order test

Add a test verifying that the first three feature channels are:

```text
channel 0: diff_hat = 0.5 * (phi - psi) / scale
channel 1: res_hat  = (f - (phi + psi)) / scale
channel 2: f_hat    = f / scale
```

Suggested test:

```python
def test_five_stencil_coupler_point_features_use_diff_res_f_not_phi_psi():
    torch.set_default_dtype(torch.float64)

    bsz = 1
    n = 3
    m = n + 2

    phi = torch.tensor(
        [[[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0]]],
        dtype=torch.float64,
    )

    psi = torch.tensor(
        [[[0.5, 1.5, 2.5],
          [3.5, 4.5, 5.5],
          [6.5, 7.5, 8.5]]],
        dtype=torch.float64,
    )

    f = torch.tensor(
        [[[2.0, 1.0, 3.0],
          [4.0, 2.0, 1.0],
          [1.5, 2.5, 3.5]]],
        dtype=torch.float64,
    )

    raw_int = torch.zeros((bsz, 2, n, n), dtype=torch.float64)
    raw_int[:, 0] = phi
    raw_int[:, 1] = psi.transpose(-1, -2)

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
    )

    expected_scale = torch.sqrt(
        torch.mean(f * f, dim=(-1, -2), keepdim=True) + coupler.eps
    )
    expected_diff_hat = 0.5 * (phi - psi) / expected_scale
    expected_res_hat = (f - (phi + psi)) / expected_scale
    expected_f_hat = f / expected_scale

    assert q.shape == (bsz, 9, n, n)
    torch.testing.assert_close(scale, expected_scale)
    torch.testing.assert_close(phi0, phi)
    torch.testing.assert_close(psi0, psi)

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

If you choose not to expose `_build_canonical_point_features`, use an equivalent test strategy that verifies the same channel construction. Do not use brittle tests based only on source-code string matching.

---

### 3. Optional: add a hook-based behavior test

A direct feature test is enough, but you may also add a behavior test that replaces `local_mlp` with a simple module using only `q5[:, 0]`, i.e. the center `diff_hat` channel, then verifies the output correction. This is optional.

---

## Documentation Updates

Update any documentation that says or implies the coupler uses independent `phi` and `psi` local state channels.

Preferred wording:

```text
The optional five-stencil coupler uses projection-aware local state features:
diff = 0.5 * (phi - psi), balance residual res = f - (phi + psi), normalized source f,
and canonical coefficient channels ax, ay, bx, by, cx, cy. It does not feed phi and psi
as independent state channels. The coupler still predicts one scalar delta and applies
(phi, psi) -> (phi + delta, psi - delta), so phi + psi is unchanged before the fixed
balance projection.
```

The `README.md` description should mention this update if it currently only says the coupler gathers generic features.

---

## Config Requirements

Do not change the nested config schema unless strictly necessary.

The existing config should remain valid:

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

No additional config field is required for this feature-coordinate change.

---

## Validation Commands

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

If `mypy` or `ruff` fails because of pre-existing unrelated issues, report that explicitly and include the relevant failure summary.

---

## Acceptance Criteria

The implementation is correct if all of the following hold:

1. `FiveStencilStencilMLPCoupler` remains the coupler class.
2. The network architecture remains explicit five-stencil gather plus shared pointwise MLP.
3. `point_features` remains 9 and `in_channels` remains 45.
4. The first three point-feature channels are:
   - `diff_hat`
   - `res_hat`
   - `f_hat`
5. Independent `phi_hat` and `psi_hat` are not used as local state input channels.
6. The scalar correction still applies:
   - `phi1 = phi0 + delta`
   - `psi1 = psi0 - delta`
7. `phi + psi` remains exactly preserved by the coupler before balance projection.
8. `coupler.enabled=false` behavior remains unchanged.
9. The existing nested config remains valid.
10. Tests document and verify the new feature construction.

---

## Things Not to Do

Do not:

- perform git checkout, pull, merge, rebase, or branch creation
- replace the coupler with a plain 2D CNN
- include diagonal stencil points
- change the coupler to output two independent corrections
- change the balance projection
- change the `CouplingNet.forward()` insertion point
- change the default coupler hyperparameters unless required by tests
- remove existing compatibility tests
- remove zero initialization of the final coupler layer

---

# Codex Self-Review Checklist

Before finishing, review your patch as if you only had this instruction.

## Implementation Clarity

- [ ] Did I update the existing `FiveStencilStencilMLPCoupler` rather than adding a second competing coupler?
- [ ] Did I avoid git branch operations?
- [ ] Did I keep the nested `coupling_model.coupler` schema intact?
- [ ] Did I keep `point_features = 9` and `in_channels = 45`?

## Mathematical Correctness

- [ ] Is `psi0` converted to canonical orientation using `.transpose(-1, -2)`?
- [ ] Is `diff_field = 0.5 * (phi0 - psi0)`?
- [ ] Is `balance_residual = f - (phi0 + psi0)`?
- [ ] Are `diff`, `res`, and `f` normalized by the same source scale?
- [ ] Are `phi_hat` and `psi_hat` no longer used as independent local input channels?
- [ ] Does the output update remain `phi + delta`, `psi - delta`?
- [ ] Does the coupler preserve `phi + psi` before projection?

## Testing

- [ ] Do the old coupler tests still pass?
- [ ] Is there a new test that directly verifies the `diff/res/f` feature order?
- [ ] Does the test check canonical orientation for `psi`?
- [ ] Does the test avoid source-code string matching?
- [ ] Did I run `PYTHONPATH=src pytest test/test_coupling.py -q`?

## Documentation

- [ ] Did I update docs that described `phi` and `psi` as independent coupler features?
- [ ] Did I explain that the coupler operates on the projection-invariant difference coordinate?

## Final Expected Summary

When reporting completion, summarize the change as:

```text
Updated the existing five-stencil StencilMLP coupler so its local state channels are
diff = 0.5 * (phi - psi), balance residual res = f - (phi + psi), and normalized f,
rather than independent phi/psi channels. The null-space scalar update and nested config
schema remain unchanged. Added/updated tests verifying the new feature construction and
preserved existing identity/null-space behavior.
```
