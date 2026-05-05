# Codex Instruction: Add Input-Side Direct Source 5-Stencil Lift to CouplingNet

## Scope

You are modifying the existing local `Coupler` branch of `jjhong0608/GreenLearning`.

This task is to add an **input-side learned source stencil lift** to the main CouplingNet.

This is **not** an output-side coupler.  
Do **not** add or modify a post-projection \((\phi,\psi)\mapsto(\phi+\delta,\psi-\delta)\) correction module for this task.

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

Add an optional input-side learned 5-stencil source lift:

\[
f \longrightarrow g_\theta(f)
\]

and use the lifted scalar source field \(g\) as the **source branch input** instead of the original normalized source line.

The source lift should be:

- source-only,
- learned,
- 5-stencil based,
- scalar-output per interior grid point,
- direct replacement of the source branch input,
- interior-only,
- independent of the physical RHS used in projection and loss.

The physical PDE source \(f\) must remain unchanged everywhere except the source branch input.

---

## Mathematical Definition

Let the physical source be available on a canonical full grid:

\[
f^{full}_{ij},\qquad i,j=0,\dots,N+1.
\]

The interior grid is:

\[
i,j=1,\dots,N.
\]

The full line length is:

\[
M=N+2.
\]

The existing config field `branch_input_dim` represents this boundary-including length \(M\).

---

## Encoder Normalization

Compute the encoder source normalization directly from `rhs_raw`, not from `rhs_tilde`.

Use sample-wise interior RMS:

\[
s_{\mathrm{enc}}
=
\sqrt{
\operatorname{mean}_{i,j=1,\dots,N}
\left[
(f^{full}_{ij})^2
\right]
+
\varepsilon
}.
\]

Then normalize the entire full grid using this interior scale:

\[
\tilde f^{full}_{ij}
=
f^{full}_{ij}/s_{\mathrm{enc}}.
\]

Important:

- The scale is computed only from interior values.
- Boundary values are normalized using the same interior scale.
- Do not mean-center the source.
- Do not use existing `rhs_tilde` as the encoder input.

---

## Interior 5-Stencil

For each interior point \((i,j)\), gather a cross-shaped 5-stencil:

\[
S_{ij}
=
[
\tilde f_{ij},
\tilde f_{i+1,j},
\tilde f_{i-1,j},
\tilde f_{i,j+1},
\tilde f_{i,j-1}
].
\]

Use order:

```text
center, east, west, north, south
```

or another order only if it is documented and consistently tested.

Boundary source values must be used for adjacent interior stencils.

Example:

\[
S_{1,j}
=
[
\tilde f_{1,j},
\tilde f_{2,j},
\tilde f_{0,j},
\tilde f_{1,j+1},
\tilde f_{1,j-1}
].
\]

Do not use artificial padding for interior stencils.  
Use the actual boundary source data already present in the line-aligned source representation.

Corners may be filled with zero because the interior 5-stencil never uses corner values.

---

## Learned Scalar Source Lift

Define a learned pointwise encoder:

\[
E_\theta:\mathbb{R}^{5}\rightarrow\mathbb{R}.
\]

For each interior point:

\[
g^{raw}_{ij}=E_\theta(S_{ij}),
\qquad i,j=1,\dots,N.
\]

The lifted source \(g\) is defined **only on interior points**.

Do not create boundary values of \(g\).  
Do not use boundary identity.  
Do not create boundary \(g\) by padding.  
Do not pass boundary \(g\) to the source branch.

---

## \(g\)-Normalization

Normalize \(g^{raw}\) using sample-wise interior RMS:

\[
s_g
=
\sqrt{
\operatorname{mean}_{i,j=1,\dots,N}
\left[
(g^{raw}_{ij})^2
\right]
+
\varepsilon
}.
\]

Then

\[
\tilde g_{ij}
=
g^{raw}_{ij}/s_g.
\]

Important:

- Use interior RMS.
- Do not mean-center \(g\).
- The normalized \(\tilde g\) should have interior RMS close to 1.
- If `use_g_normalization=false`, skip this step, but the default should use normalization.

---

## Direct Replacement

The lifted source \(\tilde g\) directly replaces the source branch input.

Existing source branch path conceptually:

\[
B_f(\tilde f(\cdot,y_j)).
\]

New enabled source lift path:

\[
B_g(\tilde g(\cdot,y_j)).
\]

Do not concatenate \(\tilde f\) and \(\tilde g\):

\[
Q_f \neq [\tilde f,\tilde g].
\]

Do not use residual form:

\[
\tilde g \neq \tilde f+\alpha E_\theta(\cdot).
\]

The intended design is direct replacement:

\[
Q_f=\tilde g.
\]

---

## Physical RHS Must Remain Unchanged

The learned source lift affects only the source branch input.

Do not use \(g\) for:

- projection RHS,
- balance projection,
- balance residual,
- output denormalization,
- loss target,
- reference solution,
- Green representation source.

The physical RHS remains the original source \(f\).

The balance projection must still enforce:

\[
\phi+\psi=f.
\]

Output denormalization must continue to use the existing `rhs_norm`.

---

# Required Model Changes

## Add a Source Stencil Lift Module

Add a small learned module for source lifting. Recommended name:

```python
FiveStencilSourceLift
```

or another name consistent with the local code style.

Recommended architecture:

\[
5 \rightarrow 32 \rightarrow 32 \rightarrow 1.
\]

Default settings:

```text
hidden_dim = 32
depth = 2
activation = "gelu"
use_bias = true
dropout = 0.0
use_g_normalization = true
eps = 1e-12
```

Do not zero-initialize the final layer.  
Use standard/default initialization.  
Direct replacement with zero initialization would produce nearly zero lifted source and can destabilize training.

The module should:

1. build a canonical full source grid from `rhs_raw`,
2. compute encoder normalization from interior source values,
3. gather interior 5-stencils,
4. apply the learned encoder pointwise,
5. normalize \(g\) over interior points,
6. return lifted source in the same axis layout expected by source branch input.

---

## Add Config Block

Add this under `coupling_model`, not under training config:

```json
"source_stencil_lift": {
  "enabled": false,
  "hidden_dim": 32,
  "depth": 2,
  "activation": "gelu",
  "use_bias": true,
  "dropout": 0.0,
  "use_g_normalization": true,
  "eps": 1e-12
}
```

Do not add normalization mode fields such as:

```text
encoder_norm
g_norm
```

The normalization policy is fixed in code:

```text
encoder normalization: sample interior RMS
g normalization: sample interior RMS
```

Do not add config fields for residual mode, concat mode, or output channels.

The output dimension is fixed to 1.

---

## Source Branch Input Dimension

Keep existing config:

```json
"branch_input_dim": M
```

where \(M=N+2\), the boundary-including line length.

Do not add a separate config field for `branch_rhs_input_dim`.

Instead, compute internally during model initialization:

```python
branch_a_input_dim = config.branch_input_dim

if config.source_stencil_lift.enabled:
    branch_rhs_input_dim = config.branch_input_dim - 2
else:
    branch_rhs_input_dim = config.branch_input_dim
```

Add a defensive check:

```python
if config.source_stencil_lift.enabled and config.branch_input_dim <= 2:
    raise ValueError(
        "source_stencil_lift requires branch_input_dim > 2 because "
        "the lifted source branch uses the interior length branch_input_dim - 2."
    )
```

---

## Coefficient Branch Must Keep Boundary Points

The coefficient branch must remain unchanged.

\[
B_a:\mathbb{R}^{M}\rightarrow\mathbb{R}^{H}.
\]

The source branch changes only when source lift is enabled:

\[
B_g:\mathbb{R}^{M-2}\rightarrow\mathbb{R}^{H}.
\]

The MIONet-style product should remain:

\[
B_a(a)\odot B_g(g)\odot T(x,y).
\]

When source lift is disabled, keep the existing source branch path:

\[
B_f(\tilde f).
\]

Enabled and disabled checkpoints do not need to be compatible.

---

## Expected Lifted Source Shape

If `rhs_raw` and source data use existing axis layout:

```text
rhs_raw: (B, 2, n_lines, M)
```

then the lifted source branch input should have shape:

```text
lifted_rhs: (B, 2, n_lines, M - 2)
```

where:

```text
n_lines = M - 2 = N
```

Axis 0 should correspond to x-lines.  
Axis 1 should correspond to y-lines.  
The layout must match the existing `rhs_tilde` axis convention except that the last dimension is interior-only.

For x-lines:

\[
\tilde g^x_j
=
[\tilde g_{1,j},\dots,\tilde g_{N,j}].
\]

For y-lines:

\[
\tilde g^y_i
=
[\tilde g_{i,1},\dots,\tilde g_{i,N}].
\]

Be careful about transposes. The lifted source must align with the existing `rhs_tilde` convention.

---

# Canonical Full Source Grid Reconstruction

The source lift module must reconstruct a canonical full source grid from `rhs_raw`.

Conceptually:

\[
f^{full}\in\mathbb{R}^{B\times(N+2)\times(N+2)}.
\]

Use the local code’s actual axis convention, but satisfy these requirements:

1. Interior values must match the physical source interior.
2. Vertical boundary values must come from x-line boundary entries.
3. Horizontal boundary values must come from y-line boundary entries.
4. Corners may be set to zero.
5. Corners must not affect any interior 5-stencil output.

If the code already has a helper for canonical source reconstruction, reuse it if correct. Otherwise, implement a small private helper in the source lift module.

Recommended helper name:

```python
_build_canonical_full_source(rhs_raw)
```

or local-style equivalent.

---

# Forward Path Requirements

## Disabled Path

When `source_stencil_lift.enabled=false`:

- use existing `rhs_tilde` as source branch input,
- source branch input dimension is `branch_input_dim`,
- output must remain backward compatible.

Conceptual path:

```python
branch_rhs_in = rhs_tilde
branch_rhs_input_dim = config.branch_input_dim
```

## Enabled Path

When `source_stencil_lift.enabled=true`:

- build lifted source from `rhs_raw`,
- use lifted source as source branch input,
- source branch input dimension is `branch_input_dim - 2`,
- keep coefficient branch input unchanged.

Conceptual path:

```python
lifted_rhs, source_lift_aux = self.source_stencil_lift(rhs_raw=rhs_raw)

branch_rhs_in = lifted_rhs
# shape: (B, 2, n_lines, M - 2)

branch_a_in = a_vals
# shape unchanged: (B, 2, n_lines, M)
```

Do not modify:

```python
raw_int = flux_tilde * rhs_norm.unsqueeze(-1)
```

or the balance projection formula.

---

# Source Lift Diagnostics

Log the following diagnostics during both training and validation when source lift is enabled.

The definitions are over the interior grid.

## `source_lift_corr_g_f`

\[
\mathrm{corr}(g,f)
=
\frac{
\langle \tilde g,\tilde f_{\mathrm{int}}\rangle
}{
\|\tilde g\|_2
\|\tilde f_{\mathrm{int}}\|_2
+\varepsilon
}.
\]

## `source_lift_rel_diff_g_f`

\[
\frac{
\|\tilde g-\tilde f_{\mathrm{int}}\|_2
}{
\|\tilde f_{\mathrm{int}}\|_2+\varepsilon
}.
\]

## `source_lift_g_rms`

\[
\sqrt{
\operatorname{mean}_{interior}(\tilde g^2)
}.
\]

If \(g\)-normalization is enabled, `source_lift_g_rms` should be close to 1.

The exact plumbing is left to local code style. You may:

- return diagnostics through model intermediates,
- return `lifted_rhs` and `f_enc_interior` through an aux dictionary,
- or compute metrics in the trainer using source lift helper outputs.

But the metrics must be logged in both train and validation outputs if source lift is enabled.

---

# Do Not Change Training Objective

This task should not change the loss function.

Keep the existing training objective, scheduler, optimizer, Green representation, and projection logic unless a minimal change is required to feed the lifted source into the branch network.

The purpose of this change is to test whether **source representation** improves validation behavior under the same loss.

---

# Tests to Add or Update

Add tests in the most appropriate existing test files, likely:

```text
test/test_coupling.py
test/test_io_config.py
```

Do not remove existing tests.

---

## Test 1: Config Parsing

Verify the new `coupling_model.source_stencil_lift` block parses correctly.

Expected defaults:

```text
enabled = false
hidden_dim = 32
depth = 2
activation = "gelu"
use_bias = true
dropout = 0.0
use_g_normalization = true
eps = 1e-12
```

Do not require `encoder_norm` or `g_norm` fields.

---

## Test 2: Disabled Path Compatibility

When `source_stencil_lift.enabled=false`, the model should use the existing source branch input path.

The forward output shape and behavior should match the previous baseline path.

If an exact numerical equality test is practical, use it. Otherwise, verify that the source lift module is not instantiated or not called, and that branch RHS input dimension remains `branch_input_dim`.

---

## Test 3: Source Branch Input Dimension

With source lift enabled:

```text
branch_a_input_dim = M
branch_rhs_input_dim = M - 2
```

Verify this explicitly if the model exposes the branch MLP input features, or indirectly by checking the first linear layer input dimension.

---

## Test 4: Lifted Source Shape

Given a batch with full line length \(M=N+2\), verify:

```text
lifted_rhs.shape == (B, 2, N, N)
```

where \(N=M-2\).

---

## Test 5: Corner Values Are Not Used

Construct two `rhs_raw` tensors that are identical except for the four canonical corner values.

Set corners to very large different values in one tensor.

Run the source lift module.

Verify the interior lifted source output is unchanged.

This test ensures corners do not affect the interior 5-stencil.

---

## Test 6: Boundary Values Are Used

Construct two `rhs_raw` tensors that differ only in a boundary source value used by an adjacent interior stencil.

Example: change a left-boundary value corresponding to \(f_{0,j}\).

Verify that the adjacent interior lifted value \(g_{1,j}\) changes.

This test ensures true boundary source data is used in adjacent interior 5-stencils.

---

## Test 7: \(g\)-Normalization

With `use_g_normalization=true`, verify:

\[
\sqrt{\operatorname{mean}_{interior}(\tilde g^2)}\approx 1.
\]

Use a reasonable tolerance.

---

## Test 8: Physical RHS Unchanged

Verify that enabling source lift does not replace the physical RHS used by projection.

A practical test is:

1. enable source lift,
2. run the model forward,
3. check that the final projected interior still satisfies the balance condition using original `rhs_raw`:

\[
\phi+\psi=f.
\]

Use the same canonical orientation convention as the existing projection test.

Do not check balance against \(g\).

---

## Test 9: Source Lift Diagnostics

If source lift is enabled, verify the trainer or model metrics include:

```text
source_lift_corr_g_f
source_lift_rel_diff_g_f
source_lift_g_rms
```

At minimum, verify these quantities are computable and finite.

---

# Documentation Updates

Update documentation only where relevant.

Add a concise description:

```text
The optional source stencil lift is an input-side learned preprocessing module.
It maps the physical source f to an interior scalar lifted source g using a 5-point stencil.
The lifted source g directly replaces rhs_tilde only as the source branch input.
The physical RHS used for balance projection, output denormalization, loss, and evaluation remains unchanged.
```

Also mention:

```text
The coefficient branch keeps boundary-including coefficient lines.
The source branch uses only interior lifted source lines when source_stencil_lift is enabled.
```

---

# Validation Commands

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

1. `coupling_model.source_stencil_lift` config exists and defaults to disabled.
2. The source lift is input-side only.
3. The source lift uses source \(f\) only.
4. The source lift performs direct replacement of source branch input.
5. No concat form is added.
6. No residual form is added.
7. The encoder maps a 5-stencil to one scalar per interior point.
8. Encoder input normalization is computed from `rhs_raw` interior values.
9. Existing `rhs_tilde` is not used as encoder input.
10. \(g\)-normalization uses interior RMS and no centering.
11. \(g\) is defined only on interior points.
12. Source branch input length is `branch_input_dim - 2` when source lift is enabled.
13. Coefficient branch input length remains `branch_input_dim`.
14. Existing `branch_input_dim` config remains boundary-including length \(M=N+2\).
15. No separate `branch_rhs_input_dim` config is added.
16. Physical RHS for projection remains original `rhs_raw`.
17. Output denormalization remains existing `rhs_norm`.
18. Corners are not used by the interior stencil.
19. Actual boundary source values are used by adjacent interior stencils.
20. Source lift diagnostics are logged for train and validation.

---

# Things Not to Do

Do not:

- run or suggest git branch operations,
- add an output-side \((\phi,\psi)\) coupler,
- use \(a,b,c\) inside the source lift encoder,
- concatenate \(\tilde f\) and \(g\) for the source branch,
- use residual source lifting,
- create boundary \(g\) values,
- pass boundary \(g\) to the source branch,
- add normalization mode config fields,
- add a separate `branch_rhs_input_dim` config field,
- replace `rhs_raw` with \(g\) in projection,
- replace `rhs_norm` with the encoder normalization scale,
- change the training loss,
- use artificial padding for interior 5-stencils,
- let corner values influence interior \(g\).

---

# Instruction Review from Codex Perspective

Before completing the patch, review against this checklist.

## Architecture

- [ ] Did I implement an input-side source lift, not an output-side coupler?
- [ ] Does the source lift use only \(f\)?
- [ ] Does it directly replace the source branch input?
- [ ] Did I avoid concat and residual modes?
- [ ] Is the source lift output scalar per interior point?

## Normalization

- [ ] Is encoder normalization computed from `rhs_raw` interior values?
- [ ] Is `rhs_tilde` avoided as source lift encoder input?
- [ ] Is \(g\)-normalization interior RMS?
- [ ] Is mean-centering avoided?

## Shapes

- [ ] Is `branch_input_dim` still the boundary-including length \(M\)?
- [ ] Is coefficient branch input dimension still \(M\)?
- [ ] Is source branch input dimension \(M-2\) when enabled?
- [ ] Is lifted source shape `(B, 2, N, N)`?

## Boundary and Corner Handling

- [ ] Are corners ignored by the interior stencil?
- [ ] Are physical boundary values used for adjacent interior stencil points?
- [ ] Did I avoid creating boundary \(g\)?

## Physical RHS

- [ ] Does projection still use original `rhs_raw`?
- [ ] Does output denormalization still use existing `rhs_norm`?
- [ ] Does loss/evaluation still use original physical source/reference?

## Logging

- [ ] Are `source_lift_corr_g_f`, `source_lift_rel_diff_g_f`, and `source_lift_g_rms` logged?
- [ ] Are these metrics available for both training and validation?

## Tests

- [ ] Did I add config parsing tests?
- [ ] Did I add shape tests?
- [ ] Did I test corner-not-used behavior?
- [ ] Did I test boundary-used behavior?
- [ ] Did I test \(g\)-normalization?
- [ ] Did I test physical RHS remains unchanged?

## Final Expected Summary

When reporting completion, summarize the change like this:

```text
Added an optional input-side direct source 5-stencil lift. When enabled, the model
builds a canonical full source grid from rhs_raw, computes an interior-RMS normalized
source, applies a learned 5-stencil scalar encoder on interior points, normalizes the
resulting lifted source over the interior, and uses the lifted source as the RHS/source
branch input. The coefficient branch still uses boundary-including lines. The physical
rhs_raw, rhs_norm, balance projection, loss, and evaluation remain unchanged.
```
