# Codex Implementation Instruction: Five-Stencil StencilMLP Coupler for CouplingNet

## One-sentence objective

Implement an optional `FiveStencilStencilMLPCoupler` inside `CouplingNet` that performs a projection-preserving local 2D five-stencil null-space residual correction **before** the existing balance projection, with a nested `coupling_model.coupler` config block.

---

## Operating assumptions

- The repository is already checked out in the intended working branch, created from `main`.
- **Do not run any git branch, checkout, pull, merge, rebase, commit, or push commands.**
- Modify code only in the current working tree.
- Preserve existing behavior when the coupler config block is absent or when `coupler.enabled == false`.
- Use the current `main` branch code shape as the baseline.

---

## Current code context to preserve

The current `CouplingNet.forward()` computes the interior raw flux-divergence tensor:

```python
flux_tilde = combined.reshape(b, axis, n_lines, m_points - 2)
norm_exp = rhs_norm.unsqueeze(-1)
raw_int = flux_tilde * norm_exp
projected_int = self._apply_balance_projection(raw_int, rhs_raw)
```

Insert the coupler **between** `raw_int = flux_tilde * norm_exp` and `projected_int = self._apply_balance_projection(raw_int, rhs_raw)`.

The existing balance projection uses the canonical orientation

```python
phi = flux_int[:, 0]
psi_t = flux_int[:, 1].transpose(-1, -2)
res = rhs_x_int - (phi + psi_t)
phi = phi + 0.5 * res
psi_t = psi_t + 0.5 * res
```

Do not remove or change this projection. The new coupler must operate before it.

---

## Mathematical design

The existing model is a line-wise 1D operator learner. The new coupler injects a local 2D finite-difference-style correction before projection.

Use a **five-point stencil** only:

\[
(i,j),\quad (i+1,j),\quad (i-1,j),\quad (i,j+1),\quad (i,j-1).
\]

Do **not** use a standard \(3 \times 3\) CNN with corner points. Do **not** flatten the full grid into a global MLP.

The coupler must compute one scalar correction \(\delta_{ij}\) and apply it in the null-space direction:

\[
\phi^1_{ij} = \phi^0_{ij} + \delta_{ij},
\]

\[
\psi^1_{ij} = \psi^0_{ij} - \delta_{ij}.
\]

Therefore, in canonical orientation,

\[
\phi^1 + \psi^1 = \phi^0 + \psi^0.
\]

This is required. The coupler should redistribute mass between \(\phi\) and \(\psi\), not change their sum before the balance projection.

---

## Config schema

Add a nested coupler config under `coupling_model`.

### Dataclasses

In `src/greenonet/config.py`, add a new dataclass before `CouplingModelConfig`:

```python
@dataclass
class CouplerConfig:
    """Optional local 2D coupler inserted before CouplingNet balance projection."""

    enabled: bool = False
    type: Literal["five_stencil_stencil_mlp"] = "five_stencil_stencil_mlp"
    hidden_channels: int = 64
    depth: int = 2
    activation: Literal["tanh", "relu", "gelu", "rational"] = "gelu"
    dropout: float = 0.0
    residual_scale_init: float = 0.05
    padding: Literal["replicate", "zero"] = "replicate"
    eps: float = 1.0e-12
```

Then update `CouplingModelConfig`:

```python
@dataclass
class CouplingModelConfig:
    """Architecture settings for CouplingNet."""

    branch_input_dim: int = 4
    trunk_input_dim: int = 2
    hidden_dim: int = 64
    depth: int = 4
    activation: Literal["tanh", "relu", "gelu", "rational"] = "tanh"
    use_bias: bool = True
    dropout: float = 0.0
    dtype: torch.dtype = torch.float64
    coupler: CouplerConfig = field(default_factory=CouplerConfig)
```

`config.py` already imports `field` and `Literal`, so reuse them.

### JSON config

Update `configs/default_coupling.json` so `coupling_model` contains a nested `coupler` block.

Use this as the default in the `coupler` branch:

```json
"coupling_model": {
  "hidden_dim": 256,
  "depth": 4,
  "activation": "rational",
  "use_bias": true,
  "dropout": 0.0,
  "branch_input_dim": 129,
  "trunk_input_dim": 2,
  "dtype": "float64",

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

Important compatibility rule:

- If `coupling_model.coupler` is missing, instantiate `CouplerConfig()` and keep the coupler disabled.
- If `coupling_model.coupler.enabled` is false, `CouplingNet` must behave as before.

---

## Config parsing updates

### `cli/train.py`

Update config parsing so the nested `coupler` object is converted into `CouplerConfig`.

Import `CouplerConfig` from `greenonet.config`.

Add a helper similar to existing nested config builders:

```python
@staticmethod
def _build_coupler_config(
    raw_coupler: object | None,
    section_name: str,
) -> CouplerConfig:
    if raw_coupler is None:
        return CouplerConfig()

    if not isinstance(raw_coupler, dict):
        raise TypeError(f"{section_name}.coupler must be an object.")

    return CouplerConfig(**dict(raw_coupler))
```

Then replace the current direct `CouplingModelConfig(**coupling_model_kwargs)` path with logic like this:

```python
coupling_model_kwargs = dict(raw.get("coupling_model", {}))

coupler_raw = coupling_model_kwargs.pop("coupler", None)
coupler_cfg = self._build_coupler_config(coupler_raw, "coupling_model")

cm_dtype = coupling_model_kwargs.pop("dtype", "float64")
coupling_model_kwargs["dtype"] = getattr(torch, cm_dtype)

coupling_model_cfg = CouplingModelConfig(
    coupler=coupler_cfg,
    **coupling_model_kwargs,
)
```

Keep all existing dtype behavior.

### `src/greenonet/io.py`

Because checkpoints serialize dataclasses with `asdict`, nested `CouplerConfig` will become a dictionary in metadata. Update deserialization so checkpoint round-trips work.

Import `CouplerConfig`:

```python
from greenonet.config import CouplerConfig, CouplingModelConfig, ModelConfig
```

Update `_deserialize_config`:

```python
def _deserialize_config(
    payload: dict[str, Any],
    config_cls: type[ModelConfig] | type[CouplingModelConfig],
) -> ModelConfig | CouplingModelConfig:
    data = dict(payload)

    if "dtype" in data and isinstance(data["dtype"], str):
        data["dtype"] = _parse_dtype(data["dtype"])

    if config_cls is CouplingModelConfig:
        coupler_raw = data.get("coupler")
        if isinstance(coupler_raw, dict):
            data["coupler"] = CouplerConfig(**coupler_raw)

    allowed_keys = {field.name for field in fields(config_cls)}
    filtered = {key: value for key, value in data.items() if key in allowed_keys}
    return config_cls(**filtered)
```

This must remain backward compatible with old checkpoints that do not have `coupler`.

---

## Coupler implementation location

Implement the new class in `src/greenonet/coupling_model.py`, near the existing `MLP` and `CouplingNet` classes.

Add:

```python
import torch.nn.functional as F
```

Update the config import:

```python
from greenonet.config import CouplerConfig, CouplingModelConfig
```

Implement:

```python
class FiveStencilStencilMLPCoupler(nn.Module, ActivationFactoryMixin):
    ...
```

Use `ActivationFactoryMixin.build_activation()` so `"gelu"` and `"rational"` are both supported.

---

## Coupler pseudo code

The pseudo code below is intentionally close to the requested implementation.

```python
class FiveStencilStencilMLPCoupler(nn.Module, ActivationFactoryMixin):
    """Explicit five-stencil gather + shared pointwise MLP null-space coupler.

    Input:
        raw_int:  (B, 2, N, N)
        a_vals:   (B, 2, N, N + 2)
        b_vals:   (B, 2, N, N + 2)
        c_vals:   (B, 2, N, N + 2)
        rhs_raw:  (B, 2, N, N + 2)

    Output:
        coupled_int: (B, 2, N, N)

    Canonical orientation:
        phi = raw_int[:, 0]
        psi = raw_int[:, 1].transpose(-1, -2)

    Null-space correction:
        phi_new = phi + delta
        psi_new = psi - delta
    """

    point_features: int = 9

    def __init__(self, config: CouplerConfig) -> None:
        super().__init__()

        if config.type != "five_stencil_stencil_mlp":
            raise ValueError(f"Unsupported coupler type: {config.type}")

        if config.hidden_channels <= 0:
            raise ValueError("coupler.hidden_channels must be positive.")

        if config.depth <= 0:
            raise ValueError("coupler.depth must be positive.")

        if config.padding not in {"replicate", "zero"}:
            raise ValueError("coupler.padding must be 'replicate' or 'zero'.")

        self.padding = config.padding
        self.eps = float(config.eps)

        in_channels = 5 * self.point_features

        layers: list[nn.Module] = []
        ch = in_channels

        for _ in range(config.depth):
            layers.append(
                nn.Conv2d(
                    in_channels=ch,
                    out_channels=config.hidden_channels,
                    kernel_size=1,
                    bias=True,
                )
            )
            layers.append(self.build_activation(config.activation))

            if config.dropout > 0.0:
                layers.append(nn.Dropout2d(config.dropout))

            ch = config.hidden_channels

        final = nn.Conv2d(
            in_channels=ch,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

        # Start as an exact identity correction.
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        layers.append(final)

        self.local_mlp = nn.Sequential(*layers)

        # Nonzero initial scale lets gradients reach the final layer immediately.
        self.residual_scale = nn.Parameter(
            torch.tensor(float(config.residual_scale_init))
        )

    def _pad(self, q: torch.Tensor) -> torch.Tensor:
        if self.padding == "replicate":
            return F.pad(q, (1, 1, 1, 1), mode="replicate")

        if self.padding == "zero":
            return F.pad(q, (1, 1, 1, 1), mode="constant", value=0.0)

        raise RuntimeError(f"Unexpected padding mode: {self.padding}")

    def _gather_5_stencil(self, q: torch.Tensor) -> torch.Tensor:
        """Gather center, i+1, i-1, j+1, j-1 features.

        q:
            (B, C, N, N)

        return:
            (B, 5*C, N, N)

        No corner entries are allowed.
        """
        q_pad = self._pad(q)

        center = q_pad[:, :, 1:-1, 1:-1]
        plus_i = q_pad[:, :, 2:, 1:-1]
        minus_i = q_pad[:, :, :-2, 1:-1]
        plus_j = q_pad[:, :, 1:-1, 2:]
        minus_j = q_pad[:, :, 1:-1, :-2]

        return torch.cat(
            [center, plus_i, minus_i, plus_j, minus_j],
            dim=1,
        )

    @staticmethod
    def _canonicalize_flux(
        raw_int: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phi = raw_int[:, 0]
        psi = raw_int[:, 1].transpose(-1, -2)
        return phi, psi

    @staticmethod
    def _decanonicalize_flux(
        phi: torch.Tensor,
        psi: torch.Tensor,
        template: torch.Tensor,
    ) -> torch.Tensor:
        out = template.clone()
        out[:, 0] = phi
        out[:, 1] = psi.transpose(-1, -2)
        return out

    def _validate_inputs(
        self,
        raw_int: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
    ) -> None:
        if raw_int.dim() != 4:
            raise ValueError("raw_int must have shape (B, 2, N, N).")

        bsz, axis, n_i, n_j = raw_int.shape

        if axis != 2:
            raise ValueError(f"Expected raw_int axis dimension 2, got {axis}.")

        if n_i != n_j:
            raise ValueError(f"Expected square raw_int interior grid, got {n_i} x {n_j}.")

        expected_full = (bsz, 2, n_i, n_i + 2)
        for name, value in (
            ("a_vals", a_vals),
            ("b_vals", b_vals),
            ("c_vals", c_vals),
            ("rhs_raw", rhs_raw),
        ):
            if tuple(value.shape) != expected_full:
                raise ValueError(
                    f"{name} must have shape {expected_full}, got {tuple(value.shape)}."
                )

    def forward(
        self,
        raw_int: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(raw_int, a_vals, b_vals, c_vals, rhs_raw)

        # 1. Canonical 2D orientation.
        phi0, psi0 = self._canonicalize_flux(raw_int)

        # 2. Source in canonical orientation.
        f = rhs_raw[:, 0, :, 1:-1]

        # 3. Coefficient fields in canonical orientation.
        ax = a_vals[:, 0, :, 1:-1]
        ay = a_vals[:, 1, :, 1:-1].transpose(-1, -2)

        bx = b_vals[:, 0, :, 1:-1]
        by = b_vals[:, 1, :, 1:-1].transpose(-1, -2)

        cx = c_vals[:, 0, :, 1:-1]
        cy = c_vals[:, 1, :, 1:-1].transpose(-1, -2)

        # 4. Per-sample source scale.
        scale = torch.sqrt(torch.mean(f * f, dim=(-1, -2), keepdim=True) + self.eps)

        phi_hat = phi0 / scale
        psi_hat = psi0 / scale
        f_hat = f / scale

        # 5. Pointwise local features:
        # [phi_hat, psi_hat, f_hat, ax, ay, bx, by, cx, cy]
        q = torch.stack(
            [
                phi_hat,
                psi_hat,
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

        # 6. Explicit five-stencil gather.
        q5 = self._gather_5_stencil(q)

        # 7. Shared pointwise MLP.
        delta_hat = self.local_mlp(q5).squeeze(1)

        # 8. Dimensional residual correction.
        delta = self.residual_scale * scale * delta_hat

        # 9. Null-space update.
        phi1 = phi0 + delta
        psi1 = psi0 - delta

        # 10. Return to raw CouplingNet axis layout.
        return self._decanonicalize_flux(phi1, psi1, raw_int)
```

---

## CouplingNet integration pseudo code

Inside `CouplingNet.__init__`, after constructing the existing branches and trunk, add:

```python
if config.coupler.enabled:
    self.coupler: FiveStencilStencilMLPCoupler | None = FiveStencilStencilMLPCoupler(
        config.coupler
    )
else:
    self.coupler = None
```

Inside `CouplingNet.forward()`, insert this immediately after `raw_int` is computed:

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

Do not change the final zero-padding logic:

```python
projected_flux[:, :, :, 1:-1] = projected_int
```

---

## Explicit non-goals

Do **not** implement any of the following:

- No global flattened-grid MLP.
- No plain \(3 \times 3\) CNN.
- No diagonal/corner stencil points.
- No replacement of the existing balance projection.
- No removal of the existing line-wise branch/trunk readout.
- No changes to the existing `branch_b` and `branch_c` behavior in `CouplingNet.forward()`.
- No git operations.
- No broad checkpoint loading changes unless required by tests. The essential checkpoint change is nested config deserialization.

---

## Tests to add

Add focused tests, preferably in `test/test_coupling.py` or a new `test/test_coupler.py`.

### Test 1: coupler shape preservation

```python
def test_five_stencil_coupler_shape_preservation():
    torch.set_default_dtype(torch.float64)

    bsz = 3
    n = 7
    m = n + 2

    raw_int = torch.randn(bsz, 2, n, n)
    a_vals = torch.randn(bsz, 2, n, m)
    b_vals = torch.randn(bsz, 2, n, m)
    c_vals = torch.randn(bsz, 2, n, m)
    rhs_raw = torch.randn(bsz, 2, n, m)

    cfg = CouplerConfig(enabled=True)
    coupler = FiveStencilStencilMLPCoupler(cfg).to(dtype=torch.float64)

    out = coupler(raw_int, a_vals, b_vals, c_vals, rhs_raw)

    assert out.shape == raw_int.shape
```

### Test 2: zero-initialized coupler is identity

```python
def test_five_stencil_coupler_initial_identity():
    torch.set_default_dtype(torch.float64)

    bsz = 2
    n = 5
    m = n + 2

    raw_int = torch.randn(bsz, 2, n, n)
    a_vals = torch.randn(bsz, 2, n, m)
    b_vals = torch.randn(bsz, 2, n, m)
    c_vals = torch.randn(bsz, 2, n, m)
    rhs_raw = torch.randn(bsz, 2, n, m)

    cfg = CouplerConfig(
        enabled=True,
        hidden_channels=64,
        depth=2,
        activation="gelu",
        residual_scale_init=0.05,
    )
    coupler = FiveStencilStencilMLPCoupler(cfg).to(dtype=torch.float64)

    out = coupler(raw_int, a_vals, b_vals, c_vals, rhs_raw)

    torch.testing.assert_close(out, raw_int)
```

### Test 3: null-space preservation for nonzero correction

Force a nonzero final bias so the coupler actually changes \(\phi\) and \(\psi\), then check the canonical sum.

```python
def test_five_stencil_coupler_preserves_phi_plus_psi_for_nonzero_delta():
    torch.set_default_dtype(torch.float64)

    bsz = 2
    n = 5
    m = n + 2

    raw_int = torch.randn(bsz, 2, n, n)
    a_vals = torch.randn(bsz, 2, n, m)
    b_vals = torch.randn(bsz, 2, n, m)
    c_vals = torch.randn(bsz, 2, n, m)
    rhs_raw = torch.randn(bsz, 2, n, m)

    cfg = CouplerConfig(enabled=True, residual_scale_init=0.1)
    coupler = FiveStencilStencilMLPCoupler(cfg).to(dtype=torch.float64)

    # Force a nonzero delta while keeping the test deterministic.
    final_conv = coupler.local_mlp[-1]
    assert isinstance(final_conv, nn.Conv2d)
    with torch.no_grad():
        final_conv.bias.fill_(0.25)

    out = coupler(raw_int, a_vals, b_vals, c_vals, rhs_raw)

    phi0 = raw_int[:, 0]
    psi0 = raw_int[:, 1].transpose(-1, -2)

    phi1 = out[:, 0]
    psi1 = out[:, 1].transpose(-1, -2)

    torch.testing.assert_close(
        phi1 + psi1,
        phi0 + psi0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    # Also check it is not just identity after forcing the bias.
    assert not torch.allclose(out, raw_int)
```

### Test 4: CouplingNet disabled compatibility

When the coupler is disabled or absent, the forward path should match the old behavior.

Suggested form:

```python
def test_coupling_net_with_disabled_coupler_matches_baseline():
    torch.set_default_dtype(torch.float64)

    # Build a small synthetic input with n_lines + 2 == m_points.
    bsz = 2
    n = 5
    m = n + 2

    coords = torch.randn(2, n, m, 2)
    a_vals = torch.randn(bsz, 2, n, m)
    b_vals = torch.randn(bsz, 2, n, m)
    c_vals = torch.randn(bsz, 2, n, m)
    rhs_raw = torch.randn(bsz, 2, n, m)
    rhs_tilde = torch.randn(bsz, 2, n, m)
    rhs_norm = torch.rand(bsz, 2, n) + 0.1

    torch.manual_seed(123)
    cfg_without = CouplingModelConfig(
        branch_input_dim=m,
        trunk_input_dim=2,
        hidden_dim=16,
        depth=1,
        activation="gelu",
        dtype=torch.float64,
    )
    model_without = CouplingNet(cfg_without)

    torch.manual_seed(123)
    cfg_disabled = CouplingModelConfig(
        branch_input_dim=m,
        trunk_input_dim=2,
        hidden_dim=16,
        depth=1,
        activation="gelu",
        dtype=torch.float64,
        coupler=CouplerConfig(enabled=False),
    )
    model_disabled = CouplingNet(cfg_disabled)

    out_without = model_without(
        coords, a_vals, b_vals, c_vals, rhs_raw, rhs_tilde, rhs_norm
    )
    out_disabled = model_disabled(
        coords, a_vals, b_vals, c_vals, rhs_raw, rhs_tilde, rhs_norm
    )

    torch.testing.assert_close(out_disabled, out_without)
```

### Test 5: nested config parsing

Update or add config tests so `cli/train.py` parses `coupling_model.coupler`.

Expected assertions:

```python
assert coupling_model_cfg.coupler.enabled is True
assert coupling_model_cfg.coupler.type == "five_stencil_stencil_mlp"
assert coupling_model_cfg.coupler.hidden_channels == 64
assert coupling_model_cfg.coupler.depth == 2
assert coupling_model_cfg.coupler.activation == "gelu"
assert coupling_model_cfg.coupler.padding == "replicate"
```

Also test missing block:

```python
assert coupling_model_cfg.coupler.enabled is False
```

### Test 6: checkpoint config round-trip

Update `test/test_io_config.py` or add a new test to ensure nested `CouplerConfig` survives serialization/deserialization.

Minimum requirement:

- Construct `CouplingModelConfig(coupler=CouplerConfig(enabled=True, hidden_channels=32))`.
- Serialize using the existing config metadata path if available.
- Deserialize using the existing loading path.
- Assert the returned config has `coupler.enabled is True` and `coupler.hidden_channels == 32`.

If using private helpers is consistent with the existing tests, test `_serialize_config` and `_deserialize_config` directly. Otherwise, use `save_model_with_config` / `load_model_with_config` with a tiny `CouplingNet`.

---

## Documentation updates

Update `README.md` because the repository guidelines require documentation updates after code changes.

Add a concise bullet near the existing CouplingNet bullets:

```markdown
* CouplingNet optional five-stencil coupler: set `coupling_model.coupler.enabled=true`
  to insert a projection-preserving local 2D five-stencil StencilMLP between the
  line-wise CouplingNet readout and the fixed balance projection. The coupler
  gathers center, i±1, and j±1 features and applies a null-space correction
  `(phi, psi) -> (phi + delta, psi - delta)`, so `phi + psi` is unchanged before
  projection.
```

---

## Validation commands

Run these from the repository root. Use the project virtual environment if present.

```bash
PYTHONPATH=src pytest test/test_coupling.py test/test_io_config.py
PYTHONPATH=src pytest test
ruff check src test
ruff format src test
mypy src
```

If a command fails because of unrelated pre-existing issues, report that clearly and include the relevant failure summary.

---

## Implementation checklist

Before finishing, verify the following manually:

- [ ] No git commands were run.
- [ ] `CouplerConfig` exists and is nested inside `CouplingModelConfig`.
- [ ] Missing `coupling_model.coupler` defaults to disabled.
- [ ] `configs/default_coupling.json` contains the nested `coupler` block.
- [ ] `CouplingNet.forward()` calls the coupler only after `raw_int` and before `_apply_balance_projection`.
- [ ] The coupler uses exactly five stencil positions: center, \(i+1\), \(i-1\), \(j+1\), \(j-1\).
- [ ] The coupler does not use diagonal/corner points.
- [ ] The coupler does not flatten the full grid.
- [ ] The coupler predicts one scalar `delta` per grid point.
- [ ] The update is exactly `phi1 = phi0 + delta`, `psi1 = psi0 - delta`.
- [ ] `phi + psi` is preserved in canonical orientation before projection.
- [ ] The final `Conv2d` layer is zero-initialized.
- [ ] `residual_scale` is an `nn.Parameter` initialized to `coupler.residual_scale_init`.
- [ ] Tests cover shape, identity initialization, null-space preservation, disabled compatibility, nested config parsing, and checkpoint round-trip.
- [ ] README is updated.
- [ ] Formatting, linting, type checking, and tests were run or failures were reported.

---

# Review from Codex's perspective

## Can I implement this from the instruction alone?

Yes. The instruction gives:

- The exact target files.
- The exact nested config schema.
- The exact insertion point in `CouplingNet.forward()`.
- The canonical tensor orientation.
- The exact null-space update rule.
- Pseudo code close to final PyTorch code.
- Tests and validation commands.
- Explicit non-goals that prevent accidental plain CNN or global MLP implementations.

## Potential ambiguity found

### 1. Whether to place the class in a separate file

Resolved by the instruction: implement `FiveStencilStencilMLPCoupler` directly in `src/greenonet/coupling_model.py`.

### 2. Whether `depth` means total layers or hidden layers

Resolved by the instruction: `depth` is the number of hidden `1x1 Conv2d + activation` blocks before the final scalar-output `1x1 Conv2d`.

### 3. Whether to use 5-stencil or 9-stencil

Resolved by the instruction: gather only center, \(i+1\), \(i-1\), \(j+1\), \(j-1\); no corners.

### 4. Whether the coupler may change `phi + psi`

Resolved by the instruction: it must preserve `phi + psi` in canonical orientation through the update `(phi + delta, psi - delta)`.

### 5. Whether Codex should run git commands

Resolved by the instruction: no git commands.

## Review result

The instruction is sufficiently specific for implementation. The highest-risk area is checkpoint compatibility because nested dataclass deserialization must explicitly convert the nested `coupler` dictionary into `CouplerConfig`. This is already called out in the `src/greenonet/io.py` section and covered by the checkpoint round-trip test.
