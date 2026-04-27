# Codex 작업 지시문: CouplingNet flux-consistency loss를 physical energy loss로 교체

## 0. 작업 대상

Repository:

```text
https://github.com/jjhong0608/GreenLearning.git
```

Branch:

```text
main
```

이 지시문은 **main branch 기준**으로 작업하라. 다른 브랜치, 특히 `GreenNetModify` 브랜치의 코드 구조를 가정하지 말고, 반드시 main branch의 실제 파일과 함수명을 확인한 뒤 수정하라.

---

## 1. 작업 목표

CouplingNet training 코드에서 현재 flux-consistency loss를 **face-based physical energy loss**로 교체하라.

현재 loss가 node-centered central difference 또는 line-based finite difference를 사용하고 있다면, 그것을 제거해도 된다. 기존 flux-consistency loss의 외부 API, metric key, logging key는 유지하되, 내부 수식은 아래의 physical energy form으로 바꿔라.

수학적으로 residual을

\[
r := u_x - u_y
\]

라고 할 때, 원하는 loss는

\[
\mathcal L_{\mathrm{energy},h}
=
\sum_{x\text{-faces}} h_xh_y\,a_{x,f}
\left|
\frac{r_{i+1,j}-r_{i,j}}{h_x}
\right|^2
+
\sum_{y\text{-faces}} h_xh_y\,a_{y,f}
\left|
\frac{r_{i,j+1}-r_{i,j}}{h_y}
\right|^2.
\]

즉 구현해야 하는 것은

\[
\boxed{
\sum_{\text{faces}} a_f |D_f r|^2
}
\]

이다.

반드시 주의하라:

```python
density = a_face * dr.pow(2)
```

이어야 한다.

다음처럼 구현하면 안 된다:

```python
density = (a_face * dr).pow(2)
```

후자는

\[
a_f^2 |D_f r|^2
\]

이므로 physical energy가 아니라 flux-magnitude norm이다.

---

## 2. main branch에서 먼저 확인할 것

수정 전 다음 파일들을 확인하라.

우선순위 높은 파일:

```text
src/greenonet/coupling_trainer.py
src/greenonet/config.py
src/greenonet/numerics.py
```

필요하면 다음도 확인하라.

```text
src/greenonet/coupling_model.py
src/greenonet/coupling_data.py
```

확인할 사항:

1. CouplingTrainer의 flux-consistency loss 함수 위치와 이름  
   예상 후보: `_flux_consistency_loss`

2. represented solution tensor orientation  
   예상:
   ```python
   u_phi_x  # x-representation on common grid
   u_psi_y  # y-representation, may require transpose(-1, -2)
   ```

3. coefficient tensor `a_vals` shape  
   예상:
   ```python
   a_vals.shape == (B, 2, n_lines, m_points)
   n_lines + 2 == m_points
   ```

4. x/y coordinate arrays and grid spacing retrieval  
   예상:
   ```python
   hx = uniform_spacing(x_axis)
   hy = uniform_spacing(y_axis)
   ```

5. 기존 loss weight, config, logging, metric keys  
   이 부분은 깨뜨리지 말 것.

---

## 3. 유지해야 할 외부 동작

다음은 유지하라.

- 기존 flux-consistency loss 함수의 public/internal call signature
- `_step_loss`에서 `loss_flux_consistency`를 계산하는 흐름
- config의 `flux_consistency.enabled`, `weight`, `weight_mode`
- metric/logging key:
  ```text
  loss_flux_consistency
  weight_flux_effective
  rel_flux
  rel_sol
  ```
- checkpoint, scheduler, optimizer, validation 흐름
- device, dtype 처리
- autograd compatibility

중요: 기존 central-difference 또는 line-FD flux-consistency 구현은 삭제해도 된다. 다만 함수 이름과 호출 지점은 유지하라. 즉 사용자는 기존처럼 flux-consistency를 켜면, 이제 physical energy loss가 계산되어야 한다.

---

## 4. residual \(r\) 정의

represented solution을 다음처럼 둔다.

\[
u_x := u_\phi^{(x)}, \qquad u_y := u_\psi^{(y)}.
\]

common grid에서 residual은

\[
r = u_x - u_y
\]

이다.

현재 main branch 코드가 이전 구조와 같다면, y-side representation은 transpose가 필요할 가능성이 높다.

예상 구현:

```python
if u_phi_x.shape != u_psi_y.transpose(-1, -2).shape:
    raise ValueError(
        "u_phi_x and u_psi_y must define the same common grid after transpose."
    )

r = u_phi_x - u_psi_y.transpose(-1, -2)
```

단, main branch 코드에서 이미 `u_psi_y`가 common grid orientation으로 들어온다면 중복 transpose하지 말고 실제 shape에 맞춰 구현하라. shape check를 반드시 명시적으로 넣어라. 조용한 broadcasting을 허용하지 말라.

---

## 5. gradient 계산 방식: face-based forward edge difference

central difference를 사용하지 말라.

face-based forward edge difference를 사용하라. 여기서 "forward"는 edge orientation을 정하는 의미다. forward와 backward를 둘 다 별도 loss term으로 더하지 말라. 각 grid edge는 한 번만 세어야 한다.

### 5.1 x-face derivative

\[
(D_x^+ r)_{i+\frac12,j}
=
\frac{r_{j,i+1}-r_{j,i}}{h_x}.
\]

예상 구현:

```python
dr_dx_face = (r[..., :, 1:] - r[..., :, :-1]) / hx
```

shape:

```text
(B, m, m-1)
```

### 5.2 y-face derivative

\[
(D_y^+ r)_{i,j+\frac12}
=
\frac{r_{j+1,i}-r_{j,i}}{h_y}.
\]

예상 구현:

```python
dr_dy_face = (r[..., 1:, :] - r[..., :-1, :]) / hy
```

shape:

```text
(B, m-1, m)
```

---

## 6. interior slicing과 coefficient shape

이 부분은 main branch의 실제 tensor shape에 맞게 구현하라. 이전 branch 구조와 같다면 다음과 같이 처리하는 것이 맞다.

보통 represented solutions는 boundary 포함 common grid:

```text
r.shape == (B, m, m)
```

이고 coefficient `a_vals`는 axial lines만 포함:

```text
a_vals[:, 0].shape == (B, n, m)  # x-lines, interior y rows, all x nodes
a_vals[:, 1].shape == (B, n, m)  # y-lines, interior x cols, all y nodes
n == m - 2
```

그러면 x-face derivative는 transverse y direction에서 interior rows만 사용해야 한다.

```python
dr_dx_face = dr_dx_face[:, 1:-1, :]   # (B, n, m-1)
```

y-face derivative는 transverse x direction에서 interior columns만 사용해야 한다.

```python
dr_dy_face = dr_dy_face[:, :, 1:-1]   # (B, m-1, n)
```

위 slicing 후 coefficient face tensors와 정확히 같은 shape가 되어야 한다. shape mismatch가 발생하면 `ValueError`를 내라.

---

## 7. face coefficient \(a_f\) 계산

physical energy는 face 위에서 계산되므로 coefficient도 face value가 필요하다.

기본은 arithmetic average를 사용하라.

### 7.1 helper 함수

가능하면 다음 helper를 `CouplingTrainer` 내부 static method 또는 private method로 추가하라.

```python
@staticmethod
def _face_average_arithmetic(values: torch.Tensor, dim: int) -> torch.Tensor:
    left = values.narrow(dim, 0, values.shape[dim] - 1)
    right = values.narrow(dim, 1, values.shape[dim] - 1)
    return 0.5 * (left + right)
```

### 7.2 x-face coefficient

x-line coefficient:

```python
a_x_nodes = a_vals[:, 0]  # expected (B, n, m)
a_x_face = self._face_average_arithmetic(a_x_nodes, dim=-1)
```

Expected shape:

```text
(B, n, m-1)
```

### 7.3 y-face coefficient

y-line coefficient must be aligned to common grid orientation.

If `a_vals[:, 1]` is shaped `(B, n, m)` with `n` interior x-lines and `m` y-nodes, transpose it to common orientation:

```python
a_y_nodes_common = a_vals[:, 1].transpose(-1, -2)  # expected (B, m, n)
a_y_face = self._face_average_arithmetic(a_y_nodes_common, dim=-2)
```

Expected shape:

```text
(B, m-1, n)
```

Again, verify shapes explicitly:

```python
if a_x_face.shape != dr_dx_face.shape:
    raise ValueError(...)

if a_y_face.shape != dr_dy_face.shape:
    raise ValueError(...)
```

---

## 8. physical energy density and quadrature

Compute physical energy density as

```python
density_x = a_x_face * dr_dx_face.pow(2)
density_y = a_y_face * dr_dy_face.pow(2)
```

Do not write:

```python
density_x = (a_x_face * dr_dx_face).pow(2)
density_y = (a_y_face * dr_dy_face).pow(2)
```

Use face-grid rectangle quadrature:

```python
loss_x_per_batch = density_x.sum(dim=(-1, -2)) * hx * hy
loss_y_per_batch = density_y.sum(dim=(-1, -2)) * hx * hy

loss = (loss_x_per_batch + loss_y_per_batch).mean()
```

Return `loss`.

Do not use Simpson integration for this face-based loss unless you deliberately redesign the face quadrature. The face grid has different cardinality from the nodal grid, so simple rectangle rule is the intended discretization here.

---

## 9. Suggested replacement function

Adapt the following to the actual main branch code.

```python
def _flux_consistency_loss(
    self,
    u_phi_x: torch.Tensor,
    u_psi_y: torch.Tensor,
    a_vals: torch.Tensor,
    x_axis: torch.Tensor,
    y_axis: torch.Tensor,
) -> torch.Tensor:
    """
    Physical-energy flux consistency loss.

    This implements

        sum_faces a_face * |D_face (u_phi_x - u_psi_y_common)|^2 * hx * hy

    using face-based forward edge differences. Each edge is counted once.
    This is physical energy int a |grad r|^2, not flux magnitude
    int |a grad r|^2.
    """
    if u_phi_x.shape != u_psi_y.transpose(-1, -2).shape:
        raise ValueError(
            "u_phi_x and u_psi_y must define the same common grid after transpose."
        )

    if a_vals.dim() != 4 or a_vals.shape[1] != 2:
        raise ValueError("a_vals must have shape (B, 2, n_lines, m_points).")

    hx = uniform_spacing(x_axis)
    hy = uniform_spacing(y_axis)

    r = u_phi_x - u_psi_y.transpose(-1, -2)  # expected (B, m, m)

    if r.dim() != 3:
        raise ValueError(f"expected residual r to be 3D (B, m, m), got {tuple(r.shape)}")

    # Face differences.
    dr_dx_face = (r[..., :, 1:] - r[..., :, :-1]) / hx      # (B, m, m-1)
    dr_dy_face = (r[..., 1:, :] - r[..., :-1, :]) / hy      # (B, m-1, m)

    # Restrict to interior transverse lines to match a_vals.
    dr_dx_face = dr_dx_face[:, 1:-1, :]   # (B, n, m-1)
    dr_dy_face = dr_dy_face[:, :, 1:-1]   # (B, m-1, n)

    a_x_nodes = a_vals[:, 0]                          # (B, n, m)
    a_y_nodes_common = a_vals[:, 1].transpose(-1, -2) # (B, m, n)

    a_x_face = self._face_average_arithmetic(a_x_nodes, dim=-1)          # (B, n, m-1)
    a_y_face = self._face_average_arithmetic(a_y_nodes_common, dim=-2)   # (B, m-1, n)

    if a_x_face.shape != dr_dx_face.shape:
        raise ValueError(
            f"x-face coefficient shape {tuple(a_x_face.shape)} does not match "
            f"x-face derivative shape {tuple(dr_dx_face.shape)}."
        )

    if a_y_face.shape != dr_dy_face.shape:
        raise ValueError(
            f"y-face coefficient shape {tuple(a_y_face.shape)} does not match "
            f"y-face derivative shape {tuple(dr_dy_face.shape)}."
        )

    # Physical energy: a_face * |D_face r|^2.
    # Do NOT square a_face.
    density_x = a_x_face * dr_dx_face.pow(2)
    density_y = a_y_face * dr_dy_face.pow(2)

    loss_x_per_batch = density_x.sum(dim=(-1, -2)) * hx * hy
    loss_y_per_batch = density_y.sum(dim=(-1, -2)) * hx * hy

    return (loss_x_per_batch + loss_y_per_batch).mean()
```

If main branch uses a different signature for `_flux_consistency_loss`, preserve that signature and adapt the body accordingly.

---

## 10. Tests and sanity checks

Add or update tests if a test suite exists. If no test suite exists, add a lightweight test file or at least a minimal script/test function that can be run locally.

### 10.1 Shape and differentiability test

Create synthetic tensors with representative shapes, for example:

```text
B = 2
m = 5
n = m - 2
u_phi_x.shape = (B, m, m)
u_psi_y.shape = (B, m, m)
a_vals.shape = (B, 2, n, m)
```

Check:

```python
loss = trainer._flux_consistency_loss(...)
assert loss.ndim == 0
loss.backward()
```

Use tensors requiring grad for `u_phi_x` and `u_psi_y`.

### 10.2 Constant residual test

If \(r\) is constant, all face differences should be zero. Therefore loss should be zero.

Construct `u_phi_x` and `u_psi_y` so that

```python
r = u_phi_x - u_psi_y.transpose(-1, -2)
```

is constant. Verify:

```python
assert torch.allclose(loss, torch.zeros_like(loss), atol=...)
```

### 10.3 Linear residual test

For

\[
r(x,y)=x
\]

on a uniform grid and \(a\equiv1\), expected:

\[
D_x r = 1, \qquad D_y r = 0.
\]

The loss should equal the discrete face area sum over the included x-faces.

You do not need an overly fragile exact scalar assertion if indexing is complex, but verify that:

- x contribution is positive,
- y contribution is zero or near zero,
- total loss is finite and positive.

### 10.4 Coefficient scaling test: critical

This test verifies physical energy uses \(a\), not \(a^2\).

Run the same residual with

```python
a_vals = ones
```

and

```python
a_vals = 2 * ones
```

The second loss should be approximately **2 times** the first loss.

If it is approximately **4 times**, the implementation is wrong because it is using \(a^2|D r|^2\).

### 10.5 \(a\equiv1\) equivalence check

When \(a\equiv1\), physical energy and flux-magnitude coefficient weighting coincide. The density should reduce to:

\[
|D_f r|^2.
\]

### 10.6 No edge double counting

Ensure the loss counts each edge once. Do not add a separate backward-difference term. If both forward and backward terms are included, the loss will approximately double-count interior edges.

---

## 11. Acceptance criteria

The task is complete only if all of the following are true.

1. The flux-consistency loss now computes physical energy:
   \[
   \sum_{\text{faces}} a_f |D_f r|^2 h_xh_y.
   \]

2. The code uses face-based forward edge differences:
   \[
   r_{i+1,j}-r_{i,j},\qquad r_{i,j+1}-r_{i,j}.
   \]

3. The code does **not** use node-centered central differences for the new flux-consistency loss.

4. The code does **not** compute:
   \[
   (a_fD_fr)^2.
   \]
   It computes:
   \[
   a_f(D_fr)^2.
   \]

5. Each grid edge is counted exactly once.

6. Existing training call sites, logging keys, and config keys continue to work.

7. Shape mismatches raise explicit errors rather than relying on accidental broadcasting.

8. At least the coefficient scaling sanity check passes:
   \(a=2\) should produce roughly 2x the \(a=1\) loss for the same residual, not 4x.

---

## 12. Notes for implementation judgment

- If coefficient \(a\) is known to be discontinuous or high contrast, consider harmonic face averaging instead of arithmetic averaging. For this task, use arithmetic averaging unless the existing codebase already uses harmonic averaging for diffusion coefficients.
- Do not introduce a new high-order 5-point or 7-point derivative stencil.
- Do not add both forward and backward differences.
- Do not change the mathematical meaning of cross-consistency or represented-solution consistency.
- Do not change checkpoint filenames, logging schemas, or evaluation metrics unless strictly necessary.
- Keep dtype/device consistent with input tensors.
- Avoid `.item()` inside the loss computation except for logging outside the computation graph.

---

# 지시문 자체 검토: Codex 관점에서 충분한가?

## A. Codex가 알아야 하는 핵심 수학이 포함되어 있는가?

예. 지시문은 기존 논의를 모르는 상태에서도 다음을 알 수 있도록 작성되어 있다.

- residual 정의:
  \[
  r=u_x-u_y
  \]
- 목표 loss:
  \[
  \sum a_f |D_f r|^2
  \]
- 금지할 잘못된 loss:
  \[
  \sum |a_fD_f r|^2
  \]
- derivative 방식:
  face-based forward edge difference
- forward/backward를 둘 다 쓰면 안 된다는 점
- coefficient는 face average를 써야 한다는 점
- face quadrature는 rectangle rule이라는 점

## B. 코드 위치에 대한 정보가 충분한가?

대체로 충분하다. 다만 main branch의 실제 파일 구조가 이전 브랜치와 다를 수 있으므로, 지시문은 Codex에게 먼저 main branch의 실제 함수명과 shape를 확인하라고 명시하고 있다.

이 지시문은 특정 줄 번호에 의존하지 않는다. 따라서 main branch의 약간 다른 구조에도 대응할 수 있다.

## C. 가장 위험한 부분은 무엇인가?

가장 위험한 부분은 tensor orientation이다.

특히

```python
u_psi_y.transpose(-1, -2)
```

가 main branch에서 필요한지 여부는 실제 코드를 보고 확인해야 한다.

또 하나는 `a_vals[:, 1]`의 orientation이다. 지시문에서는 예상 구조를 제공했지만, Codex는 main branch 코드에서 실제 shape를 확인해야 한다.

이를 위해 shape check와 `ValueError`를 반드시 넣도록 했다.

## D. 테스트가 원하는 수정을 검출할 수 있는가?

예. 특히 coefficient scaling test가 중요하다.

- 올바른 physical energy:
  \(a=2\)이면 loss가 2배
- 잘못된 flux magnitude:
  \(a=2\)이면 loss가 4배

이 테스트는 핵심 오류를 직접 검출한다.

## E. 지시문만 보고 Codex가 작업을 수행할 수 있는가?

가능하다고 판단한다.

확신도:

\[
\boxed{0.88}
\]

낮추는 이유는 main branch의 실제 코드 구조와 tensor shape를 아직 이 지시문 작성 시점에서 완전히 확인하지 못했기 때문이다. 그러나 지시문이 그 불확실성을 처리하도록 “main branch를 먼저 inspect하라”, “shape check를 넣어라”, “broadcast하지 말라”를 명확히 요구하므로, Codex가 repository를 직접 볼 수 있다면 작업 가능성이 높다.
