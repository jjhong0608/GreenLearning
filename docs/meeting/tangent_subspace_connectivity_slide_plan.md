# Tangent Subspace and Geometry Connectivity Slide Contract

## Delivery Contract

- **Deck title:** From Exact Balance to Geometry-Aware Response Alignment
- **Subtitle:** Matrix-Free Tangent Subspaces and Structural K-Connectivity
- **Logical size:** 47 main slides, including the title; no backup slides.
- **Visible language: English.**
- **Speaker notes: Korean**, exactly one notes block per slide.
- **Output:** offline Quarto Reveal.js HTML at 1600x900, verified again at
  1280x720. No presentation PDF is stored.
- **Animation rule:** one fragment exposes one mathematical or evidential step.
  Korean `Click` cues must match the actual fragment indices.
- **Evidence labels:** `EXACT ALGEBRA`, `PRODUCTION ALGORITHM`,
  `STRUCTURAL PROXY`, and `EMPIRICAL RESULT` are kept visually distinct.

The deck is a follow-up to the existing Annulus transition deck. It does not
replace or modify that deck. Its central question is how to reduce directional
Green-response mismatch while remaining on the exact source-balance plane, and
how to interpret the tangent dimension K without confusing the production
algorithm with a geometry-only reach proxy.

## Slide Sequence

### Slides 1-7: Motivation and Exact Balance

1. **From Exact Balance to Geometry-Aware Response Alignment**
   - Frame the talk around three layers: exact balance algebra, matrix-free
     response alignment, and structural/empirical evidence.
   - Notes introduce this as the continuation of the Annulus diagnosis.
2. **Previous Observation: Transition-Localized Directional Error**
   - Reuse frozen Annulus Poisson sample 47 errors for phi, psi, u_phi, u_psi,
     and u_pred.
   - State that the transition is evidence of a general response-alignment
     problem, not the definition of the method.
3. **Exact Source Balance Is Not Equal Response**
   - Contrast `phi + psi = f` with `H_x phi = H_y psi`.
   - Introduce source space versus reconstructed-response space.
4. **Raw Output, Physical Proposal, and Pull-Back**
   - Show `P,Q -> p=P/L_x^2,q=Q/L_y^2 -> projection -> Phi=L_x^2 phi,
     Psi=L_y^2 psi`.
5. **Symmetric Projection Preserves the Difference Mode**
   - Derive `tilde_phi=1/2[f+(p-q)]` and
     `tilde_psi=1/2[f-(p-q)]`.
   - Prove exact sum and preserved raw difference.
6. **The Balance Plane Has One Tangent Direction per Point**
   - Draw normal `(1,1)` and tangent `(1,-1)` at one point.
   - Separate projection onto the plane from motion within the plane.
7. **The Tangent Variable Is a Global Source Field**
   - Define sample-dependent `delta in R^P` and
     `phi=tilde_phi+delta`, `psi=tilde_psi-delta`.
   - Emphasize pointwise exact balance for every delta.

### Slides 8-13: Response Least Squares

8. **Directional Green Operators Map Sources to Responses**
   - Define H_x and H_y as frozen axial source-to-response operators.
9. **A Tangent Source Perturbation Produces a Sum Operator**
   - Define `m0=H_x tilde_phi-H_y tilde_psi`, `S=H_x+H_y`, and
     `m(delta)=m0+S delta`; explain the plus sign.
10. **The Tangent Objective Measures Physical Response Mismatch**
    - Define `J(delta)=1/2 ||m0+S delta||^2_MOmega` and the physical mass
      inner product.
11. **The Gradient Pulls Response Mismatch Back to Source Space**
    - Derive `g(delta)=S^T M_Omega(m0+S delta)` and component interpretation
      `<S e_j,m>_M`.
12. **The Hessian Is a Response Gram Operator**
    - Define `A=S^T M_Omega S` and `A_ij=<S e_i,S e_j>_M`.
    - Explain positive semidefiniteness and response correlation.
13. **The Production Path Never Assembles the Global Hessian**
    - Contrast `A delta*=-g0` with matrix-free `Az=S^T M_Omega(Sz)`.
    - State that no global matrix or sample-wise dense solve is used.

### Slides 14-22: K=1 and the General Matrix-Free Subspace

14. **A Positive Column-Gain Surrogate Pre-Conditions the Gradient**
    - Define the separable denominator
      `D=gamma_x^2+gamma_y^2+lambda_damp`.
    - State explicitly that D is not the exact diagonal of A because the
      cross-axis diagonal term is omitted.
15. **K=1 Chooses One Source Direction and One Response Direction**
    - Define `z0=D^-1 g0`, `v0=S z0`, and `delta(eta)=-eta z0`.
16. **The Scalar Line Search Is Exact Only on the Chosen Line**
    - Derive `eta*=<m0,v0>_M/(<v0,v0>_M+eps)`.
    - Distinguish a one-dimensional exact minimizer from the full delta*.
17. **K=1 Is Globally Supported but One-Dimensional**
    - Explain why dense g0 can move every point while still spanning only one
      correction pattern.
18. **The Remaining Mismatch Generates the Next Direction**
    - Build `m1`, `g1`, `z1_raw`, and `v1_raw` sequentially.
19. **Orthogonality Belongs in Response Space**
    - Orthogonalize v_k in the M_Omega inner product and apply the same linear
      combination to z_k so `S z_k=v_k` remains true.
20. **Two-Pass Modified Gram-Schmidt Protects Numerical Independence**
    - Explain the two passes, activity tolerance, and safe degenerate fallback.
21. **The General K-Step Recurrence Minimizes Along Nested Directions**
    - Present c_k, delta and mismatch updates, and
      `delta_K=-sum c_k z_k` with code-consistent signs.
22. **K Counts Independent Response-Correction Patterns**
    - Prove `phi_K+psi_K=f` for all K.
    - Define K by subspace dimension, not by a literal neighborhood radius.

### Slides 23-32: Krylov Interpretation and Geometry-Only Reach

23. **Standard Krylov Space Is a Conceptual Bridge**
    - Show `K_K(A,g0)=span{g0,Ag0,...,A^(K-1)g0}`.
    - Name production a preconditioned, residual-driven, Krylov-like nested
      response subspace rather than claiming a literal power basis.
24. **Dense Support and Correlation Capacity Are Different**
    - State that the production gradient is generally dense.
    - Explain that increasing K enriches independent geometry-mediated patterns,
      not merely nonzero support.
25. **A Localized Seed Is a Structural Probe, Not a Training Sample**
    - Introduce `g0=e_i` only for geometry visualization.
26. **The Axial Point Graph Encodes Connected-Line Incidence**
    - Connect points sharing one connected horizontal or vertical axial segment;
      disconnected same-coordinate intervals remain separate.
27. **One Conceptual A-Action Spans Forward and Adjoint Mixing**
    - Define `d_A(i,j)=ceil(d_L(i,j)/2)` and
      `K_first(i,j)=d_A(i,j)+1` as the geometry proxy.
28. **Cumulative Reach Produces a Geometry-Only K Rule**
    - Define C_i(K), C_global(K), the lower-5% tail and 99% thresholds.
29. **Unit Square and Disk Reach Every Point Pair at K=2**
    - Use paired representative-seed figures and exact reach values.
30. **The Annulus Hole Delays Structural Reach to K=4**
    - Representative reach: 0.00927%, 31.2106%, 98.2666%, 100%.
31. **Pentagram Tips Create a Long Structural Tail**
    - Representative reach: 0.02187%, 83.6177%, 98.8189%, 99.8688%.
    - State that full A-distance diameter is eight even though K=4 satisfies the
      global/tail 99% selection rule.
32. **Structural Reach Is an Interpretation, Not an Accuracy Proof**
    - List what the proxy establishes and what it cannot establish.

### Slides 33-40: Pentagram Benchmark and Trained Evidence

33. **Pentagram Benchmark: Variable-Coefficient CDR on a Nonconvex Domain**
   - Define the filled Pentagram, homogeneous Dirichlet problem, and the fixed
     diffusion, counter-clockwise convection, and reaction coefficients.
   - Use mesh views of `a`, `|b|`, and `c`.
34. **Pentagram q50 Test Source and Directional Split**
   - Use best-energy K=4 sample 79, selected by the median `rel_sol`.
   - Compare `f`, reference/predicted phi, reference/predicted psi, and signed
     directional errors on the visualization mesh.
   - State that reference directional sources are evaluation-only and that
     black source boundaries mean "not evaluated," not zero.
35. **Pentagram Reconstruction for the Same Test Source**
   - Compare reference solution, predicted solution, and signed error using one
     shared solution range and a zero-centered error range.
   - Report sample-level `rel_sol`, `rel_flux`, canonical energy, and post/pre
     tangent-response mismatch.
36. **Pentagram Test-Set Distribution Exposes the Tail**
   - Show ECDFs for `rel_sol` and `rel_flux`, plus distributions of canonical
     energy and tangent response mismatch over all 100 test samples.
   - Report mean plus/minus std, median, p90, p95, and maximum.
   - Mark sample 79 on the ECDFs.
   - Do **not** add separate phi or psi error distributions.
37. **Pentagram Accuracy Improves from K=1 through K=4**
   - Show trained best-energy rel_sol, rel_u_phi, rel_u_psi, rel_flux and the
     reference-free objective trends.
38. **K=4 Maximizes Accuracy; K=3 Is the Cost-Quality Knee**
   - Compare 141.373/211.807/282.183/361.773 ms tangent-core
     forward+backward times with rel_sol.
39. **Three Mechanisms Can Move Together without Being Equivalent**
   - Separate structural reach, spectral/subspace enrichment and trained
     accuracy; keep the separate-run causal caveat visible.
40. **Revised Interpretation: General Response Alignment**
   - Replace the narrow transition-repair narrative with balance-preserving
     directional response alignment on general geometries.

### Slides 41-47: Unit-Square Benchmark and Experimental Protocol

41. **Unit-Square Benchmark: Poisson on the Complex-Geometry Path**
   - Define `Omega=(0,1)^2`, `-Delta u=f`, and homogeneous Dirichlet data.
   - Show the constant `a=1` coefficient mesh and record `b=0`, `c=0`.
42. **Unit-Square q50 Test Source and Directional Split**
   - Use the 4,800-source seed-0 best-energy sample 11 selected by median
     `rel_sol`.
   - Use the same source/directional mesh layout and color contract as the
     Pentagram slide.
43. **Unit-Square Reconstruction for the Same Test Source**
   - Compare reference solution, prediction, and signed error and report the
     four sample-level diagnostics.
44. **Unit-Square Test Errors Are Tightly Concentrated**
   - Show the 100-sample canonical-run ECDF and energy/mismatch distributions.
   - Keep this within-run sample variation distinct from the following
     four-seed source-count variation.
   - Do **not** add separate phi or psi error distributions.
45. **Unit-Square Source-Count Study Uses a Fixed Compute Budget**
   - Four seeds, 2,400 optimizer steps, N=600/1200/2400/4800, mean rel_sol
     0.4255/0.3931/0.3696/0.3505%.
46. **A Common Source Budget: 4,800**
   - All four paired seeds improve at every doubling; 4,800 is the only tested
     setting satisfying the stated saturation rule.
47. **Three Levels of Evidence Support One Method**
    - Summarize exact balance algebra, structural K proxy and empirical accuracy.
    - Close with K=4 as the accuracy default and K=3 as an engineering option,
      without claiming geometry reach alone proves numerical optimality.

## Frozen Evidence and Provenance

- Historical Annulus: `docs/meeting/annulus_transition_error/assets/`
- Geometry reach: `checkpoints/geometry_k_connectivity_visualization/`
- Pentagram trained K and runtime:
  `checkpoints/pentagram/tangent_topology_k_analysis/`
- Pentagram K=4 field and test-distribution evidence:
  `checkpoints/pentagram/coupling11/artifacts_best_energy/`
- Unit-square source-count study:
  `checkpoints/numerical_examples/unit_square_poisson/training_size_analysis/`
- Unit-square field and test-distribution evidence:
  `checkpoints/numerical_examples/unit_square_poisson/`
  `coupling_train4800_seed0/artifacts_best_energy/`

The deck-local asset builder reads only frozen HTML/JSON/CSV/NPZ/PNG
presentation evidence. It does not load a model, checkpoint, training dataset,
or runtime tangent context. Coefficient and representative-field slides embed
the artifact exporter's frozen mesh PNGs in self-contained static HTML grids;
this preserves the exported color ranges and boundary policy without exhausting
browser WebGL contexts. Test distributions and the pre-existing quantitative
comparisons remain offline Plotly figures. `assets/manifest.json` records every
source path, metric key, source SHA-256 and generated asset SHA-256.

## Verification Contract

1. Run the asset/deck focused tests before rendering.
2. Rebuild assets with the dedicated builder and render the QMD with Quarto.
3. Inspect every final and intermediate fragment state at 1600x900 and 1280x720.
4. Reject overflow, clipped formulas, incoherent overlap, page errors, broken
   iframes and external network requests.
5. Run the full regression suite, Ruff, mypy and `git diff --check` without
   modifying the existing Annulus meeting deck.
