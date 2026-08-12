# Modified GreenONet

Axial-inspired neural solver for the 2D Poisson equation with Dirichlet boundaries. The project follows the AGENTS guidelines: class-first design, rich logging mirrored to disk, Plotly visualizations, and TDD.

## Setup
- Activate the virtual environment: `source .venv/bin/activate`
- Install runtime deps: `pip install -e .`
- On Linux `x86_64` with CPython `3.14`, the editable install now pins Torch to the official PyTorch `2.11.0+cu126` wheel from `download.pytorch.org`. Other environments fall back to `torch>=2.11.0`, so install a different PyTorch CUDA build manually if you need one.
- See dev tools (ruff/mypy/pytest): `pip install -e .[dev]`
- Ensure `PYTHONPATH` includes `src` when running commands in this repo.
- FEniCSx sample generation is intentionally isolated from the main `green_net` training environment. Create the optional solver environment with `conda env create -f environment-fenicsx.yml`, then verify it with `conda run -n green_fenicsx python -c "import dolfinx, gmsh, petsc4py, torch"`. Do not add FEniCSx to the main `pyproject.toml` dependencies.

## Usage
- Train with the sample config: `PYTHONPATH=src python cli/train.py --config configs/default.json --work-dir checkpoints/run`
- Logs: Rich console output plus `training.log` in the chosen `work-dir`.
- Artifacts: `loss_curve.html`, `green_heatmap.html`, and weights `model.safetensors` in `work-dir`.
- Config: the CLI writes `config_used.json` in the `work-dir`. GreenNet runs materialize the resolved `training.optimizer` block plus top-level Green optimizer/scheduler provenance, and complex CouplingNet runs materialize the resolved Coupling optimizer block and provenance. Unit-square CouplingNet-only runs preserve the input JSON unless another resolved runtime section requires materialization.
- Checkpoints: `*.safetensors` include JSON-encoded model config metadata; use `greenonet.io.load_model_with_config` to restore model+config.
- Loss: Green's-function reconstruction — integrates the learned Green kernel against the source and matches the recovered solution (no direct output MSE).
- GreenNet structure: the analytic Green wrapping remains active when `model.use_green=true`, while the learned correction now uses fused line encoders for `a`, `a'`, `b`, and `c` plus a hybrid trunk with smooth `(x, xi)` features.
- Green analytic coefficients: under the conservative operator `-d_x(a(x) d_x u) + b(x) d_x u + c(x) u = f`, the implemented wrapping uses `A(x, xi) = 1 / a(x)` and `B(x, xi) = (a'(x) + b(x)) / a(x)^2`, so both coefficients depend on the evaluation-side `x` values.
- Exact Green references: `greenonet.greens.ExactGreenFunction` keeps its default diffusion-only reference for `forward()`, `__call__()`, and `error()`. GreenNet `rel_green` uses the exact/reference line kernel selected from sampled coefficients: diffusion reference when `b=0, c=0`, convection-diffusion reference via `convection_diffusion(b)` when `b!=0, c=0`, and skip/invalid when reaction `c` is nonzero. Exact kernels follow the reconstruction convention `G[row=x, col=xi]`; for convection-diffusion lines, pass the axis-local convection slice (`bx` for x-lines, `by` for y-lines).
- Integration rule: set `training.integration_rule` and `coupling_training.integration_rule` to `"simpson"` or `"trapezoid"` to control sampled-data quadrature in Green/Coupling training, evaluation metrics, and coupling RHS normalization. Green synthetic samplers reuse `training.integration_rule` for sample normalization.
- Optimizers: both unit-square and complex GreenNet use AdamW by default. Set `training.optimizer.name="soap"` to opt into the pinned SOAP implementation; explicit `training.optimizer.name="adam"` is rejected because Adam has been removed from the GreenNet runtime. The optional multi-epoch LBFGS fine-tuning stage remains controlled by `TrainingConfig.lbfgs_*` and is unchanged. CouplingNet also defaults to AdamW, while SOAP remains complex CouplingNet-only under `coupling_training.optimizer.name="soap"`.
- AMUSE research status: AMUSE (Anytime MUon with Stable gradient Evaluation) is not implemented. A current-code applicability review is in `docs/amuse_optimizer_applicability.md`. AMUSE and SOAP share the broad goal of improving matrix-parameter optimization, but SOAP runs Adam in a Shampoo-derived eigenbasis whereas AMUSE combines per-step Muon orthogonalization with Schedule-Free gradient evaluation and averaged inference. The model is structurally suitable because 99.61% of its trainable parameters are 2D matrices, but AMUSE needs optimizer `train()`/`eval()` lifecycle integration, averaged-iterate checkpointing, optimizer-step warmup without cosine decay, and an explicit `bfloat16` Newton-Schulz precision policy. It must remain a complex-only opt-in pilot rather than replacing AdamW or SOAP.
- Torch compile: set `training.compile.enabled=true` to wrap GreenONet with `torch.compile`, and set `coupling_training.compile.enabled=true` to do the same for CouplingNet. The flags are independent, optional, and checkpoint saving still unwraps compiled models to keep load/save compatibility.
- Terminal logging: set top-level `terminal.width` to a positive integer such as `250` to fix Rich console log width for all project loggers. Omit `terminal` or set `width=null` to keep Rich's automatic terminal-width detection; disk `training.log` output is unchanged.
- CouplingNet: a shared branch/trunk MIONet consumes axial-line inputs `(a, b, c, f)` together with interior coordinates and predicts axial flux-divergences `(phi_x, psi_y)` through a single shared DeepONet-style readout followed by optional balance projection.
- Complex geometry mode: set `dataset.geometry_mode="complex"` to train/evaluate CouplingNet on a precomputed non-rectangular valid-point geometry. The default remains `unit_square`, so existing configs keep the original `CouplingDataset`, `CouplingNet`, `CouplingTrainer`, evaluator, and artifact exporter. Complex mode requires `dataset.geometry_path` and interprets `dataset.training_path`, `dataset.validation_path`, and `dataset.test_path` as directories of full-grid sample `.npz` files. `coupling_model.branch_input_dim` is reused as the number of fixed unit-interval branch samples per segment, while `hidden_dim`, `depth`, `activation`, `dropout`, `use_bias`, and `dtype` are reused by `ComplexCouplingNet`.
- Unit square through the complex path: the complex geometry contract also accepts a unit square encoded as one connected segment per interior horizontal/vertical line. On `[0,1]^2`, every segment has `L_x=L_y=1`, so the physical/reference response scaling reduces to the identity. Use `cli/make_rectangular_geometry.py` to create this metadata and `examples/rectangle_gmsh.py` to build the matching single-surface FEniCSx mesh. A paired training config and generated sample bundle remain separate follow-up assets.
- Unit-square vs complex architecture notes: see `docs/unit_square_vs_complex_geometry.md` for the implementation-aware comparison of the preserved unit-square GreenNet/CouplingNet core path and the separate complex geometry path. See `docs/unit_square_vs_complex_geometry_math.md` for the code-surface-free mathematical comparison focused on branch, trunk, projection, and Green reconstruction. See `docs/complex_geometry_greennet_couplingnet_technical_report.md` for a conference-preparation technical report that explains only the complex-geometry framework from its PDE, axial decomposition, GreenNet output composition, Dirac-delta/Heaviside analytic wrapping, forward supervised dataset construction, CouplingNet primary/transverse convection context, axial/transverse trunk roles, learnable rational activation, projection, Energy-Norm Error Bound Proposition, and reconstruction principles. See `docs/wccm_eccomas_2026_presentation_outline.md` for an English slide-outline planning document for the WCCM-ECCOMAS 2026 MS165 talk, including merged MOR/axial-reduction, a figure-first graphic abstract, and pull-back/scaling slides, fixed-coefficient source-to-solution framing, vector-convection notation, a directional split operator slide, a physical-1D-operator pull-back slide, a normalized Green operator action formula, an independent GreenNet analytic structure slide with \(G_0\) and \(J_0\) antiderivative identities, a GreenNet source-to-solution supervision slide with GP-generated target solutions, sources generated from target solutions, and expected reconstruction loss, a CouplingNet directional source-split transition slide with \(\phi=L_xu\), \(\psi=L_yu\), and \(\phi+\psi=f\), a CouplingNet branch/trunk context slide, a physical balance projection and Green reconstruction slide, an Energy-Norm Error Bound Proposition slide, a separated-asset GreenNet kernel-structure evidence slide, a separated-asset CouplingNet 5-by-4 quantile evidence slide, and a final `Takeaway: Coupled Axial Green Solvers` slide whose closing line is that GreenNet supplies line-wise Green inverses while CouplingNet learns the source split that turns them into a 2D elliptic solver. See `docs/wccm_eccomas_2026_slide_content_plan.md` for the detailed slide-by-slide title, subtitle, required-content, optional-content, visual, equation, speaker-emphasis, and Auto-Animate animation blueprint for the early reduction slides, Green-operator-action reveal, term-by-term GreenNet analytic-structure reveal, GreenNet GP-supervision reveal, CouplingNet directional source-split reveal with compact \(L_x,L_y\) definitions, CouplingNet branch/trunk context reveal with branch nets for source, coefficient profiles, and line geometry plus trunk nets for \(t_{\parallel}\) and \(t_{\perp}\), CouplingNet physical balance projection pipeline reveal, staged split-energy error-bound proposition reveal, the separated GreenNet kernel evidence assets plus slide-native diagnostic card, the separated CouplingNet source/reference/prediction/signed-error quantile matrix plus slide-native metric card, final takeaway reveal, and concrete Q&A backup slide specs for Dirac/Heaviside derivation, imperfect Green perturbation, and connected-interval pull-back, using the same fixed-coefficient, vector-convection, Green-operator-action, and analytic-Green-ingredient conventions. See `docs/wccm_eccomas_2026_speaker_script.md` for the approved canonical approximately 13-minute, 1,246-word English speaker script synchronized with the Quarto `::: {.notes}` blocks; future wording changes should update both the Markdown source and the slide notes, and `field` remains reserved for vector-valued quantities. See `docs/wccm_eccomas_2026_slide_deck_critical_review.md` for a Korean critical review of the current WCCM deck, including high-priority revision risks, slide-by-slide critiques, and concrete fixes for narrative, math, visual, animation, and evidence issues. See `docs/complex_geometry_energy_consistency_analysis.md` for the rigorous continuous-domain energy-consistency theorem with connected-interval Green reconstructions, final solution error-bound corollaries, full-domain admissibility discussion, and imperfect Green reconstruction perturbation.
- WCCM-ECCOMAS 2026 deck source: the backup-free main-talk Quarto + Reveal.js deck is `docs/presentations/wccm_eccomas_2026/wccm_eccomas_2026.qmd`, while `docs/presentations/wccm_eccomas_2026/wccm_eccomas_2026_with_backup.qmd` preserves the Backup/Q&A menu and Backup A/B/C; both use presentation-specific styles in `docs/presentations/wccm_eccomas_2026/styles.scss`. The title slide lists Junhong Jo with affiliation `National Institute for Mathematical Sciences`, visually emphasizes the initials `N`, `I`, `M`, and `S`, uses "Joint work with" for Taeyoung Ha (NIMS) and Chang-Ock Lee (KAIST), states the presentation date Tuesday, July 21, 2026, and frames the method as a two-dimensional elliptic solver using axial Green operators and a learned directional source split. Slide 2 shows the fixed-operator source-to-solution map as \(\mathcal S_{a,\mathbf b,c}:f\mapsto u\), includes homogeneous Dirichlet boundary condition \(u=0\) on \(\partial\Omega\), and introduces the physical \(L_x,L_y\) split. Slide 3 is a figure-first graphic abstract: two-dimensional forcing \(f(x,y)\) on a general domain \(\Omega\) \(\rightarrow\) axial interval intersections \(\rightarrow\) a GreenNet KaTeX math card \(v(t)=\int_0^1G_\theta(t,\eta)\rho(\eta)d\eta\) for a generic line-source profile \(\rho\) \(\rightarrow\) a CouplingNet KaTeX math card for \(f\mapsto(\phi,\psi)\), split paths through \(G_x,G_y\), and 2D solution \(u\), with a bottom takeaway that GreenNet supplies line-wise Green inverses and CouplingNet learns the directional source split. Slide 4 shows the physical 1D operator on \(s\in[s_0,s_1]\), interval labels \(s_0,s_1\), \(L=s_1-s_0\), the unit interval pull-back and pull-back map in the same reveal step, a visually separated length-preservation callout, scaling rule, and expanded normalized equation. Slides 5-7 cover GreenNet operator action, analytic Green structure, and GP-based source-to-solution supervision; Slide 6 follows the submitted-abstract motivation that variable-coefficient Green kernels are rarely available in closed form, so the analytic component supplies the delta-induced jump, flux-jump behavior, and boundary structure rather than learning the Green singularity from scratch. Slide 7 contrasts GreenNet supervised one-dimensional source-to-solution reconstruction with CouplingNet, which is trained without reference-solution or split labels. Numerical Evidence I appears immediately after GreenNet III, uses Disk_CD convection-diffusion GreenNet kernel assets with no visible `Kernel-level evidence` label, fixed \(\eta=0.75\), and a three-state reveal; Slides 9-11 cover horizontal/vertical line-wise flux-divergence or source components, branch/trunk context with fixed coefficients but line-varying axial profiles, and projection plus Green reconstruction; Slide 12 states the Energy-Norm Error Bound Proposition; Slide 13 uses separated CouplingNet solution and error assets in a 5-by-4 relative-error quantile matrix with a slide-native metric card and no separate evidence-label pill; Slide 14 is titled `Takeaway: Coupled Axial Green Solvers`, states that CouplingNet learns \(\phi,\psi\) without reference-solution or split labels, notes that reference solutions are evaluation-only, connects unsupervised split consistency to final solution error under assumptions, and closes with the message that GreenNet supplies line-wise Green inverses while CouplingNet learns the source split that turns them into a 2D elliptic solver; Slide 15 is the backup menu. The deck hides Quarto auto-generated title-slide `h2` so the custom title appears only once. Speaker-note `Click` cues follow the actual Reveal fragment sequence and may be more granular than the canonical Markdown script; after manual cue edits, rerender and verify that QMD and HTML cue counts match. Render the backup-free version with `quarto render docs/presentations/wccm_eccomas_2026/wccm_eccomas_2026.qmd`, and render the Q&A version with `quarto render docs/presentations/wccm_eccomas_2026/wccm_eccomas_2026_with_backup.qmd`.
- WCCM-ECCOMAS 2026 speaker-note wording: Slide 1 opens with `Good afternoon.` because the scheduled talk begins after 3:00 p.m. Slide 2 says that \(a\), \(\mathbf b\), and \(c\) are prescribed while \(f\) varies across samples, avoiding the phrases `one coefficient problem` and `heterogeneous operator`. Slide 10 says that trunk nets encode pointwise positions within the primary and transverse axial intervals, without separately arguing what a profile-level branch cannot provide. Keep this wording synchronized between the canonical Markdown script and the Quarto notes.
- WCCM-ECCOMAS 2026 deck validation: Decktape is installed in the user Node environment (`decktape version` reports `3.16.1`) and Chrome for Testing is installed under `~/.local/share/decktape-browsers` (current executable: `~/.local/share/decktape-browsers/chrome/linux-150.0.7871.46/chrome-linux64/chrome`). After rendering the Quarto deck, validate PDF and screenshots with `decktape --chrome-path "$CHROME_PATH" --chrome-arg=--no-sandbox --size 1600x900 --screenshots --screenshots-directory checks/screenshots reveal wccm_eccomas_2026.html wccm_eccomas_2026_decktape.pdf` from `docs/presentations/wccm_eccomas_2026/`, then move the generated PDF to `docs/presentations/wccm_eccomas_2026/checks/wccm_eccomas_2026_decktape.pdf`. This avoids Decktape nesting screenshot paths from a long output PDF path.
- WCCM GreenNet evidence assets: regenerate the GreenNet kernel-structure figures with `PYTHONPATH=src ~/.conda/envs/green_net/bin/python plot_wccm_green_evidence_panel.py --artifact-root checkpoints/Disk_CD/green/artifacts --outdir docs/presentations/wccm_eccomas_2026/assets --interval-index 158 --eta 0.75 --basename greennet_cd_evidence_interval158_eta075 --overwrite`. The script writes the legacy composite panel plus separated slide assets for reference kernel, learned kernel, signed error, and fixed-η slice; the Quarto deck uses the separated PNGs and a slide-native diagnostic card. This GreenNet evidence block is placed immediately after `GreenNet III: Source-to-Solution Supervision` and before CouplingNet, so the line-wise kernel is validated before the source-coupling model is introduced. The slide is titled `Numerical Evidence I: GreenNet Kernel Approximation` with subtitle `Capturing the singular Green structure on an axial interval`; it has no visible `Kernel-level evidence` box. The visible problem strip shows the reaction-free convection-diffusion disk problem \(\Omega=\{x^2+y^2<0.5^2\}\), \(-\nabla\cdot(a\nabla u)+\mathbf b\cdot\nabla u=f\), \(c=0\), \(a=1+\frac12\sin(2\pi x)\sin(2\pi y)\), and \(\mathbf b=(\frac12\sin(\pi x)\cos(\pi y),-\frac12\cos(\pi x)\sin(\pi y))\). A compact line-context tag states `Axial line: y-directed interval at x=-0.25, L=0.866`; it identifies the physical axial line without showing the artifact interval id. The deck uses a three-state reveal: reference/learned/signed-error heatmaps plus diagnostics, then a large fixed-η=0.75 slice while signed error and diagnostics remain visible, then the takeaway: "The learned kernel captures the singular Green structure, and the signed error is not concentrated near the singular diagonal." The diagnostic card defines the diagonal band in normalized unit-coordinate length, `|t-η| ≤ 5/128 = 0.0391`, and reports `Diagonal error mass / area` and `Diagonal mean / off-band mean` for clarity. The generated HTML versions can be used for interactive Q&A, but the timed main deck uses static PNGs for reliable projection and PDF export.
- WCCM CouplingNet evidence assets: regenerate the Slide 13 quantile field matrix assets with `PYTHONPATH=src ~/.conda/envs/green_net/bin/python plot_wccm_coupling_evidence_panel.py --artifact-root checkpoints/Disk_CDR/coupling/artifacts --outdir docs/presentations/wccm_eccomas_2026/assets --basename coupling_cdr_evidence_rel_sol_quantiles --overwrite`. The script reads `selected_raw_arrays.npz`, `summary.json`, and `metrics/per_sample_metrics.csv`, then writes separated field PNG/PDF/HTML/JSON files for the `min`, `q25`, `q50`, `q75`, and `max` relative-solution-error samples. Slide 13 uses this Disk_CDR convection-diffusion-reaction setup: \(\Omega=\{x^2+y^2<0.5^2\}\), \(-\nabla\cdot(a\nabla u)+\mathbf b\cdot\nabla u+c u=f\), \(a=1+\frac12\sin(2\pi x)\sin(2\pi y)\), \(\mathbf b=(\frac12\sin(\pi x)\cos(\pi y),-\frac12\cos(\pi x)\sin(\pi y))\), and \(c=\frac12(1+\frac12\cos(2\pi x)\cos(2\pi y))\). The deck assembles these assets into a 5-by-4 slide-native table with rows `Source`, `Reference`, `Prediction`, and `Signed error`; individual panels intentionally omit title, axis, colorbar, sample id, and artifact-specific text. Source appears in the initial slide state, then `Reference`, `Prediction`, and `Signed error` reveal one row at a time. Source fields use per-sample scales; reference and prediction share one solution scale within each selected sample but not across different samples; the signed-error row uses one zero-centered diverging scale across selected samples for error-magnitude comparison. The slide-native metric card reports relative solution error, labeled `rel. sol. err.`, and split-energy loss from the initial slide state. The Slide 13 takeaway states that quantile-selected CDR samples support 2D solution reconstruction across the observed relative-error range, not only on a single favorable case.
- WCCM-ECCOMAS 2026 remaining-fixes revision: the active deck under `docs/presentations/wccm_eccomas_2026/` keeps `docs/presentations_backup/` untouched and tightens the critical-review partial items. Slide 3 visually downweights the source card and emphasizes the axial interval and CouplingNet directional-coupling cards; Slide 5 adds the contrast strip "Previous step: normalize the operator. This step: learn its Green inverse."; Slide 6 keeps only the three main analytic roles in the timed talk and moves \(A/B\) coefficient-factor details to Backup A. Backup A now uses the operator-action heading "Operator application creates two effects" and separates Dirac jump, Heaviside leftover, and analytic compensation; Backup B separates directional mismatch \(\varepsilon_x-\varepsilon_y\) from common bias \(\varepsilon_x+\varepsilon_y\); Backup C uses a deck-native non-square slice visual and states that connected intervals are not merged across outside-domain gaps.
- DTE 2027 abstract source: `docs/dte2027_abstract/DTE2027_abstracts.tex` uses the official `DTE2027_abstracts.cls` template for the minisymposium `MS018 - Scientific Machine Learning for PDEs in Complex Geometries`. The abstract title is `An Operator-Learning Framework Based on Coupled One-Dimensional Green's Functions for 2D Elliptic PDEs on Complex Domains`, with Junhong Jo as presenting author and Taeyoung Ha / Chang-Ock Lee as coauthors. The address block follows the official template by listing National Institute for Mathematical Sciences, Daejeon 34047, Korea with `jjhong0608@nims.re.kr` and `tha@nims.re.kr`, and Department of Mathematical Sciences, KAIST, Daejeon 34141, Korea with `colee@kaist.edu`. The abstract starts from the Green-function source-to-solution integral representation, cites the axial Green's function method plus the DD29 axial Green surrogate proceeding, and frames the contribution around complex-domain axial intersections decomposed into connected components, unit-interval pull-back, the line-wise Green-kernel model GreenNet, and the directional source-coupling model CouplingNet for 2D reconstruction; it does not foreground label-free training, energy-norm theory, a specific numerical domain family, or neural-operator positioning. Its keyword line uses `Operator learning`, not `Neural operators`. Compile from `docs/dte2027_abstract/` with `latexmk -pdf DTE2027_abstracts.tex` or repeated `pdflatex DTE2027_abstracts.tex` when a LaTeX toolchain is available.
- Annulus transition-error meeting deck: the approved narrative and evidence
  contract remain in `docs/meeting/annulus_transition_error_slide_plan.md`; the
  canonical Quarto source is
  `docs/meeting/annulus_transition_error/annulus_transition_error.qmd`, and the
  rendered offline Reveal.js deck is
  `docs/meeting/annulus_transition_error/annulus_transition_error.html`. The
  deck has exactly 20 logical slides: an 18-slide main discussion plus two Q&A
  backups. Overall presentation time is not fixed; note-level timing labels are
  advisory. Visible titles and body text are English, while every slide
  has one Korean speaker-note script. Fragments reveal one claim, formula stage,
  or figure comparison per click. Slides 6-8 expand geometry-only compact C2,
  mismatch-detected seam C2, and local weak-residual reliability with their
  construction equations and frozen sample-0 fields. Poisson evidence precedes
  CDR evidence, and Slides 11-14 use `u_weak_residual_reliability` rather than the standard
  equal-mean `u_pred`. Pre-projection fuser experiments, auxiliary split/weak
  training losses, trace/gluing work, optimizer comparisons, learned blend
  weights, and multi-orientation charts remain outside this deck.
- Annulus meeting deck revision contract: presentation text and Korean notes use
  formulation names without model/output-contract version labels. Slide 2
  separates the `2.19x` line-length ratio from the dominant `4.80x`
  response-scale ratio; Slide 4 defines the RPS quantities `d0`, `kappa`, and
  `d_RPS`; Slide 5 compares all three reconstruction weights by signal,
  sample adaptivity, and operator awareness; and Slide 11 uses a two-line
  sidebar formula for the weak prediction. Shared Poisson/CDR result-field
  colorbar titles are placed to the right of the bars with explicit padding and
  right margin. Slide 15 now treats the physical symmetric split as an
  orthogonal projection onto `C_f={(phi,psi):phi+psi=f}`: it removes the
  `(1,1)` common-mode defect, preserves the `(1,-1)` difference mode, and then
  exposes the remaining directional mismatch
  `m0=H_x*phi_tilde-H_y*psi_tilde`. Slide 16 starts from that exact-balanced
  pair and defines the evaluated fixed tangent correction
  `g=(H_x+H_y)^T M_Omega m0`, `delta=-eta*g/D`,
  `phi=phi_tilde+delta`, `psi=psi_tilde-delta`. The fixed column diagonals
  `diag(H_s^T M_Omega H_s)` appear only in the Jacobi denominator `D`; they do
  not allocate `f-p-q`. Slide 16 records `eta=0.01`, `lambda_rel=0.01`, and the
  direct audit ratio `rho=||m_post||/||m_pre||`. Slides 17-18 use frozen Poisson
  `coupling18` and CDR `coupling8` artifacts: mean response mismatch decreases
  by `65.0%` and `63.0%`, respectively, with `50/50` improved samples in each
  run. The result slides keep mismatch reduction distinct from equal-mean and
  weak-final `rel_sol`; they do not claim causal improvement over a separately
  symmetric-trained baseline. The five directional-response factors remain in
  the balanced 3+2 block grid.
- Annulus meeting asset regeneration: run
  `PYTHONPATH=src ~/.conda/envs/green_net/bin/python docs/meeting/annulus_transition_error/build_assets.py --overwrite`,
  then render with
  `quarto render docs/meeting/annulus_transition_error/annulus_transition_error.qmd`.
  The builder reads the named frozen Poisson/CDR artifact archives and geometry
  metadata only; it never loads a CouplingNet checkpoint or reruns inference.
  It deterministically reconstructs the mismatch-seam sensor from stored
  `u_phi/u_psi` for the method-detail figure. It writes local
  Plotly HTML files, a local `plotly.min.js`, `build_assets.log`, and
  `assets/manifest.json`, which records every source path, sample id, metric,
  field key, and SHA-256 digest used by the slides. The two tangent result assets
  are `poisson_tangent_result_q50.html` and `cdr_tangent_result_q50.html`; each
  shares one zero-centered color range between its pre/post mismatch maps and
  uses an independent range for final solution error. No final presentation PDF
  is part of the deliverable.
- Annulus meeting deck QA: run
  `node docs/meeting/annulus_transition_error/qa_reveal.js --html docs/meeting/annulus_transition_error/annulus_transition_error.html --outdir docs/meeting/annulus_transition_error/screenshots/qa --slides 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18`.
  The QA script checks both 1600x900 and 1280x720 viewports, requires 20 slides,
  rejects content overflow, layout-region overlap, and external HTTP requests,
  and captures the key
  fragment states for all 18 main slides. The same browser QA run captures all
  20 final states under
  `docs/meeting/annulus_transition_error/screenshots/1600x900/` and
  `docs/meeting/annulus_transition_error/screenshots/1280x720/`; no PDF is
  required or stored for this deck.
- Complex geometry schema: the geometry `.npz` must contain `coords_valid`, `valid_grid_y_index`, `valid_grid_x_index`, `x_segment_id`, `y_segment_id`, `x_local_t`, `y_local_t`, `x_segment_left`, `x_segment_right`, `x_segment_y`, `x_segment_length`, `y_segment_bottom`, `y_segment_top`, `y_segment_x`, `y_segment_length`, `x_recon_ptr`, `x_recon_t`, `x_recon_weight`, `x_recon_valid_index`, `y_recon_ptr`, `y_recon_t`, `y_recon_weight`, `y_recon_valid_index`, `x_edges`, `y_edges`, `hx`, and `hy`. Reconstruction arrays use `valid_index == -1` for hard-zero segment endpoints; valid points must be strictly interior in segment-local coordinates.
- Rectangular geometry generation: use `PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_rectangular_geometry.py --step-size 0.0078125 --out data/geometry/unit_square_1_128.npz` for the default unit square `[0,1]^2`. General axis-aligned rectangles use `--x-min`, `--x-max`, `--y-min`, and `--y-max`; the single uniform step size must divide both side lengths and leave at least one interior point per axis. The generator excludes boundary points, stores one full-width segment per interior y-line and one full-height segment per interior x-line, writes unit-coordinate trapezoid reconstruction weights with hard-zero endpoints, records CCW boundary vertices and rectangle provenance, and validates the saved NPZ by default. Pass `--overwrite` to replace an existing file or `--no-validate` to skip loader validation.
- Circular geometry generation: use `PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_circular_geometry.py --step-size 0.05 --radius 1.0 --out data/geometry/unit_circle_h005.npz` to generate a centered circular complex geometry. The default radius is `1.0`; non-unit circles use `--radius R` and a filename such as `circle_r2_h005.npz`. The generator requires `2 * radius / step_size` to be an integer, builds the full grid on `[-radius, radius]`, excludes boundary grid points and degenerate boundary lines, stores only axial chord segments with valid interior points, writes unit-local nonuniform trapezoid reconstruction weights, and validates the saved `.npz` with `load_complex_geometry` by default. Pass `--overwrite` to replace an existing output and `--no-validate` to skip the post-write schema check.
- Annulus geometry generation: use `PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_annular_geometry.py --inner-radius 0.5 --outer-radius 1.0 --step-size 0.05 --out data/geometry/annulus_r05_r10_h005.npz` to generate the 2D "torus" shape used by this PDE code path. The domain is the centered annulus `inner_radius < sqrt(x^2+y^2) < outer_radius`, the grid is built on `[-outer_radius, outer_radius]`, and `2 * outer_radius / step_size` must be an integer. Axial lines that cross or touch the inner hole are stored as disconnected segment rows, so edges and reconstruction arrays never bridge across the hole. The saved metadata includes `domain_type="annulus"`, `inner_radius`, `outer_radius`, `center`, `step_size`, `boundary_tol`, `grid_x`, and `grid_y`.
- Pentagram geometry generation: use `PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_pentagram_geometry.py --outer-radius 1.0 --step-size 0.05 --out data/geometry/pentagram_r10_h005.npz`. The domain is the hole-free, filled simple concave 10-gon centered at the origin with top vertex `(0, R)`. Its orientation is fixed at `pi/2`, and its alternating inner radius is derived as `R / phi^2`; neither orientation nor inner radius is a CLI option. The generator stores strict interior grid points, exact polygon-intersection endpoints, disconnected axial segments, and `boundary_vertices`. Those saved vertices are the common source of truth for both Cartesian geometry and the Gmsh boundary.
- Complex sample schema: full-reference `.npz` files contain full-grid `rhs` and `sol` arrays indexed as `[row=y, col=x]`; optional flux targets use preferred keys `phi` and `psi`, with legacy `uxx` and `uyy` accepted as fallback. Source-only train/validation `.npz` files contain only full-grid `rhs`. Full-grid values are gathered into valid-point order using `valid_grid_y_index` and `valid_grid_x_index`. Test evaluation and artifact export always use the full-reference test path.
- FEniCSx complex sample generation: use `cli/make_fenicsx_samples.py` from the optional `green_fenicsx` environment to generate CouplingNet sample `.npz` files for complex geometry. The geometry `.npz` must include `grid_x` and `grid_y` because the generator writes full-grid arrays, while the existing `coords_valid` and valid index arrays define which grid values belong to the domain. Domain input must be exactly one of `--gmsh-script <path>` or `--msh <path>`. A Gmsh script must define `build_domain(gmsh, context)` and return `{"surface_tags": [...]}`; disconnected multi-surface domains must also return `point_surface_tags`, one surface tag per valid geometry point, so the generator can embed each valid point in the correct surface. Script mode embeds valid grid points as internal Gmsh points by default and requires vertex coverage by default; `.msh` mode evaluates at valid points and does not require vertex coverage unless `--require-valid-points-in-mesh` is passed. `examples/rectangle_gmsh.py` validates the saved bounds and four CCW boundary vertices before returning one rectangular plane surface. `examples/unit_circle_gmsh.py` is unit-radius by default but reads `radius` from the geometry `.npz`, so non-unit circular geometry and the FEniCSx disk mesh stay aligned. `examples/annulus_gmsh.py` reads `inner_radius` and `outer_radius` from annulus geometry metadata and returns one Gmsh surface with an inner hole. `examples/pentagram_gmsh.py` validates and consumes the exact saved `boundary_vertices`, then returns one filled concave surface and its 10 boundary curves. None of these single-surface scripts needs `point_surface_tags`.
- FEniCSx sample command example:
  ```
  PYTHONPATH=src conda run -n green_fenicsx python cli/make_fenicsx_samples.py \
    --geometry data/geometry/unit_circle_h005.npz \
    --out data/complex_samples/unit_circle_h005 \
    --gmsh-script examples/unit_circle_gmsh.py \
    --num-train 1000 \
    --num-valid 100 \
    --num-test 100 \
    --lengthscale 0.2 \
    --amplitude 1.0 \
    --seed 0 \
    --mesh-size 0.03
  ```
  The output layout is `<out>/train/sample_000000.npz`, `<out>/valid/sample_000000.npz`, `<out>/test/sample_000000.npz`, plus `<out>/make_fenicsx_samples.log` and `<out>/generation_summary.json`. Every sample stores full-grid `rhs`, `sol`, `phi`, and `psi` arrays with shape `(len(grid_y), len(grid_x))`; values outside `coords_valid` are zero-filled. `rhs` is sampled from the same shared separable squared-exponential GP and indexed-seed core used by source-only generation, then zero-filled outside the valid domain points before the FEM solve. This refactor does not change the FEniCSx PDE solve or reference schema. `sol`, `phi`, and `psi` are evaluated from FEniCSx functions at valid grid points and written back to full-grid arrays.
- Rectangular FEniCSx sample command example:
  ```
  PYTHONPATH=src conda run -n green_fenicsx python cli/make_fenicsx_samples.py \
    --geometry data/geometry/unit_square_1_128.npz \
    --out data/complex_samples/unit_square_1_128_poisson \
    --gmsh-script examples/rectangle_gmsh.py \
    --num-train 1000 \
    --num-valid 100 \
    --num-test 100 \
    --lengthscale 0.2 \
    --amplitude 1.0 \
    --mean 0.0 \
    --seed 0 \
    --mesh-size 0.02 \
    --solution-degree 3 \
    --target-degree 2 \
    --coefficients coefficients/Pure_Poisson.py
  ```
- Annulus FEniCSx sample command example:
  ```
  PYTHONPATH=src conda run -n green_fenicsx python cli/make_fenicsx_samples.py \
    --geometry data/geometry/annulus_r05_r10_h005.npz \
    --out data/complex_samples/annulus_r05_r10_h005_smoke \
    --gmsh-script examples/annulus_gmsh.py \
    --num-train 1 \
    --num-valid 0 \
    --num-test 0 \
    --lengthscale 0.2 \
    --amplitude 1.0 \
    --mean 0.0 \
    --seed 0 \
    --mesh-size 0.03 \
    --solution-degree 3 \
    --target-degree 2 \
    --coefficients coefficients/Pure_Poisson.py
  ```
- Pentagram FEniCSx sample command example:
  ```
  PYTHONPATH=src conda run -n green_fenicsx python cli/make_fenicsx_samples.py \
    --geometry data/geometry/pentagram_r10_h005.npz \
    --out data/complex_samples/pentagram_r10_h005_smoke \
    --gmsh-script examples/pentagram_gmsh.py \
    --num-train 1 \
    --num-valid 0 \
    --num-test 0 \
    --lengthscale 0.2 \
    --amplitude 1.0 \
    --mean 0.0 \
    --seed 0 \
    --mesh-size 0.03 \
    --solution-degree 3 \
    --target-degree 2 \
    --coefficients coefficients/Pure_Poisson.py
  ```
- FEniCSx sample-level parallelism: add `--num-workers N --sample-seed-policy indexed` to split independent samples across spawned Python worker processes. Each worker performs single-process FEniCSx solves; this is not MPI domain decomposition. `num_workers=1` keeps the legacy sequential RNG path. Parallel mode requires indexed seeds, so each `(split, index)` sample is reproducible regardless of worker scheduling. By default existing sample files fail fast; pass `--overwrite` to replace them or `--skip-existing` to resume without rewriting existing samples. Only the parent process writes `generation_summary.json`.
  ```
  PYTHONPATH=src conda run -n green_fenicsx python cli/make_fenicsx_samples.py \
    --geometry data/geometry/unit_circle_h01.npz \
    --out data/complex_samples/unit_circle_h01_small_parallel \
    --gmsh-script examples/unit_circle_gmsh.py \
    --num-train 32 \
    --num-valid 8 \
    --num-test 8 \
    --lengthscale 0.2 \
    --amplitude 1.0 \
    --mean 0.0 \
    --seed 0 \
    --mesh-size 0.025 \
    --solution-degree 3 \
    --target-degree 2 \
    --coefficients coefficients/Pure_Poisson.py \
    --num-workers 4 \
    --sample-seed-policy indexed
  ```
- FEniCSx `phi`/`psi` convention: the generator solves the weak form `int a grad(u).grad(v) + (b.grad(u)) v + c u v = int f v` with homogeneous Dirichlet boundary conditions, then projects the direction-split targets `phi=-d_x(a d_x u)+b_x d_x u+0.5 c u` and `psi=-d_y(a d_y u)+b_y d_y u+0.5 c u`. `generation_summary.json` records the valid-point relative balance residual for `phi + psi ~= rhs`.
- Circular-domain smoke and small dataset workflow:
  ```
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_circular_geometry.py \
    --step-size 0.25 \
    --out data/geometry/unit_circle_h025.npz \
    --overwrite

  PYTHONPATH=src conda run -n green_fenicsx python cli/make_fenicsx_samples.py \
    --geometry data/geometry/unit_circle_h025.npz \
    --out data/complex_samples/unit_circle_h025_smoke \
    --gmsh-script examples/unit_circle_gmsh.py \
    --num-train 1 \
    --num-valid 0 \
    --num-test 0 \
    --lengthscale 0.2 \
    --amplitude 1.0 \
    --mean 0.0 \
    --seed 0 \
    --mesh-size 0.035 \
    --solution-degree 3 \
    --target-degree 2 \
    --coefficients coefficients/Pure_Poisson.py

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/validate_complex_samples.py \
    --geometry data/geometry/unit_circle_h025.npz \
    --sample-root data/complex_samples/unit_circle_h025_smoke \
    --splits train \
    --coefficients coefficients/Pure_Poisson.py \
    --branch-input-dim 4 \
    --max-balance-residual 1e-2

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_circular_geometry.py \
    --step-size 0.1 \
    --out data/geometry/unit_circle_h01.npz \
    --overwrite

  PYTHONPATH=src conda run -n green_fenicsx python cli/make_fenicsx_samples.py \
    --geometry data/geometry/unit_circle_h01.npz \
    --out data/complex_samples/unit_circle_h01_small \
    --gmsh-script examples/unit_circle_gmsh.py \
    --num-train 32 \
    --num-valid 8 \
    --num-test 8 \
    --lengthscale 0.2 \
    --amplitude 1.0 \
    --mean 0.0 \
    --seed 0 \
    --mesh-size 0.025 \
    --solution-degree 3 \
    --target-degree 2 \
    --coefficients coefficients/Pure_Poisson.py

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/validate_complex_samples.py \
    --geometry data/geometry/unit_circle_h01.npz \
    --sample-root data/complex_samples/unit_circle_h01_small \
    --splits train valid test \
    --coefficients coefficients/Pure_Poisson.py \
    --branch-input-dim 4 \
    --max-balance-residual 5e-2
  ```
  The circular smoke path uses `h=0.25`, `mesh_size=0.035`, `solution_degree=3`, `target_degree=2`, and one train sample. The small dataset path uses `h=0.1`, `mesh_size=0.025`, the same FEM degrees, and `32/8/8` train/valid/test samples. Start with `coefficients/Pure_Poisson.py`; move to diffusion, reaction, and convection families only after the circular balance residual is stable.
  With these defaults, the refined `h=0.25` smoke sample has been observed to pass `1e-2` (`max ~= 7.8e-3`), while the `h=0.1` small dataset is a schema/loadability and residual-distribution check (`max ~= 3.7e-2`). Refine `mesh_size`, `solution_degree`, or `target_degree` before treating the small dataset as a strict `1e-2` quality dataset.
- Complex GreenNet training: when `dataset.geometry_mode="complex"` and `pipeline.run_green=true`, GreenNet trains on the connected x/y segments from `dataset.geometry_path` as a flat interval list `N = Sx + Sy`. Every connected interval is an independent 1D domain, so disconnected components on the same axial line remain separate rows. `dataset.samples_per_line` and `dataset.validation_samples_per_line` mean samples per connected interval. `dataset.training_path`, `dataset.validation_path`, and `dataset.test_path` are ignored by GreenNet in complex mode and remain CouplingNet full-grid sample directories. During L-BFGS fine-tuning, new complex GreenNet runs log `train_rel_sol`, validation `val_rel_sol` when enabled, and `rel_green` when the exact/reference policy supports it.
- Complex GreenNet split quadrature: set `training.green_quadrature.enabled=true` to use split Gauss-Legendre integration for complex GreenNet reconstruction loss, training/validation `rel_sol`, `evaluate(...)`, and complex Green artifact reconstruction. The supported v1 rule is `split_gauss_legendre`; `source_interpolation` can be `"linear"` or `"cubic"`, where `"cubic"` means natural cubic spline interpolation of the fine source values at Gaussian source nodes. The default is `"linear"` for stable backward-compatible behavior, while `"cubic"` is opt-in for smoother source experiments. `source_sampling_factor` controls the optional fine source grid used at Gaussian source nodes; `order=4` and `source_sampling_factor=4` are a recommended starting point. This quadrature is unit-coordinate only and does not multiply by an additional segment length. `rel_green` remains the existing uniform-grid diagnostic, and CouplingNet reconstruction/loss/evaluation paths do not use this setting.
- Complex coefficient normalization: each physical segment is mapped to the unit interval before GreenNet training and Green reconstruction. The unit coefficients are `a_unit=a_phys`, `ap_unit=L*ap_phys`, `b_unit=L*b_phys`, and `c_unit=L^2*c_phys`. GreenNet training sources use `f_unit=L^2*f_phys`; GreenNet trunk coordinates are always unit `(t, eta) in [0,1]^2`, and unit reconstruction integrates `G_unit(t, eta) f_unit(eta)` over `eta` without another segment-length factor. Complex CouplingNet reuses the unit-scaled Green coefficient branches but keeps its mandatory source branch in normalized physical source space.
- Complex CouplingNet behavior: output contract v6 is source-conditioned and response-valued. Full-grid `rhs` is gathered to valid points, lifted into segment-local physical source profiles with endpoint hard-zero values, and normalized by the length-independent amplitude `A=sqrt(integral_0^1 f_phys(s(t))^2 dt)`. The model returns raw directional reference responses `[P,Q]` with shape `(B,2,P)`, using deterministic output scales `P=L_x^2*A_x*P_tilde` and `Q=L_y^2*A_y*Q_tilde`. The coefficient branch is controlled by `coupling_model.coefficient_terms` in active `[a,b_primary,b_transverse,c]` order: when `convection=true`, x/Phi segments receive `[L_x*b_x, L_x*b_y]` and y/Psi segments receive `[L_y*b_y, L_y*b_x]`; Green reconstruction branches remain `[a,ap,b_primary,c]`. The geometry branch consumes `[s_left,s_right,s_mid,L,L^2,1/L]`, the fixed-line transverse branch consumes globally normalized transverse Fourier features, and the primary trunk consumes segment-local `t`.
- Complex pointwise length context: output contract v6 requires `axis_1d_trunk.transverse_trunk.enabled=true` and `length_context=true`. One shared four-input transverse MLP receives `[t_perp, log(L_perp/L_ref), log(L_parallel/L_perp), kappa]`, where `kappa=4*L_parallel^2*L_perp^2/(L_parallel^2+L_perp^2)^2` and `L_ref` is the larger global geometry extent. The x/Phi path uses `(L_parallel,L_perp,t_perp)=(L_x,L_y,y_local_t)`; the y/Psi path swaps the axes.
- Complex physical balance projection: output contract v6 requires `balance_projection.enabled=true`. The backward-compatible default is `mode="physical_symmetric"`; the complex-only opt-in alternatives are `mode="column_diagonal_green_response"` and `mode="symmetric_tangent_green_response"`. All modes first map raw reference responses to physical directional-source proposals, `p=P/L_x^2` and `q=Q/L_y^2`, impose `phi+psi=rhs` in physical source space, and then pull back with `Phi=L_x^2*phi`, `Psi=L_y^2*psi`. Green reconstruction consumes `Phi/Psi` directly, so it applies no additional `L^2` factor. Physical symmetric preserves `d=p-q` with `phi=0.5*(rhs+d)`, `psi=0.5*(rhs-d)`. Retired complex `response_space`, `response_preconditioned`, `symmetric`, `smooth_mask`, and `geometry_weighted` configs still fail fast; unversioned and v5-or-older complex CouplingNet checkpoints are rejected, while GreenNet checkpoints remain reusable.
  ```json
  "balance_projection": {
    "enabled": true,
    "mode": "physical_symmetric"
  }
  ```
- Complex optional column-diagonal Green-response projection: set `mode="column_diagonal_green_response"` to distribute the raw physical balance residual `r=rhs-p-q` using each source point's downstream solution-response cost. For direction `s`, production reconstruction defines `H_s=K_s W_s L_s^2` and the cached gain is `gamma_s^2=diag(H_s^T M_Omega H_s)` with `M_Omega=(hx*hy)I`. This is the squared norm of each source column, not a row/evaluation sensitivity. With `gx_bar=gamma_x^2+eps` and `gy_bar=gamma_y^2+eps`, the fixed tempered weight is `w_phi=sigmoid(alpha*(log(gy_bar)-log(gx_bar)))`, `w_psi=1-w_phi`; then `delta_phi=w_phi*r` and `delta_psi=w_psi*r`. `gain_exponent=0` gives the physical symmetric correction, while `gain_exponent=1` uses the legacy direct column-diagonal ratio. The default is `1.0`, so configs that omit the field preserve the existing numerical path. Intermediate values such as `0.25` temper gain anisotropy without adding a learnable parameter, sample-dependent gate, row method, or full-Gram solve. The frozen GreenNet context and fixed weights are built segment-by-segment once per trainer/evaluator instance. Artifact export writes the exponent, run-level gains, and weights to `data/column_diagonal_green_response_fields.npz`, selected-sample correction diagnostics to `data/selected_raw_arrays.npz`, and gain/weight Plotly figures under `figures/balance_projection/`. Exponent comparisons require paired retraining from the same initialization and data; changing alpha only at export time is not a fair comparison. See `docs/complex_column_diagonal_green_response_projection.md` for the full contract.
- Complex optional symmetric-tangent Green-response projection: set `mode="symmetric_tangent_green_response"` to start from the exact-balanced symmetric pair `p_tilde=(rhs+p-q)/2`, `q_tilde=(rhs-p+q)/2`. With `m0=H_x p_tilde-H_y q_tilde`, the reference-free gradient is `g=(H_x+H_y)^T M_Omega m0`, and the cached Jacobi denominator is `D=gamma_x^2+gamma_y^2+(relative_lambda+denominator_relative_eps)*mean(gamma_x^2+gamma_y^2)`. `subspace_dimension` defaults to `1`, preserving the existing fixed or closed-loop scalar line-search path exactly. The backward-compatible `eta_strategy="fixed"` uses `delta=-eta*g/D`; `eta_strategy="closed_loop_exact_line_search"` forms `z=g/D`, `v=(H_x+H_y)z`, computes sample-wise `eta_star=(g^T z)/(<v,v>_M+eps_sample)`, and applies `eta_applied=min(eta_star,eta_cap)`. Here `eta` is the final K=1 safety cap, not a learned parameter. When the CouplingNet LR schedule is enabled, the K=1 training cap uses a half-cosine rise over the same effective warmup epochs and then remains at `eta`; LR cosine decay is not reused. Validation, checkpoint selection, standalone evaluation, and artifact export use the final K=1 cap. All tangent paths use `phi=p_tilde+delta`, `psi=q_tilde-delta`, preserve `phi+psi=rhs`, and use no reference target, row norm, global response matrix, full-Gram matrix, or linear solve. Frozen segment-local response blocks are built once and reused. For K=1, `eta=0` is exactly the physical-symmetric ablation.
- A hypothetical full-Gram alternative would solve the per-sample normal equation `A*delta_b=-g_b`, with `A=(H_x+H_y)^T M_Omega (H_x+H_y)`. For fixed geometry, coefficients, and GreenNet, `A` is shared: it can be assembled and factorized once, then applied to every current or future sample through a different right-hand side. This gives the exact discrete least-squares minimizer of the learned response mismatch, not an exact unavailable target directional split. It also turns projection into a solver-in-the-loop and can make the CouplingNet proposal irrelevant when `A` is nonsingular, so production deliberately retains the matrix-free one-step Jacobi path.
- The opt-in `subspace_dimension` accepts `2`, `3`, or `4` for a nested fixed-rank matrix-free tangent correction. The first two directions retain the existing K=2 arithmetic exactly: `z0=D^-1*g0`, exact scalar `c0`, residual-preconditioned `z1_raw`, response orthogonalization against `z0`, and exact scalar `c1`. K=3 and K=4 repeatedly form `z_k_raw=D^-1*g_{k-1}`, apply two-pass modified Gram-Schmidt against every previous response direction, and compute one sample-wise exact scalar `c_k`. The final update is `delta=-sum_k c_k*z_k`; it requires no global response matrix, dense `K x K` solve, or reference target. A numerically degenerate new response direction is deactivated with `c_k=0`, so the result falls back to the preceding nested subspace. Because each active subspace contains its predecessor, adjacent response cost is checked not to increase beyond float64 tolerance. Every K>=2 path deliberately ignores `eta`, the tangent eta cap, and the tangent eta warmup schedule; those fields remain in the shared schema only for K=1 compatibility. Coefficients are deterministic, sample-wise, differentiable, and are not model parameters. K=2, K=3, and K=4 remain opt-in paired-training choices rather than defaults.
  ```json
  "balance_projection": {
    "enabled": true,
    "mode": "symmetric_tangent_green_response",
    "symmetric_tangent_green_response": {
      "subspace_dimension": 2,
      "eta": 0.01,
      "eta_strategy": "closed_loop_exact_line_search",
      "line_search_relative_eps": 1e-12,
      "relative_lambda": 0.01,
      "denominator_relative_eps": 1e-12
    }
  }
  ```
  `configs/complex_coupling_soap_tangent.json` is the K=1 SOAP adaptive-tangent experiment config. `configs/complex_coupling_soap_tangent_k2_pentagram.json`, `configs/complex_coupling_soap_tangent_k3_pentagram.json`, and `configs/complex_coupling_soap_tangent_k4_pentagram.json` preserve the same Pentagram pilot conditions and change only the production tangent dimension. Existing canonical, column-diagonal, and user experiment configs remain unchanged. K=1 logs/CSV retain scheduled-cap and eta statistics. K>=2 logs/CSV instead record every `c_k`, direction activity, nested response costs, adjacent cost ratios, response orthogonality, and explicitly report that eta scheduling is not applicable. Selected artifact NPZ files store stacked source directions, directional responses, response directions, coefficients, active masks, intermediate deltas/mismatches/costs, the response Gram matrix, final residual gradient, and final mismatch; legacy K2 aliases remain available. Every dimension requires paired retraining for a training comparison, while frozen-checkpoint audits remain diagnostic evidence.
- Complex optional source-normalized post-line-search stationarity: `coupling_training.post_line_search_stationarity.enabled=true` is available only with symmetric-tangent `eta_strategy="closed_loop_exact_line_search"`. For K=1, let `S=H_x+H_y`, `A=S^T M_Omega S`, `g=S^T M_Omega m0`, `z=D^-1 g`, and `r_stat=g-eta_star*A*z`. For K>=2, stationarity reuses the final nested-subspace residual gradient `r_stat=S^T M_Omega m_K` already computed by the projection. In both cases the optimized per-sample regularizer is `(r_stat^T D^-1 r_stat)/(E_f+eps)`, where `E_f=||H_x(f/2)||_M^2+||H_y(f/2)||_M^2`; its fixed `weight` adds the batch mean to the reference-free objective. The initial-gradient-relative ratio `(r_stat^T D^-1 r_stat)/(g^T D^-1 g+eps)` remains the diagnostic `tangent_post_line_search_stationarity_ratio` and is not optimized. K>=2 needs no additional eta or adjoint action because the final residual gradient is reused. Logs and CSV also report `tangent_post_line_search_stationarity_source_normalized`, `tangent_stationarity_initial_source_ratio`, and `tangent_source_response_energy`. The option defaults to disabled and does not change model or checkpoint tensor keys.
  ```json
  "post_line_search_stationarity": {
    "enabled": true,
    "weight": 0.001,
    "eps": 1e-12
  }
  ```
  Best-energy selection remains based on `loss_energy_optimized`; best-physics selection uses the stationarity-augmented total loss. The stationarity term measures the learned Green-response surrogate and therefore complements rather than replaces canonical energy.
- Complex optional response-trust loss: `coupling_training.response_trust.enabled=true` is available only with symmetric-tangent `eta_strategy="closed_loop_exact_line_search"` and may be enabled together with source-normalized stationarity. Let `m_pre=H_x*p_tilde-H_y*q_tilde`, let `delta` be the actual K=1 capped correction or final K>=2 subspace correction, and let `m_post=m_pre+(H_x+H_y)*delta`. Per sample, response trust evaluates `||m_post||_M^2/(E_f+eps) + trust_weight*||(H_x+H_y)delta||_M^2/(E_f+eps)`, with `trust_weight=0.01` by default. This directly penalizes the applied post-correction mismatch and discourages a large response correction without reading `sol` or target `phi/psi`. Response trust and stationarity share one `H_x(f/2),H_y(f/2)` source normalization. K=1 stationarity uses uncapped `eta_star` while response trust uses the capped applied correction; K>=2 uses its final `m_K`, correction response `m_K-m0`, and post-K residual gradient, with no eta semantics. Logs, CSV, and artifacts record both components and the shared source-response provenance.
  ```json
  "post_line_search_stationarity": {"enabled": false},
  "response_trust": {
    "enabled": true,
    "weight": 0.001,
    "trust_weight": 0.01,
    "eps": 1e-12
  }
  ```
  The block defaults to disabled, so existing training and checkpoint tensors are unchanged. A conservative first paired ablation can use `weight=1e-3`; this is an experiment setting rather than the dataclass default. `configs/complex_coupling_soap_tangent_response_trust_stationarity.json` is the Pentagram combined pilot with response-trust weight `1e-3`, stationarity weight `1e-4`, and `trust_weight=0.1`. Best-energy selection remains on `loss_energy_optimized`, while best-physics selection uses the complete auxiliary-augmented total loss.
- Pentagram response-objective comparison: `checkpoints/pentagram/coupling4`,
  `coupling5`, and `coupling6` use the same 4,800/300 indexed-GP source split,
  tangent projection, boundary-off canonical energy, and weak-residual final
  reconstruction. Their actual response-trust weights are `0.001`, `1.0`, and
  `0.1`; only `coupling6` optimizes source-normalized stationarity with weight
  `0.01`. On the common 100-sample test set at best-energy checkpoints, mean
  `rel_sol` is `3.103%`, `3.281%`, and `2.691%`, while mean `rel_flux` is
  `38.603%`, `46.873%`, and `47.118%`. `coupling6` lowers source-normalized
  stationarity by `70.7%` versus `coupling4` and improves `rel_sol` by `0.412`
  percentage points, but its flux error rises by `8.514` points. The matched
  causal stationarity ablation is still missing because response-trust weight
  changes between runs; `coupling4` also ran on CUDA while `coupling5/6` ran on
  CPU. See
  `checkpoints/pentagram/coupling4_6_comparison/analysis_report.md` for paired
  bootstrap intervals, checkpoint-selection effects, sample-15 spatial
  diagnostics, CSV summaries, and Plotly figures.
- Pentagram six-run directional comparison: the common best-energy checkpoints
  from `coupling` through `coupling6` were re-evaluated on the same 100 test
  samples, including full-test `u_phi` and `u_psi` errors that are absent from
  the standard per-sample artifact CSV. The re-evaluated `rel_sol/rel_flux`
  agree with every stored artifact to at most `3.1e-15`. Mean
  `rel_sol/rel_u_phi/rel_u_psi/rel_flux` is respectively
  `2.691%/5.142%/4.869%/47.118%` for `coupling6`; the best individual values are
  `coupling6/coupling/coupling2/coupling4`. Thus the joint objective gives the
  best common reconstruction and tail error but not the best x-direction
  reconstruction or physical directional-source fidelity. The local weak-
  residual final blend improves equal mean on 100/100 `coupling6` samples.
  See `checkpoints/pentagram/coupling1_6_comparison/analysis_report.md` for the
  common-checkpoint protocol, directional tables, sample-paired intervals,
  training trends, CSVs, and Plotly figures.
- Pentagram `coupling7` trust-strength comparison: `coupling7` changes only the
  saved `response_trust.trust_weight` from `0.01` to `0.1` relative to
  `coupling6`; outer trust weight `0.1`, source-normalized stationarity weight
  `0.01`, data, optimizer, tangent projection and final reconstruction remain
  the same. On the shared 100-sample best-energy evaluation, correction response
  ratio decreases `69.2%` and tangent-delta RMS decreases `35.2%`, so the
  stronger inner trust does reduce correction dependence. Mean `rel_sol`
  changes only `2.691% -> 2.706%`, while p90 improves `4.656% -> 4.195%` and
  maximum improves `9.144% -> 9.012%`. The directional tradeoff is unfavorable:
  mean `rel_u_phi/rel_u_psi/rel_flux` changes from
  `5.142%/4.869%/47.118%` to `5.341%/5.152%/56.570%`, and flux is worse on
  99/100 samples. Treat `coupling7` as a tail-robust but over-constrained
  correction ablation, not as a replacement for `coupling6`. The saved configs
  do not fix model initialization seed, so this remains a one-run observational
  comparison despite the one-field config diff. See
  `checkpoints/pentagram/coupling1_7_comparison/analysis_report.md` for full
  directional tables, paired bootstrap intervals, tangent diagnostics and
  reproducible Plotly/CSV outputs.
- Pentagram `coupling8` zero-inner-trust comparison: `coupling8` differs from
  `coupling6/7` only in saved `response_trust.trust_weight=0`, versus
  `0.01/0.1`. The post-response mismatch term remains active; only the explicit
  correction-response penalty is removed. On the shared 100-sample best-energy
  evaluation, `coupling8` has the lowest eight-run mean/median `rel_sol`
  (`2.678%/2.114%`), lowest mean `rel_u_psi` (`4.832%`), and lowest split
  mismatch (`7.886%`). Versus `coupling6`, mean `rel_sol` improves only `0.46%`
  relative and its paired interval includes zero, while `rel_u_phi` and split
  mismatch improve `2.32%` and `2.09%`. Removing inner trust raises correction
  response ratio `33.6%` and tangent-delta RMS `13.6%`; the maximum `rel_sol`
  worsens `9.144% -> 10.317%`. Thus zero trust improves distribution center and
  directional balance but gives up correction suppression and worst-sample
  robustness. The model initialization seed is not recorded, so a fixed-seed
  `0/0.005/0.01` sweep is required before selecting a default. See
  `checkpoints/pentagram/coupling1_8_comparison/analysis_report.md` for all eight
  runs, paired intervals, training-component trends, CSVs, and Plotly figures.
  The current evidence favors a functional role split: the network supplies a
  balance-plane proposal, while the deterministic tangent step removes response
  mismatch. Inner trust is only a proxy that asks this correction to stay
  small; it does not directly measure distance from an unavailable ideal
  directional split. Keep gradients through the tangent step even when this
  correction-size proxy is disabled.
- Pentagram `coupling9` trained-K=2 comparison: `coupling9` preserves the
  `coupling8` dataset, SOAP optimizer, boundary-off canonical objective,
  response-trust outer weight `0.1`, inner trust `0`, source-normalized
  stationarity weight `0.01`, and local weak-residual reconstruction. The only
  saved-config difference is
  `symmetric_tangent_green_response.subspace_dimension=2`. On the shared 100
  sample best-energy protocol, mean
  `rel_sol/rel_u_phi/rel_u_psi/split_mismatch/rel_flux` changes from
  `2.678%/5.022%/4.832%/7.886%/46.558%` to
  `1.590%/3.289%/2.711%/4.740%/39.658%`. `rel_sol`, `rel_u_psi`, and split
  mismatch improve on all 100 samples; median/p90/max `rel_sol` falls from
  `2.114%/4.643%/10.317%` to `1.297%/2.572%/5.104%`. Trained K=2 also improves
  mean `rel_sol` by `20.00%` and optimized energy by `39.72%` relative to the
  frozen `coupling8+K2` post-hoc result. The tradeoff is stronger deterministic
  correction: tangent-delta RMS rises `102.91%` and correction/symmetric-pair
  norm rises `89.67%` versus trained K=1. Interpret this as successful
  network/correction role separation, not recovery of the target directional
  source split; mean `rel_flux` remains about `40%`. The model initialization
  seed is not saved, so K=2 remains opt-in pending a same-seed multi-run
  comparison. Reproducible CSV, Plotly, JSON, and the full report are under
  `checkpoints/pentagram/coupling1_9_comparison/`.
- Pentagram trained-subspace comparison: `coupling8/9/10/11` use production
  `K=1/2/3/4`, respectively. On the shared 100-sample best-energy protocol,
  mean `rel_sol` is `2.678%/1.590%/1.234%/1.112%`, mean
  `rel_u_phi` is `5.022%/3.289%/2.552%/2.394%`, mean `rel_u_psi` is
  `4.832%/2.711%/2.120%/1.865%`, and mean `rel_flux` is
  `46.558%/39.658%/35.261%/31.930%`. The adjacent `rel_sol` reductions are
  `40.64%`, `22.39%`, and `9.87%`, while the isolated CPU forward+backward
  tangent/auxiliary block costs are `141.4/211.8/282.2/361.8 ms` for
  `K=1/2/3/4`. K=4 has the best global and tail metrics, but K=3 is the current
  cost-quality knee: K3-to-K4 improves optimized energy by only `4.20%`, the
  transition solution-error jump by only `0.75%`, and slightly worsens the
  split transition jump, while adding `28.2%` to the isolated block cost.
  Original training wall times are not directly comparable because `coupling10`
  used `cuda:1` on a different host and the other runs used CPU. Model
  initialization seed is also not recorded, so this ranking requires a
  same-device, fixed-seed multi-run confirmation before changing a default.
  The full protocol and tables are in
  `checkpoints/pentagram/coupling8_11_k_comparison/analysis_report.md`.
  ```json
  "balance_projection": {
    "enabled": true,
    "mode": "column_diagonal_green_response",
    "column_diagonal_green_response": {
      "gain_squared_eps": 1e-12,
      "gain_exponent": 0.25
    }
  }
  ```
  A frozen-checkpoint response audit can compare several fixed exponents on the
  same raw directional response. It builds the column gains once, reconstructs
  each correction with the existing GreenNet, and reports both the diagonal
  surrogate
  `sum(gx_bar*delta_phi^2 + gy_bar*delta_psi^2)` and the actual learned-Green
  response cost
  `(hx*hy)*(||H_x delta_phi||_2^2 + ||H_y delta_psi||_2^2)`. It also separates
  source-correction jumps from reconstructed correction jumps on cross-axis
  transition edges. No row norm, full Gram matrix, or global solve is used.
  Reference `sol/phi/psi` appears only in optional evaluation metrics.

  ```bash
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python \
    cli/audit_complex_projection_response.py \
    --config checkpoints/annulus_CDR/coupling7/config_used.json \
    --coupling-checkpoint \
      checkpoints/annulus_CDR/coupling7/complex_coupling_model_best_energy.safetensors \
    --green-checkpoint checkpoints/annulus_CDR/green/model.safetensors \
    --alphas 0 0.25 0.5 1
  ```

  The default output directory is
  `<coupling-checkpoint-parent>/projection_response_posthoc_audit`. It contains
  `summary.json`, a per-sample CSV, selected raw NPZ data, editable Plotly
  figures, and `diagnosis_report.md`. Because the checkpoint was trained under
  one configured exponent, solution/flux accuracy at other post-hoc exponents
  measures frozen-network compensation, not a fair end-to-end training
  comparison. Paired retraining remains necessary for model selection.
  The same run also writes
  `metrics/per_sample_directional_candidate_audit.csv` and selected candidate
  figures. These compare raw physical proposals `(p,q)`, the exact-balanced
  raw-difference pair
  `p_tilde=0.5*(rhs+p-q), q_tilde=0.5*(rhs-p+q)`, and the configured projected
  `(phi,psi)` against optional sample directional targets. Reported raw balance
  defect and correction-to-projected ratios test whether `(p,q)` behave as
  physical directional candidates or only as projection-dependent latent
  coordinates. Directional targets remain evaluation-only.

  A second frozen-checkpoint diagnostic starts from the symmetric-balanced
  candidate and tests a matrix-free Green-response tangent correction:
  `m0=H_x*p_tilde-H_y*q_tilde`,
  `g=(H_x+H_y)^T*M_Omega*m0`,
  `delta=-eta*D^{-1}*g`, with
  `phi=p_tilde+delta` and `psi=q_tilde-delta`. The diagonal damping is
  `D=gamma_x^2+gamma_y^2+(lambda_rel+eps)*mean(gamma_x^2+gamma_y^2)`.
  This preserves `phi+psi=rhs` exactly, uses segment-local Green operator and
  transpose actions, and performs no global matrix assembly or solve. The
  eta/lambda sweep reports response mismatch and canonical energy as primary
  reference-free metrics; `rel_sol` and `rel_flux` remain evaluation-only.
  Add `--closed-loop` to include the production-equivalent sample-wise exact
  line search on the same frozen raw output. Its defaults are final cap `0.01`,
  `lambda_rel=0.01`, and relative epsilon `1e-12`; override them with
  `--closed-loop-eta-cap`, `--closed-loop-relative-lambda`, and
  `--line-search-relative-eps`. The post-hoc path uses the final evaluation cap,
  not the training warmup schedule, and records each sample's `eta_star`,
  applied eta, cap hit, and line-search numerator/denominator in CSV and NPZ.
  For a shared-context cap sweep, pass
  `--closed-loop-eta-caps 0.01 0.0125 0.015 0.02 inf`; `inf` (or `uncapped`)
  applies `eta_star` directly. The response operator, tangent direction, and
  sample-wise `eta_star` are built once and reused for every listed cap. The
  audit then writes `figures/aggregate/closed_loop_cap_sweep.*` in addition to
  the per-sample CSV and selected raw arrays.

  ```bash
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python \
    cli/audit_symmetric_tangent_response.py \
    --config checkpoints/annulus_CDR/coupling7/config_used.json \
    --coupling-checkpoint \
      checkpoints/annulus_CDR/coupling7/complex_coupling_model_best_energy.safetensors \
    --green-checkpoint checkpoints/annulus_CDR/green/model.safetensors \
    --closed-loop
  ```

  The default output directory is
  `<coupling-checkpoint-parent>/symmetric_tangent_response_audit`. This is a
  post-hoc ablation of fixed and optional closed-loop tangent steps. It does not
  modify the training
  projection or establish the result of paired retraining. The diagnostic
  accepts checkpoints trained with either `physical_symmetric` or
  `column_diagonal_green_response`. For a symmetric-trained checkpoint the
  configured baseline is the symmetric pair itself; for a column-trained
  checkpoint the configured column projection is included as a separate
  baseline. Cross-checkpoint comparisons should use response mismatch,
  canonical energy, equal-mean `rel_sol`, and directional flux metrics because
  optional cross-axis reconstruction settings may differ. The summary always
  records both the configured reconstruction `rel_sol` and
  `rel_sol_equal_mean`, so the same frozen candidates can be audited with
  `local_weak_residual_reliability` enabled in a diagnostic config without
  changing the checkpoint or directional projection.

  A dedicated frozen-checkpoint audit compares symmetric balance, the configured
  capped `K=1`, uncapped exact-line-search `K=1`, and nested matrix-free
  unconstrained subspaces through a selected `K<=4` on identical raw output.
  `--max-subspace-dimension` accepts `2`, `3`, or `4` and defaults to `2`, so the
  existing K1/K2 command and outputs are preserved. K3/K4 use two-pass modified
  Gram-Schmidt in response space, sample-wise scalar coefficients, and no global
  matrix or dense solve. The audit reports response mismatch, optimized and full
  canonical energy, `rel_sol`, directional `rel_u_phi/rel_u_psi`, `rel_flux`,
  correction norm, response orthogonality, p90/p95/max tails, and paired sample
  counts.

  ```bash
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python \
    cli/audit_tangent_subspace.py \
    --config checkpoints/pentagram/coupling8/config_used.json \
    --coupling-checkpoint \
      checkpoints/pentagram/coupling8/complex_coupling_model_best_energy.safetensors \
    --green-checkpoint checkpoints/pentagram/green/model.safetensors \
    --outdir checkpoints/pentagram/coupling8/tangent_subspace_k1_k2_audit \
    --device cpu --batch-size 10
  ```

  On the 100-sample Pentagram `coupling8` test set, `K=2` versus production
  `K=1` changes mean response mismatch by `-33.028%`, optimized bulk energy by
  `-16.246%`, `rel_sol` by `-25.794%`, `rel_u_phi` by `-17.891%`, and
  `rel_u_psi` by `-22.385%`; response mismatch and all three solution metrics
  improve on `100/100` samples. Mean `rel_flux` changes only `-0.139%` and
  improves on `72/100`, while relative correction norm increases `13.211%`.
  Only two samples hit the production `eta=0.015` cap, so capped and uncapped
  `K=1` are nearly identical: the gain comes from the second response direction,
  not cap removal. The p95/max `rel_sol` falls from `5.131%/10.317%` to
  `3.820%/6.411%`. These are strong frozen-checkpoint findings, but `K=2` was
  not used during that checkpoint's training. The production implementation is
  therefore opt-in and requires paired retraining before any default change.
  Outputs are under
  `checkpoints/pentagram/coupling8/tangent_subspace_k1_k2_audit/`.

  The same audit was extended through K4 on the frozen Pentagram `coupling9`
  best-energy checkpoint, which was trained with K2:

  ```bash
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python \
    cli/audit_tangent_subspace.py \
    --config checkpoints/pentagram/coupling9/config_used.json \
    --coupling-checkpoint \
      checkpoints/pentagram/coupling9/complex_coupling_model_best_energy.safetensors \
    --green-checkpoint checkpoints/pentagram/green/model.safetensors \
    --outdir checkpoints/pentagram/coupling9/tangent_subspace_k1_k4_audit \
    --device cpu --batch-size 10 --max-subspace-dimension 4
  ```

  On the same frozen raw output, K3 versus K2 changes mean response cost by
  `-22.609%`, optimized energy by `-12.942%`, and
  `rel_sol/rel_u_phi/rel_u_psi` by `-14.834%/-11.009%/-15.027%`. K4 versus K3
  changes them by `-21.252%`, `-5.064%`, and
  `-9.371%/-12.218%/-8.560%`. Overall K4 versus K2 improves mean
  `rel_sol/rel_u_phi/rel_u_psi` by `22.815%/21.883%/22.300%` on `100/100`
  samples and lowers optimized energy by `17.351%`, while correction/symmetric
  pair norm grows only `3.147%`. K2-to-K4 `rel_sol` p95/max falls from
  `2.866%/5.104%` to `2.147%/3.471%`. All four directions are active on all
  samples, adjacent response cost is monotone on `100/100`, maximum normalized
  response non-orthogonality is below `4.8e-13`, and balance error is below
  `6.3e-15`. Mean `rel_flux` improves only `0.309%`, so the larger subspace still
  primarily improves learned directional-response alignment rather than target
  directional-source recovery. This is strong evidence for a paired K3
  retraining experiment and useful evidence for K4, but it is not a trained-K3
  or trained-K4 result. Reproducible outputs are under
  `checkpoints/pentagram/coupling9/tangent_subspace_k1_k4_audit/`.
- Paired fixed-vs-closed-loop CDR result: `annulus_CDR/coupling8` and
  `annulus_CDR/coupling9` have identical saved configs except that coupling8
  uses fixed `eta=0.01`, while coupling9 uses sample-wise
  `closed_loop_exact_line_search` with final cap `0.01`. The best validation
  canonical-energy epochs are 46 and 56, respectively. On their paired
  best-energy checkpoints and the same 50-sample test set, coupling9 changes
  mean response mismatch by `-0.735%`, canonical energy by `+0.680%`, mean
  `rel_sol` from `2.970490%` to `2.998864%`, split-transition jump RMS by
  `+3.532%`, and mean `rel_flux` from `13.676697%` to `13.297429%`. The flux
  gain comes from regular points (`-3.405%`); transition flux error increases
  by `+1.527%`. In coupling9, mean `eta_star=0.011414` and 68% of samples hit
  the `0.01` cap. Frozen-checkpoint decomposition confirms that closed-loop
  reduces its intended response mismatch but increases canonical and
  transition energies relative to fixed eta. Therefore fixed `eta=0.01`
  remains the preferred tangent baseline for energy/solution/transition work;
  closed-loop remains an optional ablation. The reproducible report, paired
  CSVs, and Plotly figures are under
  `checkpoints/annulus_CDR/coupling8_vs_coupling9_analysis/`.
- CDR `coupling10` boundary-off result: the run combines closed-loop tangent
  correction with final cap `0.015`, local weak-residual reconstruction, and
  `canonical_energy.boundary_weight=0.0`. Its epoch-43 best-bulk-energy
  checkpoint obtains 50-sample mean `rel_sol=2.494922%`, the lowest archived
  CDR result, versus `2.970490%` for coupling8 and `2.998864%` for coupling9.
  Relative to coupling9, bulk energy decreases by `31.298%` and `rel_sol` by
  `16.804%`, but mean `rel_flux` increases from `13.297429%` to `15.266857%`
  and boundary energy increases by `22.806%`. The boundary median is slightly
  lower while its maximum is about three times coupling9, so boundary-off
  primarily introduces a heavy outlier tail. The final epoch-100 checkpoint is
  overtrained: versus the best checkpoint, test `rel_sol` is `2.764%` worse and
  `rel_flux` is `83.659%` worse. Local weak reconstruction improves the best
  checkpoint's equal-mean `rel_sol` by `6.652%` and wins on all 50 samples, but
  directional transition seams remain visible. Coupling10 is not a clean
  boundary-weight ablation because it also changes the tangent cap from `0.01`
  to `0.015`; causal attribution requires a same-seed `boundary_weight=0/1`
  pair at fixed cap. Full analysis is in
  `checkpoints/annulus_CDR/coupling10/analysis_report.md`.
- Complex optional pre-projection fusion: set `coupling_model.pre_projection_fusion.enabled=true` to insert one small nonlinear MLP between the two axis-conditioned raw responses and the selected physical balance projection. The block first forms `p0=P0/L_x^2`, `q0=Q0/L_y^2`, `d_base=p0-q0`, and `A=sqrt((A_x^2+A_y^2)/2)`. Its only inputs are the normalized physical values `z=[d_base/A_safe,rhs/A_safe]`, where `A_safe=max(A,eps)`. The backward-compatible default `mode="residual"` uses `d_fused=d_base+A_safe*h_theta(z)` and therefore retains the fixed identity skip. The opt-in `mode="absolute"` uses `d_fused=A_safe*h_theta(z)`, so the MLP predicts the complete fused difference without adding `d_base`. Both modes construct `phi_pre=0.5*(rhs+d_fused)` and `psi_pre=0.5*(rhs-d_fused)`, preserving exact source balance before the configured physical projection. There is no learned linear branch, learned gate, convex combination, or direct coordinate/geometry/line-length input.
  `final_layer_init_scale` scales the final `torch.nn.Linear` layer's standard initialized weight and bias: `0.0` gives zero initialization and `1.0` leaves the standard initialization unchanged. With scale zero, residual mode starts from the disabled/base path, while absolute mode starts from the symmetric split `d_fused=0`; these are intentionally different initial conditions. Zero source amplitude forces the physical MLP output and fused difference to zero in both modes. This complex-only option adds no reference-target loss and leaves output contract v6, physical symmetric projection, reconstruction, GreenNet, and NPZ schemas unchanged. Existing unmarked v6 single-MLP checkpoints are interpreted as residual mode for compatibility, but marked residual and absolute checkpoints cannot be cross-loaded. Checkpoints trained with the retired split linear/nonlinear fuser remain incompatible and require retraining. `configs/complex_coupling_soap_absolute.json` is the paired SOAP example for the absolute-mode experiment.
  ```json
  "pre_projection_fusion": {
    "enabled": true,
    "mode": "absolute",
    "hidden_dim": 16,
    "depth": 1,
    "eps": 1e-12,
    "final_layer_init_scale": 0.0
  }
  ```
- Complex canonical split energy: v6 evaluates the full-domain physical audit energy for `r=u_phi-u_psi`. Every same-segment `x_edges` and `y_edges` contribution is summed once with the physical spacing, `hx*hy` area, and arithmetic face diffusion coefficient. The boundary part adds the missing P1 edge from each endpoint of every connected x/y segment to that segment's nearest represented interior node, with hard-zero endpoint value and contribution `a_i * r_i^2 * h_perp / d_endpoint`. The always-reported audit metric is `loss_energy_consistency = loss_energy_bulk + loss_energy_boundary`; the actual base objective is `loss_energy_optimized = loss_energy_bulk + boundary_weight * loss_energy_boundary`. The complex-only `coupling_training.canonical_energy.boundary_weight` defaults to `1.0`, exactly preserving the canonical objective. Setting it to `0.0` is a boundary-off ablation that optimizes only the bulk term while still reporting canonical and boundary total/x/y diagnostics. Edges are not classified as regular or transition, and this positive energy uses no reference target.
- Retired gluing/carrier objective: `coupling_training.admissibility_gluing`, global self-trace gluing, and transition-only cross-axis carriers have been removed. Null-space analysis showed that the carrier is unnecessary once every connected-segment endpoint participates in the canonical boundary energy. Old configs containing `admissibility_gluing` fail fast instead of silently changing the objective.
- Complex energy null-space diagnostic: run `PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/analyze_complex_energy_nullspace.py --geometry data/geometry/annulus_02_05_1_128.npz --outdir checkpoints/Annulus_poisson/energy_nullspace_analysis`. It compares bulk gradient nullity with the canonical all-segment boundary anchors and writes `summary.json`, `nullspace_stages.csv`, `analysis_report.md`, and a mirrored Rich/file log. For the current 10,788-point Annulus geometry, the connected bulk graph has one constant null mode; the 712 general endpoint anchors reduce nullity from one to zero, so no carrier objective is required for residual coercivity.
- Persistent-jump refinement audit: run `cli/audit_complex_energy_refinement.py` with at least three annulus geometry files ordered arbitrarily through `--geometries`. The audit uses a fixed-amplitude interior patch jump that remains zero near the physical boundary, measures bulk/boundary/canonical energy, fits the log-log slope, and verifies both a default slope range `[-1.25,-0.75]` and stability of `h*E_h`. The `inner_radius=0.2`, `outer_radius=0.5`, `h={1/32,1/64,1/128}` audit measured canonical energies `{28,60,124}`, slope `-1.0734`, and relative `h*E_h` spread `0.1011`.
- Complex relative split consistency: `coupling_training.relative_split_consistency.enabled=true` replaces the raw optimized-energy split objective by a per-sample, source-normalized objective. Its numerator is `E_optimized + mass_weight * D_ref^-2 * h_x*h_y*sum((u_phi-u_psi)^2)`, where `E_optimized=E_bulk+boundary_weight*E_boundary`, and its denominator is `h_x*h_y*sum(rhs^2)+eps`, with `D_ref` equal to the larger global geometry extent. The value/mass term penalizes constant and low-frequency split mismatch that derivative energy cannot see. `weight` scales the complete relative split objective. The option is complex-only, defaults to disabled in the dataclass, and never reads reference `sol/phi/psi`.
- Complex directional weak closure: `coupling_training.weak_operator_closure.enabled=true` adds a reference-free variational residual for the common prediction `u_pred=0.5*(u_phi+u_psi)`. Every connected axial segment is assembled with P1 nodal test functions, true boundary endpoints fixed to zero, physical element lengths, and coefficients evaluated directly at physical element midpoints. The x residual uses `a`, `bx`, and `c/2` against projected physical `phi`; the y residual uses `a`, `by`, and `c/2` against projected physical `psi`. Residuals are normalized by lumped nodal mass and source energy, then multiplied by `weight`. This follows the variational-residual and integration-by-parts principle used by [VPINNs](https://arxiv.org/abs/1912.00873), while retaining this repository's segment-wise P1 discretization.
- Complex total objective: `loss = loss_split_objective + loss_weak_operator_closure + loss_tangent_response_trust + loss_tangent_post_line_search_stationarity`, omitting each disabled optional term. Response trust and optimized stationarity are independent fixed-weight options and can run jointly; the former evaluates the capped applied correction and the latter evaluates the uncapped tangent-line stationarity residual. With relative split disabled, the split objective is `loss_energy_optimized`. The trainer and evaluator record the fixed `boundary_weight`, optimized energy, unweighted canonical energy, bulk, boundary total/x/y, and any enabled optional metrics. Boundary diagnostics remain available when the boundary weight is zero; transition-specific metrics are not generated.
- Canonical complex training config: `configs/complex_coupling.json` is the energy-only v6 configuration. It explicitly includes `pre_projection_fusion.enabled=false` as an architecture-ablation switch and intentionally omits the disabled `relative_split_consistency`, `weak_operator_closure`, `best_physics_checkpoint`, and `best_rel_sol_checkpoint` blocks; their dataclass defaults are disabled. It also omits the Green `training` section because `pipeline.run_green=false`, and omits a null `coupling_pretrained_path` so each run starts a new complex CouplingNet while reusing only the configured GreenNet checkpoint.
- Boundary-off paired experiment: `configs/complex_coupling_soap_tangent_boundary_off.json` differs from `configs/complex_coupling_soap_tangent.json` only by `coupling_training.canonical_energy.boundary_weight=0.0`. It is an ablation of possible overlap between tangent correction and endpoint anchoring, not a replacement for the production default. Remove the block or set the weight to `1.0` to restore the canonical objective without changing model or checkpoint tensor keys.
- Complex reference-free checkpointing: v6 requires `coupling_training.best_rel_sol_checkpoint.enabled=false`; `sol` and optional `phi/psi` targets remain detached evaluation metrics only. `best_energy_checkpoint.enabled=true` saves `complex_coupling_model_best_energy.safetensors` from the smallest validation `loss_energy_optimized`, so boundary-off also applies to model selection. `best_physics_checkpoint.enabled=true` independently saves `complex_coupling_model_best_physics.safetensors` from the smallest validation total reference-free `loss`, including optional relative-split, weak, response-trust, and stationarity terms. Reference targets do not affect gradients, scheduling, early stopping, or either checkpoint criterion.
- Complex fixed-source train/validation: `dataset.coupling_source.mode` selects deterministic source-only `"npz"` files or the runtime `"indexed_gp"` provider. Both use the shared separable squared-exponential GP core and `SeedSequence([base_seed, split_id, sample_index])`, with split IDs `train=0`, `valid=1`, `test=2`. Therefore a stored source and a runtime source with the same geometry, GP settings, split, and index are bitwise identical. Runtime GP covariance factors are built once per provider; individual samples are regenerated by index without caching the entire dataset. DataLoader shuffle and repeated epochs do not change sample identity.
- Explicit training reproducibility: every stage started by `cli/train.py` requires its own uint32 base seed. Use `training.seed` for GreenNet and `coupling_training.seed` for CouplingNet. SHA-256 namespaced sub-seeds isolate Green train data, Green validation data, model initialization, runtime/dropout, shuffled first-stage loaders, and Green LBFGS loaders. CouplingNet has independent model, runtime, and shuffled-loader streams, so running GreenNet first or changing the Green training sample count does not shift CouplingNet initialization. The CLI records all resolved sub-seeds in `config_used.json` and `training.log`; Green and complex Coupling artifact summaries reproduce the same provenance.
  ```json
  "training": {
    "seed": 0,
    "deterministic_algorithms": true
  },
  "coupling_training": {
    "seed": 0,
    "deterministic_algorithms": true
  }
  ```
- Training seed and source seed are separate controls. `dataset.coupling_source.indexed_gp.seed` fixes the source realization assigned to each `(split,index)`, while `coupling_training.seed` fixes CouplingNet initialization, shuffle, and runtime RNG. A controlled architecture or optimizer ablation must keep both values, data split, hardware, software stack, and all non-ablated settings fixed. Strict deterministic mode enables PyTorch deterministic algorithms, disables cuDNN benchmarking, and configures deterministic cuBLAS workspace behavior before CUDA initialization; unsupported nondeterministic operations fail instead of silently falling back. Cross-version or cross-hardware bitwise identity is not guaranteed. Safetensors checkpoints remain model-only and do not store optimizer or RNG state, so resuming at an intermediate epoch is not an exact continuation.
- Complex reference diagnostics: `dataset.reference_diagnostics.training` and `.validation` default to `true`, preserving existing full-reference NPZ behavior. Set both to `false` for source-only training. A disabled split reads only `rhs`, supplies shape-compatible internal zero placeholders, and omits `rel_sol` and `rel_flux` from console logs and `complex_training_metrics.csv`. The placeholders are not training targets. Predicted `u_phi/u_psi`, projected physical `phi/psi`, canonical energy, projection, and Green reconstruction remain active and reference-free. The test path is always loaded as full-reference NPZ regardless of the train/validation backend.
- Source-only NPZ generation does not require FEniCSx:
  ```bash
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_complex_sources.py \
    --geometry data/geometry/annulus_02_05_1_128.npz \
    --out data/complex_sources/annulus_gp \
    --num-train 4000 \
    --num-valid 200 \
    --lengthscale 0.2 \
    --amplitude 1.0 \
    --mean 0.0 \
    --seed 0
  ```
  The CLI writes `train/sample_XXXXXX.npz` and `valid/sample_XXXXXX.npz` with only a float64 full-grid `rhs`, zeros outside the domain, plus `generation_summary.json` and `make_complex_sources.log`. Existing files fail unless `--overwrite` is supplied; validation is enabled by default and can be disabled with `--no-validate`.
- Source-only NPZ config:
  ```json
  "coupling_source": {"mode": "npz"},
  "reference_diagnostics": {"training": false, "validation": false}
  ```
  Set `dataset.training_path` and `dataset.validation_path` to the generated split directories. Keep `dataset.test_path` pointed at the FEniCSx full-reference test split.
- Runtime indexed GP config:
  ```json
  "coupling_source": {
    "mode": "indexed_gp",
    "indexed_gp": {
      "num_train": 4000,
      "num_valid": 200,
      "seed": 0,
      "lengthscale": 0.2,
      "amplitude": 1.0,
      "mean": 0.0
    }
  },
  "reference_diagnostics": {"training": false, "validation": false}
  ```
  Omit `dataset.training_path` and `dataset.validation_path` in this mode. Keep `dataset.geometry_path` and the full-reference `dataset.test_path`. A positive validation count is required when best-energy or best-physics checkpointing is enabled.
- SOAP paired-pilot config: `configs/complex_coupling_soap.json` uses the fixed
  runtime `indexed_gp` backend with `num_train=4800`, `num_valid=400`,
  `seed=0`, `lengthscale=0.15`, `amplitude=1.0`, and `mean=0.0`. With
  `batch_size=400`, `epochs=125`, and `warmup_epochs=13`, the run keeps about
  1,500 optimizer steps and 156 warmup steps while reducing repeated exposure
  to each fixed source. Training and validation reference diagnostics are
  disabled; its full-reference `dataset.test_path` remains available for
  detached test evaluation.
- Fixed-source runs must be interpreted through the validation-selected
  best-energy checkpoint rather than the final epoch. In the exploratory
  `coupling11_2000_train` run, validation energy was lowest at epoch 133 and
  then increased by 5.22x while train energy continued to fall. The generated
  diagnostic bundle is under
  `checkpoints/Annulus_poisson/coupling11_2000_train/analysis/`. This run also
  changed `hidden_dim` from 256 to 384, so it is not a controlled source-count
  ablation.
- The completed `coupling12` Annulus run used 4,800 fixed indexed-GP train
  sources, 300 validation sources, the original 1,814,587-parameter model, and
  1,600 SOAP steps. Validation canonical energy reached its minimum
  `4.366951e-4` at epoch 65 and finished only 2.19% higher, so late overfitting
  was mild. Its best-energy checkpoint reduced mean test energy by 24.03%,
  `rel_sol` by 5.07%, and `rel_flux` by 9.37% relative to `coupling11`.
  Inner-radius transition seams nevertheless remain visible; the full report,
  CSV/JSON diagnostics, and Plotly figures are under
  `checkpoints/Annulus_poisson/coupling12/analysis/`.
- A same-checkpoint ablation of the trained `coupling12` legacy split
  linear/nonlinear pre-projection fuser
  confirms that it is functionally important: bypassing only the fuser raises
  mean test canonical energy from `4.050432e-4` to `2.568269e-3` (6.34x) and
  mean `rel_sol` from 5.602% to 12.578% (2.25x), with the enabled path better on
  all 50 test samples. Mean transition solution-error RMS falls by 58.94% with
  the fuser, but the transition/bulk ratio falls by only 3.65%, so the fuser
  reduces the seam amplitude without eliminating its structural concentration.
  Mean `rel_flux` improves by only 1.02%. This inference-time bypass measures
  the retired fuser's contribution inside the co-adapted checkpoint; an
  architecture claim still requires a separately trained disabled control.
  These values do not describe the current single residual MLP. The report and
  paired Plotly artifacts are under
  `checkpoints/Annulus_poisson/coupling12/pre_projection_fuser_ablation/`.
- Complex disabled features: `cross_consistency`, `smooth_mask`, `balance_loss`, `source_stencil_lift`, and `green_response_feature` are unit-square-only surfaces. Complex trainer/evaluator/artifact paths do not compute, log, serialize, or export cross-related metric keys or placeholder fields.
- CouplingNet coefficient terms: in the standard branch path (`source_stencil_lift.enabled=false`), `coupling_model.coefficient_terms` controls which operator coefficients enter the generic `branch_coefficient`. Unit-square mode concatenates enabled terms in `[a, b, c]` order from `diffusion`, `convection`, and `reaction`; complex mode expands enabled convection into `[b_primary, b_transverse]` while preserving the same config surface. The default `diffusion=true, convection=false, reaction=false` preserves the diffusion-only coefficient branch. If all three are false, CouplingNet uses a source-only pure Poisson branch path and skips `branch_coefficient`.
- CouplingNet branch fusion: set `coupling_model.branch_fusion.mode` to choose how branch features are combined before the trunk readout. The default `product` keeps the existing multiplicative fusion of source, coefficient, and optional transverse branch features. The experimental `product_fuser` mode concatenates the active branch features with their component-wise product and passes them through a learned fuser, preserving the product bias while allowing a learned final branch representation.
- CouplingNet optional source stencil lift: set `coupling_model.source_stencil_lift.enabled=true` to add input-side learned source and coefficient 5-point stencil encoders. It reconstructs canonical full grids from `rhs_raw` and `a_vals`, feeds normalized source `f` stencils to a source encoder and raw coefficient `a` stencils to a separate coefficient encoder, optionally normalizes the source lifted scalar field by interior RMS, and sends the two lifted fields through separate source and coefficient branch networks before multiplying their branch features. Set `coupling_model.source_stencil_lift.encoder_type` to `"mlp"` for nonlinear five-stencil encoders or `"linear"` for direct affine maps from each five-stencil to its scalar field. The coefficient lift keeps RMS output normalization by default with `coefficient_normalization="rms"`; opt into the bounded coefficient output `beta * tanh(r_coef)` with `coefficient_normalization="tanh"` and `coefficient_tanh_beta`. `b_vals` and `c_vals` are not part of this first coefficient encoder version, and the physical `rhs_raw`, `rhs_norm`, balance projection, losses, and evaluation targets remain unchanged.
- CouplingNet optional Green response feature: set `coupling_model.green_response_feature.enabled=true` to append the frozen axial Green response `G(rhs_tilde)` to the existing normalized source branch input, so the branch sees `[rhs_tilde, G(rhs_tilde)]`. The trainer and evaluator compute this feature from the current `green_kernel`; `CouplingNet` does not own the Green kernel and still uses `rhs_raw` for balance projection. This first version is axis-local only, has no separate normalization option, and cannot be enabled together with `source_stencil_lift`.
- CouplingNet optional trunk positional encoding: set `coupling_model.trunk_positional_encoding.enabled=true` to replace raw unit-square trunk coordinates with deterministic coordinate features. The default `mode="fourier"` appends axis-aligned Fourier features `[sin(2*pi*f*x), cos(2*pi*f*x), sin(2*pi*f*y), cos(2*pi*f*y)]` with log-spaced frequencies from `1` to `max_frequency`; the defaults `num_frequencies=4` and `max_frequency=8.0` give `[1, 2, 4, 8]`. Set `mode="boundary_algebraic"` to append Dirichlet/domain-aware algebraic features `[x(1-x), y(1-y), x*y, x^2, y^2, x(1-x)y(1-y)]`. Set `include_input=false` to drop raw `(x, y)` from either encoded trunk input. This feature is unit-square-only and cannot be enabled in complex geometry mode.
- CouplingNet optional shared axis-1D trunk: set `coupling_model.axis_1d_trunk={"enabled": true, "boundary_aware_modes": k}` to use one shared 1D trunk for both axes in unit-square mode. In this mode `phi` evaluates the shared trunk on `x`, `psi` evaluates the same trunk on `y`, and a separate transverse branch receives only boundary-aware features `Enc_k(t)=[sin(n*pi*t), cos(n*pi*t)]` for `n=1..k`; raw transverse coordinate `t` is not included. For `phi`, `t` is the fixed line coordinate `y`; for `psi`, `t` is the fixed line coordinate `x`. In complex v6 mode, the primary trunk is always segment-local 1D `t`, and `axis_1d_trunk.num_frequencies` / `max_frequency` control Fourier features of the globally normalized fixed-line transverse branch. Complex v6 additionally requires `axis_1d_trunk.transverse_trunk={"enabled": true, "fusion": "product"|"product_fuser", "length_context": true}` so the shared pointwise transverse trunk receives the cross-axis local coordinate and both pointwise segment lengths. The generic dataclass defaults remain backward compatible for unit-square configs, but complex v6 validation rejects a disabled transverse trunk or `length_context=false`.
- Balance projection: CouplingNet defaults to `coupling_model.balance_projection={"enabled": true, "mode": "symmetric", "mask": "quadratic"}`, the fixed interior `0.5/0.5` residual split. Set `mode="smooth_mask"` to use smooth transverse masks that preserve exact interior `phi + psi = f` while damping the raw difference mode near transverse boundaries. `balance_projection.mask` selects `"quadratic"` masks `m_phi=y(1-y)`, `m_psi=x(1-x)` or `"sin"` masks `m_phi=sin(pi*y)`, `m_psi=sin(pi*x)`; omitted masks and legacy string configs such as `balance_projection="smooth_mask"` still load as `"quadratic"`. For `"quadratic"`, `smooth_mask_normalize=true` scales the masks by `4`; `"sin"` is already normalized and ignores that scaling. For either mask, `coupling_model.smooth_mask_power` applies the mask exponent `p`, and `coupling_model.smooth_mask_diff_power` applies the difference-mode exponent `q` in `beta = 0.5 * (2 * alpha_soft)^q`; both default to `1.0`, which recovers the previous projection. Set `coupling_model.smooth_mask_diff_power_trainable=true` to learn only `q` as a bounded scalar with `smooth_mask_diff_power_min <= q <= smooth_mask_diff_power_max` (defaults `0.25` and `2.0`); this requires projection `enabled=true` and `mode="smooth_mask"`. Set `enabled=false` to return the raw CouplingNet `(phi, psi)` output without projection; `mode` and `mask` remain config metadata but are not used in the forward pass.
- Sine smooth-mask projection example: use `coupling_model.balance_projection={"enabled": true, "mode": "smooth_mask", "mask": "sin"}` to select `sin(pi*y)` / `sin(pi*x)` masks while keeping the same balance projection and `smooth_mask_power` / `smooth_mask_diff_power` controls.
- Coupling losses: Coupling training uses independently controlled losses under `coupling_training.losses`: represented-solution L2 consistency, energy consistency, cross consistency, `balance_loss`, and `symmetric_boundary_loss`. The nested loss config is the only supported schema. `balance_loss` penalizes the raw common-grid residual `f - phi - psi` and is allowed only when `coupling_model.balance_projection.enabled=false`; enabling it with projection on fails fast instead of being ignored. `symmetric_boundary_loss` trains the symmetric projection's raw difference mode at transverse boundaries by penalizing `(phi_raw - psi_raw) + f` on `phi` boundaries `y=0,1` and `(phi_raw - psi_raw) - f` on `psi` boundaries `x=0,1`. It is allowed only with `coupling_model.balance_projection.enabled=true`, `mode="symmetric"`, and without `source_stencil_lift` or `green_response_feature`.
- Coupling source-lift diagnostics: when the source stencil lift is enabled, training and validation log `source_lift_corr_g_f`, `source_lift_rel_diff_g_f`, and `source_lift_g_rms` to track how the learned interior source branch input compares with the normalized physical source.
- Coupling optimizer config: `coupling_training.learning_rate` and `coupling_training.weight_decay` control the main CouplingNet AdamW group. Set `coupling_training.source_stencil_lift_learning_rate` or `coupling_training.source_stencil_lift_weight_decay` to create a separate optimizer group for the input-side 5-stencil source encoder; omitted source values fall back to the main group values. If trainable smooth-mask `q` is enabled, it is excluded from the main group and placed in a separate `smooth_mask_diff_power` group with the main learning rate and zero weight decay. The shared LR schedule applies the same multiplicative factor to all groups.
- Complex CouplingNet SOAP optimizer: omit `coupling_training.optimizer` to preserve the existing AdamW defaults, or set `optimizer.name="soap"` as shown in `configs/complex_coupling_soap.json`. SOAP is vendored from the official repository at commit `a1e553530fde97d0e6b307d7c82ac6d38b072340`; attribution is in `THIRD_PARTY_NOTICES.md`. The supported SOAP options are `betas`, `eps`, `shampoo_beta`, `precondition_frequency`, `max_precondition_dim`, `merge_dims`, `precondition_1d`, `normalize_grads`, and `correct_bias`. Frequency is counted in optimizer steps, not epochs. The official first `step()` initializes the preconditioner and intentionally performs no parameter update.
- SOAP telemetry and provenance: set `coupling_training.optimizer.profile_step_time=true` to record optimizer-step mean/p95/max milliseconds, step count, periodic basis-refresh count, and peak allocated CUDA memory in `training.log` and `complex_training_metrics.csv`. `optimizer_peak_memory_mib` measures `torch.cuda.max_memory_allocated(...)` only on CUDA; CPU runs report `0.0` as a not-measured sentinel and do not measure process RSS or CPU allocator memory. CUDA synchronization is used only while profiling. Every complex run writes resolved optimizer metadata to `optimizer_provenance.json`, and complex artifact summaries include the same metadata. Checkpoints remain model-only safetensors and do not support interrupted optimizer-state resume.
- SOAP experiment command: `PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/train.py --config configs/complex_coupling_soap.json --work-dir checkpoints/Annulus_poisson/coupling_soap_pilot`. Start with a 300-500 epoch paired pilot before any long run, and compare against AdamW using the same dataset, batch order, initialization seed, GreenNet checkpoint, loss, projection, clipping, scheduler, optimizer-step budget, and wall-clock budget. This integration does not make SOAP the default.
- SOAP stability screening: interpret warmup in optimizer steps rather than epoch count. With 800 training samples and `batch_size=400`, one epoch contains only two optimizer calls, so `warmup_epochs=3` supplies six calls and is too short for the observed Annulus SOAP pilot. Before a long run, use a 20-30 epoch abortable screen with a lower peak learning rate and longer warmup; the current conservative starting point is `learning_rate=2e-4`, `warmup_epochs=50`, `betas=[0.95,0.99]`, `shampoo_beta=0.95`, and `precondition_frequency=5`. Keep `normalize_grads=false` at this small frequency. Treat these values as project-specific pilot settings, not new defaults, and stop the screen if validation energy exceeds ten times its best finite value.
- SOAP stabilized-pilot interpretation: the Annulus `coupling10` rerun with the conservative settings above reached epoch 71 with 70 consecutive decreases in validation canonical energy (`3.481542e-1` to `3.148627e-1`). The first 50 warmup epochs reduced that energy by only about 1.5%, while epochs 50-71 reduced it by another 8.2%; therefore the early trace can look stalled even though the post-warmup run is converging. Do not classify this pilot as plateaued before roughly epoch 120-150. As a practical diagnostic, continue while the latest 20-epoch validation energy reduction remains at least 2%; regard a reduction below 1% together with flat or worsening detached `rel_sol` as evidence of a plateau. These thresholds are experiment diagnostics, not optimizer defaults.
- SOAP stabilized-pilot late-stage interpretation: by epoch 199 the same run had reduced validation canonical energy monotonically on all 198 epoch transitions, from `3.481542e-1` to `1.260071e-2`. However, detached validation `rel_flux` reached its minimum `4.325040e-1` at epoch 141 and `rel_sol` reached its minimum `3.012693e-1` at epoch 154; by epoch 199 they had worsened to `4.490581e-1` and `3.154724e-1`, respectively, while energy continued to fall. Training metrics turn at nearly the same epochs, so this is not ordinary train/validation overfitting. Treat it as objective-to-solution metric misalignment: SOAP is converging to the configured canonical-energy objective, but lower energy beyond the mid-run optimum is not producing a better reconstructed solution. Reference metrics remain evaluation-only and must not be used as training losses or checkpoint selectors.
- Annulus AdamW comparison snapshot: an independently executed AdamW run reached copied-log epoch 97 with validation energy `3.580980e-3`, `rel_sol=1.467270e-1`, and `rel_flux=3.957876e-1`. These values are already below the SOAP run's full observed minima through epoch 303. This is strong practical evidence in favor of AdamW for the current workload, but it is not a controlled optimizer-only comparison: the AdamW log uses `learning_rate=2e-3`, `epochs=3000`, and AdamW betas, whereas SOAP uses `learning_rate=2e-4`, `epochs=500`, and SOAP-specific betas/preconditioning. The runs also execute on different computers, so do not compare wall-clock time. The copied AdamW logs and local SOAP files do not share a live work directory or checkpoint.
- SOAP high-LR single-variable ablation: to separate the original short-warmup failure from learning-rate sensitivity, repeat the stabilized SOAP recipe with only `learning_rate` changed from `2e-4` to `2e-3`. Keep `epochs=500`, `warmup_epochs=50`, `min_lr=1e-5`, `betas=[0.95,0.99]`, `shampoo_beta=0.95`, `precondition_frequency=5`, batch size, seed, data order, clipping, weight decay, model, and hardware unchanged. Use a separate work directory. Abort on any non-finite value, validation energy above ten times the best finite value, or three consecutive epochs above twice the best finite value. This ablation tests high-LR stability under the stabilized SOAP conditions; it does not by itself establish a fair wall-clock comparison with AdamW.
- SOAP high-LR ablation snapshot: the `coupling10_2` run differs from the stabilized low-LR `coupling10` resolved config only by `learning_rate=2e-3` instead of `2e-4`. At epoch 81 it remained finite after the 50-epoch warmup and reached validation energy `2.494793e-3`, `rel_sol=1.259009e-1`, and `rel_flux=3.395570e-1`; all three were the run's observed minima at that snapshot. Relative to the complete observed low-LR SOAP trace through epoch 309, these values were lower by about 79.4%, 58.2%, and 21.5%, respectively. This result shows that `2e-3` is stable under the longer-warmup, `betas=[0.95,0.99]`, `shampoo_beta=0.95`, and frequency-5 recipe, so the earlier epoch-4 failure cannot be attributed to peak learning rate alone. Continue to monitor detached metrics after epochs 100-160 because the low-LR run's energy/solution-quality misalignment appeared later; do not promote SOAP to the default from this early snapshot.
- SOAP high-LR late-stage interpretation: the continuing `coupling10_2` run shows genuine objective overfitting by the epoch-317 snapshot. Validation canonical energy reached its minimum `5.400105e-4` at epoch 194 and validation `rel_sol` reached `5.755842e-2` at epoch 199, but by epoch 317 they had worsened to `8.833431e-4` and `6.438856e-2` while train energy continued down to `2.923218e-4`. The latest validation/train energy ratio is `3.02`, led by a boundary-energy ratio of `7.67` versus `2.49` for bulk energy. In contrast, train and validation `rel_flux` remain nearly identical and continue improving to about `1.691e-1`. This is therefore a boundary-dominated energy/solution generalization gap rather than optimizer divergence or the earlier low-LR objective-to-metric misalignment. Use `complex_coupling_model_best_energy.safetensors`, last updated at epoch 194, for reference-free model selection instead of the current or final weights.
- SOAP preconditioner options: covariance factors are updated on every optimizer step, while `precondition_frequency` controls only the periodic eigenbasis/QR refresh; a smaller value therefore means a fresher but more expensive basis. With two optimizer steps per epoch, frequencies `5`, `2`, and `1` refresh approximately every 2.5 epochs, every epoch, and every step. For a controlled follow-up to `coupling10_2`, test `5 -> 2` before trying `1` and keep every other setting fixed. `max_precondition_dim=1024` builds a Shampoo factor only for tensor axes no larger than 1024; this includes both axes of every current 2D CouplingNet weight. `merge_dims=false` preserves original tensor axes and is effectively irrelevant for the current 2D-only weight matrices. `precondition_1d=false` leaves bias and activation vectors on the original-basis Adam update. `normalize_grads=false` avoids forcing each projected-back parameter update to unit RMS; keep it false at the current small refresh frequency. `correct_bias=true` applies standard Adam first/second-moment bias correction and should remain enabled.
- Coupling gradient clipping: CouplingNet training clips gradients with `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` by default. Set `coupling_training.gradient_clip_max_norm` to a positive value to change the threshold, or `null` to disable clipping.
- Coupling validation mode: validation and standalone evaluation run with `model.eval()` and `torch.no_grad()`, then restore the model's previous train/eval mode. Dropout remains active for training batches but is inactive for validation metrics and best-validation checkpoint selection.
- Coupling periodic checkpoints: set `coupling_training.periodic_checkpoint.enabled=true` and `coupling_training.periodic_checkpoint.every_epochs=<int>` to save additional AdamW-phase checkpoints as `coupling_model_epoch_XXXX.safetensors`.
- Coupling best-validation checkpoint: unit-square mode can continue to use `coupling_training.best_rel_sol_checkpoint.enabled=true` for `coupling_model_adam_best_rel_sol.safetensors`. Complex v6 rejects that option because reference `sol` cannot select a training checkpoint. Use `coupling_training.best_energy_checkpoint.enabled=true` for validation `loss_energy_optimized` and/or `best_physics_checkpoint.enabled=true` for validation total reference-free physics loss.
- Coupling loss config: use `coupling_training.losses.l2_consistency`, `coupling_training.losses.energy_consistency`, `coupling_training.losses.cross_consistency`, `coupling_training.losses.balance_loss`, and `coupling_training.losses.symmetric_boundary_loss`, each with `enabled` and `weight`. The sample config keeps projection on with `mode="smooth_mask"` and leaves `balance_loss.enabled=false` and `symmetric_boundary_loss.enabled=false`; a raw-output run should set `coupling_model.balance_projection.enabled=false` before enabling `coupling_training.losses.balance_loss`. A symmetric-boundary run should set `coupling_model.balance_projection.enabled=true`, `mode="symmetric"`, and then enable `coupling_training.losses.symmetric_boundary_loss`.
- Energy consistency auxiliary loss: `energy_consistency` computes the face-based physical energy of the represented-solution residual `r = u_phi^(x) - u_psi^(y)`, using forward edge differences, arithmetic face coefficients, and the density `a_face * |D_face r|^2`. Complex v6 sums this density over every same-segment x/y edge and separately evaluates the general connected-segment endpoint boundary contribution. `boundary_weight` controls only how those two values enter `loss_energy_optimized`; the unweighted canonical diagnostic remains their sum. It does not rebalance a transition subset. The optional relative split objective augments the optimized energy with a source-normalized value/mass term; unit-square loss behavior is unchanged.
- Cross consistency auxiliary loss: `cross_consistency` penalizes the cross-operator terms `L_x(u_psi^(y)) <-> phi` and `L_y(u_phi^(x)) <-> psi`, reusing the same represented solutions, conservative stencil, common-grid slicing, and quadrature rule as the rest of the Coupling trainer.
- Green hybrid trunk: the learned Green correction always sees smooth handcrafted features `x`, `xi`, `x*xi`, `x^2`, `xi^2`, `x-xi`, `(x-xi)^2`, and `sqrt((x-xi)^2 + eps)`. If `model.use_fourier=true`, the Fourier embedding is appended to that structured trunk basis instead of replacing it.
- Coupling LR schedule: both unit-square and complex CouplingNet trainers honor `coupling_training.use_lr_schedule=true` with `warmup_epochs` and `min_lr`. The shared schedule applies linear warmup, then cosine decay from `learning_rate` to `min_lr`; it steps once after each epoch's optimizer updates, validation, and checkpoint writes. Complex training records the learning rate actually used by every train/validation epoch in `complex_training_metrics.csv` and `training.log`. With `use_lr_schedule=false`, AdamW keeps the configured fixed `learning_rate`.
- GreenNet optimizer config: omit `training.optimizer` to use AdamW with `betas=[0.9,0.999]`, `eps=1e-8`, and `training.weight_decay` (default `0.0`). Set `training.optimizer.name="soap"` for either GreenNet geometry path; supported SOAP options and pinned source are shared with complex CouplingNet. `configs/complex_green_soap.json` is the paired-pilot example, while `configs/default_green.json` and `configs/complex_green.json` remain AdamW examples.
- GreenNet LR schedule: both unit-square and complex GreenNet honor `training.use_lr_schedule=true` with `warmup_epochs` and `min_lr`. Linear warmup followed by cosine decay is applied only to the AdamW/SOAP first stage and steps once after each completed epoch. The pre-LBFGS model is saved as `model_pre_lbfgs.safetensors`; LBFGS then uses its existing independent optimizer, learning rate, closure, and strong-Wolfe line search without any scheduler.
- GreenNet optimizer audit: every run writes `green_optimizer_provenance.json` and `green_training_metrics.csv`, records the actual first-stage learning rate in `training.log`, and includes resolved Green optimizer/scheduler metadata in `config_used.json` and Green artifact summaries. `training.optimizer.profile_step_time=true` enables the same optimizer-step timing, SOAP basis-refresh, and CUDA peak-memory telemetry used by complex CouplingNet. Green checkpoints remain model-only safetensors; optimizer/scheduler resume state is not stored.
- Coefficients: the training CLI resolves `a_fun`, `apx_fun`, `apy_fun`, directional convection coefficients `bx_fun`/`by_fun`, and `c_fun`, then forwards the same function set into the axial sampler and Coupling datasets. Internally `b_vals[0]` stores x-direction convection and `b_vals[1]` stores y-direction convection.
- Coefficient functions: set `dataset.coefficient_functions_path` to a Python file that defines callable `a_fun`, `apx_fun`, `apy_fun`, `bx_fun`, `by_fun`, and `c_fun` with signature `(x, y) -> Tensor`; `configs/sinusoidal_coefficients.py` is a default-equivalent example. Legacy files that define only `b_fun` are interpreted as `bx_fun = b_fun` and `by_fun = b_fun`; mixing `b_fun` with `bx_fun`/`by_fun` or defining only one directional convection function fails fast.
- Annulus convection-diffusion-reaction coefficients: `coefficients/Annulus_Convection_Diffusion_Reaction.py` keeps the diffusion and reaction fields from `Convection_Diffusion_Reaction.py` and replaces convection by a smooth counter-clockwise tangential field for the centered annulus with `inner_radius=0.2` and `outer_radius=0.5`. Its radial polynomial envelope makes the full vector field zero on both circular boundaries, and `CONVECTION_AMPLITUDE=0.5` preserves the original coefficient family's convection scale. Point `dataset.coefficient_functions_path` to this file for the matching Annulus experiment.
- Pentagram convection-diffusion-reaction coefficients: `coefficients/CDR_pentagram.py` targets the centered regular pentagram with `outer_radius=0.5`. It uses the radius-normalized diffusion `a=1+0.5*sin(pi*x/R)*sin(pi*y/R)`, the divergence-free counter-clockwise field `b=0.5*(-y/R,x/R)`, and the full physical reaction `c=1+0.5*cos(pi*x/R)*cos(pi*y/R)`, whose range is `[0.5,1.5]`. The coefficient module does not apply axial `L` or `L^2` scaling and does not alter the existing directional reaction split; point `dataset.coefficient_functions_path` to this file for matching FEniCSx, GreenNet, and CouplingNet runs.
- Sampler mode: set `dataset.sampler_mode` to `"forward"` (sample `u`, derive `f`) or `"backward"` (sample `f`, solve `-d/dx(a u') + b u' + c u = f` with `scipy.integrate.solve_bvp` to recover `u`).
- Green validation reconstruction: set `training.compute_validation_rel_sol=true` to generate a separate synthetic validation dataset for GreenONet and log `val_rel_sol` alongside training `rel_sol`.
- Green validation dataset controls: use `dataset.validation_samples_per_line` to choose how many validation samples are generated per axial line, `dataset.validation_scale_length` to override the validation sampler length scale, and `dataset.validation_sampler_mode` to override the validation sampler type. If either validation override is omitted, Green validation reuses the training-side `dataset.scale_length` and `dataset.sampler_mode`.
- Green per-line export: `per_line_metrics.csv` now includes validation reconstruction columns `val_rel_sol_line`, `val_rel_sol_line_mean`, `val_rel_sol_line_min`, `val_rel_sol_line_max`, and `val_rel_sol_line_std` when Green validation is enabled; `per_line_metrics_summary.json` also includes validation summary entries.
- Dataset domain: the CLI ignores `dataset.domain` (sampling defaults to the unit square), so configs can omit it safely.
- Dataset split symlinks: use `python make_splits.py --source <npz_dir> --out <split_dir> --train-ratio 0.8 --valid-ratio 0.1 --test-ratio 0.1 --seed 42` to create `train/`, `valid/`, and `test/` symlink views. Ratios must be non-negative and sum to `1.0`; zero-ratio splits are created as empty directories. Keep the source `.npz` files in place because outputs are symlinks, not copies.
- For closer parity with the original `/home/jjhong0608/Documents/GreenONet`, the entrypoint `greenonet.runner.run_green_o_net` mirrors the classic `run_green_o_net.py` API and now includes `sampler_mode` for forward/backward synthetic data generation.

## Evaluation

- Exporting paper-oriented GreenNet checkpoint artifacts:
  ```
  PYTHONPATH=src python cli/export_green_artifacts.py \
    --checkpoint checkpoints/run_green_net/model.safetensors \
    --config checkpoints/run_green_net/config_used.json \
    --outdir checkpoints/run_green_net/green_artifacts \
    --eval-seed 12345 \
    --eval-split validation_like \
    --eval-samples-per-line 10 \
    --line-indices 0 32 64 96 128 \
    --xi-fractions 0.25 0.5 0.75 \
    --device cpu \
    --theme plotly_white
  ```
  The exporter reloads the GreenONet checkpoint, regenerates evaluation source/solution samples from the config's Green sampler distribution, and writes `summary.json`, metric CSVs, selected raw arrays, Green-kernel heatmaps, fixed-`xi` one-dimensional Green slices, coefficient slices, and axial reconstruction figures. It uses `training.device` from the config unless `--device` is provided. Every Plotly figure is saved as `.html`, editable Plotly `.json`, `.png`, and `.pdf`; if static export fails because Kaleido/Chrome is unavailable, `.html` and `.json` are still written. Use `--eval-seed` and the saved metadata to distinguish same-distribution evaluation from reusing the exact training samples. The paper-facing `rel_green` metric is exported when an exact/reference line Green kernel is supported: diffusion-only runs use `rel_green_reference="diffusion"`, reaction-free convection-diffusion runs use `rel_green_reference="convection_diffusion"`, and reaction runs record a skip reason.
- Plotting GreenONet logs: `python plot_green_logs.py --logs checkpoints/run_green_net/training.log --outdir plots_green --theme plotly_white` (supports multiple logs, `--labels`, and Plotly templates via `--theme`; outputs `loss`, `train_rel_sol`, `val_rel_sol`, and `rel_green` figures as HTML/JSON plus PNG/PDF if available).
- Plotting paper-facing CouplingNet logs: `python plot_coupling_logs.py --logs checkpoints/coupling_run/training.log checkpoints/coupling_run_2/training.log --labels run1 run2 --outdir plots_coupling --theme plotly_white`. Add `--show-annotations` to label each trace's last value and minimum value near the corresponding curve points. This intentionally writes only `loss`, `l2_consistency`, `energy_consistency`, `rel_flux`, and `rel_sol` training curves, each with train/validation traces when present. Complex CouplingNet logs with separate `_log_epoch - epoch ... train/val ...` lines are supported for `loss`, `loss_energy_consistency`, `rel_sol`, and `rel_flux`; `l2_consistency` is skipped when no L2 values are present, and cross metrics remain absent. Every figure is saved as `.html` and editable Plotly `.json`, plus `.png`/`.pdf` when static export is available.
- Plotting recent Coupling logs from the current `_run_training_phase - epoch ...` format: `python plot_logs.py --logs checkpoints/test_diffusion/coupling/single_unknown/backward/training.log --outdir plots_coupling_recent`. This plots total loss, L2 consistency, energy consistency, cross consistency, optional `balance_loss`, optional `symmetric_boundary_loss`, `rel_flux`, and `rel_sol` from the current Coupling trainer log lines and ignores compile/checkpoint noise lines.
- Workshop CouplingNet boxplots: `plot_coupling_rel_sol_boxplots.py` reads the four problem-level `*_per_sample_metrics.csv` files, displays `rel_sol` as percent, and exports Plotly HTML/JSON/PNG/PDF. `--rel-sol-percentile p` keeps exactly `max(1, floor(n*p/100))` lowest samples per problem; the default `100` keeps all samples.
- Complex GreenNet training example:
  ```
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_circular_geometry.py \
    --step-size 0.05 \
    --out data/geometry/unit_circle_h005.npz

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/train.py \
    --config configs/complex_green.json \
    --work-dir checkpoints/complex_green

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/export_green_artifacts.py \
    --checkpoint checkpoints/complex_green/model.safetensors \
    --config checkpoints/complex_green/config_used.json \
    --outdir checkpoints/complex_green/green_artifacts \
    --device cpu
  ```
  The complex GreenNet path writes the standard training curves and `model.safetensors`, plus `per_interval_metrics.csv`, `per_interval_metrics_summary.json`, and a first-interval `green_heatmap.html`. Artifact export records `geometry_mode="complex"`, the geometry path, interval counts, per-interval metrics, selected flat-interval kernels, coefficient slices, and unit reconstruction figures.
- Plotting complex GreenNet per-interval metrics:
  ```
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python plot_complex_green_interval_metrics.py \
    --csv checkpoints/complex_domain_poisson/green/artifacts/metrics/per_interval_metrics.csv \
    --outdir checkpoints/complex_domain_poisson/green/artifacts/figures/per_interval_metrics \
    --overwrite
  ```
  This reads `per_interval_metrics.csv` with pandas and writes Plotly figures for solution reconstruction error versus transverse coordinate and segment length, axis-wise distributions, Green-kernel error summaries, and x/y chord maps colored by `rel_sol_interval_mean`. Outputs are saved as `.html` and editable `.json`, plus `.png`/`.pdf` when static Plotly export is available. `metrics_summary.json` records row counts, axis counts, metric summaries, source CSV, and generated figure names.
- Plotting complex CouplingNet per-sample test metrics:
  ```
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python plot_complex_coupling_sample_metrics.py \
    --csv checkpoints/complex_domain_poisson/coupling/metrics/test_per_sample_metrics.csv \
    --outdir checkpoints/complex_domain_poisson/coupling/figures/test_per_sample_metrics \
    --overwrite
  ```
  This reads `test_per_sample_metrics.csv` with pandas and writes Plotly figures for `rel_sol`, `rel_flux`, `loss`, log-scaled loss, relative-metric distributions, `rel_sol` versus `rel_flux`, and best/worst sample rankings by `rel_sol`, `rel_flux`, and `loss`. Relative errors are stored as raw fractions in the CSV and summary, but figures display them as percentages. Outputs are saved as `.html` and editable `.json`, plus `.png`/`.pdf` when static Plotly export is available. `metrics_summary.json` records raw and percent summaries, metric correlations, best/worst sample records, source CSV, and generated figure names.
- Exporting paper-oriented CouplingNet checkpoint artifacts:
  ```
  PYTHONPATH=src python cli/export_coupling_artifacts.py \
    --config checkpoints/run_coupling/config_used.json \
    --coupling-checkpoint checkpoints/run_coupling/coupling_model.safetensors \
    --green-checkpoint checkpoints/run_green_net/model.safetensors \
    --outdir checkpoints/run_coupling/coupling_artifacts \
    --device cpu \
    --theme plotly_white
  ```
  The exporter reads `dataset.test_path` as a `CouplingDataset`, uses `dataset.coefficient_functions_path` unless `--coefficients` overrides it, and uses `coupling_training.device` unless `--device` is provided. If `--selected-samples` is omitted, selected heatmaps default to five `rel_sol` representatives: min, q25, q50, q75, and max; pass `--selected-samples 0 5 12` to override this policy explicitly. It writes `summary.json`, per-sample and aggregate metric CSVs, selected raw `.npz` archives, coefficient/source context figures, solution figures, flux-divergence figures, and balance figures. Reference/prediction non-error comparison heatmaps share color ranges within each selected sample for `u/u_pred/u_pred_x/u_pred_y`, `phi/phi_pred`, and `psi/psi_pred`. Paper-facing error heatmaps are signed differences only: `u_pred - u`, `u_pred_x - u`, `u_pred_y - u`, `u_pred_x - u_pred_y`, `phi_pred - phi`, and `psi_pred - psi`. The paper exporter does not create null-space or closure diagnostic figures; those remain in the debug evaluator path.
- CouplingNet debug evaluation on test data (per-sample metrics and diagnostic plots):
  1. Set `dataset.test_path` in your config.
  2. Run:
     ```
     python cli/eval_coupling.py \
       --config configs/default.json \
       --coupling-checkpoint checkpoints/your_coupling.safetensors \
       --green-checkpoint checkpoints/run_green_net/model.safetensors \
       --work-dir checkpoints/eval_run
     ```
  3. Outputs: per-file metrics CSV (relative L2 for solution and flux-divergence via the configured `coupling_training.integration_rule`) and Plotly heatmaps (Times New Roman, bold) for exact/pred/error of solution and flux-divergence saved as pdf/png/html.
- Complex geometry training/evaluation example:
  ```
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/make_circular_geometry.py \
    --step-size 0.05 \
    --out data/geometry/unit_circle_h005.npz

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/train.py \
    --config configs/complex_geometry.json \
    --work-dir checkpoints/complex_geometry_run

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/eval_coupling.py \
    --config checkpoints/complex_geometry_run/config_used.json \
    --coupling-checkpoint checkpoints/complex_geometry_run/complex_coupling_model.safetensors \
    --green-checkpoint checkpoints/run_green_net/model.safetensors \
    --work-dir checkpoints/complex_geometry_eval

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/export_coupling_artifacts.py \
    --config checkpoints/complex_geometry_run/config_used.json \
    --coupling-checkpoint checkpoints/complex_geometry_run/complex_coupling_model.safetensors \
    --green-checkpoint checkpoints/run_green_net/model.safetensors \
    --outdir checkpoints/complex_geometry_run/complex_artifacts \
    --device cpu \
    --coefficient-vector-max-points 400
  ```
  The complex artifact exporter writes `summary.json`, `metrics/per_sample_metrics.csv`, `data/selected_raw_arrays.npz`, and Plotly valid-point scatter figures on `coords_valid`. Primary fields include `rhs`, `sol`, `u_pred=0.5*(u_phi+u_psi)`, `u_phi`, `u_psi`, projected physical `phi`/`psi`, signed solution errors `u_pred_error`, `u_phi_error`, `u_psi_error`, split mismatch `u_split_mismatch`, and optional target flux fields plus signed `phi_error`/`psi_error` when sample flux targets are available. Solution reference/prediction scatters share their full range within each selected sample. Directional-value scatters use one robust range for `target_phi/phi` and another for `target_psi/psi`; each range is the lower and upper quantile of the pooled finite valid-point values. Directional errors use a zero-centered range whose magnitude is the selected quantile of `abs(error)`. Complex mode defaults to `--directional-color-quantile 0.99`; use `1.0` to recover full min/max ranges. The same precomputed ranges are reused by corresponding scatter and mesh figures, while hover always reports the raw unclipped value with `x` and `y`. `summary.json` records full extrema, displayed extrema, and saturation counts without modifying raw arrays or metrics. The v6 raw archive stores raw/projected reference responses, pre-projection raw physical fields, projected physical fields, `L_x^2/L_y^2`, response and physical balance residuals, pointwise cross-axis length-context features, and canonical boundary endpoint coordinates, distances, nearest valid indices, and split residuals. It does not store production transition masks or length-jump scores. When optional pre-projection fusion is enabled, it additionally stores `fusion_base_difference`, normalized difference/source inputs, generic normalized and physical MLP outputs, `fusion_fused_difference`, `fusion_delta_from_base`, the constructed pre-projection physical pair, raw/safe source scales, and the pre-projection balance residual. Residual mode also preserves the legacy `fusion_residual_normalized` and `fusion_residual_physical` aliases; absolute mode omits those aliases because its MLP output is an absolute difference rather than a residual. Matching signed figures show the base difference, physical MLP output, fused difference, and fused-minus-base delta. Relative split and weak closure retain their existing optional archives.

  All complex selected-sample scatter, coefficient/vector, and projection-context figures overlay the domain boundary by default. Interior markers remain filled and colored only by values evaluated at `coords_valid`; boundary endpoints come from the cached canonical boundary-energy context and are rendered as neutral `circle-open` geometry markers without scalar values, colorscale participation, or metric participation. Mesh figures use the field-specific boundary policy described next instead of these open scatter markers. Use `--no-show-domain-boundary` to reproduce the previous interior-only scatter figures and to omit unavailable-value outlines from source/directional meshes. `summary.json` records the overlay state, endpoint source, marker representation, and boundary-point count under `domain_boundary_overlay`. This visualization option does not change `figure_count`, `figure_fields`, raw NPZ arrays, model inference, or unit-square artifacts.

  Complex scalar fields can additionally be exported on a conforming triangle mesh. Generate the reusable cache once in the optional `green_fenicsx` environment, then pass it to the normal artifact command in `green_net`:

  ```bash
  PYTHONPATH=src ~/.conda/envs/green_fenicsx/bin/python \
    cli/make_complex_visualization_mesh.py \
    --geometry data/geometry/annulus_02_05_1_128.npz \
    --gmsh-script examples/annulus_gmsh.py \
    --boundary-size-factor 3.0 \
    --max-auxiliary-fraction 0.001 \
    --out data/visualization_mesh/annulus_02_05_1_128_mesh.npz

  PYTHONPATH=src ~/.conda/envs/green_net/bin/python \
    cli/export_coupling_artifacts.py \
    --config checkpoints/Annulus_poisson/coupling/config_used.json \
    --coupling-checkpoint checkpoints/Annulus_poisson/coupling/complex_coupling_model_best_energy.safetensors \
    --green-checkpoint checkpoints/Annulus_poisson/green/model.safetensors \
    --outdir checkpoints/Annulus_poisson/coupling/artifacts \
    --visualization-mesh data/visualization_mesh/annulus_02_05_1_128_mesh.npz \
    --directional-color-quantile 0.99 \
    --device cpu
  ```

  Use `examples/rectangle_gmsh.py`, `examples/unit_circle_gmsh.py`, `examples/annulus_gmsh.py`, or `examples/pentagram_gmsh.py` with its matching geometry. The cache embeds every `coords_valid` as a distinct vertex, validates the geometry SHA-256, preserves holes and concave boundaries through explicit Gmsh connectivity, and fails if auxiliary interior vertices exceed the configured fraction. The complex-path unit-square cache generated from `data/geometry/unit_square_h_1_128.npz` is `data/visualization_mesh/unit_square_h_1_128_mesh.npz`. Selected-sample outputs are additive Plotly `Mesh3d` figures for `sol`, `u_pred`, `u_pred_error`, `rhs`, `phi`, and `psi`, plus `target_phi`, `target_psi`, `phi_error`, and `psi_error` when flux targets exist. Solution fields use exact valid-point values, prescribed homogeneous Dirichlet zero at boundary vertices, and no black boundary outline. Source and directional fields do not invent boundary scalar values: cell colors average only non-boundary vertices, all-boundary triangles fail fast, and the boundary is drawn as a neutral dark outline when domain-boundary display is enabled. Permitted auxiliary interior vertices use the cached mesh-adjacency stencil. Exact hover values come from `coords_valid`; solution hover additionally exposes prescribed boundary zero. Existing scatter figures and `figure_fields` remain unchanged. All scalar meshes preserve the physical x/y aspect ratio and use a shared scene scale of `1.5`, so the domain occupies more of the fixed `900x800` canvas without changing coordinates or values. Mesh boundary and interpolated values never enter `rel_sol`, transition metrics, checkpoint selection, or `selected_raw_arrays.npz`. With generated-data saving enabled, the validated cache is copied to `data/visualization_mesh.npz`, and `summary.json` records its provenance, field list, transfer policy, boundary policy, robust color ranges, hover policy, and scene scale. Omitting `--visualization-mesh` removes only the additive mesh output; unit-square export rejects both the mesh option and the complex-only directional color-range option.

  Complex export also evaluates the physical PDE coefficients directly at `coords_valid`, without interpolation, pull-back, or segment-length scaling. It writes the run-level archive `data/coefficient_fields.npz` with `a`, `bx`, `by`, `b_magnitude`, `c`, and the deterministic quiver indices. Plotly outputs under `figures/coefficients/` include `diffusion_a`, optional `reaction_c`, and optional `convection_bx`, `convection_by`, `convection_magnitude`, and `convection_vector`; diffusion is always shown, while zero reaction/convection figures are omitted unless their CouplingNet coefficient branch term is enabled. When `--visualization-mesh` is supplied, matching scalar meshes are added under `figures/coefficients/mesh/` for `a`, `c`, `bx`, `by`, and `|b|`. These values are evaluated directly at every physical mesh vertex, including boundary and auxiliary vertices, use vertex intensity and exact-value hover, and share the corresponding scatter colorscale and color range. Coefficient boundary values are physical function values rather than prescribed zero or unavailable outlines. The vector figure remains the existing subsampled quiver on the full valid-point convection-magnitude field; no vector `Mesh3d` is generated, and `--coefficient-vector-max-points` continues to set the arrow limit. `coupling_model.coefficient_terms` controls model branch inputs, not whether a coefficient exists in the physical PDE, so `summary.json` records physical field status, branch activation, mesh evaluation policy, and mesh figure counts separately.
- Complex line-length response diagnostic: use the checkpoint-backed command below to locate the first stage at which annulus transition-line error appears. The diagnostic reruns only inference, never training, and requires test samples with `phi/psi` targets. Under v6 it compares the pre-projection physical proposals, projected physical directional-source errors, projected reference-response errors, predicted-source exact Green reconstruction, production learned-Green reconstruction, and target-source exact closure. The production reconstruction consumes the projected response directly. The diagnostic-only exact path also audits the mathematically equivalent physical-interval and unit-interval integrals at `1e-10` by default.

  ```bash
  PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/diagnose_complex_length_response.py \
    --config checkpoints/Annulus_poisson/coupling5/config_used.json \
    --coupling-checkpoint checkpoints/Annulus_poisson/coupling5/complex_coupling_model_best_energy.safetensors \
    --green-checkpoint checkpoints/Annulus_poisson/green/model.safetensors \
    --selected-samples 47
  ```

  The default output is `<coupling-checkpoint-parent>/length_response_diagnostics`. It contains `summary.json`, per-sample/per-segment/transition-zone CSVs, `data/selected_diagnostic_arrays.npz`, grouped Plotly figures, and `diagnose_complex_length_response.log`. Sample 47 is selected by default, and unique min/q25/q50/q75/max `rel_sol` representatives are added unless `--no-include-rel-sol-quantiles` is used. The diagnostic decomposition is `learned total error = exact source response + exact target closure + learned-minus-exact Green contribution`; reference `sol/phi/psi` are evaluation-only and never enter a training loss.
- Annulus CDR CouplingNet reference audit (2026-07-29):
  `checkpoints/annulus_CDR/coupling` uses fixed indexed-GP sources
  (`4800` train, `100` validation), canonical-energy-only training, output
  contract v6, physical-symmetric projection, optional pre-projection fusion,
  and SOAP. Validation energy reached its minimum at epoch 82
  (`4.2700e-4`) and ended at `4.3083e-4` at epoch 100. The final checkpoint's
  50-sample test means are `rel_sol=5.1905%`, `rel_flux=17.1055%`, and
  canonical energy `4.4634e-4`. Exported artifacts use the epoch-82
  best-energy checkpoint, verified by its learned fusion gate, and therefore
  have slightly different aggregate metrics. Across the selected artifact
  samples, the annulus `|x|`/`|y| ~= 0.2` transition band occupies `8.12%` of
  valid points but contains `21.85%` of the largest 1% solution errors and
  roughly `25-27%` of the largest split/flux errors. Canonical energy and
  detached `rel_sol` have only weak sample ranking correlation, so both must
  be reported when interpreting this run. The selected fusion diagnostics also
  show that the raw nonlinear correction has `6.20x` the transition-edge jump
  RMS of the linear correction. This does not transfer directly to the final
  field: the learned nonlinear gate is only `0.06854`, and the pooled blended
  transition jump is `0.987x` the linear-only jump. Therefore the nonlinear
  artifact is visibly more discontinuous, but this run alone does not show
  that it increases the final solution seam. This retired nonlinear fuser is
  transition-aware, not transition-regularizing: it can detect discontinuous
  length features but has no structural continuity constraint. Because the
  physical-symmetric projection preserves the fused physical difference mode,
  any seam that survives blending also survives projection.
  This paragraph describes the retired split-fuser architecture used by that
  checkpoint, not the current single residual MLP. Its nonlinear path is a
  shared pointwise MLP: the linear path maps
  normalized `(base_difference, rhs)` from `2` to `1`, while the nonlinear
  path augments those values with six local geometry features and applies
  `8 -> hidden_dim -> 1` for depth one. It has no neighboring-line input,
  continuity operator, pointwise gate, or coefficient field input. That legacy
  implementation supported `residual_correction` and
  `absolute_difference` modes. Absolute mode can combine an identity-initialized
  linear candidate with a scaled nonlinear residual, or take a true convex
  average of two absolute candidates; its artifact metadata distinguishes these
  component semantics and recorded whether the outer base residual was used.
  Internally, the fuser converts base reference responses to physical
  directional sources, applies fusion, and temporarily converts the fused
  values back to reference responses because the model forward contract
  returns reference-space tensors. The projection immediately converts that
  temporary response back to physical source space, so this intermediate
  multiply/divide pair cancels algebraically. It is distinct from the required
  post-projection pull-back used by Green reconstruction.
- Annulus CDR single-residual-MLP comparison (2026-07-30):
  `checkpoints/annulus_CDR/coupling3/comparison_analysis/analysis_report.md`
  compares the retired split fusers in `coupling`/`coupling2` with the current
  single nonlinear residual MLP in `coupling3`. The new MLP reduced sample-0
  correction transition-jump RMS by `33.7%` versus `coupling` and `69.3%`
  versus `coupling2`, so removing explicit geometry inputs did make the learned
  correction smoother. It did not remove the final seam: the fixed identity
  skip preserves a transition-bearing `d_base`, and the fused-difference
  transition/regular jump ratio remained `2.831`. On the common 50-sample
  best-energy test, `coupling3` had the lowest mean `rel_sol` (`5.1411%`) but
  the highest mean `rel_flux` (`17.4703%`) and canonical energy (`4.5120e-4`);
  these are small mixed changes rather than a clear end-to-end win. Sample 39
  dominates its energy tail through a boundary-x failure. The three configs do
  not pin CouplingNet initialization, so causal architecture claims require
  paired fixed-seed repetitions.
- Annulus CDR absolute-fuser comparison (2026-07-31):
  `checkpoints/annulus_CDR/coupling4/comparison_analysis/analysis_report.md`
  compares the single-MLP residual mode in `coupling3` with absolute mode in
  `coupling4`. The data, 384-width backbone, SOAP settings, scheduler, and test
  set match; the CouplingNet initialization seed is not pinned. Absolute mode
  increased best validation canonical energy by `4.110x`. On the same 50 test
  samples, mean canonical energy increased by `298.2%`, `rel_sol` increased
  from `5.164%` to `9.092%`, and `rel_flux` increased from `16.988%` to
  `37.830%`; all 50 samples were worse on all three primary metrics. This is
  mainly slow optimization and loss of the direct identity path, not late
  overfitting: absolute validation energy still improved through epoch 99.
  With a zero-initialized final layer, absolute mode initially blocks the
  difference-path gradient to the base axis networks. Its unused base
  difference then becomes an unanchored latent that the small pointwise MLP
  must compress back to a physical split. In the common selected samples 0 and
  37, mean base RMS grew from `0.675` to `9.219`, while fused-to-target relative
  error grew from `0.234` to `0.586`. Residual mode remains the production
  baseline; rerunning absolute mode requires an identity-preserving
  initialization or another direct gradient path rather than merely extending
  the current cosine schedule.
- Annulus CDR fuser-off comparison (2026-07-31):
  `checkpoints/annulus_CDR/coupling5/comparison_analysis/analysis_report.md`
  compares the disabled control with the single residual MLP in `coupling3`,
  the failed absolute MLP in `coupling4`, and both retired split-fuser runs.
  The focused `coupling3/4/5` configs differ only in the pre-projection fusion
  fields; model initialization is still not seed-pinned. Disabling the fuser
  recovers the absolute regression completely: on the common 50-sample
  best-energy test, `coupling5` has mean canonical energy `4.4593e-4`,
  `rel_sol=5.1562%`, and `rel_flux=17.8986%`, versus
  `1.8212e-3`/`9.0959%`/`37.8407%` for `coupling4`. Relative to the residual
  `coupling3`, energy is 1.17% lower and `rel_sol` is 0.29% higher, with paired
  wins close to 50/50 and no supported difference. Flux is the exception:
  fuser-off is 2.45% worse on average and worse on 45/50 samples. Thus the
  residual fuser provides a small, consistent directional-split benefit but no
  demonstrated energy/solution benefit. Fuser-off does not remove the annulus
  seam: its five selected `u_pred_error` fields have transition/regular
  edge-jump ratios `5.01` to `6.04`.
- Fixed Smooth Cross-Axis Reconstruction Blend diagnostic: run
  `PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/diagnose_fixed_smooth_cross_axis_blend.py --config checkpoints/annulus_CDR/coupling5/config_used.json --coupling-checkpoint checkpoints/annulus_CDR/coupling5/complex_coupling_model_best_energy.safetensors --green-checkpoint checkpoints/annulus_CDR/green/model.safetensors --outdir checkpoints/annulus_CDR/coupling5/fixed_smooth_cross_axis_blend`.
  This is a standalone post-hoc diagnostic and does not change production
  training, projection, reconstruction, or checkpoint contracts. It constructs
  sample-independent weights from smoothed transverse changes of
  `log(Lx^2)` and `log(Ly^2)`, then compares
  `0.5*(u_phi+u_psi)` with `w_phi*u_phi+w_psi*u_psi` on the full-reference test
  set. The fixed untuned preset on `coupling5` lowers mean `rel_sol` from
  `5.1562%` to `5.1164%` and lowers transition trace-error jump RMS by `54.3%`
  on average, but increases the broad transition-zone error RMS by `0.95%`.
  The result is therefore promising but mixed; it is not yet enabled in the
  production estimator. Outputs include `summary.json`,
  `diagnosis_report.md`, a per-sample CSV, selected raw NPZ arrays, and Plotly
  weight/comparison figures.
- Compact C2 topology-distance variant: add
  `--weight-construction compact_c2_ramp --ramp-gamma 0.5 --ramp-width 0.03125 --compact-sweep`
  to the same command. This variant detects transverse coordinates where
  adjacent axial lines change connected-segment multiplicity and uses a
  compact quintic ramp
  `B(s)=1-10*s^3+15*s^4-6*s^5` for `0 <= s < 1`. The recommended
  `gamma=0.5`, `width=4h` preset on the frozen `coupling5` checkpoint changes
  mean `rel_sol` from `5.156246%` to `5.054177%`, transition-zone error RMS by
  `-4.225%`, and transition trace-error jump RMS by `-44.956%`; it wins global
  `rel_sol` on `36/50` samples. The maximum neighboring weight jump is `0.1124`
  instead of `0.3216`, and the correction is exactly zero outside compact
  support. The optional gamma/width sweep is explicitly test-target exploratory
  and must not be presented as independent validation or automatic production
  tuning. Canonical outputs are under
  `checkpoints/annulus_CDR/coupling5/compact_c2_cross_axis_blend_fine_sweep/`.
- Poisson compact C2 transfer check: the same fixed `gamma=0.5`, `width=4h`
  preset was evaluated on all 50 full-reference samples from
  `checkpoints/Annulus_poisson/coupling15` without retraining or changing the
  checkpoint. Mean `rel_sol` changes from `5.569825%` to `5.470165%`
  (`-1.789%`), transition-zone RMS changes by `-3.412%`, and transition
  trace-error jump RMS changes by `-45.308%`. It improves `rel_sol` on `35/50`
  samples and trace jump on `50/50`; paired-bootstrap 95% intervals for all
  three aggregate changes lie below zero. Because Poisson and CDR use the same
  annulus geometry, their geometry-only weights are identical, while their test
  sources are independent (`seed=2732` versus `2222`). This supports
  cross-equation reuse on the same geometry, not cross-domain generalization.
  Canonical outputs are under
  `checkpoints/Annulus_poisson/coupling15/compact_c2_cross_axis_blend/`.
- Poisson four-way reconstruction-blend comparison: the frozen `coupling15`
  checkpoint was evaluated on all 50 test samples with the same fixed CDR
  presets and no parameter sweep. Equal mean, geometry-only compact C2,
  mismatch-detected seam C2, and local weak-residual reliability obtain mean
  `rel_sol` values of `5.569825%`, `5.470165%`, `5.312818%`, and `4.860579%`,
  respectively. Relative to equal mean, the three adaptive estimators change
  mean `rel_sol` by `-1.789%`, `-4.614%`, and `-12.734%`; they win on `35/50`,
  `44/50`, and `50/50` samples. Geometry-only C2 is the most controlled
  topology-specific rule, mismatch-detected C2 gives a stronger global
  correction but weaker trace-jump suppression, and local weak-residual
  reliability gives the best aggregate result while retaining axial stripe
  structure in its weights. These are post-hoc final-estimator diagnostics,
  not retrained models or production defaults. Canonical outputs are under
  `checkpoints/Annulus_poisson/coupling15/weak_residual_reliability_blend_comparison/`.
- Prediction-only cross-axis estimator comparison: run
  `PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/compare_cross_axis_blend_estimators.py --config checkpoints/annulus_CDR/coupling5/config_used.json --coupling-checkpoint checkpoints/annulus_CDR/coupling5/complex_coupling_model_best_energy.safetensors --green-checkpoint checkpoints/annulus_CDR/green/model.safetensors --outdir checkpoints/annulus_CDR/coupling5/cross_axis_blend_estimator_comparison`.
  The mismatch estimator constructs sample-dependent directional sensors from
  normalized axial-edge jumps of `u_phi-u_psi`. It uses
  `theta=gamma*activation*(Jx-Jy)/(Jx+Jy+eps)` and
  `w_phi=(1+theta)/2`, `w_psi=1-w_phi`; it does not read segment lengths,
  transition coordinates, `sol`, or flux targets. The fixed exploratory preset
  `gamma=0.5`, two 50:50 graph-smoothing steps, and normalized activation
  interval `[0.15,0.35]` changes full-test mean `rel_sol` from `5.156246%` to
  `4.999177%`, transition-zone error RMS by `-7.345%`, and transition
  trace-error jump RMS by `-28.278%`, winning `rel_sol` on `43/50` samples.
  The fixed geometry-only compact ramp remains the stronger trace-jump
  suppressor (`-44.956%`) but has higher mean `rel_sol` (`5.054177%`). These
  thresholds were selected from prediction-only scale inspection on the same
  test run, so the comparison is post-hoc exploratory evidence rather than an
  independent production preset. Outputs include a three-estimator CSV,
  selected raw NPZ arrays, Plotly sensor/error figures, `summary.json`, and
  `diagnosis_report.md`.
- Mismatch-detected seam C2 comparison: the same CLI also supports a fourth
  estimator that separates detection from weight construction. It reduces the
  axial edge jumps of `u_phi-u_psi` to one x-profile and one y-profile, smooths
  those profiles, selects at most `--seam-max-per-axis` peaks with physical
  non-maximum suppression, and uses the selected coordinates only as seam
  centers. The reconstruction weights are then rebuilt with the compact
  quintic profile
  `B(s)=1-10*s^3+15*s^4-6*s^5`, using
  `theta=gamma*(B_x-B_y)`, `w_phi=(1+theta)/2`, and
  `w_psi=1-w_phi`. The detector does not read segment lengths, topology
  transition coordinates, `sol`, or target fluxes; known annulus transition
  coordinates are used only for localization audit. The default physical NMS
  separation is four ramp widths so a secondary peak from the same transition
  band does not consume both seam slots.
- The frozen-checkpoint exploratory command
  `PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/compare_cross_axis_blend_estimators.py --config checkpoints/annulus_CDR/coupling5/config_used.json --coupling-checkpoint checkpoints/annulus_CDR/coupling5/complex_coupling_model_best_energy.safetensors --green-checkpoint checkpoints/annulus_CDR/green/model.safetensors --outdir checkpoints/annulus_CDR/coupling5/cross_axis_blend_detected_seam_c2_comparison --geometry-gamma 0.5 --mismatch-gamma 0.5 --seam-gamma 0.3 --seam-ramp-width 0.09375 --seam-peak-relative-threshold 0.25 --seam-profile-smoothing-steps 1 --seam-sweep`
  changes mean `rel_sol` from equal-mean `5.156246%` and direct-mismatch
  `4.999177%` to `4.943157%`. Relative to equal mean, broad transition-zone
  error RMS changes by `-7.391%` and trace-error jump RMS by `-22.425%`.
  Its maximum neighboring weight jump is `0.023329`, versus `0.242220` for
  direct mismatch. Relative to direct mismatch, mean `rel_sol` improves by
  `1.121%`, broad transition RMS is effectively tied (`-0.051%`), and trace
  jump is `8.160%` worse. Thus separating detection from a wide C2 profile
  removes the sensor's sharp weight variation and improves this run's global
  solution metric, but it does not dominate the direct sensor or the
  geometry-only ramp on every seam diagnostic.
- The 80-combination seam sweep is test-target sensitivity analysis only.
  `gamma=0.3,width=12h` gives the lowest mean `rel_sol`,
  `gamma=0.3,width=6h` gives the lowest broad transition RMS, and
  `gamma=0.5,width=6h` gives the lowest trace jump. Peak-relative thresholds
  from `0.15` to `0.30` select the same top-two seams in this run. None of
  these values is a production default without frozen calibration and
  independent data. Canonical outputs are under
  `checkpoints/annulus_CDR/coupling5/cross_axis_blend_detected_seam_c2_comparison/`.
- General local weak-residual reliability comparison: run
  `PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/compare_weak_residual_reliability_blend.py --config checkpoints/annulus_CDR/coupling5/config_used.json --coupling-checkpoint checkpoints/annulus_CDR/coupling5/complex_coupling_model_best_energy.safetensors --green-checkpoint checkpoints/annulus_CDR/green/model.safetensors --outdir checkpoints/annulus_CDR/coupling5/weak_residual_reliability_blend_comparison --geometry-gamma 0.5 --seam-gamma 0.3 --seam-ramp-width 0.09375 --weak-gamma 0.5 --weak-smoothing-steps 2 --weak-relative-floor 0.1 --weak-sweep`.
  This frozen-checkpoint diagnostic compares the equal mean, geometry-only C2
  ramp, mismatch-detected seam C2 ramp, and a geometry-transition-independent
  local reliability blend. For each candidate `u`, the last estimator applies
  the existing directional P1 weak operators and forms
  `R(u)=Rx(u;phi)+Ry(u;psi)` without solving a global matrix system. Smoothed
  mass-normalized squared residuals define
  `theta=gamma*(eta_psi^2-eta_phi^2)/(eta_phi^2+eta_psi^2+2*floor)` and the
  partition `w_phi=(1+theta)/2`, `w_psi=1-w_phi`. The weights use predictions,
  projected directional sources, coefficients, and axial geometry, but not
  `sol` or target `phi/psi`.
- On the frozen `coupling5` 50-sample test set, the fixed weak-residual preset
  changes mean `rel_sol` from `5.156246%` to `4.564722%` (`-11.472%` relative),
  transition-zone error RMS by `-9.870%`, and transition trace-error jump RMS
  by `-48.785%`. It improves `rel_sol` on `49/50` samples and trace jump on
  `50/50`; the paired-bootstrap 95% interval for aggregate `rel_sol` change is
  `[-12.736%,-10.261%]`. Its pointwise weights still contain axial stripe
  structure and have maximum neighboring jump `0.315646`, so this is evidence
  for a useful general reliability signal, not proof that continuity has been
  structurally restored. The 36-case parameter sweep ranks candidates using
  the same test reference and is sensitivity analysis only. Outputs include
  per-sample and sweep CSVs, selected raw NPZ arrays, Plotly figures,
  `summary.json`, and `diagnosis_report.md` under the command's output folder.
- Optional complex final reconstruction: `coupling_model.cross_axis_reconstruction`
  can enable the same local weak-residual reliability rule in production
  evaluation and artifact export. The backward-compatible default is disabled,
  so `u_pred=0.5*(u_phi+u_psi)`. With
  `mode="local_weak_residual_reliability"`, both candidates are tested with the
  existing full directional P1 weak operators,
  `R(v)=R_x(v;phi)+R_y(v;psi)`, and the mass-normalized indicators
  `eta_v^2=R(v)^2/(m_x+m_y+eps)` are smoothed for two 50:50 steps on
  `x_edges union y_edges`. The resolved weights are
  `theta=gamma*(eta_psi^2-eta_phi^2)/(eta_phi^2+eta_psi^2+2*tau)`,
  `w_phi=(1+theta)/2`, and `w_psi=1-w_phi`; the default preset uses
  `gamma=0.5` and `relative_floor=0.1`. The option is complex-only and affects
  only the detached trainer `rel_sol`, evaluator `u_pred`, and complex
  artifacts. It does not alter the training objective, checkpoint selection,
  projection, directional sources, directional reconstructions, `rel_flux`,
  or model state keys; it reads no solution/flux targets and performs no
  global matrix solve. Geometry-only compact C2 and mismatch-detected seam C2
  remain post-hoc diagnostics, not production modes. See
  `docs/complex_local_weak_residual_reliability_reconstruction.md` for the full
  contract.
- Local weak-residual smoothing ablation: frozen-checkpoint evaluation compared
  `smoothing_steps=0` with the production preset `smoothing_steps=2`, holding
  `gamma=0.5`, `relative_floor=0.1`, and all directional reconstructions fixed.
  On Poisson `coupling18`, two smoothing steps reduce mean `rel_sol` from
  `3.432811%` to `3.405442%` (`-0.797%` relative, `37/50` samples), transition
  trace-jump RMS by `11.297%`, and the maximum neighboring weight jump by
  `35.247%`; broad transition error RMS instead increases by `2.176%`. On CDR
  `coupling8`, smoothing reduces mean `rel_sol` from `3.025313%` to `2.970490%`
  (`-1.812%`, `44/50`), broad transition RMS by `0.376%`, transition trace-jump
  RMS by `12.129%`, and maximum neighboring weight jump by `26.065%`. The
  unsmoothed estimator still improves over equal mean in both PDEs, so smoothing
  is not required for validity; it remains the preferred default because it
  consistently improves global solution error and trace regularity. Evidence is
  under `coupling18/weak_residual_smoothing_ablation` and
  `annulus_CDR/coupling8/weak_residual_smoothing_ablation`, with the paired
  no-smoothing outputs in the adjacent `weak_residual_no_smoothing` directories.
- `annulus_CDR/coupling6` production audit: the best-energy checkpoint is epoch
  50. Its local weak-residual reconstruction lowers 50-sample mean `rel_sol`
  from `5.225503%` (equal mean) to `4.680390%` (`-10.432%`), transition RMS by
  `-9.161%`, and transition trace-jump RMS by `-49.401%`; it wins global
  `rel_sol` and trace jump on all 50 samples. The concurrently introduced
  column-diagonal Green-response projection lowers boundary energy versus the
  earlier physical-symmetric `coupling5` run but raises mean `rel_flux` from
  `17.898558%` to `22.001710%`. Treat the final reconstruction as supported and
  the projection as an unresolved boundary/flux tradeoff. See
  `checkpoints/annulus_CDR/coupling6/weak_residual_reliability_analysis/diagnosis_report.md`.
- Coupling null-space diagnostics: evaluation also exports `null_sol_x`, `null_sol_y`, and `null_sol_residual` heatmaps, where `q` is inferred from the flux errors and integrated with the pretrained Green kernels to visualize the hidden null-space contribution in solution space.
- Coupling closure diagnostics: evaluation also exports `closure_phi_residual` and `closure_psi_residual` heatmaps for the exact-flux baseline `L_fd(G(phi_exact)) - phi_exact` and `L_fd(G(psi_exact)) - psi_exact`, where `L_fd` is the conservative stencil for `-d_s(a d_s u) + b d_s u + c u`.
- Evaluation batching: CouplingNet evaluation uses `coupling_training.batch_size` to batch computations while still saving plots per sample.
- Plot export parallelism: CouplingNet evaluation exports all per-sample heatmaps with a process pool (default `plot_workers=4`).
- Per-line CSV bar comparison: use `python plot_per_line_bars.py --csv-a <run_a/per_line_metrics.csv> --csv-b <run_b/per_line_metrics.csv> --label-a run_a --label-b run_b --outdir <output_dir>` to generate grouped-bar figures for `(x, rel_sol_line_mean)`, `(x, val_rel_sol_line_mean)`, `(x, rel_green_line_mean)`, `(y, rel_sol_line_mean)`, `(y, val_rel_sol_line_mean)`, `(y, rel_green_line_mean)`. Training and validation solution figures include error bars from `rel_sol_line_std` and `val_rel_sol_line_std`; Green-function figures use `rel_green_line_mean` only. Outputs are saved as `.png` and `.pdf`.
- WCCM GreenNet evidence layout: all three Auto-Animate states align the bottom of the left main visual with the bottom of the right signed-error/diagnostics stack. The enlarged fixed-η state is the sizing reference; the kernel state expands its two kernel cards to the shared state height, while the takeaway state constrains the side stack within the slice row and keeps the takeaway as a separate row.

## Development

- Tests first: `PYTHONPATH=src pytest test`
- Lint/format/type-check: `ruff check src`, `ruff format src`, `mypy src`
- Key dirs: `src/` core code (including axial-line sampler and cleaned runner), `cli/` CLIs, `configs/` JSON configs, `checkpoints/` outputs, `test/` tests.

## References
- Axial Green's Function Method (see `refenreces/` PDFs) as conceptual inspiration for the axial decomposition in `GreenONetModel`.
