# Annulus Diffusion-Reaction Reconstruction Study

## Status: Configs Finalized, GreenNet and Test Data Pending

The four CouplingNet JSON configs use the confirmed radius-0.2/0.5 Annulus and Smooth
variable diffusion-reaction coefficient. All paths are project-root-relative
versions of the absolute paths selected by the user. The coefficient exists,
but the new GreenNet checkpoint and reference test data still need to be
generated: **configuration is complete, but CouplingNet training must wait for
those inputs**. The standalone GreenNet config below needs only the existing
geometry and coefficient, not the 2D reference dataset.

| Config key | Confirmed path |
| --- | --- |
| `dataset.coefficient_functions_path` | `coefficients/Smooth_Variable_Diffusion_Reaction.py` |
| `pipeline.green_pretrained_path` | `checkpoints/numerical_examples/annulus/green/model.safetensors` |
| `dataset.test_path` | `data/complex_samples/annulus_02_05_1_128_reaction_diffusion/test` |

The physical operator is `-div(a grad u) + c u = f`, with homogeneous Dirichlet
conditions on both circles. The unchanged coefficient file defines
`a = 1 + 0.5*sin(2*pi*x)*sin(2*pi*y)`, zero convection, and
`c = 0.5*(1 + 0.5*cos(2*pi*x)*cos(2*pi*y))`. This choice preserves x/y exchange
symmetry in the coefficient, so the reconstruction study does not introduce
the extra y-direction diffusion frequency of `Diffusion_Reaction_Ver2.py`.
The physical reaction definition and existing directional c/2 split are unchanged.
The new GreenNet checkpoint and reference test data must use this coefficient and
`data/geometry/annulus_02_05_1_128.npz`. Do not substitute the existing Annulus
Poisson/CDR checkpoints or the radius-0.5/1.0 diffusion-reaction dataset.

The matching visualization mesh is now available at
`data/visualization_mesh/annulus_02_05_1_128_mesh.npz`, generated from the shared
geometry with `examples/annulus_gmsh.py`. All four reconstruction configs set
`coupling_artifacts.visualization_mesh` to this path, so automatic best-energy
export includes mesh figures in addition to scatter figures. The GreenNet-only
config is unchanged. For a manual mesh export, pass
`--visualization-mesh data/visualization_mesh/annulus_02_05_1_128_mesh.npz`.

## Shared Training Protocol

### Shared GreenNet

`annulus_green.json` trains one seed-0 GreenNet for all four CouplingNet seeds.
It derives from `configs/complex_green_soap.json`, with the Annulus geometry and
Smooth coefficient. Explicit `warmup_steps=20` and `validation_every_steps=1`
follow the completed Disk GreenNet experiment's step-based schedule. Other
optimizer, model, quadrature, and LBFGS settings remain those of the template;
saved-run provenance is not copied into the input config.

Run from the project root (this command is provided for the user, not executed
as part of config preparation):

```bash
PYTHONPATH=src ~/.conda/envs/green_net/bin/python cli/train.py \
  --config numerical_examples/annulus/annulus_green.json \
  --work-dir checkpoints/numerical_examples/annulus/green
```

The Green-only pipeline writes `model.safetensors` directly under this work
directory, matching the checkpoint path in every CouplingNet config. It also
saves `model_pre_lbfgs.safetensors`, config/provenance, and training logs.
Use a fresh work directory rather than overwriting an earlier run.

GreenNet uses float64 on `cuda:1`, seed 0, 25 forward-generated training samples
and 5 validation samples per connected axial interval, and 129 branch nodes.
The model has hidden dimension 128, depth 4, rational activation, and Fourier
dimension 64. SOAP runs for 200 epochs with batch size 20000, base learning rate
0.005, minimum learning rate 1e-5, and 20-step linear warmup followed by cosine
decay. Subsequent LBFGS runs for 100 epochs at learning rate 1.0, with at most
100 internal iterations per optimizer step and history size 100; the SOAP
scheduler does not apply to LBFGS. Green reconstruction uses order-16 split
Gauss-Legendre quadrature with cubic interpolation and source sampling factor 4.
No FEniCSx reference samples, CouplingNet source generator, or tangent settings
are required to train this GreenNet. Kernel accuracy must be checked after
training before using the checkpoint in the paper experiments.

### CouplingNet Seeds

Train `annulus_reconstruction_seed{0,1,2,3}.json` once each, in separate work
directories. Only `coupling_training.seed` differs between the four files.
The indexed-GP source seed stays at zero, so all runs use the same fixed
4,800 training and 300 validation sources. Confirm that the eventual reference
test sources are independent of both sets before using the study in the paper.

The starting template is
`numerical_examples/pentagram/nvidia_a40/seed0/pentagram_k4_seed0.json`.
The model keeps source/coefficient branches with a concat fuser and the
primary/transverse C-trunk. Geometry branch, transverse branch, pre-projection
fuser, stationarity loss and response-trust loss are disabled. Diffusion and
reaction conditioning are active; convection conditioning is disabled.

All runs use float64, explicit `cuda:1`, SOAP, batch size 200, 100 epochs
(2,400 optimizer calls), 240 warmup steps, and validation every 24 steps.
The loss is canonical energy with boundary weight zero. Fixed, uncapped K=4
uses the separable tangent preconditioner; geometry auto-selection is disabled.
On this discretization, K=4 saturates the geometry-only structural reach, not
necessarily numerical convergence. No additional K ablation is included here.

## One Checkpoint, Two Reconstruction Rules

Select one validation best-energy checkpoint per training seed. Do not train
separate equal-mean and weak-reliability models or choose a checkpoint by either
test solution error. Reconstruction is detached from the training objective.

The configured weak preset is `gamma=0.5`, `smoothing_steps=2`,
`smoothing_relaxation=0.5`, `relative_floor=0.1`, and `eps=1e-12`.
Production evaluation already reports both `rel_sol` (weak reconstruction) and
`rel_sol_equal_mean` from the same directional predictions. Automatic export
writes `artifacts_best_energy` under each work directory and retains the
equal-mean fields and errors alongside the official weak reconstruction.
The reusable tangent context is saved separately in the work directory.

The paper comparison should also inspect directional weights and errors near
horizontal/vertical segment-topology transitions, distinguish overlapping
transition bands, and report bulk and tail accuracy. Geometry may define the
diagnostic bands but must not replace the production weak-residual weight rule.
Reference solutions are only for test diagnostics, never training or weighting.

## Verification Boundary

`test/test_annulus_numerical_configs.py` checks strict parsing, the four-seed
matrix, the Green-only pipeline, and the fixed model/objective/reconstruction
settings without loading
a model, building a response context, running training, or requiring untracked
checkpoints. It also locks the three confirmed input paths. Passing it does not
certify the pending PDE inputs. Before launch,
check their provenance, the GreenNet architecture, test NPZ geometry/coefficient
consistency, and train/validation/test source separation.
