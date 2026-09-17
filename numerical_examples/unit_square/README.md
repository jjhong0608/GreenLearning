# Example 1: transverse-trunk ablation

Unit-square Poisson, complex-model path. Eight paired training configs:

| Hardware | Seeds | Configs per seed | Device |
|---|---|---|---|
| nvidia_a40 | 0, 2 | unit_square_trunk_off_seedN.json / unit_square_trunk_on_seedN.json | cuda:1 |
| mac_studio | 1, 3 | unit_square_trunk_off_seedN.json / unit_square_trunk_on_seedN.json | cpu |

Four additional `unit_square_primary_w428_trunk_off_seedN.json` configs use the
same hardware/seed folders. Each is its original off config plus only
`coupling_model.primary_trunk_hidden_dim=428`. Source width and latent output
remain 256. Counts: original off 560,182; on 955,994; wide off 958,018.
Use separate work directories for the wide controls; they require new training
and cannot load the original 256-width checkpoint. None is launched by this change.

Within each original pair only `coupling_model.axis_1d_trunk.transverse_trunk.enabled`
differs. Across seeds only that flag, training seed and device differ.
The on condition uses concat fusion; the off condition uses the primary trunk alone.
Disabled trunk's length_context setting is retained but has no runtime effect.

Common settings: float64, 4800/300 fixed indexed-GP sources (source seed 0),
K=2 explicit/separable/uncapped, canonical bulk-energy-only objective, SOAP,
batch 200, 100 epochs, 2400 optimizer calls, 240 warmup steps, validation every
24 steps. Reference diagnostics for training/validation remain off.
Weak reconstruction is the primary prediction; artifact export also records
equal-mean and directional results. Best-energy artifacts are written under each
work directory's `artifacts_best_energy/`; tangent context is saved per run.

Paths are project-relative. Shared Green checkpoint:
`checkpoints/poisson_unit_square/green/model.safetensors`.
The Poisson coefficient file duplicates the unchanged definition in
`numerical_examples/unit_square_old/coefficients.py`. Previous experiments remain
under `unit_square_old/`; their configs are not rewritten.

Mac launcher paths (not JSON settings):
- Project: `/Users/jjhong0608/Documents/ComplexGeometryGreenNet`
- Python: `/Users/jjhong0608/.local/share/mamba/envs/sciml/bin/python`

Use a distinct work directory for each config. Concurrency, CPU thread limits,
and CUDA MPS settings belong in a future launcher, not these scientific configs.
Compare paired architectures within hardware; do not pool wall-clock costs across
A40 and Mac. Same seed does not guarantee identical shared-layer initialization
across different architectures. Mac code/assets/environment must be synchronized;
local parser checks do not certify the remote runtime or torch.compile backend.

Static validation only (does not train or export):
`python -m pytest test/test_unit_square_trunk_configs.py`
