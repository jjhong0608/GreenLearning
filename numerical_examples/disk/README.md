# Example 2: Disk Preconditioning Study

Eight paired training configs compare identity and separable at fixed K=2.
Seeds 0/2 run on A40 cuda:1; seeds 1/3 run on Mac CPU. Both variants of one
seed stay on the same machine. All paths are project-relative.

- Layout: `nvidia_a40/seed{0,2}/disk_{identity,separable}_seedN.json` and
  `mac_studio/seed{1,3}/disk_{identity,separable}_seedN.json`.
- Base architecture and training protocol: Example 1 primary256/trunk-on
  config, with source/coefficient concat fusion and concat transverse trunk.
- PDE: radius-0.5 Disk, asymmetric diffusion
  `a=1+0.5*sin(2*pi*x)*sin(4*pi*y)`, no convection or reaction.
  `coefficients.py` preserves the definition from `disk_old/coefficient.py`.
- Shared Green checkpoint:
  `checkpoints/numerical_examples/disk_diffusion/green/model.safetensors`.
- Shared test set:
  `data/complex_samples/circle_radius_05_1_128_sinusoidal_diffusion/test`.
  The historical `circle` name is intentional; geometry uses `disk_radius_05_1_128.npz`.
- New response normalization and independence/line-search thresholds 1e-12
  are identical across conditions. Identity selects exact D=I; separable keeps
  relative_lambda=0.01. K is explicit, not selected automatically.
- Only the training seed varies across seed pairs. Fixed indexed-GP source seed
  is zero, with 4800 training / 300 validation sources, lengthscale 0.15.
- SOAP, batch 200, 100 epochs, 2400 optimizer calls, warmup 240 steps,
  validation every 24 steps. Canonical bulk energy only; auxiliary losses off.
- Existing weak reconstruction settings are retained, not an experimental axis.
  Use matching reconstruction metrics when comparing preconditioners.
- Best-energy artifact export and per-run tangent context sidecar are enabled.
  Artifact output is `<work-dir>/artifacts_best_energy`.

No training has been launched. Mac root:
`/Users/jjhong0608/Documents/ComplexGeometryGreenNet`; Python:
`/Users/jjhong0608/.local/share/mamba/envs/sciml/bin/python`.
Copy the new response-normalization implementation and shared input files to
both machines before running. These configs do not start an MPS server or impose
an MPS resource limit; concurrency belongs to the launcher, not the config.
