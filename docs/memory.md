# Coupler Repo Memory

이 문서는 Coupler repo에서 사용자와 Codex가 합의한 연구 맥락, 모델 구조,
coefficient 의미, 실험 설계 기준, 논문용 데이터/figure 생성 기준을 기록한다.
`README.md`는 사용법, `AGENTS.md`는 작업 규칙, 이 문서는 연구/실험 의사결정
기억을 담당한다.

## Stable Project Context

- 이 repo의 중심 목표는 axial Green function 관점에서 2D PDE solution을
  재구성하는 GreenONet/CouplingNet 실험을 수행하는 것이다.
- GreenONet은 axial Green kernel 또는 그 근사를 학습/제공한다.
- CouplingNet은 axial decomposition을 이용해 solution/flux reconstruction을
  학습한다.
- 주요 PDE family는 Poisson, variable diffusion, diffusion-reaction,
  convection-diffusion이다.
- 논문용 결과 생성에서는 coefficient family, model variant, train/eval split,
  metric, figure 저장 규칙을 명확히 분리한다.

## Stable Modeling Decisions

- Convection coefficient는 scalar `b_fun`보다 vector form
  `bx_fun(x, y)`, `by_fun(x, y)`를 권장 API로 사용한다.
- Legacy coefficient file이 `b_fun(x, y)`만 제공하면
  `bx_fun = b_fun`, `by_fun = b_fun`으로 해석한다.
- 내부 tensor convention은 `b_vals[0] = b_x`, `b_vals[1] = b_y`를 유지한다.
  별도 `b_x`/`b_y` tensor field를 만들지 않는다.
- `branch_coefficient`는 diffusion/convection/reaction coefficient를 통합하는
  generic coefficient branch이다.
- `coupling_model.coefficient_terms` config가 diffusion, convection, reaction
  입력 여부를 제어한다.
- `coupling_model.branch_fusion.mode`는 CouplingNet branch feature 결합 방식을
  제어한다. 기본값 `product`는 기존 multiplicative fusion을 보존하고,
  실험 옵션 `product_fuser`는 active branch features와 그 component-wise product를
  concat한 뒤 learned fuser로 최종 branch representation을 만든다.
- `source_stencil_lift.enabled=true` path와 standard coefficient branch path는
  의미가 다르므로 섞지 않는다.
- `terminal.width`는 top-level runtime/logging config이며 Rich terminal wrapping을
  제어한다. `training.log` file output과는 별도 surface로 취급한다.

## Coefficient Families

- `Pure_Poisson.py`: constant diffusion, zero convection, zero reaction.
- `Sinusoidal_Diffusion_Only.py`: variable diffusion, zero convection,
  zero reaction.
- `Sinusoidal_Diffusion_Only_Ver2.py`: alternate smooth variable diffusion,
  zero convection, zero reaction.
- `Diffusion_Reaction_Ver2.py`: variable diffusion with reaction,
  zero convection.
- `Smooth_Variable_Diffusion_Reaction.py`: smooth variable diffusion with reaction,
  zero convection.
- `Convection_Diffusion_Reaction.py`: variable diffusion with convection and reaction.
- `Annulus_Convection_Diffusion_Reaction.py`: `Convection_Diffusion_Reaction.py`와
  동일한 diffusion/reaction을 사용하고, `inner_radius=0.2`,
  `outer_radius=0.5`에서 0이 되는 smooth counter-clockwise tangential
  convection을 사용한다. Convection amplitude scale은 `0.5`이고, radial
  polynomial envelope 때문에 annulus 내부 vector field는 divergence-free이다.
- `Divergence_Free_Convection_Diffusion.py`: variable diffusion,
  divergence-free convection with amplitude `2.0`, zero reaction.
- 새 coefficient file을 추가할 때는 `a_fun`, `apx_fun`, `apy_fun`,
  `bx_fun`, `by_fun`, `c_fun`을 정의한다.

## Important Mathematical Checks

- `apx_fun`, `apy_fun`은 반드시 `a_fun`의 해석적 편미분과 일치해야 한다.
- Divergence-free convection은 `d bx / dx + d by / dy = 0`을 만족해야 한다.
- Reaction-free 문제는 `c_fun(x, y) = 0`이어야 한다.
- Green accuracy가 좋아도 solution reconstruction이 나쁠 수 있다. 이런 경우
  kernel 사용 경로, coefficient alignment, axis convention, normalization,
  quadrature/integration path를 우선 의심한다.
- Convection-diffusion 문제에서 `bx_fun`은 x-direction line/operator에,
  `by_fun`은 y-direction line/operator에 들어간다.

## GreenNet / GreenONet Notes

- GreenNet은 특정 coefficient problem에 대한 Green's function을 학습한다.
- 이 repo의 논문용 문제 설정에서는 2D coefficient field를 문제별로 고정한다.
- 하지만 GreenNet이 다루는 것은 full 2D Green's function 하나가 아니라 axial
  decomposition에서 생기는 1D Green's function들이다.
- 따라서 같은 2D coefficient problem 안에서도 각 x-axis/y-axis line이 보는
  1D operator coefficient slice가 달라질 수 있고, 각 축선의 Green's function도
  서로 다르다고 해석해야 한다.
- 현재 GreenNet 학습의 training/validation dataset은 저장된 외부 dataset을
  읽는 방식이 아니라 config의 sampling 설정으로 `ForwardSampler` 또는
  `BackwardSampler`가 즉석 생성한 `TrainingData`를 `AxialDataset`으로 감싸는
  방식이다.

## Model And Loss Notes

- `balance_projection.enabled=false`이면 CouplingNet은 projection 없이 raw
  output을 반환하는 path를 사용한다.
- `balance_loss`는 projection off일 때만 허용한다.
- `symmetric_boundary_loss`는 symmetric projection on일 때 boundary condition을
  raw difference mode에 학습시키기 위한 loss이다.
- `symmetric_boundary_loss` 계산은 `CouplingTrainer._symmetric_boundary_loss`에
  있으며, projection output이 아니라 `CouplingNet.raw_flux_at_coords`로 얻은
  raw flux를 사용한다. Phi boundary residual은
  `(phi_raw_boundary - psi_raw_endpoint) + f`, psi boundary residual은
  `(phi_raw_endpoint - psi_raw_boundary) - f`이고, 각 edge 방향으로 적분한 뒤
  `0.5 * (phi_loss + psi_loss)`로 합친다.
- `smooth_mask` projection은 mask option을 갖는다. 기본은 `quadratic`,
  추가 option으로 `sin`을 사용할 수 있다.
- `axis_1d_trunk.enabled=true`이면 shared 1D trunk를 사용하고, transverse 정보는
  boundary-aware sin/cos branch로 넣는다.
- Axis-1D trunk의 boundary-aware encoding은 raw coordinate `t`를 포함하지 않고
  `sin(n*pi*t)`, `cos(n*pi*t)` for `n=1..k`만 사용한다.
- Complex geometry mode는 `dataset.geometry_mode="complex"`일 때만 사용한다.
  기존 unit-square path는 기본값 `unit_square`로 보존한다.
- Complex geometry v1 입력 계약은 precomputed geometry `.npz`와 full-grid sample
  `.npz` 조합이다. geometry 추출 자체는 이 repo의 v1 구현 범위가 아니다.
- Complex geometry full-reference sample은 full-grid `rhs`, `sol`을 필수로 갖고,
  flux target은 `phi`/`psi`를 우선 사용하며 legacy `uxx`/`uyy`를 fallback으로
  해석한다. Source-only train/validation sample은 `rhs`만 저장할 수 있다. 모든
  full-grid array는 `[row=y, col=x]` convention으로 valid point에 gather한다.
- Complex geometry Green interval normalization은 `a_unit=a_phys`,
  `ap_unit=L*ap_phys`, `b_unit=L*b_phys`, `c_unit=L^2*c_phys`,
  `f_unit=L^2*f_phys`를 사용한다. Unit reconstruction에서 Green kernel에
  추가 segment length factor를 곱하지 않는다.
- Complex geometry GreenNet은 geometry `.npz`의 connected x/y segment를
  flat interval list `N=Sx+Sy`로 펼쳐 학습한다. 각 connected interval은
  독립 1D domain이며, 같은 fixed coordinate에 놓인 disconnected segment도
  합치지 않는다. Trunk coordinate는 항상 unit `(t, eta) in [0,1]^2`이고,
  `dataset.samples_per_line`은 complex GreenNet mode에서 connected interval당
  synthetic sample 수를 뜻한다.
- Complex GreenNet의 `training.green_quadrature.enabled=true`는
  reconstruction loss, train/validation `rel_sol`, `evaluate(...)`, complex Green
  artifact reconstruction에만 split Gauss-Legendre 적분을 적용한다. Fine source는
  `source_sampling_factor`로 생성하고 Gaussian source node에서 `linear` 또는
  natural cubic source interpolation을 선택해 사용한다. Default는 `linear`이고,
  `cubic`은 smooth source 실험용 opt-in이다. `rel_green`은 기존 uniform-grid
  diagnostic으로 유지하며, CouplingNet reconstruction/loss/evaluation에는 이 설정을
  적용하지 않는다.
- Complex CouplingNet은 unit-square CouplingNet처럼 source-conditioned model이다.
  Full-grid `rhs`를 valid point로 gather한 뒤 endpoint hard-zero를 포함한
  segment-local physical source profile로 interpolation한다. Source amplitude는
  `sqrt(integral_0^1 f_phys(s(t))^2 dt)`이고, normalized physical profile을 branch에
  넣는다. Complex CouplingNet output contract version 5는 `(B,2,P)` directional
  response `[P,Q]`이며, normalized model output을 각각 `L_x^2*A_x`, `L_y^2*A_y`로
  scale한다. Unversioned 및 version 4 이하 complex CouplingNet checkpoint는 tensor
  shape가 같아도 load하지 않고 재학습한다. GreenNet checkpoint contract는 바뀌지 않는다.
- Complex CouplingNet coefficient branch는 `coefficient_terms`에 따라 active
  `[a,b_primary,b_transverse,c]` 순서로 구성한다. `convection=true`이면 x/Phi
  path는 `[L_x*b_x, L_x*b_y]`, y/Psi path는 `[L_y*b_y, L_y*b_x]`를 넣는다.
  GreenNet과 Green reconstruction branch는 primary convection만 사용하며
  `[a,ap,b_primary,c]` contract를 유지한다. `a'`는 GreenONet reconstruction
  query용 branch에는 보관하지만 CouplingNet coefficient branch에는 넣지 않는다.
  Technical/comparison 문서에서도 이 구분을 유지한다: transverse convection은
  CouplingNet의 source-split prediction context이고 Green reconstruction coefficient가 아니다.
- Complex CouplingNet primary trunk는 항상 segment-local 1D `t`를 사용한다.
  Fixed-line transverse branch는 global geometry extent 기준으로 normalized
  `r_hat`을 만들고, `axis_1d_trunk.num_frequencies`와 `max_frequency`로
  Fourier encoding한다. v6는 `axis_1d_trunk.transverse_trunk.enabled=true`와
  `length_context=true`를 필수로 둔다. 하나의 shared transverse MLP는
  `[t_perp,log(L_perp/L_ref),log(L_parallel/L_perp),kappa]` 네 feature를 함께 받으며,
  `kappa=4*L_parallel^2*L_perp^2/(L_parallel^2+L_perp^2)^2`이다. x/Phi path는
  `(L_parallel,L_perp,t_perp)=(L_x,L_y,y_local_t)`, y/Psi path는 축을 바꾼다.
  `fusion`은 `product` 또는 `product_fuser`이다.
  `trunk_positional_encoding`은 unit-square 2D trunk coordinate encoding이므로
  complex mode에서는 사용하지 않는다.
- Complex CouplingNet v6 balance projection은 `enabled=true`에서
  backward-compatible default `physical_symmetric`와 complex-only opt-in
  `column_diagonal_green_response`를 허용한다. Network raw output은
  reference response `P,Q`이고 `sigma_x=Lx^2`, `sigma_y=Ly^2`를 사용해 먼저
  `p=P/sigma_x`, `q=Q/sigma_y`로 physical directional-source proposal을 만든다.
  Default mode는 physical raw difference `d=p-q`를 보존하면서
  `phi=(rhs+d)/2`, `psi=(rhs-d)/2`로 symmetric projection을 적용한 뒤에만
  `Phi=sigma_x*phi`, `Psi=sigma_y*psi`로 reference response에 pull-back한다.
  Green reconstruction은 projected response `Phi/Psi`를 직접 사용하며 추가
  `L^2` scaling을 하지 않는다. Complex `response_space`,
  `response_preconditioned`, `symmetric`, `smooth_mask`, geometry-weighted mode는
  폐기되었고 v6에서 fail fast한다. Unversioned 및 v5 이하 complex CouplingNet
  checkpoint는 재학습 오류로 거부하며 GreenNet checkpoint는 그대로 재사용한다.
- Optional column-diagonal Green-response mode는 production reconstruction과 동일한
  connected-segment operator `H_s=K_s W_s L_s^2`와
  `M_Omega=(hx*hy)I_valid`를 사용해
  `gamma_s^2=diag(H_s^T M_Omega H_s)`를 계산한다. 이는 source column 하나가
  전체 directional solution을 움직이는 비용이며 evaluation row norm이 아니다.
  `gx_bar=gamma_x^2+eps`, `gy_bar=gamma_y^2+eps`와 fixed
  `gain_exponent=alpha`에 대해
  `w_phi=sigmoid(alpha*(log(gy_bar)-log(gx_bar)))`, `w_psi=1-w_phi`로
  correction을 나누고 마지막에 `psi=rhs-phi`로 exact balance를 유지한다.
  기본 exponent는 `1.0`이고 기존 direct ratio path를 그대로 사용한다. `alpha=0`은
  physical symmetric correction이며 `0<alpha<1`은 column anisotropy만 완화한다.
  Exponent는 fixed sample-independent scalar이고 learnable/sample-dependent alpha는
  지원하지 않는다. Frozen GreenNet gain과 tempered weight는 trainer/evaluator
  instance당 한 번 segment-wise 계산해 cache하며 full response/Gram matrix와 global
  solve는 만들지 않는다.
  Canonical config는 계속 symmetric이고, 성능 비교는 같은 초기화와 학습 조건으로
  두 mode 및 각 exponent를 각각 재학습해야 한다. 기존 checkpoint의 post-hoc
  projection/exponent 비교는 공정한 학습 비교로 해석하지 않는다.
- Frozen column-diagonal response audit는
  `cli/audit_complex_projection_response.py`로 실행한다. 동일한 frozen CouplingNet
  raw response와 residual에 fixed exponent들을 적용하고, Green-response context는
  한 번만 만든다. Primary metric은 diagonal surrogate와 실제 learned-Green
  correction response cost
  `hx*hy*(||H_x delta_phi||^2+||H_y delta_psi||^2)`이며, cross-axis transition에서
  source correction jump와 reconstructed correction jump를 별도로 기록한다.
  Row norm, full Gram matrix, global solve는 사용하지 않고 reference
  `sol/phi/psi`는 optional evaluation metric에만 사용한다. 이 결과는 projection의
  즉시 효과를 공정하게 비교하지만, 특정 exponent로 학습된 network의 end-to-end
  alpha 선택에는 paired retraining이 필요하다.
- 같은 frozen audit는 raw directional candidate identifiability도 별도로 기록한다.
  `metrics/per_sample_directional_candidate_audit.csv`에서 raw physical `(p,q)`,
  raw difference를 보존한 exact-balanced
  `p_tilde=(rhs+p-q)/2`, `q_tilde=(rhs-p+q)/2`, configured projected
  `(phi,psi)`를 optional target `phi/psi`와 비교한다. Global 및 transition-point
  pair relative error, raw balance defect/rhs norm, projection correction/projected
  norm을 함께 저장한다. Target fields는 이 diagnostic에서만 사용하며 training,
  projection, checkpoint selection에는 사용하지 않는다. 이 비교는 raw pair가
  physical candidate인지 projection-dependent latent coordinate인지 판별하기 위한
  evaluation-only audit이다.
- `annulus_CDR/coupling7` best-energy checkpoint의 frozen audit에서 alpha가
  `0 -> 0.25 -> 0.5 -> 1`로 증가할 때 mean actual correction-response cost ratio는
  `1 -> 0.561529 -> 0.353580 -> 0.239688`, directional transition response-jump
  ratio는 `1 -> 0.478732 -> 0.327955 -> 0.325758`이었다. 반대로 weight transition
  jump RMS는 `0 -> 0.122851 -> 0.185828 -> 0.247887`로 증가했다. 따라서 불연속
  weight가 곧바로 더 큰 reconstructed correction seam을 뜻하지 않으며, 이 run에서는
  의도된 response preconditioning으로 작동했다. 다만 alpha=0.25로 학습된 frozen
  network의 equal-mean `rel_sol`은 alpha=0.25에서만 `5.1622%`였고 alpha=1 post-hoc은
  `140.8773%`였으므로, response-cost 감소를 accuracy 개선으로 해석해서는 안 된다.
- 같은 `coupling7` checkpoint의 50-sample directional-candidate audit에서 valid-point
  pair relative target error의 mean은 raw `(p,q)` `188.03%`, symmetric-balanced
  `(p_tilde,q_tilde)` `50.39%`, configured alpha=0.25 projected `(phi,psi)`
  `17.27%`였다. Symmetric stage는 raw보다 50/50 sample에서, configured stage는
  symmetric보다 50/50 sample에서 작았다. Raw balance defect/rhs norm은 평균
  `226.95%`, configured correction/projected-pair norm은 `188.75%`였으므로 이
  checkpoint의 raw `(p,q)`를 작은 correction만 필요한 독립 physical directional
  candidate로 해석하지 않는다. Raw difference는 유용한 split signal을 포함하지만,
  raw pair 전체는 projection-dependent latent proposal이고 최종 projected
  `(phi,psi)`가 physical directional component이다. Configured target error는
  transition-point에서 `25.04%`, regular point에서 `16.51%`여서 projection이 전역
  candidate error를 크게 줄인 뒤에도 Annulus transition은 더 어려운 영역으로 남았다.
- Symmetric-balanced directional candidate를 출발점으로 하는 후속 post-hoc audit는
  `cli/audit_symmetric_tangent_response.py`로 실행한다. 목적함수는
  `0.5*||H_x(p_tilde+delta)-H_y(q_tilde-delta)||_M^2`이고, 한 번의
  matrix-free tangent update에서
  `g=(H_x+H_y)^T*M*m0`, `delta=-eta*D^{-1}g`를 사용한다. 여기서
  `D=gamma_x^2+gamma_y^2+(lambda_rel+eps)*mean(gamma_x^2+gamma_y^2)`이다.
  Segment-local Green blocks로 forward/transpose action만 수행하며 full Gram matrix,
  global matrix, solve는 만들지 않는다. 모든 후보는 `phi+psi=rhs`를 정확히 보존하고,
  reference `sol/phi/psi`는 update나 method selection이 아니라 evaluation에만 쓴다.
- `annulus_CDR/coupling7`의 50-sample, 72-combination frozen sweep에서 mean absolute
  response mismatch와 canonical energy를 함께 최소화한 tangent 설정은
  `eta=0.015`, `lambda_rel=0.1`이었다. Symmetric 대비 mean per-sample response ratio는
  `0.133889`, canonical-energy ratio는 `0.115515`, production-blend `rel_sol`은
  `0.973226 -> 0.274304`, `rel_flux`는 `0.514398 -> 0.405362`, transition solution-error
  jump RMS는 `0.00430626 -> 0.00116966`이었다. 반면 configured column alpha=0.25는
  각각 `0.005345`, `0.003374`, `0.045711`, `0.176215`, `0.000109589`로 tangent보다
  여전히 크게 우수했고, tangent는 이 configured method를 어떤 sample에서도
  이기지 못했다. Mean per-sample normalized response ratio만 최소화한 설정은
  `eta=0.01`, `lambda_rel=0.001`이지만 `rel_flux=0.637308`로 symmetric보다 악화됐다.
  따라서 damping은 source/solution consistency tradeoff에 필수이고, 현재 one-step
  tangent는 symmetric baseline 개선에는 유효하지만 frozen 결과만으로 기본 training
  projection으로 채택할 근거는 부족하다. 따라서 production에서는 optional mode로만
  제공하고, 최종 판단에는 projection별 paired retraining을 사용한다. 또한 coupling7은
  configured column projection으로 학습됐기 때문에 post-hoc 수치만으로 mode를
  선택하지 않는다.
- 위 training-projection bias를 분리하기 위해 동일 diagnostic은
  `physical_symmetric` checkpoint도 허용한다. `annulus_CDR/coupling5`는 coupling7과
  projection 및 optional cross-axis reconstruction 외의 config가 같고, symmetric로
  학습된 비교군이다. Coupling5 symmetric baseline은 response mismatch
  `8.17954e-7`, canonical energy `4.45929e-4`, equal-mean `rel_sol=0.0515625`,
  `rel_flux=0.178986`였다. 이는 coupling7 raw output을 post-hoc symmetric로 적용한
  값 `1.87090e-4`, `0.213606`, `1.028401`, `0.514398`보다 압도적으로 좋고,
  coupling7 configured column의 `8.42383e-7`, `4.40646e-4`, `0.0516218`,
  `0.176215`와 거의 같은 수준이다. 따라서 coupling7의 symmetric baseline 악화는
  column-trained raw representation과 symmetric projection의 mismatch가 주원인이다.
- Coupling5에서 balanced tangent 설정 `eta=0.01`, `lambda_rel=0.01`은 symmetric
  대비 response mismatch `-45.94%`, canonical energy `-4.54%`, equal-mean rel_sol
  `-27.61%`, rel_flux `-0.71%`였지만, transition solution-error jump RMS는
  `+5.08%`, split-transition jump RMS는 `+4.02%`였다. 즉 symmetric-trained
  checkpoint에서도 global response correction은 유효하지만 Annulus transition을
  개선한다는 증거는 없다. Coupling7에서 관찰된 큰 tangent improvement를 그대로
  training benefit으로 해석하지 않고, production integration은 보류한다.
- Coupling5 frozen candidates에 coupling7과 같은
  `local_weak_residual_reliability` reconstruction만 post-hoc 적용하면 symmetric
  baseline `rel_sol`은 equal mean `0.0515625`에서 weak `0.0456472`로 `11.47%`
  감소한다. Balanced tangent `eta=0.01`, `lambda_rel=0.01`은 equal mean
  `0.0373254`에서 weak `0.0343297`로 추가 `8.03%` 감소하며, weak symmetric보다
  `24.79%` 낮고 49/50 sample에서 개선된다. `eta=0.01`, `lambda_rel=0.1`은 weak
  `rel_sol=0.0346418`이고 50/50 sample에서 weak symmetric를 개선한다. Weak blend는
  symmetric transition solution-error jump RMS도 `2.17443e-4 -> 1.11364e-4`로
  약 절반 줄이지만, tangent `eta=0.01`, `lambda_rel=0.01`에서는
  `1.16880e-4`로 weak symmetric보다 `4.95%` 높다. 따라서 weak blend와 tangent는
  global rel_sol 측면에서는 상보적이지만, tangent가 transition을 추가 개선한다는
  결론은 여전히 성립하지 않는다.
- Complex CouplingNet production은 optional
  `balance_projection.mode="symmetric_tangent_green_response"`를 지원한다. Raw
  response를 physical `(p,q)`로 옮긴 뒤 symmetric-balanced
  `p_tilde=(rhs+p-q)/2`, `q_tilde=(rhs-p+q)/2`를 기준으로 삼는다. Frozen response
  operator에서 `m0=H_x*p_tilde-H_y*q_tilde`,
  `g=(H_x+H_y)^T*M_Omega*m0`를 계산하고,
  `D=gamma_x^2+gamma_y^2+(lambda_rel+eps_rel)*mean(gamma_x^2+gamma_y^2)`,
  `delta=-eta*g/D`를 한 번 적용한다. 최종 source는
  `phi=p_tilde+delta`, `psi=q_tilde-delta`여서 exact balance를 유지한다.
- Tangent `eta`, `relative_lambda`, `denominator_relative_eps`는 fixed
  sample-independent scalar이며 learnable parameter가 아니다. 기본값은 각각
  `0.01`, `0.01`, `1e-12`이고 `eta=0`은 physical symmetric와 동일하다. Row norm,
  full Gram, global response matrix와 matrix solve는 사용하지 않는다. Reference
  `sol/phi/psi`도 tangent update, loss, checkpoint selection에 사용하지 않는다.
- Tangent runtime은 GreenNet의 segment-local physical-source response blocks를 한 번
  구축하고 forward, adjoint, final reconstruction에 동일하게 재사용한다. Blocks는
  frozen이지만 batch matvec는 detach하지 않으므로 CouplingNet으로 first-order
  autograd가 전달된다. Model architecture, output contract version, safetensors key는
  바뀌지 않는다. 별도 paired experiment config는
  `configs/complex_coupling_soap_tangent.json`이며 canonical/column config는 유지한다.
- Tangent artifact는 selected raw archive에 symmetric physical/source response,
  pre/post mismatch, gradient, denominator와 delta를 저장하고, run-level context는
  `data/symmetric_tangent_green_response_fields.npz`에 저장한다. Context build count,
  fixed parameters, no-reference/no-solve convention을 summary와 log에 기록한다.
- Complex CouplingNet의 `coupling_model.pre_projection_fusion`은 optional
  single nonlinear fusion MLP이며 기본값은 disabled이다. Axis network의
  base response \(P_0,Q_0\)를 \(p_0=P_0/L_x^2\), \(q_0=Q_0/L_y^2\)로 옮기고,
  \(d_{\mathrm{base}}=p_0-q_0\),
  \(A=\sqrt{(A_x^2+A_y^2)/2}\), \(A_{\mathrm{safe}}=\max(A,\varepsilon)\)로 둔다.
  유일한 MLP 입력은
  \([d_{\mathrm{base}}/A_{\mathrm{safe}},f/A_{\mathrm{safe}}]\)이다.
  Default `mode="residual"`은
  \(d_{\mathrm{fused}}=d_{\mathrm{base}}+
  A_{\mathrm{safe}}h_\theta\)로 fixed identity skip을 유지하고, opt-in
  `mode="absolute"`는
  \(d_{\mathrm{fused}}=A_{\mathrm{safe}}h_\theta\)로 MLP output만 전체
  difference로 사용한다. Pre-projection pair는
  \(\phi_{\mathrm{pre}}=(f+d_{\mathrm{fused}})/2\),
  \(\psi_{\mathrm{pre}}=(f-d_{\mathrm{fused}})/2\)로 구성해 balance를 정확히
  만족한다. Learned linear branch, gate, convex combination, direct coordinate/
  geometry/length feature는 없다. Final MLP layer는 weight와 bias를 zero
  initialization하는 것이 기본이지만, `final_layer_init_scale`을 `[0,1]`에서
  설정해 standard `torch.nn.Linear` initialization을 선형 scale한다. Scale
  `0`은 zero init, scale `1`은 standard init이다. Scale `0`에서 residual
  mode는 disabled/base path로 시작하고 absolute mode는
  \(d_{\mathrm{fused}}=0\)인 symmetric split으로 시작한다. Source amplitude가
  zero이면 physical MLP output과 fused difference도 정확히 zero다. 이 block은 새 loss나
  reference `sol/phi/psi`를 사용하지 않고 output contract v6를 유지한다.
  Existing marker 없는 v6 single-MLP checkpoint는 residual mode에서만
  backward-compatible하게 읽는다. Mode marker가 있는 residual/absolute
  checkpoint는 서로 cross-load하지 않고 fail fast한다. Retired split-fuser
  checkpoint와 config는 호환하지 않으며 재학습한다. Paired absolute SOAP
  예시는 `configs/complex_coupling_soap_absolute.json`이다.
- Complex v6 training의 base split seminorm은 full-domain canonical physical
  energy이다. Residual `r=u_phi-u_psi`에 대해 모든 same-segment `x_edges`와
  `y_edges`의 diffusion face energy를 물리 spacing과 `hx*hy` area weight로 합하고,
  모든 connected-segment endpoint의 general boundary energy를 더한다.
  Regular/transition 분류, length-jump score, group normalization은 production
  objective에서 사용하지 않는다. `loss_energy_consistency`가 유일한 base split
  objective이자 best-energy checkpoint metric이다.
- Complex v6의 `relative_split_consistency`는 opt-in이고 dataclass 기본값은
  disabled이다. Enabled이면 raw canonical energy 대신 sample별
  `(E_canonical + mass_weight*D_ref^-2*h_x*h_y*sum((u_phi-u_psi)^2)) /
  (h_x*h_y*sum(rhs^2)+eps)`를 split objective로 사용한다. `D_ref`는 global x/y
  extent 중 큰 값이다. 이 mass term은 derivative energy가 보지 못하는 constant 및
  low-frequency split mismatch를 억제한다. Reference `sol/phi/psi`는 사용하지 않는다.
- Complex v6의 `weak_operator_closure`도 opt-in이고 dataclass 기본값은 disabled이다.
  공통 trial solution `u_pred=0.5*(u_phi+u_psi)`를 사용하고, 각 connected axial
  segment의 true endpoint를 포함한 P1 nodal weak residual을 계산한다. X form은
  `(a,bx,c/2,phi)`, y form은 `(a,by,c/2,psi)`를 사용한다. Coefficient는 physical
  element midpoint에서 직접 평가하고, endpoint trial/test/source 값은 hard zero이다.
  Nodal residual은 lumped directional mass와 physical source energy로 normalize한다.
  이는 reference-free variational closure이며 sparse matrix dependency 없이
  element gather/scatter로 구현한다.
- Complex total objective는 selected split objective에 enabled weak closure를
  더한다. Relative split이 꺼져 있으면 full-domain canonical
  bulk-plus-boundary energy가 split objective다. Reported canonical, bulk,
  boundary x/y, relative split, weak x/y/total metric은 실제 objective와 audit
  contribution을 구분해 기록한다. Transition-specific training metric은 생성하지
  않는다.
  Raw-balance gauge penalty는 이번 contract에서 구현하지 않으며 후속 현상이 남을
  때만 재검토한다.
- Canonical `configs/complex_coupling.json`은 energy-only v6 실험만 표현한다.
  Disabled opt-in인 `relative_split_consistency`, `weak_operator_closure`,
  `best_physics_checkpoint`, `best_rel_sol_checkpoint`는 config에서 생략하고 dataclass
  disabled default를 사용한다. `pipeline.run_green=false`이므로 Green `training`
  section도 생략하며, null `coupling_pretrained_path`를 쓰지 않아 새 CouplingNet을
  처음부터 학습한다.
- Complex canonical split energy는 valid-point physical bulk edge energy와 general
  boundary energy의 합이다. Boundary energy는 모든 connected x/y segment의 양쪽
  hard-zero endpoint를 해당 segment의 nearest represented interior node에 연결하는
  누락된 P1 edge contribution이다. Residual `r=u_phi-u_psi`에 대해 각 anchor는
  `a_i*r_i^2*h_perp/d_endpoint`를 더한다. Coefficient는 nearest valid point의 one-sided
  값을 사용하고, x endpoint의 transverse measure는 `h_y`, y endpoint는 `h_x`다.
  `loss_energy_consistency`는 이 bulk+boundary 합이며, 별도 length-balanced
  metric은 없다.
- `admissibility_gluing`, global self-trace loss, transition-only cross-axis carrier,
  trace/carrier metric과 artifact는 폐기되었다. General endpoint boundary energy가
  residual constant null mode를 제거하므로 carrier는 coercivity에 필요하지 않다.
  이전 config의 `admissibility_gluing` key는 호환하지 않고 fail fast한다.
- `cli/analyze_complex_energy_nullspace.py`는 canonical bulk와 all-segment boundary
  anchor를 bulk component-constant space에서 비교한다. 현재
  `annulus_02_05_1_128.npz`는 10,788 valid-point DOF, one connected bulk component,
  712 boundary anchors를 가진다. Nullity는 bulk `1`, bulk+general boundary `0`이며
  carrier objective는 필요하지 않다.
- `cli/audit_complex_energy_refinement.py`는 physical boundary 근처에서 0인
  grid-independent fixed-amplitude interior jump를 세 개 이상의 annulus refinement에서
  평가한다. 현재 `inner_radius=0.2`, `outer_radius=0.5`,
  `h={1/32,1/64,1/128}`에서 canonical energy는 `{28,60,124}`, log-log slope는
  `-1.0734`, `h*E_h` relative spread는 `0.1011`로 inverse-h acceptance를 통과했다.
- Complex v6는 reference-free checkpoint selection을 강제한다.
  `best_rel_sol_checkpoint.enabled=true`는 fail fast한다.
  `best_energy_checkpoint`는 validation `loss_energy_consistency`가 가장 작은
  `complex_coupling_model_best_energy.safetensors`를 저장하고,
  `best_physics_checkpoint`는 validation total reference-free loss가 가장 작은
  `complex_coupling_model_best_physics.safetensors`를 독립 저장한다. Reference
  `sol/phi/psi`는 detached evaluation metric/artifact에만 사용하며 gradient, loss,
  scheduler, early stopping, 두 checkpoint selection에 사용하지 않는다.
- Complex CouplingNet train/validation source는 모든 epoch에서 fixed이다.
  `dataset.coupling_source.mode`는 dedicated CLI가 만드는 deterministic
  source-only `npz`와 dataset 내부 `indexed_gp` provider 중 하나를 선택한다.
  두 backend는 shared GP factorization, geometry mask, 그리고
  `SeedSequence([base_seed,split_id,sample_index])` identity를 사용하므로 같은
  설정의 같은 sample은 bitwise 동일한 full-grid `rhs`를 만든다. Split ID는
  `train=0`, `valid=1`, `test=2`이다.
- `dataset.reference_diagnostics.training/validation`은 기본값 `true`로 기존
  full-reference NPZ 동작을 보존한다. Disabled split은 `rhs`만 읽고
  `has_solution=false`, `has_flux=false`와 shape-compatible zero placeholder를
  사용하며 `rel_sol`/`rel_flux` metric key를 생성하지 않는다. Placeholder는
  objective에 들어가는 reference target이 아니다. Indexed GP mode에서는 두
  diagnostic을 반드시 끈다.
- Source-only NPZ는 `cli/make_complex_sources.py`가 `green_net` environment에서
  FEniCSx 없이 serial로 생성한다. 각 file은 float64 full-grid `rhs` 하나만
  저장하고 outside-domain은 `0.0`이다. Test evaluation과 artifact export는 source
  backend와 무관하게 FEniCSx-generated `rhs/sol/phi/psi` full-reference
  `dataset.test_path`를 계속 사용한다.
- `configs/complex_coupling_soap.json`은 fixed runtime `indexed_gp` source를
  `num_train=4800`, `num_valid=400`, `seed=0`, `lengthscale=0.15`,
  `amplitude=1.0`, `mean=0.0`으로 사용한다. `batch_size=400`, `epochs=125`,
  `warmup_epochs=13`으로 약 1,500 total optimizer steps와 156 warmup steps를
  유지하면서 source별 반복 노출을 줄인다. Periodic checkpoint는 같은 step
  간격에 가깝도록 50 epoch마다 저장한다. Train/validation reference
  diagnostics는 모두 꺼져 있고, full-reference `test_path`만 detached test
  evaluation용으로 유지한다.
- Fixed indexed-GP source를 사용한 `coupling11_2000_train` 실험은 validation
  canonical energy가 epoch 133에서 `5.320162e-4`로 최소가 된 뒤 epoch 447에서
  `2.777888e-3`까지 증가한 반면 train energy는 계속 감소했다. Late validation
  bulk는 best 대비 4.38배, boundary는 9.32배 증가했으므로 명확한 fixed-source
  overfitting이다. Artifact의 fusion gate는 epoch-133 값과 일치하므로 현재
  artifact는 stopped state가 아니라 best-energy checkpoint를 나타낸다.
  이 실험은 source 수뿐 아니라 `hidden_dim=256 -> 384`와 parameter count
  `1,814,587 -> 4,048,827`도 함께 바뀌었으므로 source-count 단독 ablation으로
  해석하지 않는다. 상세 결과는
  `checkpoints/Annulus_poisson/coupling11_2000_train/analysis/`에 둔다.
- 완료된 `coupling12` Annulus 실험은 4,800 fixed indexed-GP train source, 300
  validation source, 기존 1,814,587-parameter architecture, 1,600 SOAP
  optimizer step을 사용했다. Validation canonical energy는 epoch 65에서
  `4.366951e-4`로 최소였고 epoch 100은 그보다 2.19% 높아 late overfitting이
  경미했다. Best-energy checkpoint의 mean test energy/`rel_sol`/`rel_flux`는
  `4.050432e-4`/`5.602%`/`17.310%`이며 `coupling11` 대비 각각
  24.03%/5.07%/9.37% 개선되었다. 그러나 inner-radius line-length-squared
  scale이 4.796배 바뀌는 `|x|,|y| ~= 0.2` transition seam은 남아 있다.
  Physical balance residual은 machine precision이므로 seam은 balance violation이
  아니다. 상세 분석은 `checkpoints/Annulus_poisson/coupling12/analysis/`에 둔다.
- `coupling12` best-energy checkpoint의 retired split linear/nonlinear
  pre-projection fuser를 inference에서만
  bypass한 50-sample paired ablation에서 fuser-on mean test energy/`rel_sol`은
  `4.050432e-4`/5.602%, fuser-off는 `2.568269e-3`/12.578%였다. Energy와
  `rel_sol`은 50개 전부 fuser-on이 더 좋았고 transition solution-error RMS도
  58.94% 낮았다. 반면 transition/bulk ratio 개선은 3.65%, mean `rel_flux`
  개선은 1.02%에 그쳤다. 따라서 이 checkpoint 안에서 fuser는 solution과
  canonical energy에 필수적인 learned difference-mode correction이지만 annulus
  seam 자체를 제거하거나 supervised flux split을 크게 개선하지는 않는다.
  Backbone과 fuser가 함께 학습되었으므로 이것은 same-checkpoint functional
  ablation이며 architecture-level causality에는 separately trained no-fuser
  control이 필요하다. 이 결과는 현재 single residual MLP의 동작을 의미하지
  않는다. 상세 산출물은
  `checkpoints/Annulus_poisson/coupling12/pre_projection_fuser_ablation/`에 둔다.
- Unit-square와 complex CouplingNet trainer는 같은 linear-warmup + cosine-decay
  learning-rate schedule을 사용한다. `coupling_training.use_lr_schedule=true`이면
  `warmup_epochs` 동안 `learning_rate`까지 선형 증가한 뒤 마지막 epoch의
  `min_lr`까지 cosine decay한다. Scheduler는 epoch의 optimizer update, validation,
  checkpoint 저장이 끝난 뒤 한 번 step한다. Complex trainer의
  `complex_training_metrics.csv`와 `training.log`에는 해당 epoch optimizer update에
  실제 사용한 `learning_rate`를 기록한다. `use_lr_schedule=false`이면 고정
  learning rate를 유지한다.
- Unit-square와 complex GreenNet의 first-stage optimizer 기본값은 AdamW이다.
  `training.optimizer.name="soap"`이면 두 GreenNet geometry path 모두 pinned
  SOAP을 사용하고, 명시적인 `"adam"`은 폐기된 optimizer이므로 fail fast한다.
  `training.weight_decay`, nested optimizer `betas`/`eps`, SOAP option은
  `TrainingConfig`의 strict parser를 통해 동일하게 적용한다.
- GreenNet의 `training.use_lr_schedule=true`는 AdamW 또는 SOAP first stage에만
  linear warmup과 cosine decay를 적용한다. 실제 epoch learning rate는
  `training.log`와 `green_training_metrics.csv`에 기록하고, first stage가 끝나면
  `model_pre_lbfgs.safetensors`를 저장한다. 이후 LBFGS는 기존 독립 learning
  rate, closure, strong-Wolfe line search를 그대로 사용하며 scheduler를
  적용하지 않는다.
- GreenNet run은 `green_optimizer_provenance.json`을 저장하고
  `config_used.json`과 Green artifact summary에 resolved optimizer/scheduler
  metadata를 materialize한다. Checkpoint는 계속 model-only safetensors이며
  optimizer/scheduler state resume는 지원하지 않는다. Optional optimizer
  profiling은 step timing, SOAP basis refresh, CUDA peak allocated memory를
  기록하고 CPU peak-memory 값 `0.0`은 not-measured sentinel이다.
- Complex CouplingNet optimizer의 backward-compatible default는 AdamW이다.
  SOAP은 `coupling_training.optimizer.name="soap"`인 경우에만 사용하는
  CouplingNet complex-only opt-in이며 unit-square CouplingNet은 SOAP을 거부한다.
  Vendored source는 official SOAP commit
  `a1e553530fde97d0e6b307d7c82ac6d38b072340`에 고정되고 MIT attribution은
  `THIRD_PARTY_NOTICES.md`에 둔다. SOAP `precondition_frequency`는 epoch가 아닌
  optimizer-step 단위이고, 첫 `step()`은 preconditioner만 초기화하며 parameter를
  갱신하지 않는다.
- Complex optimizer checkpoint는 계속 model-only safetensors이다. SOAP optimizer
  state를 이용한 interrupted-training resume는 지원하지 않는다. 각 training
  run은 resolved optimizer block/provenance를 `config_used.json`에 materialize하고,
  같은 설정을 `optimizer_provenance.json`에 기록하며 complex artifact summary도
  같은 provenance를 포함한다. `optimizer.profile_step_time=true`일
  때만 optimizer-step mean/p95/max, step count, periodic basis-refresh count,
  peak allocated CUDA memory를 측정한다. CUDA synchronization에 따른 timing
  overhead는 profiling opt-in에서만 발생한다. `optimizer_peak_memory_mib`는
  CUDA에서만 `torch.cuda.max_memory_allocated`를 읽으며 CPU에서는
  not-measured sentinel `0.0`을 기록한다. 따라서 CPU RSS 또는 SOAP optimizer
  state가 실제로 0 MiB라는 의미가 아니다.
- SOAP은 canonical energy의 null space, boundary admissibility, geometry
  adjacency, projection 또는 reconstruction 오류를 해결하는 수단이 아니다.
  유지 여부는 AdamW와 같은 seed/data/loss/scheduler를 사용한 300--500 epoch
  paired pilot에서 equal-step 및 equal-wall-clock 기준을 함께 비교한 뒤 결정하며,
  기본 optimizer로 자동 승격하지 않는다.
- AMUSE (Anytime MUon with Stable gradient Evaluation)는 아직 구현하지 않은
  complex CouplingNet optimizer 연구 후보이다. 2026-07-27 조사 기준 current
  SOAP pilot architecture의 1,814,586 trainable parameter 중 1,807,506개,
  즉 99.61%가 2D matrix이므로 Muon 구조 적용성은 높다. 그러나 AMUSE는
  drop-in `optimizer.step()` 교체가 아니다. Training에서는 optimizer
  `train()` 상태의 gradient-evaluation iterate를 사용하고 validation,
  best-energy selection, checkpoint 저장에서는 optimizer `eval()` 상태의
  averaged iterate를 사용해야 한다.
- AMUSE를 후속 구현할 경우 complex CouplingNet-only opt-in으로 제한하고
  AdamW/SOAP default와 GreenNet/unit-square path를 보존한다. Official AMUSE는
  optimizer-step linear warmup 뒤 constant learning rate를 사용하므로 external
  cosine decay와 동시에 사용하지 않는다. Model-only checkpoint는 averaged
  iterate만 저장하고 optimizer resume는 지원하지 않는다고 명시한다.
  Official Newton-Schulz path가 matrix update를 `bfloat16`으로 변환하므로
  current `float64` model에서는 precision policy와 numerical test를 반드시
  provenance에 포함한다. AMUSE 적용성 상세는
  `docs/amuse_optimizer_applicability.md`를 따른다.
- SOAP과 AMUSE는 matrix-parameter optimization을 개선한다는 상위 목적만
  공유하며 같은 preconditioner 계열로 취급하지 않는다. SOAP은 Shampoo
  covariance에서 얻은 slowly changing eigenbasis로 gradient를 회전해 그
  좌표계에서 Adam moments를 사용한다. AMUSE는 matrix momentum을 매 step
  Newton-Schulz로 orthogonalize하고, fast base iterate와 averaged iterate 사이의
  time-varying evaluation point를 사용한다. 따라서 SOAP hyperparameter,
  basis-refresh frequency, cosine scheduler를 AMUSE에 그대로 이전하지 않는다.
- Annulus `coupling10` SOAP pilot은 epoch당 optimizer call이 2회인 상태에서
  `lr=2e-3`, `warmup_epochs=3`, `betas=(0.95,0.95)`,
  `precondition_frequency=10`을 사용했고, epoch 3 validation canonical energy
  `3.375519e-1` 이후 epoch 4에 `1.081258e6`으로 발산했다. 첫 periodic basis
  refresh는 epoch 6이므로 refresh event 자체가 최초 발산 원인은 아니다. SOAP
  warmup은 epoch가 아니라 optimizer-step 수로 판단한다. 다음 안정성 pilot의
  보수적 시작점은 `lr=2e-4`, `warmup_epochs=50`,
  `betas=(0.95,0.99)`, `shampoo_beta=0.95`,
  `precondition_frequency=5`이며, 이는 canonical default가 아니라 20--30 epoch
  abortable screen용 실험 설정이다. Validation energy가 best finite value의
  10배를 넘으면 해당 설정은 즉시 실패로 판정한다.
- 같은 `coupling10`을 보수적 설정으로 다시 실행한 결과, epoch 71까지 validation
  canonical energy가 70회 연속 감소해 `3.481542e-1`에서 `3.148627e-1`로
  내려갔다. Warmup epoch 1--50의 감소율은 약 1.5%에 불과했지만 epoch
  50--71에서는 추가로 약 8.2% 감소했으므로 이 run은 plateau가 아니라
  post-warmup 수렴 구간이다. 절대 정확도는 아직 `rel_sol=1.177138`로 낮아
  optimization convergence와 solution quality를 구분해야 한다. Epoch
  120--150 전에는 설정을 바꾸지 않고, 최근 20-epoch validation energy 감소율이
  2% 이상이면 계속 실행한다. 그 감소율이 1% 미만이면서 detached `rel_sol`도
  정체 또는 악화될 때만 plateau로 판정한다. 이 기준은 현재 Annulus SOAP
  pilot 진단용이며 canonical optimizer default가 아니다.
- `coupling10`의 후속 epoch 199 시점에는 validation canonical energy가 198회
  연속 감소해 `3.481542e-1`에서 `1.260071e-2`로 내려갔지만, detached
  `rel_flux`는 epoch 141의 `4.325040e-1`, detached `rel_sol`은 epoch 154의
  `3.012693e-1`을 최저점으로 다시 증가했다. Epoch 199에서는 각각
  `4.490581e-1`, `3.154724e-1`이다. Training 쪽 최저 epoch도 flux 142,
  solution 155로 거의 같으므로 일반적인 train/validation overfitting이 아니다.
  이는 SOAP 비수렴이 아니라 canonical-energy objective와 reconstructed-solution
  quality의 late-stage misalignment다. Reference metric은 계속 evaluation-only로
  두며 loss나 checkpoint selection에는 사용하지 않는다. 이 run에서 optimizer
  설정을 다시 튜닝하는 것만으로 이 objective mismatch가 해결된다고 해석하지
  않는다.
- 별도 컴퓨터의 독립 work directory에서 실행한 AdamW 로그를 비교 목적으로
  `coupling10` 폴더에 복사했다. 따라서 현재 Linux SOAP checkpoint와 AdamW
  checkpoint가 서로 덮어쓰는 문제는 없다. 복사된 AdamW epoch 97 validation은
  energy `3.580980e-3`, `rel_sol=1.467270e-1`, `rel_flux=3.957876e-1`이고,
  이는 SOAP epoch 303까지의 전체 최저값보다 각각 약 70.7%, 51.3%, 8.5%
  낮다. 현재 workload의 실용 결과는 AdamW를 강하게 지지한다. 다만 AdamW는
  `lr=2e-3`, `epochs=3000`, SOAP은 `lr=2e-4`, `epochs=500`을 사용하고
  실행 컴퓨터도 다르므로 optimizer-only causal 비교나 wall-clock 비교로
  해석하지 않는다. Strict comparison에는 같은 hardware/seed/data/order,
  learning-rate schedule과 step budget을 맞춘 paired run이 필요하다.
- 다음 SOAP ablation은 안정화된 현재 recipe에서 `learning_rate`만
  `2e-4`에서 `2e-3`으로 바꾼다. `epochs=500`, `warmup_epochs=50`,
  `min_lr=1e-5`, `betas=(0.95,0.99)`, `shampoo_beta=0.95`,
  `precondition_frequency=5`, batch, seed, data order, clipping, weight decay,
  model과 hardware는 모두 고정하고 별도 work directory를 사용한다. 이 실험은
  이전 high-LR 실패의 `warmup=3` confound를 제거한다. Non-finite 값,
  validation energy가 best finite value의 10배 초과, 또는 3 epoch 연속 2배
  초과 중 하나가 발생하면 즉시 중단한다.
- `coupling10_2`는 위 single-variable ablation으로, resolved config상
  `coupling10`과 다른 값은 `learning_rate=2e-3`뿐이다. Epoch 81 snapshot에서
  50-epoch warmup 이후에도 finite하며 validation energy `2.494793e-3`,
  detached `rel_sol=1.259009e-1`, detached `rel_flux=3.395570e-1`로 세 metric
  모두 해당 run의 관측 최저값을 갱신했다. Low-LR SOAP의 epoch 309까지 관측
  최저값보다 각각 약 79.4%, 58.2%, 21.5% 낮으므로 `lr=2e-4`는 이 workload에서
  지나치게 보수적이었다. 이전 `lr=2e-3` 발산은 learning rate 단독 효과가 아니라
  짧은 warmup과 당시 beta/frequency 설정의 결합으로 해석한다. 다만 low-LR
  run의 objective-to-solution metric misalignment가 epoch 141--154부터
  나타났으므로 high-LR run도 epoch 100--160 이후 detached metric의 turning
  point를 계속 확인하고, 이 early snapshot만으로 SOAP을 기본 optimizer로
  승격하지 않는다.
- `coupling10_2`의 epoch 317 snapshot에서는 early improvement 이후 실제
  objective overfitting이 확인된다. Validation canonical energy는 epoch 194의
  `5.400105e-4`, detached validation `rel_sol`은 epoch 199의 `5.755842e-2`가
  최저점이며, epoch 317에는 각각 `8.833431e-4`, `6.438856e-2`로 악화되었다.
  같은 기간 train energy는 계속 감소해 `2.923218e-4`가 되었고, latest
  validation/train ratio는 total energy `3.02`, bulk `2.49`, boundary `7.67`이다.
  반면 train/validation `rel_flux`는 모두 약 `1.691e-1`로 계속 개선되고 거의
  일치한다. 따라서 이는 optimizer 발산이 아니라 boundary-dominated
  energy/solution generalization gap이며, reference-free 선택에는 epoch 194에
  마지막 갱신된 `complex_coupling_model_best_energy.safetensors`를 사용한다.
- Local SOAP 구현은 Shampoo covariance factor를 매 optimizer step 갱신하고,
  `precondition_frequency`마다 eigenbasis/QR만 갱신한다. 따라서 frequency를
  낮추면 basis가 더 자주 갱신된다. 현재 epoch당 optimizer step은 2회이므로
  frequency `5`, `2`, `1`은 각각 약 2.5 epoch마다, 매 epoch, 매 step refresh다.
  `coupling10_2` 후속 single-variable ablation은 `5 -> 2`를 우선하고, `1`은
  frequency 2가 같은-step metric을 개선할 때만 검토한다.
- 현재 CouplingNet의 모든 2D weight axis는 1024 이하이므로
  `max_precondition_dim=1024`는 모든 matrix를 양쪽 축에서 precondition한다.
  Weight parameter는 2D이고 1D parameter 비중은 작으므로
  `merge_dims=false`, `precondition_1d=false`를 유지한다.
  `normalize_grads`는 raw gradient가 아니라 projected-back adaptive update를
  parameter tensor별 unit RMS로 바꾸므로 작은 frequency에서는 false를 유지한다.
  `correct_bias=true`는 Adam moment의 초기 zero bias를 보정하므로 유지한다.
- `cli/diagnose_complex_length_response.py`는 complex CouplingNet checkpoint를 다시
  추론해 line-length amplification을 단계별로 분리하는 evaluation-only 도구이다.
  v6에서는 pre-projection physical proposal, projected physical directional-source
  error, projected reference-response error, exact Green
  source response, learned-minus-exact Green contribution, target-source exact closure를
  따로 기록한다. Production reconstruction은 projected response를 직접 사용한다.
  핵심 항등식은 `learned total error = exact source response +
  target exact closure + learned-minus-exact`이다. Exact path는 production과 같은
  segment node, endpoint hard-zero, nonuniform unit weight를 사용하며,
  `G_unit*(L^2 source)*w_unit`과 `(L G_unit)*source*(L w_unit)`의 등가성을 audit한다.
  Reference `sol/phi/psi`는 이 진단과 metric에만 사용하고 training loss에는 쓰지
  않는다.
- Complex geometry mode에서는 `cross_consistency`, `smooth_mask`, `balance_loss`,
  `source_stencil_lift`, `green_response_feature`를 사용하지 않는다. Cross 관련
  key는 metric, log, artifact에 남기지 않는 것을 contract로 둔다.
- `docs/unit_square_vs_complex_geometry.md`는 unit-square core path와 complex
  geometry path의 canonical 비교 문서이다. 이 문서의 unit-square 설명은 complex
  extension과 비교하는 데 필요한 GreenNet/CouplingNet core structure만 다루며,
  complex mode에서 사용하지 않는 unit-square-only auxiliary option을 나열하지 않는
  writing convention을 유지한다.
- `docs/unit_square_vs_complex_geometry_math.md`는 코드 surface를 제거한
  수학적/architecture 중심 비교 문서이다. 이 문서는 branch, trunk, projection,
  Green reconstruction, domain representation 차이를 설명하는 데 집중하고,
  config/file/schema/tensor-size contract는 engineering note인
  `docs/unit_square_vs_complex_geometry.md`에 남긴다.
- `docs/complex_geometry_greennet_couplingnet_technical_report.md`는 학회 발표자료
  준비를 위한 complex-mode-only technical report이다. 이 문서는 unit-square 확장
  narrative 없이 complex geometry PDE, axial connected interval decomposition,
  GreenNet unit pull-back, source-conditioned CouplingNet, physical balance
  projection, Green reconstruction, training/evaluation 해석을 수식과 알고리즘
  중심으로 설명하며, 코드 surface와 complex mode에서 사용하지 않는 기능은 넣지 않는다.
- 이 technical report의 GreenNet section은 branch-trunk learned correction과
  analytic Green structure의 결합식을 포함한다. 수식은 코드 surface 없이
  \(R_\theta\), \(G_0\), \(J_0\), \(A\), \(B\), \(E\), \(M\) 중심으로 설명하고,
  reaction coefficient는 analytic factor가 아니라 learned correction을 통해 반영된다고
  기록한다.
- 같은 report의 GreenNet dataset 설명은 forward construction만 다룬다. Smooth
  target solution은 Gaussian Process sample로 설명하고, GreenNet supervision은 exact
  kernel label이 아니라 source-to-solution reconstruction target으로 기록한다. Fine
  source와 quadrature 설명은 이 dataset construction 설명에서 제외한다.
- 같은 report의 GreenNet analytic wrapping 설명은 \(A(t)G_0\)를
  Dirac-\(\delta\) property를 만드는 jump/singularity 구조로 해석하고,
  \(B(t)(J_0-\frac12E)\)를 그 과정에서 생기는 Heaviside-type contribution
  cancellation term으로 설명한다. Learned correction은 이 analytic singular/cancellation
  구조 이후에 남는 smooth residual을 담당한다고 기록한다.
- 같은 report의 CouplingNet 설명에서는 "local trunk" 대신 "axial local trunk" 용어를
  사용한다. Axial local trunk는 primary axial coordinate \(t_{\parallel}\)를 담당하고,
  pointwise transverse trunk는 \(t_{\perp}\)와 parallel/perpendicular segment-length
  context 네 feature를 통해 pointwise boundary/response context를 전달한다. Transverse branch는
  global transverse placement만 담당하며 pointwise transverse boundary 정보를 대체하지
  않는다.
- 같은 report의 activation 설명에서는 rational activation을 fixed activation이 아니라
  learnable activation으로 설명한다. \(P_\alpha\)와 \(Q_\beta\)의 coefficient는
  현재 초기값에서 시작하지만 학습 중 업데이트되는 parameter이며, activation은 analytic
  Green wrapping이나 physical symmetric balance projection을 대체하지 않고 branch/trunk 내부의
  representation nonlinearity 역할을 한다.
- 같은 report는 energy analysis 문서를 단순 reference로 넘기지 않고, energy loss가
  final solution energy error를 bound한다는 핵심 의미를 독립적인 proposition 형식으로
  설명한다. Technical report 안에서는
  \(\mathcal{E}_{\mathrm{split}}=\|u_\phi-u_\psi\|_a^2\), \(H_0^1(\Omega)\) admissibility,
  source-linear weak inverse assumption,
  \(\|u_{\mathrm{pred}}-u_*\|_a\le\frac{C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}\) exact
  bound, imperfect Green reconstruction perturbation, common Green bias limitation을
  presentation-ready mathematical structural statement로 설명한다. 이 proposition은
  proof 대체가 아니며, proof-level theorem과 final solution error-bound corollaries는
  `docs/complex_geometry_energy_consistency_analysis.md`에 유지한다. Energy loss/error-bound
  notation은 \(\mathcal{E}_{\mathrm{split}}\)로 통일하고 이전 L-subscript-E 표기는 더 이상 쓰지 않는다.
- `docs/complex_geometry_energy_consistency_analysis.md`는 complex geometry energy
  consistency를 위한 continuous-domain Markdown analysis이다. 이 문서는
  connected-interval Green reconstruction을 전제로 하고, split operator는
  \(\frac12cu\) reaction split을 사용한다. Exact Green reconstruction theorem,
  exact/perturbed final solution error-bound corollaries, imperfect Green
  reconstruction perturbation을 모두 포함한다. Green reconstruction은 source에 대해 linear하다고 가정하며,
  approximate reconstruction perturbation에서는 최소한 source-difference identity를 요구한다.
  \(u_\phi,u_\psi,u_*\in H_0^1(\Omega)\) admissibility는 theorem assumption으로 두고
  별도 discussion에서 다룬다. 특히 represented solutions \(u_\phi,u_\psi\)의
  \(H_0^1(\Omega)\) membership은 interval endpoint zero에서 자동으로 나오지 않으며,
  connected-interval Green operators가 admissible source spaces에서
  \(H_0^1(\Omega)\)로 mapping된다는 조건으로 읽는다. Section 8은 transverse
  derivative, moving interval geometry, degenerating interval length를 별도
  regularity issue로 설명한다. \(L^2\)-consistency insufficiency는 local
  high-frequency example로 설명한다.
- `docs/wccm_eccomas_2026_presentation_outline.md`는 WCCM-ECCOMAS 2026,
  MS165 - Methods and Applications of Model Order Reduction 발표를 위한 영어 slide
  outline planning document이다. 발표 제목은 "Hybrid Green's Function Learning With
  Axial Reduction for Multi-Dimensional Elliptic Problems"로 둔다. Canonical main-flow
  slide ordering은 title, merged MOR motivation + axial reduction, figure-first
  graphic abstract, merged unit-interval pull-back + operator scaling,
  GreenNet 3-slide block, CouplingNet 3-slide block,
  Energy-Norm Error Bound Proposition, separated GreenNet/CouplingNet numerical
  evidence, conclusion 순서다. Complex geometry는 axial line construction이 아니라
  non-square/non-unit interval의 unit-interval normalization motivation으로만 가볍게
  다룬다. GreenNet analytic structure and learned correction은 main-flow 독립 slide로
  두며, Main slide에서는 Dirac delta jump, Heaviside cancellation, boundary-compatible
  learned smooth correction의 role decomposition만 설명한다. CouplingNet은 directional source
  split, branches/local context for split prediction, projection/reconstruction으로
  설명한다. Numerical evidence는 GreenNet kernel approximation과 CouplingNet solution
  reconstruction으로 분리한다. Code/config/schema/checkpoint/dataset generation detail은
  outline에서 제외한다. WCCM presentation 문서는 fixed coefficient problem에서의
  source-to-solution reconstruction framing을 사용한다. Coefficient field는 문제별로
  고정된 heterogeneous operator를 정의하고, sample variation은 source \(f\) 중심으로
  설명한다. Coefficient branch/profile/context는 coefficient profiles 또는
  operator-coefficient profiles로 설명하며,
  sample-varying coefficient family learning claim으로 쓰지 않는다. WCCM presentation
  문서의 2D PDE 표기는 vector convection \(\mathbf b=(b_x,b_y)\)를 사용하고,
  Slide 2의 opening PDE block은 homogeneous Dirichlet boundary condition
  \(u=0\) on \(\partial\Omega\)를 함께 표시한다. Slide 2는 fixed 2D operator와
  physical-coordinate \(L_x,L_y\) split operator를 함께 보여준다. 이때 reaction
  split은 \(\frac12cu\) convention을 사용한다. Slide 3은 figure-first graphic
  abstract로 두고, two-dimensional forcing \(f(x,y)\) on a general domain
  \(\Omega\) \(\rightarrow\) axial interval intersections \(\rightarrow\)
  GreenNet KaTeX math card \(v(t)=\int_0^1G_\theta(t,\eta)\rho(\eta)d\eta\)
  for a generic line-source profile \(\rho\) \(\rightarrow\) CouplingNet KaTeX math card for \(f\mapsto(\phi,\psi)\), split
  paths through \(G_x,G_y\), and 2D solution \(u\) 흐름을 그림 위주로 보여준다.
  이 slide는 analytic Green structure, CouplingNet
  branch/trunk detail, energy bound를 설명하지 않고 전체 계산 흐름만 orient한다.
  Slide 4는
  generic physical 1D axial operator \(\mathcal L_{\mathrm{phys}}\)를 \(s\in[s_0,s_1]\)에서
  unit interval로 pull-back하는 step이며, \(L_x,L_y\) 전체 split을 반복하지 않는다.
  Interval visual은 endpoint를 \(s_0,s_1\)로 표기하고 \(L=s_1-s_0\)를 명시하며,
  fragment reveal로 unit interval \(0\to1\)을 아래에 보여준다.
  Unit-interval pull-back/scaling에서는 full vector가 아니라 primary axial component
  \(b_\parallel\)를 써서 \(b_{\parallel,\mathrm{unit}}=L b_{\parallel,\mathrm{phys}}\)로
  표기하고, 결론 수식으로 conservative-form normalized equation on \(t\in[0,1]\)을
  보여준다. Slide 5는 GreenNet을 one-dimensional Green kernels로 소개하고,
  center object는 \(G_{\mathrm{unit}}(t,\eta)\) with "kernel integral operator"로
  표시한다. Axial coefficient profiles define each local 1D operator라는 문구로
  coefficient profile의 역할을 설명한다. Green operator는
  \(\mathcal G_{\mathrm{unit}}[f_{\mathrm{unit}}](t)=\int_0^1G_{\mathrm{unit}}(t,\eta)f_{\mathrm{unit}}(\eta)\,d\eta\)
  형태의 normalized axial source-to-solution 적분 작용으로 정의한다. Slide 7는
  learned Green kernel \(G_\theta\)의 source-to-solution supervision을 설명한다.
  Main slide flow에서는 full-domain 2D Green function integral보다 normalized axial
  1D Green operator 수식을 사용한다. Slide 6는 \(G_0\) piecewise formula와
  \(\partial_t^2G_0=-\delta\)를 main-flow에 포함하고, \(J_0\)를 \(G_0\)의
  \(t\)-antiderivative \(\partial_tJ_0=G_0\)로 설명한다. \(S=J_0-\frac12E\)와
  \(\partial_t^2S=\partial_tG_0\), \(A/B\) coefficient factors, \(E/M\) envelope
  factors도 Slide 6의 analytic ingredient로 둔다. \(J_0\) full piecewise formula와
  full Dirac/Heaviside derivation은 main slide가 아니라 backup 또는 technical report
  수준으로 둔다. Animation plan에서는 Slide 1을 static/no-animation title slide로 두고,
  Slide 2는 Auto-Animate와 progressive reveal을 사용해 fixed 2D operator, direct
  \(f\mapsto u\) challenge, \(L_x,L_y\) directional split, axial Green reconstructions
  plus learned coupling diagram을 3-click sequence로 보여준다. Slide 4는 Auto-Animate로
  physical interval \([s_0,s_1]\) with \(L=s_1-s_0\)에서 unit interval \([0,1]\)과
  pull-back map을 같은 fragment로 reveal하고, 이후 fragments로
  "length is not discarded" callout, full scaling rule,
  expanded normalized unit equation을 4-click sequence로 보여준다. \(b_\parallel\)
  clarification은 initial state에 두지 않고 scaling/normalized-equation stage 이후에
  작은 note로 보여준다. Slide 4의 length-preservation callout은 interval visual의
  caption처럼 보이지 않도록 전용 spacing을 둔 독립 card로 보여준다. Slide 5는 light Auto-Animate continuity와 progressive reveal로
  Green operator action을 source profile \(f_{\mathrm{unit}}(\eta)\), kernel panel
  \(G_{\mathrm{unit}}(t,\eta)\), integral action, output profile \(v(t)\) 순서로
  보여준다. Slide 5에서는 analytic kernel formula와 training loss를 다루지 않고,
  각각 Slide 6와 Slide 7로 넘긴다. Slide 6는 같은 heading을 가진 3개의
  Auto-Animate state로 구현한다. State 1은
  \(G_\theta=\text{Dirac delta structure}+\text{Heaviside cancellation}+\text{learned smooth correction}\)
  thesis만 보여주고, State 2는 full formula에서 \(A(t)G_0\) Dirac delta jump와
  \(B(t)(J_0-\frac12E)\) Heaviside cancellation을 강조하며 \(G_0\), \(J_0/S\),
  \(A/B\) identities와 "The analytic terms handle the Green-function singularity before learning."
  takeaway를 보여준다. State 3은 \(E M R_\theta\) learned smooth correction을
  강조하고 analytic terms를 muted 처리하며 \(E/M\) boundary-compatible envelope와
  final note를 보여준다. 세 state 모두 아래 여백에는 compact takeaway strip을 둔다.
  \(J_0\) full piecewise formula와 full distributional proof는 main
  reveal에 넣지 않는다. Slide 7는 GreenNet source-to-solution supervision slide로,
  pointwise Green-kernel label이 아니라 GP-generated target solution과 그 target
  solution에서 생성한 source의 reconstruction loss를 보여준다. Main slide에는
  \(w(t)\sim\mathcal{GP}(0,k_\ell)\),
  \(v(t)=w(t)-((1-t)w(0)+t\,w(1))\), \(v(0)=v(1)=0\),
  \(f_{\mathrm{unit}}=\mathcal L_{\mathrm{unit}}v\),
  \(v_\theta(t)=\int_0^1G_\theta(t,\eta)f_{\mathrm{unit}}(\eta)\,d\eta\), 그리고
  \(\mathcal J_{\mathrm{Green}}\sim\mathbb E[\int_0^1|v_\theta(t)-v(t)|^2dt]\)를
  포함한다. Animation은 "No pointwise Green-kernel labels."를 먼저 보이고, GP target
  construction, source generated from the target solution, learned reconstruction,
  expected reconstruction loss, "sources generated from target solutions" takeaway를
  순서대로 reveal한다. GP covariance detail, normalization, quadrature detail은
  Slide 7 main reveal에서 제외한다. Slide 8은 GreenNet block에서 CouplingNet block으로
  넘어가는 "CouplingNet I: Directional Source Split" transition slide로 둔다. Animation은
  ghosted \(G_x[\cdot]\), \(G_y[\cdot]\) directional Green inverse context와
  "What source should each axial inverse receive?" 질문에서 시작하고,
  \(f\to(\phi,\psi)\), directional source definitions \(\phi=L_xu\), \(\psi=L_yu\),
  compact \(L_x,L_y\) formulas with \(\frac12cu\), balance \(\phi+\psi=f\), final
  takeaway "CouplingNet learns the directional source split that couples the axial Green
  reconstructions."를 4-click sequence로 reveal한다. Slide 8에서는
  source-conditioned branch/context 설명, projection, \(u_\phi\), \(u_\psi\),
  \(u_{\mathrm{pred}}\)를 다루지 않고 각각 Slide 9과 Slide 10로 넘긴다. Slide 9은 CouplingNet
  branch-net/trunk-net context slide로 둔다. Animation은 central split predictor와
  two grouped reveal stages를 사용한다. Branch nets group은 source profile,
  coefficient profiles, line-geometry structure를 처리하고, trunk nets group은
  axial local coordinate \(t_{\parallel}\)와 pointwise transverse coordinate
  \(t_{\perp}\)를 처리한다. \(t_{\parallel}\)는 primary axial interval 안의 pointwise
  coordinate이고, \(t_{\perp}\)는 같은 physical point를 지나는 transverse interval 안의
  coordinate로 설명한다. Visible slide에서는 "fixed operator context" 표현을 쓰지 않고,
  coefficient profiles가 coefficient problem은 fixed여도 axial line마다 달라질 수 있다는
  점을 speaker note로 설명한다. Final state에는
  \((\text{branch features},\text{trunk features})\to(\phi,\psi)\) conceptual map과
  \(\underbrace{\text{profiles and line geometry}}_{\text{branch nets}}+
  \underbrace{\text{pointwise coordinates}}_{\text{trunk nets}}\Rightarrow
  \text{directional source split}\) underbrace takeaway strip을 남긴다. Slide 9에서는 projection/reconstruction과 implementation surface를
  다루지 않는다. Slide 10는 CouplingNet physical balance projection and Green
  reconstruction slide로 둔다. Projection은 unit interval quantity가 아니라 physical
  split variables에서 설명한다. Layout은 dense two-stage solver pipeline을 사용한다:
  Stage 1은 CouplingNet raw split \((\phi_{\mathrm{raw}},\psi_{\mathrm{raw}})\), projection residual
  \(r=f-(\phi_{\mathrm{raw}}+\psi_{\mathrm{raw}})\), projection formula
  \(\phi=\phi_{\mathrm{raw}}+\frac12r\), \(\psi=\psi_{\mathrm{raw}}+\frac12r\), balanced
  split \(\phi+\psi=f\) 순서이고, Stage 2는 two Green reconstructions
  \(u_\phi=G_x[\phi]\), \(u_\psi=G_y[\psi]\), final average
  \(u_{\mathrm{pred}}=\frac12(u_\phi+u_\psi)\) 순서다. Animation은 raw split,
  projection step, balanced split, reconstruction stage, compact bottom pipeline
  equation 순서의 fragment reveal로 둔다. The compact bottom pipeline uses a one-line
  flow and the arrow label `projection`. Pipeline arrows must reveal with the
  destination boxes they point to, not before the boxes, so the initial state never shows
  standalone arrows inside an empty stage. Slide 10에서는 energy bound와 branch/context
  details를 다루지 않고 각각 Slide 11과 Slide 9로 넘긴다. Slide 11은
  Energy-Norm Error Bound Proposition slide로 둔다. Main claim은
  \(\mathcal{E}_{\mathrm{split}}\)가 단순 diagnostic이 아니라 structural assumptions
  아래 final solution energy error를 bound한다는 것이다. Main slide에는
  diffusion coefficient \(a(x)>0\)를 명시한
  \(\|v\|_a^2=\int_\Omega a(x)|\nabla v(x)|^2\,dx\),
  \(\mathcal{E}_{\mathrm{split}}=\|u_\phi-u_\psi\|_a^2\),
  reference solution \(\mathcal Lu_*=f,\ u_*|_{\partial\Omega}=0\),
  \(\|u_{\mathrm{pred}}-u_*\|_a\le\frac{C_E}{2}\sqrt{\mathcal{E}_{\mathrm{split}}}\)를
  포함한다. \(C_E\)는 fixed elliptic operator의 stability constant로 짧게 label한다.
  Animation은 \(u_\phi,u_\psi,u_{\mathrm{pred}}\) diagram, energy norm,
  split-energy bracket, reference solution \(u_*\), final error-bound proposition,
  final assumption footer 순서로 reveal한다. Assumption footer는 처음부터 보이지 않게
  마지막 fragment로 두며, exact/controlled Green reconstruction과
  \(H_0^1(\Omega)\)-admissible represented solutions를 조건으로 쓴다.
  Full proof, \(q_c,q_x,q_y\) proof variables, full perturbation bound는
  main slide에 넣지 않고, learned Green reconstruction errors add perturbation terms라는
  message만 verbal/backup 수준으로 둔다. GreenNet evidence는 GreenNet III 직후,
  CouplingNet 소개 전에 배치한다. 이 slide는 GreenNet-only kernel-structure evidence로
  확정하고, `plot_wccm_green_evidence_panel.py`가 `checkpoints/Disk_CD/green/artifacts`에서
  생성한 separated PNG assets를 사용한다. Current canonical command는 interval 158,
  \(\eta=0.75\), basename `greennet_cd_evidence_interval158_eta075`를 사용한다.
  Main Quarto slide는 reference Green kernel, learned Green kernel, signed error
  heatmap, fixed-η slice를 각각 별도 image로 배치하고, kernel/slice relative error와
  diagonal-band diagnostic은 slide-native diagnostic card로 구성한다. Compact line-context
  tag는 physical axial line을 \(y\)-directed interval at \(x=-0.25\), \(L=0.866\)로
  표시하고 artifact interval id는 노출하지 않는다. Visible
  `Kernel-level evidence` label은 사용하지 않는다. Slide title, subtitle, 위치,
  speaker notes가 GreenNet kernel validation임을 설명한다. Diagonal-band definition은
  grid-step count가 아니라 normalized unit-coordinate length로 표시하며,
  \(|t-η|\le 5/128=0.0391\)로 표기한다. Boundary residual은 generated summary에는
  보관하되 main slide diagnostic card에서는 제거한다.
  Training/artifact context such as sampler mode, epochs, batch size, branch input size,
  device, and checkpoint path is not shown on the slide. Instead, the slide includes a
  compact visible reaction-free convection-diffusion problem setup strip:
  \(\Omega=\{x^2+y^2<0.5^2\}\),
  \(-\nabla\cdot(a\nabla u)+\mathbf b\cdot\nabla u=f\), \(c=0\),
  \(a(x,y)=1+\frac12\sin(2\pi x)\sin(2\pi y)\), and
  \(\mathbf b=(\frac12\sin(\pi x)\cos(\pi y),-\frac12\cos(\pi x)\sin(\pi y))\).
  The GreenNet evidence uses three Reveal.js states: (1) reference/learned/signed-error
  heatmaps plus diagnostics, (2) enlarged fixed-η=0.75 slice while signed error and
  diagnostics stay visible, and (3) the takeaway. The takeaway is
  "The learned kernel captures the singular Green structure, and the signed error is not
  concentrated near the singular diagonal." The fixed-η curve draw order is an
  asset-level readability detail and is not narrated in the speaker script; the spoken
  point is only that the reference and learned curves nearly overlap, including near
  the singular point.
  Generated Plotly HTML은 interactive Q&A aid로 사용할 수 있지만,
  timed Quarto/Reveal.js main deck에서는 reliable projection, navigation, handout
  export를 위해 static PNG/PDF-ready separated assets를 사용한다. Slide-facing GreenNet
  evidence labels use the Unicode η character in text labels, not the literal word eta,
  and the generated Plotly assets use enlarged axis, tick, colorbar, legend, and card
  label fonts for projection readability. Diagonal concentration diagnostics are labeled
  as `Error mass / band area` and `Mean error / off-band mean` rather than the ambiguous
  `Diag. mass / area` and `Diag. mean ratio`. Slide 13는 CouplingNet final solver
  evidence slide로 확정하고, `plot_wccm_coupling_evidence_panel.py`가 생성한 separated
  field assets를 사용한다. Main slide는 relative-solution-error 기준 `min`, `q25`,
  `q50`, `q75`, `max` 5개 selected samples를 column으로 두고, `Source`,
  `Reference`, `Prediction`, `Signed error` 4개 row를 slide-native table로 조립한다.
  Individual field panels에는 title, axis, colorbar, sample id, artifact-specific text를
  넣지 않고, row/column label과 metric table은 Quarto slide 안에서 따로 구성한다.
  Source row는 per-sample scale을 사용한다. Reference와 prediction row는 같은 selected
  sample column 안에서만 shared solution color scale을 사용하고, 서로 다른 sample column은
  independent solution scale을 사용한다. Signed-error row는 selected samples 전체에 대해
  zero-centered shared diverging scale을 사용해 error magnitude comparison을 유지한다.
  Slide-native metric card는 relative solution
  error를 `rel. sol. err.`로 label하고 split-energy loss만 compact하게 표시하며 첫 slide
  state부터 보이게 한다. Source row와 metric card는 initial state에 두고, Reference,
  Prediction, Signed error row는 row 단위 fragment로 순차 reveal한다. Flux target/error와
  \(u_\phi-u_\psi\) split mismatch는 main slide가 복잡해질 경우 verbal/backup으로 둔다.
  Slide 13에는 별도의 `Solver-level evidence` pill을 두지 않고, slide title, CouplingNet
  pipeline 뒤의 배치, CDR problem strip, metric card로 CouplingNet final solver
  reconstruction을 검증하는 slide임을 명확히 한다. Slide 13 takeaway는 quantile-selected samples가 observed relative-error range 전반에서
  2D solution reconstruction을 support한다는 claim으로 쓰고, single favorable case만
  보여준다는 인상을 피한다.
  Numerical evidence slides는 heavy Auto-Animate보다
  figure-first minimal reveal을 사용한다. Slide 14은 `Takeaway: Coupled Axial Green
  Solvers` closing slide로 둔다. Subtitle은 "GreenNet learns line-wise Green kernels;
  CouplingNet learns the directional source split."로 두고, thesis sentence는
  "A 2D elliptic problem is solved through axial Green inversions and a learned,
  balance-preserving source decomposition."로 둔다. 이 thesis sentence는 중앙 정렬
  slogan이 아니라 left-aligned technical callout으로 표시한다. Four contribution blocks는
  Axial Green kernels, Analytic structure, Source split, Energy bound로 정리하되,
  taller cards를 사용해 slide 하단 여백을 줄인다. Card text는 각각 axial interval
  normalization/line-wise inverse kernels, singular Green behavior before neural
  correction, reference-solution 또는 split label 없이 학습되는 phi/psi source split,
  structural assumptions 아래 unsupervised split consistency와 final solution error의
  연결을 설명한다. CouplingNet training loss는 reference solution을 직접 쓰지 않는
  split-energy consistency이며, reference solution은 evidence/evaluation metric에만
  사용한다는 speaker note를 둔다. Closing line은 wide bottom banner로 배치하고
  "GreenNet supplies line-wise Green inverses; CouplingNet learns the source split
  that turns them into a 2D elliptic solver."로 둔다. Animation은 light progressive reveal만 사용하고, new
  equations, numerical metrics, proof detail, future-work list는 Slide 14에 넣지 않는다.
  Backup/Q&A slides는 숫자 기반 backup naming을 쓰지 않고
  `Backup A`, `Backup B`, `Backup C`로 통일한다.
- `docs/wccm_eccomas_2026_slide_content_plan.md`는 WCCM-ECCOMAS 2026 발표 deck을
  만들기 위한 slide-by-slide content blueprint이다. Outline 문서는 macro flow와 timing을
  담당하고, content plan은 각 slide의 title, subtitle, main claim, must include,
  optional/can omit, suggested visual, equations/notation, slide text draft, speaker
  emphasis를 정리한다. 모든 slide section은 반드시 필수 내용과 생략 가능 내용을
  구분한다. 실제 Quarto + Reveal.js deck source는 `docs/presentations/wccm_eccomas_2026/`
  아래에 생성되어 있으며, GreenNet numerical evidence는 separated kernel-structure
  assets와 slide-native diagnostic card로 확정하고 CouplingNet numerical evidence는
  separated 5-by-4 quantile field matrix와 slide-native metric card로 확정한다.
- `docs/wccm_eccomas_2026_slide_deck_critical_review.md`는 현재 WCCM-ECCOMAS 2026 deck의
  revision checklist로 사용한다. 이 문서는 한국어 critical review이며 각 critique를 Issue,
  Why it matters, Recommended fix 구조로 정리한다. 후속 slide 수정에서는 특히 backup numbering
  inconsistency, Slide 12 takeaway clipping, Slide 13 metric-card density, 초반 contribution
  sentence anchoring, Slide 6 GreenNet analytic-structure timing risk를 high-priority item으로
  우선 확인한다.
- WCCM-ECCOMAS 2026 backup slides는 main 14-slide timing에 포함하지 않는 Q&A 전용
  support material이다. Ready backup은 Dirac/Heaviside derivation sketch, imperfect
  Green reconstruction perturbation, connected-interval pull-back detail의 3장으로
  구성한다. Extra GreenNet reconstruction evidence와 extra CouplingNet split/energy
  evidence는 final numerical figures가 준비될 때까지 deferred figure-dependent backup으로
  둔다. Backup slides는 self-contained하게 작성하되 code/config/schema/checkpoint나
  geometry/sample generation detail은 넣지 않는다.
- WCCM-ECCOMAS 2026 deck remaining-fixes revision convention: active deck만 수정하고
  `docs/presentations_backup/`은 baseline으로 유지한다. Slide 3 graphic abstract는 source
  card를 작게 두고 axial interval view와 CouplingNet directional coupling card의 비중을
  키운다. Slide 5는 "Previous step: normalize the operator. This step: learn its
  Green inverse." contrast strip으로 Slide 4와 역할을 분리한다. Slide 6 timed-talk
  main state에는 Dirac jump, Heaviside cancellation, boundary-compatible learned smooth
  correction의 세 역할과 최소 \(G_0,J_0,S,E,M\) identity만 두고, \(A/B\) coefficient
  factor details와 operator-application interpretation은 Backup A로 이동한다. Backup A는
  "Operator application creates two effects" heading 아래 Dirac jump, Heaviside leftover,
  analytic compensation을 구분하고, Backup B는 directional mismatch
  \(\varepsilon_x-\varepsilon_y\)와 common bias \(\varepsilon_x+\varepsilon_y\)를 two-channel
  visual로 구분하며, Backup C는 non-square slice SVG와 "connected intervals are not merged
  across outside-domain gaps" 메시지를 우선 보여준다.
- WCCM GreenNet Evidence state alignment uses a shared outer-height contract rather than
  unrelated child-card heights. The second fixed-η state remains the visual sizing reference;
  the first state stretches both kernel cards to the full left/right row height, and the third
  state constrains the signed-error/diagnostics stack to the fixed-slice row while leaving the
  takeaway in a separate row. Preserve the existing Auto-Animate `data-id` values when adjusting
  these dimensions.
- WCCM-ECCOMAS 2026 submitted-abstract alignment: Slide 6 GreenNet II는
  "Do not learn the Green singularity from scratch."를 compact visible motivation으로
  두고, variable-coefficient Green kernels are rarely available in closed form,
  source-point singularity와 homogeneous boundary constraints are hard for a plain
  neural network, analytic component supplies the delta-induced jump, flux-jump
  behavior, and boundary structure before learning이라고 speaker note에 둔다.
  Slide 7은 GreenNet이 one-dimensional source-to-solution reconstruction pairs로
  supervised되고, CouplingNet은 reference-solution 또는 split label 없이 balance, Green
  reconstruction, split-energy consistency로 학습된다는 contrast를 명시한다. Slide 8은
  CouplingNet이 solution을 직접 예측하는 것이 아니라 horizontal/vertical line-wise
  flux-divergence 또는 source components를 예측한다고 설명한다. Slide 9는 coefficient
  field는 problem에 대해 fixed이지만 axial profiles가 line마다 달라지고, 같은
  operator-learning model이 line-wise로 재사용된다는 phrasing을 사용한다. Slide 14는
  reference solutions are used only for evaluation; CouplingNet is trained through
  balance, Green reconstruction, and split consistency라는 wording을 유지한다.
- WCCM-ECCOMAS 2026 실제 발표 deck source는
  `docs/presentations/wccm_eccomas_2026/wccm_eccomas_2026.qmd`인 Quarto + Reveal.js
  deck이고, deck-local style은 `docs/presentations/wccm_eccomas_2026/styles.scss`에 둔다.
  Render command는
  `quarto render docs/presentations/wccm_eccomas_2026/wccm_eccomas_2026.qmd`로 고정한다.
  Decktape validation은 user Node environment의 `decktape` 3.16.1과 Chrome for Testing
  executable
  `~/.local/share/decktape-browsers/chrome/linux-150.0.7871.46/chrome-linux64/chrome`를
  사용한다. Decktape는 output PDF path를 screenshot basename에 포함시키므로,
  `docs/presentations/wccm_eccomas_2026/` 안에서 basename PDF로 export한 뒤
  `checks/wccm_eccomas_2026_decktape.pdf`로 이동한다. Check screenshots는
  `docs/presentations/wccm_eccomas_2026/checks/screenshots/`에 둔다.
  Title slide metadata는 presenter `Junhong Jo` with affiliation
  `National Institute for Mathematical Sciences`,
  affiliation의 `N`, `I`, `M`, `S` initials를 bold로 강조해 acronym을 드러내고,
  `Joint work with Taeyoung Ha (NIMS) and Chang-Ock Lee (KAIST)`, presentation date
  `Tuesday, July 21, 2026`로 둔다. Title slide subtitle은 reduced solver가 아니라
  two-dimensional elliptic solver, axial Green operators, learned directional source split을
  드러내고, Reduce/Recover cards는 각각 "a multi-dimensional elliptic problem into axial
  Green subproblems"와 "coupled axial Green operators" 표현을 사용한다. Quarto heading에서
  생성되는 title-slide `h2`는 custom title과 중복되지 않도록 deck-local CSS에서 숨긴다.
  Slide 2 title은 전통적인 MOR로 과하게 보이지 않도록 "Axial Reduction of Elliptic
  Operators"로 둔다. Fixed-coefficient framing은 긴 설명문보다
  \(\mathcal S_{a,\mathbf b,c}:f\mapsto u\) source-to-solution map 수식으로 직접 보여주고,
  "Operator fixed; source varies." caption은 작고 secondary하게 둔다. 같은 수식을 Direct
  map card에 반복하지 않고, card에는 "one global source-to-solution representation"처럼
  역할 설명만 둔다. Reduced view card는 줄바꿈을 줄이기 위해 "directional 1-D operators
  + learned coupling"으로 축약한다. GreenNet Evidence는 GreenNet III 직후에 배치하고
  Disk_CD convection-diffusion separated GreenNet evidence assets, fixed-η=0.75,
  physical axial-line context tag, three-state reveal, no visible `Kernel-level evidence` label을 사용한다. Slide 13은
  `checkpoints/Disk_CDR/coupling/artifacts`에서 생성한
  `coupling_cdr_evidence_rel_sol_quantiles` separated CouplingNet evidence assets를
  사용해 convection-diffusion-reaction solver-level evidence를 보여주며, 5-by-4
  relative-error quantile matrix와 처음부터 보이는 right-side metric card로 조립한다.
  Slide 3은 deck-native SVG/HTML graphic abstract로 구성하며, 일반 경계 domain의
  two-dimensional forcing \(f(x,y)\), clipped axial intervals, GreenNet direct
  KaTeX kernel-integration card, CouplingNet direct KaTeX split/reconstruction card,
  bottom takeaway를 사용한다. 우선 Quarto/KaTeX로 직접 수식을 렌더링하고, 수식 card가
  깨질 때만 고해상도 PNG panel fallback을 고려한다. Main content는 그림 위주로
  유지하고, 3-4 click reveal plus takeaway reveal 정도의 가벼운 animation만 사용한다.
- WCCM-ECCOMAS 2026 speaker script는 `docs/wccm_eccomas_2026_speaker_script.md`를
  canonical editable English source로 두고, 사용자 승인 후 active Quarto deck의
  `::: {.notes}` block과 동기화한다. 이 문서는
  14개 main slide에 대해 약 13분 spoken script를 목표로 하며, 각 slide section은
  target time, reveal/click cues, speaker script, compression option, transition sentence
  구조를 사용한다. Backup A/B/C는 timed talk가 아니라 Q&A-only prompt snippet으로
  유지한다. 현재 압축본은 main `Speaker script` 기준 1,246 words와
  12분 40초 발화 예산을 사용하며, 클릭과 짧은 호흡을 포함해 약 13분을 목표로 한다.
  이 압축본은 QMD notes에 삽입되었으며, 이후 wording 수정은 Markdown canonical source와
  QMD notes에 함께 반영한다. Auto-Animate로 나뉜 GreenNet analytic 및
  GreenNet evidence slide는 같은 대본을 반복하지 않고 각 state에서 실제로 말할 문장을
  분배한다. 발표가 오후 3시 이후에 시작하므로 Slide 1의 canonical greeting은
  `Good afternoon.`으로 고정한다. Slide 2에서는 \(a\), \(\mathbf b\), \(c\)가
  prescribed이고 \(f\)가 sample마다 변한다고 설명하며, `one coefficient problem`과
  `heterogeneous operator` 표현은 사용하지 않는다. Slide 9은 GreenNet이 directional
  inverses를 제공하지만 2D source를 \(x\)- 및 \(y\)-directional source components로
  split하는 방법은 결정하지 않는다고 설명하며, 모호한 `allocated` 표현은 사용하지 않는다.
  Slide 10은 trunk nets가 primary 및 transverse axial intervals 안의 pointwise
  positions를 encode한다고 한 문장으로 설명하고, profile-level branch의 한계를 별도로
  주장하지 않는다. Slide 11은 GreenNet이 normalized unit interval에서 axial inverse를
  계산하는 설명과 physical-coordinate balance projection 사이를 명시적으로 연결한다.
  Raw directional source components는 projection 전에 physical source scale로 복원하고,
  balanced components는 각 axial interval로 pull back한 뒤 \(G_x,G_y\)에 전달한다고
  설명한다. Slide 12는 CouplingNet이 reference-solution 또는 directional-split label
  없이 학습되므로 두 directional reconstructions만으로 계산 가능한 loss가 필요하다는
  동기에서 시작한다. \(\mathcal E_{\mathrm{split}}=\|u_\phi-u_\psi\|_a^2\)를
  unsupervised split-energy loss로 정의하고, \(u_*\)는 theoretical bound에만 사용되며
  CouplingNet training loss에는 들어가지 않는다고 명시한다. Error-bound proposition은
  이 unsupervised loss의 conditional structural justification으로 설명한다. 이 대본에서
  `field`는
  \(\mathbf b\)와 같은 vector-valued
  quantity에만 사용한다. Scalar quantities는 각각 source 또는 forcing \(f\), solution \(u\),
  coefficients \(a,c\), directional source components \(\phi,\psi\)로 직접 지칭한다.
  Active QMD의 speaker-note `Click` cue는 실제 Reveal fragment 순서에 맞추며,
  canonical Markdown 대본보다 세분화될 수 있다. 2026-07-21부터 active deck은
  `wccm_eccomas_2026.qmd`의 backup-free main-talk version과
  `wccm_eccomas_2026_with_backup.qmd`의 Backup/Q&A 포함 version으로 나뉜다.
  두 QMD의 main-talk prefix는 동일하다. Quarto 검증에서 backup-free version은
  18개 notes block과 41개 click cue, with-backup version은 22개 notes block과
  41개 click cue를 가지며, 각각의 rendered HTML notes와 정확히 일치했다.
- DTE 2027 abstract source는 `docs/dte2027_abstract/DTE2027_abstracts.tex`이고,
  official `DTE2027_abstracts.cls` template를 같은 폴더에 보관한다. 초록은
  `MS018 - Scientific Machine Learning for PDEs in Complex Geometries` minisymposium용이며,
  title은 "An Operator-Learning Framework Based on Coupled One-Dimensional Green's
  Functions for 2D Elliptic PDEs on Complex Domains"로 둔다. Junhong Jo는 presenting
  author로 asterisk를 붙이고, Taeyoung Ha와 Chang-Ock Lee는 coauthor로 둔다. Address
  block은 template 예시처럼 affiliation, postal address, e-mail을 직접 포함한다:
  NIMS는 `National Institute for Mathematical Sciences`, `Daejeon 34047, Korea`,
  `jjhong0608@nims.re.kr, tha@nims.re.kr`로 쓰고, KAIST는
  `Department of Mathematical Sciences, KAIST`, `Daejeon 34141, Korea`,
  `colee@kaist.edu`로 쓴다. 내용 framing은 Green-function source-to-solution
  integral representation에서 출발하고 references는 axial Green's function method와
  DD29 axial Green surrogate proceeding을 사용하며, SciML for complex geometries에 맞춰 axial intersections decomposed into
  connected components, each as an independent 1D boundary value problem,
  unit-interval pull-back, the line-wise Green-kernel model GreenNet,
  the directional source-coupling model CouplingNet, geometry-aware directional source
  decomposition, and coherent 2D reconstruction을
  중심으로 유지한다. DTE abstract에서는 label-free training이나 energy-norm theory를
  핵심 claim으로 전면화하지 않고, numerical results 문장에서도 disk-type 등 특정
  domain family를 전면화하지 않는다. Keyword line은 `Neural operators` 대신
  `Operator learning`을 사용한다.
- Circular complex geometry generator는 `cli/make_circular_geometry.py`이며
  center `(0, 0)` 고정, radius CLI option default `1.0`, `2 * radius / step_size`
  정수 조건을 사용한다. Grid interval은 `[-radius, radius]`이고, boundary grid point와
  degenerate boundary line은 제외하며, valid interior point가 있는 axial chord segment만
  저장한다. Reconstruction weight는 physical length가 아닌 segment-local unit coordinate
  기준 nonuniform trapezoid weight로 저장한다.
- Annulus complex geometry generator는 `cli/make_annular_geometry.py`이며
  2D PDE code path에서의 torus shape를 두 동심원 사이 영역으로 해석한다. Center는
  `(0, 0)` 고정이고, valid point는
  `inner_radius < sqrt(x^2+y^2) < outer_radius`인 strict interior point만 저장한다.
  Grid interval은 `[-outer_radius, outer_radius]`이고,
  `2 * outer_radius / step_size` 정수 조건을 사용한다. Inner hole을 지나거나 접하는
  axial line은 disconnected segment row로 나눠 저장해서 edge/reconstruction이 hole을
  가로지르지 않게 한다.
- FEniCSx complex sample generator는 optional `green_fenicsx` conda env에서만
  실행하는 path로 둔다. Main `green_net` training env와 `pyproject.toml`에는
  FEniCSx dependency를 섞지 않는다.
- Complex Coupling sample generator output `.npz`는 full-grid `rhs`, `sol`,
  `phi`, `psi` 배열을 저장하고, 모든 배열은 `[row=y, col=x]` convention을 따른다.
  `coords_valid` 밖의 full-grid 값은 `0.0`으로 채운다.
- FEniCSx sample generator의 `phi`/`psi`는
  `phi=-d_x(a d_x u)+b_x d_x u+0.5*c*u`,
  `psi=-d_y(a d_y u)+b_y d_y u+0.5*c*u`인 direction-split operator component이며,
  valid point에서 `phi + psi ~= rhs` balance residual을 summary에 기록한다.
- FEniCSx sample-level 병렬화는 spawned Python process worker로 독립 sample을
  나눠 생성하는 방식이다. MPI domain decomposition은 v1 범위가 아니며,
  `num_workers > 1`이면 `sample_seed_policy="indexed"`를 강제한다. Parent process만
  `generation_summary.json`을 작성하고, sample `.npz` schema는 `rhs`, `sol`,
  `phi`, `psi` 그대로 유지한다.
- FEniCSx Gmsh script input은 `build_domain(gmsh, context)`를 제공해야 하며,
  multi-surface disconnected domain에서는 valid point마다 들어갈 surface를
  `point_surface_tags`로 명시해야 한다. Script mode는 valid points를 mesh internal
  points로 embed하려고 시도하고, `.msh` mode는 vertex 보장을 기본 요구하지 않는다.
  `examples/unit_circle_gmsh.py`는 geometry `.npz`의 `radius` metadata를 읽어서
  non-unit circular geometry와 Gmsh disk mesh radius를 일치시킨다.
- Annulus FEniCSx sample generation은 `examples/annulus_gmsh.py`를 사용한다.
  이 script는 geometry `.npz`의 `inner_radius`와 `outer_radius` metadata를 읽고,
  Gmsh OCC cut으로 inner hole을 가진 single annulus surface를 반환한다. Surface가
  하나이므로 `point_surface_tags`는 필요 없고, 기존 valid-point embedding은 모든
  valid point를 같은 surface에 embed한다.
- Circular sample generation workflow는 `examples/unit_circle_gmsh.py`를 기본
  Gmsh domain script로 사용한다. Smoke default는 `h=0.25`, `mesh_size=0.035`,
  `solution_degree=3`, `target_degree=2`, `train=1`; small dataset default는
  `h=0.1`, `mesh_size=0.025`, same FEM degrees, `train/valid/test=32/8/8`이다.
  첫 coefficient는 `Pure_Poisson.py`로 고정한다. 현재 refined smoke는
  `phi + psi ~= rhs` residual `1e-2`를 통과했지만, small dataset default는
  max residual이 약 `3.7e-2`이므로 schema/loadability와 residual distribution 확인용으로
  먼저 사용하고, strict `1e-2` 품질 dataset이 필요하면 mesh/projection parameter를 더 조정한다.

## Experiment And Figure Planning

- 논문용 데이터 생성 전에 coefficient family, model variant, train/eval split,
  output directory, plot target을 먼저 확정한다.
- GreenNet 결과는 `rel_green`, `train_rel_sol`, `val_rel_sol`을 함께 본다.
- Complex GreenTrainer의 새 L-BFGS run은 Adam 단계와 같은 metric token을
  유지해 `train_rel_sol`, validation이 있으면 `val_rel_sol`, reference가 가능하면
  `rel_green`을 `training.log`에 기록한다. 과거 complex GreenNet run의 L-BFGS
  log에는 validation/`rel_green` token이 없을 수 있다.
- GreenNet 논문용 산출물은 run-level training metrics, per-line metrics,
  selected-line Green kernel data, selected-line coefficient slices,
  source-to-solution reconstruction data를 분리해서 저장하는 방향을 기본으로 한다.
- GreenNet `training.log`와 artifact summary의 `rel_green`은 exact/reference
  1D line Green kernel을 구성할 수 있는 경우에 유효하다. `c=0, b=0`이면
  diffusion reference를 쓰고, `c=0, b!=0`이면 convection-diffusion reference를
  쓴다. Reaction `c`가 nonzero이면 현재 exact/reference가 없으므로 invalid/skip
  처리한다.
- `ExactGreenFunction`의 기본 `forward()`, `__call__()`, `error()`는 diffusion-only
  reference로 유지한다. Reaction-free convection-diffusion line reference가
  필요하면 `convection_diffusion(b)`를 명시적으로 호출하고, x-lines에는 `bx`,
  y-lines에는 `by`의 line slice를 넘긴다. GreenNet `rel_green` 경로는 같은
  reference policy를 사용한다.
- Exact/reference Green kernel matrix convention은 `G[row=x, col=xi]`이다.
  Reconstruction은 마지막 dimension인 `xi` 방향으로 적분한다. Convection-diffusion
  kernel은 비대칭이므로 이 orientation을 해석식 테스트로 고정해야 한다.
- GreenNet problem selection의 현재 논의 기준은 `Pure_Poisson.py`와 두
  diffusion-only 문제를 `rel_green` 중심 주력 문제로 두고, reaction-free
  convection-diffusion 문제도 `rel_green_reference="convection_diffusion"`로
  비교할 수 있다. Reaction 포함 문제는 reconstruction 중심 보조/확장 산출물로
  다루는 것이다.
- GreenNet 논문용 산출물은 training loop에 묶기보다 checkpoint/config/dataset을
  다시 읽는 별도 재생성 script 또는 CLI로 만드는 방향을 선호한다.
- GreenNet이 주장하려는 내용이 fixed coefficient problem에서 학습한 Green
  function으로 같은 sampler/source distribution의 source term에 대한 solution을
  잘 재구성한다는 것이라면, 논문용 evaluation data는 별도 저장 dataset 없이도
  학습 때와 같은 config/sampler 설정으로 다시 생성해도 충분하다고 본다.
  다만 seed와 generation sequence는 metadata에 명시해서 training data 재사용과
  같은-distribution evaluation을 구분한다.
- GreenNet artifact exporter 구현은 새 CLI만 두기보다 dataset 재생성,
  Green kernel/reconstruction metric, fixed-`xi` slice 추출, Plotly multi-format
  저장을 작은 helper로 분리해 training loop와 독립시키는 방향이 좋다.
- GreenNet artifact exporter surface는 `cli/export_green_artifacts.py`이며,
  helper는 `src/greenonet/green_artifacts.py`와 `src/greenonet/plotly_io.py`에
  둔다. 기본 evaluation은 `validation_like`, 기본 seed는 `12345`, 기본 fixed
  `xi` fractions는 `0.25, 0.5, 0.75`이다. Device는 기본적으로 config의
  `training.device`를 따르며, 필요하면 CLI `--device`로 override한다.
- 논문용 산출물은 최소 산출물만 만드는 방향보다 가능한 한 많이 생성한 뒤,
  그중 논문에 적합한 결과와 figure를 선택하는 방향을 기본으로 한다.
- CouplingNet 논문용 run-level training curves는 `train loss`, `val loss`,
  `l2_consistency`, `energy_consistency`, `rel_flux`, `rel_sol`만 사용한다.
  Other auxiliary losses are ignored for paper-facing curves unless explicitly
  requested later.
- CouplingNet selected-sample figures are limited to source `f`, exact solution
  `u`, `u_pred`, `u_pred_x`, `u_pred_y`, signed solution errors
  `u_pred - u`, `u_pred_x - u`, `u_pred_y - u`, mismatch `u_pred_x - u_pred_y`,
  exact flux-divergences `phi`, `psi`, predicted flux-divergences `phi_pred`,
  `psi_pred`, signed flux-divergence errors `phi_pred - phi`, `psi_pred - psi`,
  and balance fields `phi + psi`, `f - phi - psi`. Error figures must be signed
  differences, not absolute values.
- CouplingNet selected-sample non-error comparison figures use shared color ranges
  within each selected sample for reference/prediction groups. Unit-square groups
  are `u/u_pred/u_pred_x/u_pred_y`, `phi/phi_pred`, and `psi/psi_pred`.
- Complex CouplingNet selected-sample artifact figures use valid-point scatter
  plots on `coords_valid`. The default fields are `rhs`, `sol`,
  `u_pred=0.5*(u_phi+u_psi)`, `u_phi`, `u_psi`, signed solution errors
  `u_pred - sol`, `u_phi - sol`, `u_psi - sol`, split mismatch `u_phi - u_psi`,
  projected physical `phi`/`psi`, and optional target `phi`/`psi` plus signed
  flux errors when sample flux targets are available. Complex error and mismatch
  scatter figures use zero-centered diverging colors. Complex non-error comparison
  groups are `sol/u_pred/u_phi/u_psi`, `target_phi/phi`, and `target_psi/psi`
  when flux targets exist; `rhs` and target-free flux diagnostics keep independent
  ranges. The raw archive stores `raw_physical_phi/psi` and
  `projected_unit_phi/psi` for audit; raw/projected-unit fields are not primary figures.
- CouplingNet selected-sample flux-divergence figures should exclude boundary
  grid values. CouplingNet predictions use zero-padding at boundaries only for
  trapezoid-rule integration compatibility with boundary-zero Green functions;
  those padded boundary values are not meaningful flux-divergence predictions
  and should not be shown in paper-facing flux-divergence heatmaps.
- CouplingNet null-space and closure diagnostics are not part of the selected
  paper-facing figure set unless explicitly re-enabled later.
- CouplingNet 논문용 산출물에는 test-set aggregate metrics, balance/projection
  diagnostics, boundary diagnostics, coefficient/source context figures,
  ablation comparison tables, and raw metric/data archive를 유지한다.
- CouplingNet artifact 구현 시 current `CouplingEvaluator`는 일부 tensor
  computation을 재사용할 수 있지만, paper-facing selected figures는 별도 path
  또는 option으로 분리해야 한다. Current evaluator uses absolute errors and
  null/closure diagnostics, while the paper-facing set requires signed errors and
  excludes null/closure figures.
- CouplingNet paper artifact exporter surface는 `cli/export_coupling_artifacts.py`
  이며 helper는 `src/greenonet/coupling_artifacts.py`에 둔다. Test data는
  `dataset.test_path`의 `.npz`를 `CouplingDataset`으로 읽고, coefficient는 기본적으로
  `dataset.coefficient_functions_path`를 사용하며 `--coefficients`로 override한다.
  Device는 기본적으로 `coupling_training.device`를 따르고 `--device`가 우선한다.
- Complex CouplingNet coefficient artifact는 selected sample과 무관한 run-level
  context로 한 번만 생성한다. `a`, `b_x`, `b_y`, `|b|`, `c`는
  `coords_valid`에서 coefficient 함수를 직접 평가한 physical field이며 pull-back,
  interpolation, segment-length scaling을 적용하지 않는다. Diffusion은 항상 scalar
  figure를 만들고, reaction/convection은 physical field가 nonzero이거나 대응
  `coefficient_terms`가 enabled일 때 figure를 만든다. Convection은 signed component,
  magnitude, subsampled quiver를 함께 저장하며 default arrow limit은 400이다.
- `coefficient_terms`는 CouplingNet branch input 여부이고 physical PDE coefficient의
  존재 여부가 아니다. Complex artifact summary는 physical nonzero/constant 상태와
  branch enabled 상태를 분리해 기록하고, raw coefficient arrays는
  `data/coefficient_fields.npz`에 sample-independent archive로 저장한다.
- `plot_coupling_logs.py`는 paper-facing run-level curve 전용으로 유지한다. 출력은
  `loss`, `l2_consistency`, `energy_consistency`, `rel_flux`, `rel_sol` 5개만 생성하며,
  optional auxiliary loss curve는 debug용 `plot_logs.py`에서 다룬다.
- `plot_coupling_logs.py`는 complex CouplingNet의 `_log_epoch - epoch ... train/val ...`
  별도 줄 형식을 지원한다. Complex log는 `loss_energy_consistency`, `rel_sol`,
  `rel_flux`를 기록하고 cross metric을 남기지 않으므로, `loss_energy_consistency`는
  `energy_consistency` curve로 그리며 값이 없는 `l2_consistency` figure는 건너뛴다.
- `plot_coupling_logs.py`는 `--show-annotations` 옵션을 켜면 각 train/val trace의
  마지막 값과 최소값을 표시한다. 기본값은 꺼짐으로 유지한다. 값 표시는
  실제 curve point에 연결된 Plotly annotation으로 두고, last/min 위치에는 작은
  marker도 함께 찍는다. y-axis가 log-scale이므로 plotted marker 위치는
  `1e-16` floor로 맞추고, annotation y coordinate는 Plotly log-axis 규칙에 맞게
  `log10(plotted_y)`를 사용한다. Annotation이 켜진 figure는 마지막 epoch label이
  오른쪽에 놓일 수 있도록 x-axis 오른쪽 padding을 둔다.
- `plot_coupling_logs_aux_temp.py`는
  `checkpoints/Pure_Poisson/expreiments/raw_output_with_balance_loss/training.log`
  확인용 root-level 임시 script이다. Paper-facing default에는 넣지 않고,
  기존 5개 curve에 `balance_loss`, `symmetric_boundary_loss`만 추가해 본다.
- Best epoch과 final epoch을 분리해서 해석한다.
- Plot은 비교 실험 단위로 같은 scale, theme, output naming을 유지한다.
- Plotly 기반 논문용 figure는 나중에 수정하기 쉽도록 `html`, `png`, `pdf`와
  함께 Plotly figure spec `json`도 저장하는 것을 기본으로 한다. Static export가
  실패해도 `html`과 `json`은 저장해야 한다.
- `plot_green_logs.py`도 GreenNet log comparison figures를 `html`, `json`으로
  항상 저장하고, 가능한 경우 `png`, `pdf`도 함께 저장한다.
- `plot_complex_green_interval_metrics.py`는 complex GreenNet의
  `per_interval_metrics.csv` 시각화 전용 root-level script이다. pandas로 schema를
  검증하고 Plotly로 coordinate/length/distribution/chord-map figure를 만들며,
  `save_plotly_figure`를 통해 `html`, `json`, 가능한 경우 `png`, `pdf`를 함께
  저장한다.
- `plot_complex_coupling_sample_metrics.py`는 complex CouplingNet의
  `test_per_sample_metrics.csv` 시각화 전용 root-level script이다. pandas로
  sample-level schema를 검증하고 Plotly로 `rel_sol`, `rel_flux`, `loss`, log-loss,
  relative-metric distribution, `rel_sol`-`rel_flux` scatter, best/worst sample
  ranking figure를 만든다. CSV의 `rel_sol`/`rel_flux`는 raw fraction으로
  유지하고, figure에서는 `%` 단위(100배)로 표시한다.
- `plot_wccm_coupling_evidence_panel.py`는 WCCM-ECCOMAS Slide 13 전용
  CouplingNet evidence asset generator이다. Script default artifact root는
  `checkpoints/Diffusion/coupling/artifacts`로 남아 있지만, 현재 WCCM deck의 canonical
  Slide 13 asset command는 `--artifact-root checkpoints/Disk_CDR/coupling/artifacts`
  와 `--basename coupling_cdr_evidence_rel_sol_quantiles`를 사용한다. 이 CDR artifact는
  \( -\nabla\cdot(a\nabla u)+\mathbf b\cdot\nabla u+c u=f \) disk problem의
  solver-level evidence이며, `selected_raw_arrays.npz`, `summary.json`,
  `metrics/per_sample_metrics.csv`를 읽어 relative-solution-error quantile roles
  `min/q25/q50/q75/max`에 대한 separated `rhs`, `sol`, `u_pred`, `u_pred_error`
  field panels를 생성한다. Panel images는 title/axis/colorbar/sample id를 제거하고,
  row/column labels와 metric table은 Quarto slide에서 따로 구성한다. Slide-facing
  deck에서는 Source row를 먼저 보이고 Reference, Prediction, Signed error row를 순차
  reveal한다.
- `plot_coupling_rel_sol_boxplots.py`는 workshop용 CouplingNet test result CSV
  비교 script이다. `checkpoints/For_Workshop/CouplingNetResults`의 네
  `*_per_sample_metrics.csv` 파일에서 `rel_sol`을 읽고 문제별 Plotly boxplot을
  `html/json/png/pdf` 네 형식으로 저장한다. 이 script는 점 표시 없이 박스 플롯만
  그리며 `rel_sol`은 `%` 단위(100배)로 표시한다.
  하위 분위수 필터를 위해 `--rel-sol-percentile` 옵션을 지원한다. 기본값 100은
  전체 샘플을 사용하고, 예를 들어 90을 주면 각 문제별 `rel_sol`에서 값이 낮은
  sample만 남겨 boxplot을 그린다. 정확한 보존 개수는
  `max(1, floor(n * percentile / 100))`이며 percentile cutoff interpolation을
  사용하지 않는다. Figure의 percent 변환 값은 Plotly tuple과 binary float이므로
  test에서는 list로 변환한 뒤 approximate comparison을 사용한다.
- Figure 후보:
  - coefficient field visualization: `a`, `bx`, `by`, `c`
  - source/solution sample visualization
  - Green kernel heatmap per selected axial line
  - fixed-`xi` 1D Green function slices per selected axial line, with boundary
    values and diagonal/singularity behavior highlighted
  - training curves for GreenONet and CouplingNet
  - metric comparison table/bar chart
  - solution reconstruction and error heatmap
  - flux reconstruction and error heatmap

## Annulus CDR CouplingNet Reference Audit

- `checkpoints/annulus_CDR/coupling`의 2026-07-29 audit 기준:
  fixed indexed-GP source는 train `4800`, validation `100`이고,
  train/validation reference diagnostics는 꺼져 있다. Training objective는
  canonical energy only이며 output contract v6, physical-symmetric projection,
  pre-projection fusion, SOAP optimizer를 사용한다.
- Validation canonical energy의 best는 epoch 82의 `4.2700e-4`이고 epoch 100
  final은 `4.3083e-4`이다. Final checkpoint의 50-sample test mean은
  `rel_sol=5.1905%`, `rel_flux=17.1055%`, canonical energy `4.4634e-4`이다.
- `artifacts/`는 epoch-82 best-energy checkpoint로 생성되었다. Artifact의
  learned fusion gate가 epoch-82 checkpoint와 일치하므로, final-model
  `metrics/test_*`와 artifact aggregate를 같은 checkpoint 결과로 혼동하지 않는다.
- Selected artifact samples에서 `|x|` 또는 `|y|`가 inner radius `0.2`에 가까운
  transition band는 valid points의 `8.12%`이지만, absolute error 상위 1% 중
  `u_pred_error`의 `21.85%`, split/flux error의 약 `25-27%`를 차지한다.
- 이 run에서는 sample-level canonical energy와 detached `rel_sol`의 상관이
  약하다. Best-energy selection은 reference-free 원칙에 맞지만, evaluation
  report에서는 canonical energy와 `rel_sol`/`rel_flux`를 함께 해석한다.
- 이 historical checkpoint의 selected pre-projection fusion diagnostics에서
  raw nonlinear correction의
  transition-edge jump RMS는 linear correction의 `6.20`배이고, regular edge
  대비 transition jump 비율은 nonlinear `13.09`, linear `2.93`이다. 그러나
  learned gate가 `0.06854`이므로 pooled blended transition jump는 linear-only의
  `0.987`배이다. Raw nonlinear figure의 큰 seam과 최종 solution seam의 인과를
  혼동하지 않으며, 기여 판단에는 fuser-off 또는 linear-only ablation이 필요하다.
- 이 retired nonlinear fuser는 transition-aware이지만 transition-regularizing
  구조는
  아니다. Discontinuous length feature를 감지해 correction에 반영할 수 있지만
  correction continuity를 강제하지 않는다. Physical-symmetric projection은
  fused physical difference mode를 그대로 보존하므로 blend에 남은 seam을 제거하지
  않는다.
- 이 audit의 retired nonlinear fuser는 모든 valid point에 공유되는 pointwise
  MLP이다. Linear
  path는 normalized `(base_difference, rhs)`의 `2 -> 1`, nonlinear path는 여기에
  local coordinate와 length context 여섯 개를 더한 `8 -> hidden -> 1` 구조다.
  Neighboring point/line coupling, coefficient direct input, pointwise gate, continuity
  constraint는 없다.
- 이 historical run의 pre-projection fusion은 config에서 선택 가능한 optional
  split-fuser mode였다. 기존
  `residual_correction`은 그대로 보존되고, `absolute_difference`는 외부
  `+d_base` 없이 전체 difference를 예측한다. Absolute linear candidate는
  normalized base difference identity weight `[1,0]`으로 초기화하며,
  nonlinear component는 scaled standard final-layer initialization을 사용한다.
  Artifact는 mode, combination, initialization, component semantics와
  `outer_base_residual_used`를 기록한다.
- 이 historical fuser는 physical source에서 correction을 계산한 뒤 model
  forward의
  reference-response return contract를 맞추기 위해 임시로 `L_x^2/L_y^2`를
  곱한다. 외부 physical projection은 즉시 같은 scale로 나누므로 이 중간
  multiply/divide pair는 대수적으로 상쇄된다. 이것은 projection 이후
  `Phi=L_x^2 phi`, `Psi=L_y^2 psi`를 만들어 Green reconstruction에 전달하는
  필수 pull-back과 구분한다.

## Annulus CDR Single Residual MLP Audit

- `checkpoints/annulus_CDR/coupling3/comparison_analysis/analysis_report.md` is
  the canonical comparison of `coupling`, `coupling2`, and `coupling3`.
- The current single nonlinear residual MLP in `coupling3` uses normalized
  physical `(d_base, rhs)` only and a fixed identity skip. On selected sample 0
  it reduced correction transition-jump RMS by `33.7%` versus the retired
  gated split fuser and by `69.3%` versus the retired absolute fuser.
- A smoother correction did not imply a smoother final difference. The
  identity skip retained the transition-bearing `d_base`; sample-0 fused
  transition/regular jump ratio was `2.831`, and the `u_pred_error` edge-jump
  ratio was `6.850`.
- On the common 50-sample best-energy artifacts, `coupling3` had mean
  `rel_sol=5.1411%`, `rel_flux=17.4703%`, and canonical energy `4.5120e-4`.
  It slightly improved solution mean/tail but did not improve flux or energy.
  Sample 39 is the main canonical-energy tail failure, especially in the
  boundary-x term.
- `artifacts/metrics/per_sample_metrics.csv` is a best-energy checkpoint result;
  `metrics/test_per_sample_metrics.csv` is a final checkpoint result. Never mix
  the two when comparing runs.
- These historical runs do not pin the CouplingNet initialization seed.
  Architecture claims require data-aligned, initialization-aligned paired
  repeats, preferably at least three seeds.

## Annulus CDR Absolute Fuser Audit

- `checkpoints/annulus_CDR/coupling4/comparison_analysis/analysis_report.md` is
  the canonical residual-versus-absolute comparison for `coupling3` and
  `coupling4`.
- Both runs use the same fixed indexed-GP data, 384-width backbone, coefficient
  branches, SOAP settings, learning-rate schedule, and 50-sample test set.
  CouplingNet initialization is not pinned, so the result is not a strict
  paired-seed architecture proof.
- Absolute mode's best validation canonical energy was `1.791162e-3` at epoch
  99, `4.110x` the residual mode's `4.358221e-4` at epoch 72. Validation was
  still slowly improving, so classic late overfitting is not the dominant
  failure.
- On the same 50 final-checkpoint test samples, absolute mode increased mean
  canonical energy from `4.580003e-4` to `1.823562e-3` (`+298.2%`), mean
  `rel_sol` from `5.1637%` to `9.0922%` (`+76.1%`), and mean `rel_flux` from
  `16.9882%` to `37.8301%` (`+122.7%`). All 50 samples were worse for each
  primary metric.
- Absolute mode with `final_layer_init_scale=0` initially has
  `d_fused=0` and zero derivative from the fuser output to `d_base`; unlike the
  residual identity skip, it does not pass the first-step difference gradient
  directly to the axis networks.
- Because absolute output does not directly use `d_base`, the base difference
  became an unanchored latent. Across the common artifact samples 0 and 37,
  mean base RMS was `9.219` for absolute versus `0.675` for residual, while
  fused-to-target relative error was `0.586` versus `0.234`. Triangular/spoke
  structure in the oversized base latent remained visible in absolute
  directional-source and solution errors.
- Keep residual mode as the production baseline. Do not interpret
  `final_layer_init_scale=1` as identity initialization; it is only standard
  random `Linear` initialization. A future absolute experiment needs an
  identity-preserving initialization or another direct gradient path and
  fixed-seed paired repeats.

## Annulus CDR Fuser-Off Audit

- `checkpoints/annulus_CDR/coupling5/comparison_analysis/analysis_report.md` is
  the canonical comparison of the fuser-disabled control against
  `coupling3` residual, `coupling4` absolute, and the two retired split-fuser
  runs.
- `coupling3`, `coupling4`, and `coupling5` match in geometry, CDR
  coefficients, fixed indexed-GP sources, test set, 384-width backbone, SOAP,
  and scheduler. Their config differences are limited to
  `pre_projection_fusion.enabled`, `mode`, and
  `final_layer_init_scale`. The model initialization seed is not pinned.
- `coupling5` reaches best validation canonical energy `4.277555e-4` at epoch
  69. Its best-energy 50-sample test means are canonical energy
  `4.459290e-4`, `rel_sol=5.1562%`, and `rel_flux=17.8986%`.
- Disabling the fuser fully recovers the `coupling4` absolute failure:
  `coupling5` lowers mean energy by 75.5%, `rel_sol` by 43.3%, and `rel_flux`
  by 52.7%, winning all 50 samples on each primary metric.
- Against `coupling3` residual, fuser-off energy is 1.17% lower and `rel_sol`
  is 0.29% higher, with sample wins near 50/50 and paired Wilcoxon
  `p>=0.625`. The energy improvement is tail-sensitive: sample 39 accounts for
  65.8% of the summed energy delta. These differences do not establish an
  energy/solution advantage.
- Directional flux is different: fuser-off mean `rel_flux` is 2.45% higher
  than residual and is worse on 45/50 best-energy samples
  (`p=1.35e-11`). The same direction remains at the final checkpoint. Treat
  the residual fuser as a possible directional-split refinement, not a proven
  solution/energy improvement.
- Fuser-off does not remove the annulus transition seam. Across its five
  selected samples, `u_pred_error` transition/regular edge-jump ratios are
  `5.01` to `6.04` with mean `5.44`. The seam therefore also exists in the
  axis-conditioned base response and is not caused solely by the
  pre-projection MLP.
- A production architecture decision requires at least three fixed-model-seed
  paired repeats of residual versus disabled. Use best-validation canonical
  energy as primary, `rel_sol` as secondary, and `rel_flux` plus transition
  edge-jump as diagnostics.

## Annulus Transition Meeting Slide Plan

- `docs/meeting/annulus_transition_error_slide_plan.md` is the approved content
  contract for the collaborator-meeting block. The implemented canonical source
  is `docs/meeting/annulus_transition_error/annulus_transition_error.qmd`, its
  deck-local style is `docs/meeting/annulus_transition_error/styles.scss`, and
  the rendered offline Reveal.js entry point is
  `docs/meeting/annulus_transition_error/annulus_transition_error.html`.
- The implemented meeting deck has exactly 18 logical slides: main Slides 1-16
  plus Backup A/B. Overall presentation time is intentionally not fixed; timing
  labels in speaker notes are advisory only. Visible slide text is
  English; each slide has exactly one Korean `::: {.notes}` script. Reveal
  fragments expose one claim, formula stage, or figure comparison per click.
  Poisson must precede CDR, and Slides 11-14 must use
  `u_weak_residual_reliability`, never the equal-mean artifact `u_pred`.
- `docs/meeting/annulus_transition_error/build_assets.py` is an artifact-only
  presentation builder. It reads frozen NPZ/CSV/JSON outputs, never loads a
  CouplingNet checkpoint or runs model inference, and writes offline Plotly HTML plus local `plotly.min.js`, a
  Rich log, and `assets/manifest.json` with source paths, sample ids, field keys,
  metrics, and SHA-256 provenance. The final delivery is HTML-only; no final
  presentation PDF is stored.
- Run `docs/meeting/annulus_transition_error/qa_reveal.js` after every render.
  It checks 1600x900 and 1280x720, exact slide count, overflow, visible
  layout-region overlap, page errors, and external HTTP requests for every
  final state and every requested intermediate fragment state. It also captures
  all 18 final states at both viewports. This deck's QA path does not require or
  store a presentation PDF.
- The main block contains sixteen slides: previously observed transition
  error; the two-short-segment to one-long-segment representation together with
  the exact raw-reference, physical-projection, and reference-pull-back
  transformation; and the stagewise error diagnosis showing strong
  amplification at the mandatory \(L^2\) pull-back; followed by a compact
  projection/output-space history from original physical-symmetric through
  geometry-weighted/RPS and response-space to current physical-symmetric;
  one method slide defining three post-hoc blend rules; three detailed
  construction slides for geometry-only compact C2, mismatch-detected seam C2,
  and local weak-residual reliability; a Poisson-first clean comparison; a CDR
  broader-operator consistency check; four final-result
  slides for Poisson/CDR fields and signed errors; one slide explaining why
  equal source correction does not imply equal Green response; and one slide
  defining the proposed column-diagonal Green-response projection.
- In Slide 15, the directional-response factors are block-level labels in a
  balanced 3+2 grid: Green kernel, coefficient, and quadrature on the first
  row; evaluation position and segment length on the second. Do not use inline
  spans here because Quarto can wrap them into one irregular paragraph.
- Slide 1 is locked to sample 47 from
  `checkpoints/Annulus_poisson/coupling/artifacts`. Generate one artifact-only
  Plotly composite directly from `data/selected_raw_arrays.npz`; do not load the
  historical checkpoint or current model/config contract. The top row contains
  signed `phi_error` and `psi_error`, and the bottom row contains
  `u_phi_error`, `u_pred_error`, and `u_psi_error`.
- The Slide 1 source-error row and solution-error row use separate
  zero-centered `RdBu` ranges and one colorbar per row. All five panels share
  the same physical extent and one-to-one aspect ratio. The implemented deck
  uses the standalone offline Plotly HTML directly; PNG or repository PDF
  copies are not presentation inputs. Artifact summary provenance identifies
  sample 47 as the maximum-\(\mathrm{rel\_sol}\) selected test sample.
- The optional FEniCSx `target_phi/target_psi` fields are numerical diagnostics,
  not exact balance labels. The slide may use `phi_error/psi_error` to locate
  directional mismatch, but speaker notes must state that their nonzero sum
  contains the target balance residual and does not imply failure of the
  prediction's exact balance projection.
- Slide 2 uses the historical notation
  \(P_0=\texttt{raw\_unit\_phi}\),
  \(Q_0=\texttt{raw\_unit\_psi}\). It defines physical raw proposals
  \(p_0=P_0/L_x^2\), \(q_0=Q_0/L_y^2\), applies the pointwise physical
  symmetric projection
  \(\phi=(f+p_0-q_0)/2\), \(\psi=(f-p_0+q_0)/2\), and then pulls back
  \(\Phi=L_x^2\phi\), \(\Psi=L_y^2\psi\) for unit-interval Green
  reconstruction.
- Slide 2 displays the measured geometry change as two separate indicators:
  `line-length ratio 2.19x` and dominant neighboring `L^2` response-scale ratio
  `4.80x`. The ratio card has a dedicated third grid track, and slide-local
  geometry, pipeline, equation, and takeaway heights use the lower slide area
  without hiding overflow.
- The expanded Slide 2 equations retain the cross-axis terms
  \(-(L_x^2/L_y^2)Q_0/2\) in \(\Phi\) and
  \(-(L_y^2/L_x^2)P_0/2\) in \(\Psi\). The historical sample-47 stored `phi`
  is reproduced by this formula with maximum discrepancy
  \(4.44\times10^{-16}\), so the slide formula is artifact-verified rather
  than inferred from the current checkpoint contract.
- Slide 3 does not repeat the full transformation. It maps physical
  directional-source errors through
  \(E_\Phi=L_x^2e_\phi\), \(E_\Psi=L_y^2e_\psi\), then compares exact-Green and
  learned-Green solution responses using the existing one-segment/split-line
  RMS ratios.
- Slide 4 is a four-card horizontal timeline. It includes original
  physical-symmetric projection, length-aware geometry-weighted/RPS variants,
  response-space, and current physical-symmetric. Presentation titles and
  Korean notes do not use formulation version numbers. Its claim is limited to
  the observation that all variants enforce balance, alter directional-error
  distribution, and nevertheless leave the transition pattern unresolved.
  The length-aware card defines
  \(d_0=(\sigma_y-\sigma_x)f/(\sigma_x+\sigma_y)\),
  \(\kappa=4\sigma_x\sigma_y/(\sigma_x+\sigma_y)^2\), and
  \(d_{\mathrm{RPS}}=d_0+\kappa d\): \(d_0\) is the equal-response baseline
  and \(\kappa\) attenuates raw difference under length imbalance.
  Response-space may report source/flux improvement with unchanged solution
  error, but existing runs must not be described as a controlled single-factor
  paired ablation.
- A compact Slide 4 note may state that transition-specific edge weighting was
  tested and removed because local improvement did not lower global solution
  error. Do not show its loss formula or metrics.
- Slide 5 defines the common post-reconstruction estimator
  \(u_{\mathrm{blend}}=w_\phi u_\phi+w_\psi u_\psi\),
  \(w_\phi+w_\psi=1\), and three fixed information contracts: known topology
  for geometry-only compact C2, prediction mismatch for detected-seam C2, and
  local full-PDE defect for weak-residual reliability. The three rules do not
  alter \(\phi,\psi,u_\phi,u_\psi\), training, GreenNet, or CouplingNet, and no
  reference target enters weight construction.
- Slide 5 compares every rule with the same `Signal`, `Sample adaptive`, and
  `Operator aware` rows. Geometry C2 is known-geometry/no/no, mismatch C2 is
  mismatch-jump/yes/no, and weak reliability is local-full-PDE-defect/yes/yes.
  A full-width invariant band states that
  \(\phi,\psi,u_\phi,u_\psi\) remain frozen and only the final partition changes.
- Slides 6-8 expand the three weight constructions with a common
  input/sensor-to-weight visual grammar. Slide 6 derives known-transition
  distances and the compact quintic bump; Slide 7 derives normalized mismatch
  edge profiles, smoothing/NMS, and a separately prescribed C2 ramp; Slide 8
  defines the weak rule by
  \(R_\phi=R_x(u_\phi;\phi)+R_y(u_\phi;\psi)\) and
  \(R_\psi=R_x(u_\psi;\phi)+R_y(u_\psi;\psi)\) from local P1 weak defects.
  The rule uses no `sol`, target flux, known transition coordinate, or global
  matrix solve and changes only the final reconstruction blend.
- Slide 9 presents the frozen Poisson `coupling15` comparison first. Equal mean,
  geometry C2, mismatch seam C2, and weak residual have mean `rel_sol`
  `5.569825%`, `5.470165%`, `5.312818%`, and `4.860579%`. Their changes versus
  equal mean are `-1.789%`, `-4.614%`, and `-12.734%`, with `35/50`, `44/50`,
  and `50/50` wins. Weak transition RMS changes by `-8.360%` and trace jump by
  `-48.280%`; its maximum neighboring weight jump remains `0.302343`.
- Slide 10 presents the frozen CDR `coupling5` consistency check. The four mean
  `rel_sol` values are `5.156246%`, `5.054177%`, `4.943157%`, and `4.564722%`,
  giving changes `-1.980%`, `-4.133%`, and `-11.472%`. The estimator ordering
  is identical to Poisson. Poisson and CDR use independent sources but the same
  Annulus geometry, so the result is cross-equation consistency rather than
  cross-domain validation.
- Poisson is presented before CDR to isolate geometry/reconstruction behavior
  from convection and reaction. Speaker notes must preserve the experimental
  provenance: fixed presets originated in CDR exploratory diagnostics and were
  transferred to Poisson without a Poisson sweep. Presentation order is not
  tuning chronology.
- Slides 11-12 present the Poisson weak-result calculation for selected q50-role
  sample 0. Slides 13-14 use the same field/error grammar for CDR sample 9 and
  include physical direct-at-`coords_valid` coefficient fields. The result
  contract is
  \(u_{\mathrm{pred}}^{\mathrm{weak}}
  =w_\phi^{\mathrm{weak}}u_\phi
  +(1-w_\phi^{\mathrm{weak}})u_\psi\). The weak rule changes only the final
  blend, so \(\phi,\psi,u_\phi,u_\psi\) remain the frozen-checkpoint fields.
  Figure builders must load the final solution from
  `u_weak_residual_reliability`; standard artifact `u_pred` is equal mean and
  must not be relabeled as the weak result.
- Slide 11 uses a dedicated two-line sidebar formula card for
  \(u_{\mathrm{pred}}^{\mathrm{weak}}\), rather than the generic equation band.
  The result-field builder places the shared `f` and `u` colorbar titles on the
  right side of the bars with explicit thickness, padding, tick font, and figure
  right margin; the same layout is used for the CDR result-field slide.
- Slides 11-14 report signed `phi/psi` differences against numerical FEniCSx
  directional targets and signed solution errors against `sol`. References are
  evaluation-only, `rel_flux` remains an unchanged directional-source
  diagnostic, and these result slides omit transition markers, line-length
  maps, trace-jump metrics, and causal-diagnosis language.
- Slide 15 starts from the symmetric correction
  \(\delta\phi=\delta\psi=(f-p-q)/2\), defines
  \(\delta u_\phi=H_x\delta\phi\), \(\delta u_\psi=H_y\delta\psi\), and makes
  explicit that equal source corrections can produce unequal reconstructed
  responses when \(H_x\ne H_y\). Slide 16 then defines fixed source-point gains
  \(\gamma_{s,j}^2=[H_s^\top M_\Omega H_s]_{jj}\), the column-diagonal
  constrained minimization, and its exact-balance closed form. This is the
  squared output-mass-weighted norm of column \(j\), so it measures how one
  source-point correction moves the whole directional solution field. The gain
  is derived from geometry, prescribed coefficients, quadrature, and the frozen
  GreenNet; it is not learned. Off-diagonal source-response correlations and
  full response-matrix solves are excluded. The visible Slide 16 method card
  must show the full system
  \((A_x+A_y)\delta\phi=A_y r\) before replacing \(A_s\) by its diagonal, so
  the reason for the no-solve approximation is explicit rather than confined
  to speaker notes. This method changes
  the directional split and therefore requires a controlled paired retraining
  experiment.
- Green-Response Preconditioning is not a revival of the retired response-space
  formulation: that formulation used analytic length scales, while the new proposal uses a
  fixed gain derived from the full axial response
  \(H_s=K_sW_sL_s^2\). Its column-diagonal definition must pass discarded
  off-diagonal-energy, positivity, grid-refinement, and boundary audits before
  implementation.
- This meeting block excludes all pre-projection fuser experiments and the
  discussion about weakening \(u_\phi,u_\psi\in H_0^1(\Omega)\). It also
  excludes relative split consistency, training-time weak operator closure,
  self-trace gluing, cross-axis carrier, null-space analysis, and
  boundary-energy development from the main slides, backups, and speaker
  notes. Slides 5-14 retain weak residual only as a post-hoc reliability
  indicator. Learned reconstruction weights/gates, Multi-Orientation Axial
  Charts, and full Green-response matrix solves are also excluded.
- The slide claim must distinguish verified diagnostics from the working
  hypothesis: exact and learned Green reconstructions propagate essentially the
  same transition structure, while insufficient inverse-\(L^2\) compensation
  by the predicted physical source split is the leading interpretation rather
  than a completed theorem.
- Slides 9-14 are measured post-hoc results and must not be described as
  production remedies. Slides 15-16 explain a proposed, unvalidated method. The
  presentation order is method definition, Poisson clean evidence, CDR
  broader-operator consistency, Poisson/CDR result fields and errors, then
  response-gain audit and paired
  preconditioned retraining. Independent geometry validation remains required
  before production integration.
- The fixed Smooth Cross-Axis Reconstruction Blend now has a standalone
  diagnostic implementation in
  `cli/diagnose_fixed_smooth_cross_axis_blend.py` and
  `src/greenonet/complex_smooth_blend_diagnostics.py`. It must remain outside
  production inference until a later explicit decision. The geometry-only
  fields use `y_edges` for
  \(J_\phi=\mathcal S(|\Delta\log L_x^2|)\), `x_edges` for
  \(J_\psi=\mathcal S(|\Delta\log L_y^2|)\), positive exponential
  reliabilities, and a partition of unity. The default preset is fixed before
  test evaluation: `alpha=1/log(2)`, two 50:50 neighbor-smoothing passes,
  reliability floor `1e-6`, transition threshold `log(2)`, and two graph-ring
  transition dilation steps.
- On the full 50-sample `coupling5` test set, the fixed preset changes mean
  `rel_sol` from `5.156246%` to `5.116388%` and wins on `30/50` samples. It
  reduces transition trace-error jump RMS from `2.174435e-4` to
  `9.931488e-5` (`-54.326%`) on average and improves that trace diagnostic on
  all 50 samples, but increases broad transition-zone error RMS from
  `3.680150e-4` to `3.715199e-4` (`+0.952%`). The result is
  `promising_but_mixed`, not sufficient evidence to change the production
  estimator. Test `sol` is evaluation-only and must not be used to retune the
  same preset. Canonical evidence is under
  `checkpoints/annulus_CDR/coupling5/fixed_smooth_cross_axis_blend/`.
- The standalone diagnostic also supports
  `weight_construction="compact_c2_ramp"`. It detects each axial family’s
  split/merge interface as the midpoint between neighboring fixed coordinates
  whose connected-segment multiplicities differ. For each point it computes
  the absolute transverse-coordinate distance to that interface and uses
  `B(s)=1-10*s^3+15*s^4-6*s^5` for `0 <= s < 1`, with zero influence outside
  the configured physical width. With
  `theta=gamma*(B_psi-B_phi)`, the final weights are
  `w_phi=(1+theta)/2`, `w_psi=(1-theta)/2`. Thus the estimator is exactly the
  equal mean outside compact support, favors `u_psi` near horizontal-chart
  transitions, favors `u_phi` near vertical-chart transitions, and cancels
  back to equal weights where both transition bands overlap.
- On the same frozen `coupling5` checkpoint, the predeclared compact preset
  `gamma=0.5`, physical width `0.03125=4h` changes mean `rel_sol` from
  `5.156246%` to `5.054177%` (`-1.980%` relative), transition-zone error RMS
  from `3.680150e-4` to `3.524676e-4` (`-4.225%`), and transition trace-error
  jump RMS from `2.174435e-4` to `1.196888e-4` (`-44.956%`). It wins global
  `rel_sol` on `36/50` samples. The weight range is
  `[0.254013,0.745987]`, the maximum neighboring weight jump is `0.112396`,
  partition residual and outside-support correction are exactly zero, and
  paired-bootstrap 95% intervals for all three aggregate relative changes lie
  below zero.
- The same fixed compact C2 preset was run post-hoc on the Poisson
  `checkpoints/Annulus_poisson/coupling15` best-energy checkpoint and all 50
  full-reference test samples. Mean `rel_sol` changes from `5.569825%` to
  `5.470165%` (`-1.789%`), transition-zone RMS from `3.310015e-4` to
  `3.197070e-4` (`-3.412%`), and transition trace-error jump RMS from
  `1.923231e-4` to `1.051847e-4` (`-45.308%`). It improves `rel_sol` on
  `35/50`, transition RMS on `31/50`, and trace jump on `50/50` samples; all
  three paired-bootstrap aggregate 95% intervals are below zero.
- The Poisson and CDR checks share the exact annulus geometry, so their compact
  C2 transition coordinates, support, and weight fields are identical:
  \(w_\phi,w_\psi\in[0.254013,0.745987]\), support size `3192`, and maximum
  neighboring weight jump `0.112396`. Their test forcing realizations are not
  paired: the Poisson generator uses seed `2732`, while CDR uses seed `2222`,
  and none of the 50 same-index `rhs` arrays matches exactly. Treat the result
  as evidence of cross-equation reuse on one geometry, not cross-domain or
  same-source paired validation. Canonical evidence is under
  `checkpoints/Annulus_poisson/coupling15/compact_c2_cross_axis_blend/`.
- The compact-ramp parameter sweep is evaluation-only sensitivity analysis.
  A broad sweep shows that `gamma=0.25` and `0.5` improve all three aggregate
  diagnostics for widths `2h,4h,6h,8h`, while `gamma>=0.75` reduces trace jump
  more aggressively but worsens solution metrics. A finer sweep over
  `gamma=0.2..0.5` and widths `2h..10h` improves all three metrics in all 30
  tested combinations; its best mean `rel_sol` occurs at
  `gamma=0.3,width=10h` and its best transition RMS at
  `gamma=0.3,width=6h`. The exact linear-plus-quadratic MSE decomposition
  estimates a target-derived optimum near `gamma=0.29..0.33`. These selected
  values are not independent confirmation and must not silently become
  production defaults. Canonical exploratory evidence is under
  `checkpoints/annulus_CDR/coupling5/compact_c2_cross_axis_blend_fine_sweep/`.
- The prediction-only mismatch-gradient comparison is implemented in
  `cli/compare_cross_axis_blend_estimators.py` and
  `src/greenonet/complex_mismatch_blend_diagnostics.py`. It compares the equal
  mean, the fixed geometry-only compact ramp, and a sample-dependent estimator
  built from normalized axial-edge jumps of `u_phi-u_psi`. The mismatch
  estimator uses
  `theta=gamma*activation*(Jx-Jy)/(Jx+Jy+eps)` and a smooth C2 activation; it
  reads axial adjacency and grid spacing but not segment lengths, transition
  coordinates, `sol`, or flux targets. Geometry-derived transition masks are
  used only to audit localization and evaluation metrics.
- On the frozen `coupling5` checkpoint, the prediction-only fixed exploratory
  preset `gamma=0.5`, two 50:50 graph-smoothing steps, and normalized activation
  interval `[0.15,0.35]` changes mean `rel_sol` from `5.156246%` to
  `4.999177%` (`-3.046%` relative), transition-zone error RMS by `-7.345%`,
  and transition trace-error jump RMS by `-28.278%`; it wins `rel_sol` on
  `43/50` samples. The geometry-only compact ramp gives `5.054177%`,
  `-4.225%`, and `-44.956%`, respectively. The prediction-only method is
  therefore better for mean solution and broad transition error in this run,
  while geometry-only remains the stronger trace-jump suppressor. Its mean
  transition activation is `0.427486` versus `0.086350` in the regular zone,
  and the transition zone contains `56.517%` of sensor energy.
- The mismatch-gradient result remains post-hoc exploratory. The activation
  thresholds were chosen from prediction-only scale inspection on the same
  test run, so they must not become production defaults without a frozen
  calibration policy and independent checkpoint/dataset confirmation.
  Canonical evidence is under
  `checkpoints/annulus_CDR/coupling5/cross_axis_blend_estimator_comparison/`.
- The mismatch-detected seam C2 estimator is implemented in the same
  `cli/compare_cross_axis_blend_estimators.py` and
  `src/greenonet/complex_mismatch_blend_diagnostics.py` path. It intentionally
  separates sample-dependent detection from weight construction. For each
  sample, normalized axial-edge jumps of `u_phi-u_psi` are reduced to x/y RMS
  profiles; after one-dimensional smoothing, physical non-maximum suppression
  selects at most two seam coordinates per axis. Only these coordinates and
  their directions survive into the weight builder.
- The seam weight builder uses a separate compact quintic C2 influence
  `B(s)=1-10*s^3+15*s^4-6*s^5` on each detected seam and
  `theta=gamma*(B_x-B_y)`. It never uses segment lengths, known topology
  transition coordinates, `sol`, or target fluxes. The annulus transition
  metadata enters only the post-hoc detection-accuracy and error audits.
  Default non-maximum separation resolves to four ramp widths, which avoids
  selecting two nearby peaks from one transition side in the tested annulus.
- The canonical frozen-`coupling5` exploratory candidate uses
  `gamma=0.3`, width `0.09375=12h`, two seams per axis, relative peak
  threshold `0.25`, and one profile-smoothing step. It changes mean `rel_sol`
  from equal-mean `5.156246%` and direct-mismatch `4.999177%` to
  `4.943157%`. Relative to equal mean, transition-zone RMS changes by
  `-7.391%`, trace jump by `-22.425%`, and global `rel_sol` by `-4.133%`.
  The maximum neighboring weight jump is `0.023329`, about one tenth of the
  direct sensor's `0.242220`.
- The detected-seam C2 estimator is not uniformly superior. Relative to direct
  mismatch it improves mean `rel_sol` by `1.121%` with paired-bootstrap 95%
  interval `[-2.196%,-0.070%]`, leaves broad transition RMS effectively tied
  at `-0.051%`, and worsens trace jump by `8.160%`. The 80-combination sweep
  chooses different candidates for mean solution (`gamma=0.3,width=12h`),
  broad transition RMS (`gamma=0.3,width=6h`), and trace jump
  (`gamma=0.5,width=6h`). Thresholds `0.15..0.30` do not change the selected
  top-two seams in this run. These are same-test exploratory results, not
  independent calibration or a production estimator decision. Canonical
  evidence is under
  `checkpoints/annulus_CDR/coupling5/cross_axis_blend_detected_seam_c2_comparison/`.
- The general local weak-residual reliability diagnostic is implemented in
  `cli/compare_weak_residual_reliability_blend.py` and
  `src/greenonet/complex_weak_residual_blend_diagnostics.py`. It compares four
  frozen-checkpoint reconstructions: equal mean, geometry-only compact C2,
  mismatch-detected seam C2, and local weak-residual reliability. For each
  directional candidate `u_phi` or `u_psi`, it applies the existing x/y P1
  weak operators and assembles the full defect
  `R=Rx(u;phi)+Ry(u;psi)` by local element gather/scatter. It never solves a
  global matrix equation and never uses reference `sol` or target fluxes to
  construct weights. Smoothed mass-normalized squared residuals select the
  lower-defect candidate through a partition of unity; equal evidence returns
  `w_phi=w_psi=0.5`.
- On frozen `annulus_CDR/coupling5`, the fixed weak preset `gamma=0.5`, two
  50:50 graph-smoothing steps, and `relative_floor=0.1` changes mean `rel_sol`
  from `5.156246%` to `4.564722%` (`-11.472%` relative), broad transition RMS
  by `-9.870%`, and transition trace-jump RMS by `-48.785%`. It wins `rel_sol`
  on `49/50` samples and trace jump on `50/50`; the paired-bootstrap 95% CI for
  aggregate `rel_sol` change is `[-12.736%,-10.261%]`. The pointwise weight
  field remains axially striped and its maximum neighboring jump is
  `0.315646`, so the result supports the reliability signal but does not prove
  transverse regularity or justify production integration. The optional
  36-case sweep is same-test target sensitivity analysis and must not define a
  default without an independent frozen calibration/validation protocol.
  Canonical evidence is under
  `checkpoints/annulus_CDR/coupling5/weak_residual_reliability_blend_comparison/`.
- The same fixed four-way comparison was run without a sweep on all 50 Poisson
  test samples from frozen `Annulus_poisson/coupling15`. Mean `rel_sol` is
  `5.569825%` for equal mean, `5.470165%` for geometry-only compact C2,
  `5.312818%` for mismatch-detected seam C2, and `4.860579%` for local
  weak-residual reliability. Their relative changes versus equal mean are
  `-1.789%`, `-4.614%`, and `-12.734%`, with `35/50`, `44/50`, and `50/50`
  sample wins. Broad transition RMS changes by `-3.412%`, `-6.954%`, and
  `-8.360%`; transition trace-jump RMS changes by `-45.308%`, `-23.553%`, and
  `-48.280%`. Thus the Poisson ordering matches the independent-source CDR
  run: weak residual is strongest globally, geometry-only C2 is the more
  controlled topology-specific trace smoother, and detected-seam C2 occupies
  the middle ground. The weak weights still have axial stripe structure and a
  maximum neighboring jump of `0.302343`, so this remains post-hoc evidence,
  not structural regularity or a production-estimator decision. Canonical
  evidence is under
  `checkpoints/Annulus_poisson/coupling15/weak_residual_reliability_blend_comparison/`.
- Complex CouplingNet supports an optional production final reconstruction at
  `coupling_model.cross_axis_reconstruction`. The absent/disabled default
  remains exact equal mean. The only supported production mode is
  `local_weak_residual_reliability`; geometry-only compact C2 and
  mismatch-detected seam C2 remain post-hoc-only. The production mode reuses
  the existing full directional axial P1 weak operator, combined nodal mass,
  and the valid-point graph `x_edges union y_edges`. It uses the locked preset
  `gamma=0.5`, two 50:50 smoothing steps, and `relative_floor=0.1` unless the
  config explicitly overrides them. It changes only detached trainer
  `rel_sol`, evaluator official `u_pred`, and complex artifacts. Canonical loss,
  checkpoint selection, projection, `phi/psi`, `u_phi/u_psi`, `rel_flux`, model
  state keys, and the output contract remain unchanged. Reliability weights
  use no `sol/target_phi/target_psi` and require no global matrix assembly or
  solve. Source-only train/validation skips the optional reconstruction.
- `annulus_CDR/coupling6` uses both column-diagonal Green-response projection
  and production local weak-residual final reconstruction, with the
  pre-projection fuser disabled. The best-energy checkpoint is exactly epoch
  50; validation energy then rises `15.87%` by epoch 100, driven mainly by an
  `87.34%` boundary-energy increase. On the frozen 50-sample test set, weak
  reconstruction changes mean `rel_sol` from `5.225503%` to `4.680390%`
  (`-10.432%`, 50/50 wins), transition RMS by `-9.161%` (38/50 wins), and
  transition trace-jump RMS by `-49.401%` (50/50 wins). Versus the separately
  trained physical-symmetric `coupling5`, `coupling6` has `-15.409%` boundary
  energy but `+22.924%` mean `rel_flux`, with all 50 flux samples worse. The
  projection weights are extreme (`w_phi<0.1` on `22.29%`, `w_phi>0.9` on
  `21.85%`) and transition-edge weight-jump RMS is `0.247887`. This supports
  the weak final estimator but leaves column-diagonal projection as an
  unresolved boundary/flux tradeoff requiring paired seeded retraining.
  Canonical report:
  `checkpoints/annulus_CDR/coupling6/weak_residual_reliability_analysis/diagnosis_report.md`.

## Verification Defaults

- `.venv`가 없으면 `/home/jjhong0608/.conda/envs/green_net/bin/python`을 사용한다.
- Focused pytest를 먼저 실행하고, 이후 touched files에 대해 `ruff check`와
  필요한 `mypy`를 실행한다.
- Repo-wide `mypy src`는 2026-06 기준 통과 상태로 정리되었다. 새 변경 이후에는
  focused mypy뿐 아니라 가능한 경우 `mypy src` 전체 통과를 유지한다.
- Markdown 파일은 `ruff` 대상이 아니다.
- 이미 dirty worktree인 경우, 요청 범위와 무관한 변경은 건드리지 않는다.

## Open Planning Items

- 논문용 coefficient family 목록 확정.
- 각 family별 dataset 규모와 random seed 확정.
- GreenONet baseline config와 CouplingNet baseline config 확정.
- Ablation 목록 확정:
  - projection on/off
  - `balance_loss`
  - `symmetric_boundary_loss`
  - `smooth_mask` mask type
  - axis-1D trunk
  - coefficient term inputs
- Figure 목록과 paper section별 배치 확정.
- 결과 저장 directory naming convention 확정.
- Run log, metric CSV, generated figure를 어떤 형식으로 archive할지 확정.

## Update Policy

- Codex는 이 repo에서 답변하기 전에 이 `memory.md`를 먼저 참고한다.
- 답변 이후에는 새로 합의된 durable decision, 실험 기준, coefficient 의미,
  figure/data planning 기준이 생겼는지 판단하고 이 파일을 업데이트한다.
- 이 파일은 모든 작업 로그를 기록하는 곳이 아니다.
- 앞으로 반복해서 참조해야 하는 수학적 의미, API convention, 실험 설계 기준,
  figure planning 기준만 추가한다.
- 일회성 command output, transient error, 구현 세부 diff는 기록하지 않는다.
- 논문용 결과 생성 과정에서 확정된 dataset/figure/run naming은 이 파일에
  누적해서 갱신한다.
