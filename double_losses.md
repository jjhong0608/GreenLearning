# Design flux-divergence balance loss and consistency loss

## Goal
- You should implement two losses
	- The first is `flux-divergence balance` loss, $\mathcal{L}_{\textrm{FDB}}=\varphi+\psi-f$ for all cross points.
	- The second is `consistency` loss, $\mathcal{L}_{\textrm{consistency}}=u^{(x)}(\bar{x},\bar{y})-u^{(y)}(\bar{x},\bar{y})=0$ for all cross points, $(\bar{x},\bar{y})$.
- Total loss is constructed as $\lambda_{\textrm{FDB}}\mathcal{L}_{\textrm{FDB}}+\lambda_{\textrm{consistency}}\mathcal{L}_{\textrm{consistency}}$
- Hear, $u^{(x)}(\bar{x},\bar{y})=\int_{0}^{1}\,G^{(x)}(\bar{x},\xi)\varphi(\xi,\bar{y})\,\textrm{d}\xi$ and $u^{(y)}(\bar{x},\bar{y})=\int_{0}^{1}\,G^{(y)}(\bar{y},\eta)\psi(\bar{x},\eta)\,\textrm{d}\eta$.
- $G^{(x)}$ and $G^{(y)}$ are pretrained Green's fucntion from GreenONet.
- Current `Cross-Integral` loss should be remained. I'll select these two-option in configuration file.