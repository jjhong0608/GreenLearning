# CouplingNet

## Description of CouplingNet
We'd solve following Poisson's equation with variable-coeffient in two-dimensional unit squre domain $[0,1]^2$.
$$
-\nabla\cdot(\kappa(x,y)\nabla{u}(x,y))=f(x,y)
$$
with homogeneous Dirichlet boundary condition. We'd use neural network model inspired by `Axial Green's function Method (AGM)`. For applying this, we should decompose the governing equation.
$$
\begin{cases}
-\partial_{x}(\kappa(x,y)\partial_{x}u(x,y))=\varphi(x,y)
&
\textrm{along $x$-axial line},
\\
-\partial_{y}(\kappa(x,y)\partial_{y}u(x,y)=\psi(x,y))
&
\textrm{along $y$-axial line},
\end{cases}
$$
where $\varphi(x,y)$ and $\psi(x,y)$ are axial flux-divergences, respectively. They have to satisfy $\varphi(x,y)+\psi(x,y)=f(x,y)$. Because we already constructed `GreenONet` to predict the Green's function, if we predict these axial flux-divergence accurately, then we could represent the solution $u$ of the governing equation as follows,
$$
\begin{cases}
u(x,\bar{y})=\int_{0}^{1}\,G(x;\xi)\varphi(\xi,\bar{y})\,\textrm{d}\xi,
&
\textrm{along $x$-axial line},
\\
u(\bar{x},y)=\int_{0}^{1}\,G(y;\eta)\psi(\bar{x},\eta)\,\textrm{d}\eta,
&
\textrm{along $y$-axial line},
\end{cases}
$$
where $\bar{x}$ and $\bar{y}$ are fixed points. So `CouplingNet` aims to predict axial flux-divergences. The model follows `MIONet` archtecture and it takes the coordinates and sampling points of the variable-coefficient $\kappa$ and source-term, $f$ on each axial line as input. The output of the model is the axial flux-divergence at the coordinates on the axial line. the shape of the coordinates is `(B,2,n_lines,m_points,2)` and the sampling points is `(B,2,n_line,m_points)`. The shape of the output, axial flux-divergence, is `(B,2,n_lines,m_points)`. Each batch is constructed from each two-dimensional problem. On the other hand, the Green's function is same for each batch because we're solving same variable-coefficient Poisson's problem. The only thing we change is the source-term. So, we should keep the Green's function from `GreenONet` with shape `(2,n_lines,m_points,m_points)`. it is not nesessary to evaluate `GreenONet` for every calculation of the loss function. It's a waste of the time. 

## Loss function construction
For axial flux-divergence, $\varphi$ and $\psi$, we could construct two cross-integral equations. For every cross points of the x/y-axial lines, $(\bar{x},\bar{y})$, 
$$
\int_{0}^{1}\,G^{(x)}(\bar{x};\xi)\varphi(\xi,\bar{y})\,\textrm{d}\xi+\int_{0}^{1}\,G^{(y)}(\bar{y};\eta)\varphi(\bar{x},\eta)\,\textrm{d}\eta=\int_{0}^{1}\,G^{(y)}(\bar{y};\eta)f(\bar{x},\eta)\,\textrm{d}\eta,
\\
\int_{0}^{1}\,G^{(x)}(\bar{x};\xi)\psi(\xi,\bar{y})\,\textrm{d}\xi+\int_{0}^{1}\,G^{(y)}(\bar{y};\eta)\psi(\bar{x},\eta)\,\textrm{d}\eta=\int_{0}^{1}\,G^{(x)}(\bar{x};\xi)f(\xi,\bar{y})\,\textrm{d}\xi.
$$
For each batch, these two-cross-integral equations should be satisfied. If we use this loss, it is unsupervised learning because we only use the source-term $f$ and variable-coefficient $\kappa$ for training.

## Traning &Validation sets
The `CouplingNet` conduct training using unsupervised learning. However, we need source-term, $f$ for training and to validate the model for each epoch. So I'll provide datasets in `2D_data` folder. All datasets have `npz` extension and they have contens `sol`, `uxx`, `uyy`, `rhs`. `sol` and `rhs` have its shape `(257, 257)` and `uxx` and `uyy` have `(255, 255)`. `sol` and `rhs` are the solution and the source-term on the uniform grid in $[0,1]^2$, respectively. `uxx` and `uyy` are x/y-axial flux-divergence on same grid except boundaries. So, `uxx` and `uyy` should be extended or padded to the boundary using constant extrapolation. Then, all datasets are have its shape `(257, 257)`. Because they are calculated on uniform grid with step size, $h=1/256$, We should construct axial lines on this grid. For example, it is possible to construct uniform x/y-axial lines with gap as $1/16$. It also should be constructed except boundaries similar to axial lines for `GreenONet`. The number of coordinates on each axial line also should be selected same rule. Because the sampling points should be selected from datasets. Then, the source-term, `rhs` is used for training and others, `sol`, `uxx`, `uyy` are used for validation. We could validate the reprented solution along x/y-axial lines with `sol` and x/y-axial flux-divergences with `uxx` or `uyy` directly. You have to note that the y-axial flux-divergence is used on y-axial lines, `uyy` should be transposed for align axial lines./

## Instruction
You must following belows.
- Don't touch `GreenONet` part. The `CouplingNet` should be separated from `GreenONet` architecture and training.
- The traninig of `CouplingNet` also should be separated from `GreenONet`. Two networks are independent. But `CouplingNet`use pretrained `GreenONet` for traning.
- Datasets also should be separated from `GreenOnet`'s one.
- All pipelines for `CouplingNet` should be separated from `GreenONet`'s pipelines.
- But they should be controlled in same configuration file.
- However, configuration file should separate two network configures.
- The configuration file should contain whether to run training for each network.
- If when `CouplingNet` conduct training but `GreenONet` does not, `GreenONet` should load pretrained model. The pretrained model path also should be given in configuration file.
- If we train both networks, then we should train `GreenONet` first, then, use this network to train `CouplingNet`.
- During the training of `CouplingNet`, training loss, validation loss and Relative L_2 error for flux-divergences and solution are should be printed to log.