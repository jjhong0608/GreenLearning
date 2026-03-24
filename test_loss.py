import torch
from torch import trapezoid

from greenonet.greens import ExactGreenFunction, EllipticGreenFunction, IntegrationEllipticGreenFunction
from greenonet.numerics import simpson

torch.set_default_dtype(torch.float64)

def test_rel(m: int) -> float:
    x = torch.linspace(0, 1, m)
    xi = torch.linspace(0, 1, m)
    u_val = torch.sin(2 * torch.pi * x)
    a_val = torch.ones_like(x)
    f_val = 4 * torch.pi ** 2 * torch.sin(2 * torch.pi * x)
    kernel = ExactGreenFunction(x, a_val)()
    source = f_val.unsqueeze(-2)
    rhs = source * kernel
    integ = simpson(rhs, x=xi, dim=-1)
    residual = u_val - integ
    residual_energy = simpson(residual.pow(2), x=x, dim=-1)
    solution_energy = simpson(u_val.pow(2), x=x, dim=-1)
    rel = torch.sqrt(residual_energy / solution_energy)
    return rel.item()

def test_green(m: int):
    x = torch.linspace(0, 1, m)
    xi = torch.linspace(0, 1, m)
    a_val = 1.0 + 0.5 * torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * xi[2])
    ap_val = torch.pi * torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * xi[2])
    kernel = ExactGreenFunction(x, a_val)()
    ekernel_fun = EllipticGreenFunction()
    iekernel_fun = IntegrationEllipticGreenFunction()
    trunk_grid = torch.meshgrid(x, xi, indexing="ij")
    trunk_grid = torch.stack(trunk_grid, dim=-1)
    ekernel = ekernel_fun(trunk_grid).squeeze(-1)
    iekernel = iekernel_fun(trunk_grid).squeeze(-1)
    ekernel0 = ekernel_fun(trunk_grid).squeeze(-1) / a_val.unsqueeze(-1)
    iekernel0 = iekernel_fun(trunk_grid).squeeze(-1) / a_val.unsqueeze(-1) * ap_val.unsqueeze(0) / a_val.unsqueeze(0)

    print(a_val[63:66])
    print(ap_val[63:66])
    print((kernel - ekernel)[64, :])
    print((kernel - ekernel)[:, 64])
    print((kernel - ekernel0 - iekernel0)[64, :])
    print((kernel - ekernel0 - iekernel0)[:, 64])
    print((ekernel - ekernel0)[64, :])
    print((ekernel - ekernel0)[:, 64])




if __name__ == '__main__':
    # m = 2 ** 6
    # print(test_rel(m + 1))
    # m = 2 ** 7
    # print(test_rel(m + 1))
    # m = 2 ** 8
    # print(test_rel(m + 1))
    # m = 2 ** 9
    # print(test_rel(m + 1))

    m = 2 ** 7
    test_green(m + 1)
