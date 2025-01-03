import torch
from safetensors.torch import load_model

from models.agm import AGM
from models.modified_greenlearning import GreenLearning
from models.oper_learn_arch import DeepOperatorLearningArchitecture
from utils import get_device


def pipeline() -> None:
    torch.set_default_dtype(torch.float64)
    device = get_device()
    green_learning = GreenLearning(
        green_layers=[2] + [50] * 4 + [1],
        # hom_layers=[1] + [50] * 4 + [1],
        work_dir="work_dir/helmholtz/DeepONet_poisson",
        device=device,
    )
    # green_learning.load_data("Examples/poisson.npz").load_valid_data(
    #     "Examples/poisson_valid.npz"
    # ).train().save_model()
    green_learning.load_data("Examples/poisson_valid.npz").load_model(
        "work_dir/helmholtz/DeepONet_poisson/20250102_162741"
    ).evaluate()
    # green_learning.load_data("Examples/poisson_valid.npz").load_model(
    #     "work_dir/helmholtz/DeepONet_poisson/20250102_145741"
    # ).plot_green_function()
    # green_learning.load_data("Examples/helmholtz.npz").train().green_function_output()


def agm() -> None:
    model = DeepOperatorLearningArchitecture(rhs_size=201, num_layers=4, hidden_size=50)
    load_model(
        model,
        "work_dir/helmholtz/DeepONet/green_network.safetensors",
        # "work_dir/helmholtz/DeepONet/model_basis_training/green_network.safetensors",
    )
    agm = AGM(model)
    agm.load_data()
    agm.construct_matrix().make_matrix().calculate_matrix().calculate_solution().plot_solution()
    # agm.plot()


if __name__ == "__main__":
    pipeline()
    # agm()
