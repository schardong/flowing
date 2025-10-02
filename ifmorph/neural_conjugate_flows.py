# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD
from ifmorph.model import SIREN, MLPv2

# -----------------------------------------------------------------------------
# Invertible Layers


def no_shuffle(x, y):
    return x, y


def shuffle(x, y):
    return y, x


def norm(x):
    return torch.linalg.norm(x, dim=-1, keepdim=True)


class AdditiveCouplingLayer(nn.Module):
    """
    Additive Coupling Layer module.
    Args:
        model (nn.Module): Sub-model. A neural network.
        initial_step_size (float or torch.Tensor, optional): Initial step size for the delta computation.
            If None, the default value is 0.1. If a float or tensor is provided, it will be used as the initial step size.
        orientation (str, optional): Orientation of the coupling layer.
            Can be either 'normal' or 'skew'. Defaults to 'normal'.
    Attributes:
        sub_model (nn.Module): Sub-model used for computing the delta.
        initial_step_size (torch.Tensor): Initial step size for the delta computation.
        shuffle (function): Function used for shuffling the input.
    Methods:
        forward(x): Forward pass of the additive coupling layer.
        inverse(x): Inverse pass of the additive coupling layer.
    """

    def __init__(self, model, *, initial_step_size=None, orientation="normal"):
        super().__init__()

        self.sub_model = model
        if initial_step_size is None:
            self.register_buffer("initial_step_size", 0.1 * torch.ones(1))
        elif torch.is_tensor(initial_step_size) or isinstance(initial_step_size, float):
            self.initial_step_size = nn.Parameter(torch.ones(1) * initial_step_size)

        if orientation == "normal":
            self.shuffle = no_shuffle
        elif orientation == "skew":
            self.shuffle = shuffle

    def forward(self, x):
        x_1, x_2 = x.chunk(2, dim=-1)
        x_1, x_2 = self.shuffle(x_1, x_2)

        delta = self.sub_model(x_2) * self.initial_step_size

        x_1 = x_1 + delta
        x_1, x_2 = self.shuffle(x_1, x_2)
        return torch.cat((x_1, x_2), dim=-1)

    def inverse(self, x):
        x_1, x_2 = x.chunk(2, dim=-1)
        x_1, x_2 = self.shuffle(x_1, x_2)

        delta = self.sub_model(x_2) * self.initial_step_size
        x_1 = x_1 - delta

        x_1, x_2 = self.shuffle(x_1, x_2)
        return torch.cat((x_1, x_2), dim=-1)


class DoubleAdditiveCouplingLayer(nn.Module):
    def __init__(self, models, *, gamma=0.1):
        super().__init__()

        self.models = nn.ModuleList(models)

        self.gamma = gamma

    def forward(self, x):
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = x_1 + self.gamma * self.models[0](x_2)
        x_2 = x_2 + self.gamma * self.models[1](x_1)

        return torch.cat((x_1, x_2), dim=-1)

    def inverse(self, x):
        x_1, x_2 = x.chunk(2, dim=-1)

        x_2 = x_2 - self.gamma * self.models[1](x_1)
        x_1 = x_1 - self.gamma * self.models[0](x_2)

        return torch.cat((x_1, x_2), dim=-1)


class TripleCouplingLayer(nn.Module):
    def __init__(self, models, *, gamma=0.1):
        super().__init__()

        assert len(models) == 3, "TripleCouplingLayer requires 3 models."
        self.models = nn.ModuleList(models)

        self.gamma = gamma

    def forward(self, x):
        x_1, x_2 = x.chunk(2, dim=-1)

        x_2 = x_2 + self.gamma * self.models[0](x_1)
        x_1 = x_1 + self.gamma * self.models[1](x_2)
        x_2 = x_2 + self.gamma * self.models[2](x_1)

        return torch.cat((x_1, x_2), dim=-1)

    def inverse(self, x):
        x_1, x_2 = x.chunk(2, dim=-1)

        x_2 = x_2 - self.gamma * self.models[2](x_1)
        x_1 = x_1 - self.gamma * self.models[1](x_2)
        x_2 = x_2 - self.gamma * self.models[0](x_1)

        return torch.cat((x_1, x_2), dim=-1)


class TimeCouplingLayer(nn.Module):
    def __init__(self, models, *, gamma=0.1):
        super().__init__()

        assert len(models) == 3, "TimeCouplingLayer requires 3 models."
        self.models = nn.ModuleList(models)

        self.gamma = gamma

    def forward(self, tx):
        t = tx[:, 0:1]
        x = tx[:, 1:]
        x_1, x_2 = x.chunk(2, dim=-1)

        x_2 = x_2 + self.gamma * self.models[0](torch.cat((t, x_1), dim=-1))
        x_1 = x_1 + self.gamma * self.models[1](torch.cat((t, x_2), dim=-1))
        x_2 = x_2 + self.gamma * self.models[2](torch.cat((t, x_1), dim=-1))

        return torch.cat((t, x_1, x_2), dim=-1)

    def inverse(self, x):
        t = x[:, 0:1]
        x = x[:, 1:]
        x_1, x_2 = x.chunk(2, dim=-1)

        x_2 = x_2 - self.gamma * self.models[2](torch.cat((t, x_1), dim=-1))
        x_1 = x_1 - self.gamma * self.models[1](torch.cat((t, x_2), dim=-1))
        x_2 = x_2 - self.gamma * self.models[0](torch.cat((t, x_1), dim=-1))

        return torch.cat((t, x_1, x_2), dim=-1)


# class AdditiveOrthogonalCouplingLayer(nn.Module):
#     def __init__(self, model, orthogonal_layer):
#         super().__init__()

#         self.sub_model = model
#         self.orthogonal_map = orthogonal_layer

#     def forward(self, x):
#         x = self.orthogonal_map(x)

#         x_1, x_2 = x.chunk(2,dim=-1)

#         delta = self.sub_model(x_2)
#         x_1 = x_1 + delta

#         x = self.orthogonal_map.inverse(torch.cat((x_1,x_2),dim=-1))
#         return x

#     def inverse(self, x):
#         x = self.orthogonal_map(x)

#         x_1, x_2 = x.chunk(2,dim=-1)

#         delta = self.sub_model(x_2)
#         x_1 = x_1 - delta

#         x = self.orthogonal_map.inverse(torch.cat((x_1,x_2),dim=-1))
#         return x


class AffineCouplingLayer(nn.Module):
    def __init__(self, model_add, model_mul, *, mul_alpha=1 / 10, orientation="normal"):
        super().__init__()

        self.sub_model_add = model_add
        self.sub_model_mul = model_mul
        self.mul_alpha = mul_alpha

        if orientation == "normal":
            self.shuffle = no_shuffle
        elif orientation == "skew":
            self.shuffle = shuffle

    def forward(self, x):
        x_1, x_2 = x.chunk(2, dim=-1)
        x_1, x_2 = self.shuffle(x_1, x_2)

        delta = self.sub_model_add(x_2)
        mul = torch.exp(self.sub_model_mul(x_2) * self.mul_alpha)

        x_1 = x_1 * mul + delta
        x_1, x_2 = self.shuffle(x_1, x_2)
        return torch.cat((x_1, x_2), dim=-1)

    def inverse(self, x):
        x_1, x_2 = x.chunk(2, dim=-1)
        x_1, x_2 = self.shuffle(x_1, x_2)

        delta = self.sub_model_add(x_2)
        mul = torch.exp(self.sub_model_mul(x_2) * -self.mul_alpha)
        x_1 = (x_1 - delta) * mul

        x_1, x_2 = self.shuffle(x_1, x_2)
        return torch.cat((x_1, x_2), dim=-1)


# -----------------------------------------------------------------------------


### Padders
class ZeroPadder(nn.Module):
    def __init__(self, n_zeroes):
        super().__init__()
        self.n_zeroes = n_zeroes

    def forward(self, x):
        dim = x.shape[-1]
        return F.pad(x, (0, self.n_zeroes), "constant", 0.0)

    def inverse(self, x):
        if not self.training:
            return x[..., : -self.n_zeroes]
        else:
            return x


class ZeroDePadder(nn.Module):
    def __init__(self, n_zeroes):
        super().__init__()
        self.n_zeroes = n_zeroes

    def forward(self, x):
        if not self.training:
            return x[..., : -self.n_zeroes]
        else:
            return x

    def inverse(self, x):
        if not self.training:
            return F.pad(x, (0, self.n_zeroes), "constant", 0.0)
        else:
            return x


class CopyPadder(nn.Module):
    def __init__(self, n_copies=2):
        super().__init__()
        self.n_copies = n_copies

    def forward(self, x):
        return x.repeat(1, self.n_copies)

    def inverse(self, x):
        if not self.training:
            return x.reshape(x.shape[0], self.n_copies, -1).mean(dim=1)
        else:
            return x


# -----------------------------------------------------------------------------

### Flows


class Flow(nn.Module):
    """Class for the Flows Psi"""

    def __init__(self):
        super().__init__()

    def forward(self, t, x):
        raise NotImplementedError

    def vector_field(self, x):
        raise NotImplementedError

    def vector_field_jacobian(self, x):
        raise NotImplementedError


class RectilinearFlow(Flow):
    """Rectilinear Flow (pure translation). Very weak.

    Parameters
    ----------
    dim: int
        Dimension of the input.
    """

    def __init__(self, dim, *, b=None):
        super().__init__()
        if b is None:
            b = torch.randn(1, dim)
            b = nn.init.xavier_normal_(b)
        else:
            assert b.shape[0] == dim, "b must be a vector of length dim"
            b = b

        self.b = nn.Parameter(b)

    def forward(self, t, x):
        t = t.view(-1, 1)
        return x + t * self.b

    def vector_field(self, x):
        return self.b.expand(*x.shape)

    def vector_field_jacobian(self, x):
        return torch.zeros(
            x.shape[0], self.b.shape[1], self.b.shape[1], device=self.b.device
        )


def general_linear(x):
    return x


def symmetric(x):
    return (x + x.T) / 2


def skew_symmetric(x):
    return (x - x.T) / 2


# Dictionary of possible matrix constraints
LIE_DICT = {
    "general_linear": general_linear,
    "symmetric": symmetric,
    "skew_symmetric": skew_symmetric,
}


class LinearFlow(Flow):
    """
    Linear Flow. Applies e^(At) to x.

    Parameters
    ----------
    - dim (int): The dimension of the linear transformation matrix A.
    - lie_group (str): The type of Lie group to use for A. Default is 'general_linear'.
    """

    def __init__(self, dim, *, A=None, lie_group="general_linear"):
        super().__init__()

        self.lie_group = LIE_DICT[lie_group]

        if A is None:
            A = torch.randn(dim, dim)
            A = nn.init.xavier_normal_(A)
            A = (A - A.T) / 2
        else:
            assert (
                A.shape[0] == dim and A.shape[1] == dim
            ), "A must be a square matrix of size dim"
            A = A

        self.A = nn.Parameter(A)

    def forward(self, t, x):
        A = self.lie_group(self.A)
        A = A.unsqueeze(0)
        t = t.view(-1, 1, 1)
        x = x.unsqueeze(2)

        At = A * t
        ex = torch.matrix_exp(At)

        return torch.bmm(ex, x).squeeze()

    def vector_field(self, x):
        A = self.lie_group(self.A)
        return torch.matmul(A.unsqueeze(0), x.unsqueeze(2)).squeeze(2)

    def vector_field_jacobian(self, x):
        A = self.lie_group(self.A)
        return A.unsqueeze(0).expand(x.shape[0], -1, -1)


class AffineFlow(Flow):
    """Linear Flow with translation.
    Parameters:
        dim (int): The dimension of the affine transformation matrix.
        lie_group (str, optional): The type of Lie group to use for the affine transformation.
            Defaults to 'general_linear'.
    """

    def __init__(self, dim, *, A=None, b=None, lie_group="general_linear"):
        super().__init__()

        self.lie_group = LIE_DICT[lie_group]

        if A is None:
            A = torch.randn(dim, dim)
            A = nn.init.xavier_normal_(A)
            A = A / (10 * dim)
        else:
            assert (
                A.shape[0] == dim and A.shape[1] == dim
            ), "A must be a square matrix of size dim"
            A = A
        if b is None:
            b = torch.randn(dim, 1) / 10
        else:
            assert b.shape[0] == dim, "b must be a vector of length dim"
            b = b

        self.A = nn.Parameter(A)
        self.b = nn.Parameter(b)
        self.register_buffer("zeros", torch.zeros(1, dim + 1))

    def forward(self, t, x, mode="batched"):
        A = self.lie_group(self.A)
        b = self.b
        A = torch.cat((A, b), dim=1)
        A = torch.cat((A, self.zeros), dim=0)
        A = A.unsqueeze(0)

        t = t.view(-1, 1, 1)

        x = nn.functional.pad(x, (0, 1), "constant", 1.0)
        x = x.unsqueeze(2)

        At = A * t
        ex = torch.matrix_exp(At)

        if mode == "batched":
            x = torch.bmm(ex, x).squeeze()
        elif mode == "single":
            x = torch.matmul(ex, x).squeeze()
        return x[..., :-1]
        return x[..., :-1]

    def vector_field(self, x):
        A = self.lie_group(self.A)
        return (
            torch.matmul(A.unsqueeze(0), x.unsqueeze(2)) + self.b.unsqueeze(0)
        ).squeeze(2)

    def vector_field_jacobian(self, x):
        A = self.lie_group(self.A)
        return A.unsqueeze(0).expand(x.shape[0], -1, -1)


class BlockAffineFlow(Flow):
    """Linear Flow with translation.
    Parameters:
        dim (int): The dimension of the affine transformation matrix.
        lie_group (str, optional): The type of Lie group to use for the affine transformation.
            Defaults to 'general_linear'.
    """

    def __init__(self, dim, *, A=None, b=None, lie_group="general_linear"):
        super().__init__()

        self.lie_group = LIE_DICT[lie_group]

        if A is None:
            A = torch.randn(dim, dim)
            A = nn.init.xavier_normal_(A)
            A = A / (10 * dim)
        else:
            assert (
                A.shape[0] == dim and A.shape[1] == dim
            ), "A must be a square matrix of size dim"
            A = A
        if b is None:
            b = torch.randn(dim, 1) / 10
        else:
            assert b.shape[0] == dim, "b must be a vector of length dim"
            b = b

        self.A = nn.Parameter(A)
        self.register_buffer("A_0", torch.zeros_like(A))
        self.b = nn.Parameter(b)
        self.register_buffer("b_0", torch.zeros_like(b))
        self.register_buffer("zeros", torch.zeros(1, 2 * dim + 1))

    def forward(self, t, x, mode="batched"):
        A = self.lie_group(self.A)
        A = torch.block_diag(A, self.A_0)
        b = self.b
        b = torch.cat((b, self.b_0), dim=0)
        A = torch.cat((A, b), dim=1)
        A = torch.cat((A, self.zeros), dim=0)
        A = A.unsqueeze(0)

        t = t.view(-1, 1, 1)

        x = nn.functional.pad(x, (0, 1), "constant", 1.0)
        x = x.unsqueeze(2)

        At = A * t
        ex = torch.matrix_exp(At)

        if mode == "batched":
            x = torch.bmm(ex, x).squeeze()
        elif mode == "single":
            x = torch.matmul(ex, x).squeeze()
        return x[..., :-1]

    def vector_field(self, x):
        A = self.lie_group(self.A)
        return (
            torch.matmul(A.unsqueeze(0), x.unsqueeze(2)) + self.b.unsqueeze(0)
        ).squeeze(2)

    def vector_field_jacobian(self, x):
        A = self.lie_group(self.A)
        return A.unsqueeze(0).expand(x.shape[0], -1, -1)


class TimeAffineFlow(Flow):
    """Linear Flow with translation.
    Parameters:
        dim (int): The dimension of the affine transformation matrix.
        lie_group (str, optional): The type of Lie group to use for the affine transformation.
            Defaults to 'general_linear'.
    """

    def __init__(self, dim, *, A=None, b=None, c=None, lie_group="general_linear"):
        super().__init__()

        self.lie_group = LIE_DICT[lie_group]

        if A is None:
            A = torch.randn(dim, dim)
            A = nn.init.xavier_normal_(A)
            A = A / (10 * dim)
        else:
            assert (
                A.shape[0] == dim and A.shape[1] == dim
            ), "A must be a square matrix of size dim"
            A = A
        if b is None:
            b = torch.randn(dim, 1) / 10
        else:
            assert b.shape[0] == dim, "b must be a vector of length dim"
            b = b
        if c is None:
            c = torch.randn(dim, 1) / 10
        else:
            assert c.shape[0] == dim, "c must be a vector of length dim"
            c = c

        self.A = nn.Parameter(A)
        self.b = nn.Parameter(b)
        self.c = nn.Parameter(c)
        self.register_buffer("zeros", torch.zeros(1, dim + 2))
        one = torch.zeros(1, dim + 2)
        one[0, -1] = 1.0
        self.register_buffer("one", one)

    def forward(self, t, x):
        A = self.lie_group(self.A)
        b = self.b
        c = self.c
        A = torch.cat((c, A, b), dim=1)
        A = torch.cat((A, self.zeros), dim=0)
        A = torch.cat((self.one, A), dim=0)
        A = A.unsqueeze(0)

        t = t.view(-1, 1, 1)

        x = nn.functional.pad(x, (0, 1), "constant", 1.0)
        x = x.unsqueeze(2)

        At = A * t
        ex = torch.matrix_exp(At)

        x = torch.bmm(ex, x).squeeze()
        return x[..., :-1]

    def vector_field(self, x):
        A = self.lie_group(self.A)
        return (
            torch.matmul(A.unsqueeze(0), x.unsqueeze(2)) + self.b.unsqueeze(0)
        ).squeeze(2)

    def vector_field_jacobian(self, x):
        A = self.lie_group(self.A)
        return A.unsqueeze(0).expand(x.shape[0], -1, -1)


# -----------------------------------------------------------------------------


MLP_ACTIVATIONS = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


### Neural Conjugate Flow
class ApproximateNeuralConjugate(nn.Module):
    """Constructs a conjugation from two NON invertible nets.

    Parameters
    ---------
    dim: int
        Input dimensions.
    layers: collection of layers
        List (or tuple) of layers.
    psi: str, optional
        Type of flow to use. Must be keys of `PSI_DICT`. Uses `matrix_exp` by
        default.
    """

    def __init__(self, H, H_inv, psi):
        super(NeuralConjugate, self).__init__()
        self.H = H
        self.H_inv = H_inv
        self.psi = psi

    def forward(self, tx, preserve_grad=False):
        if not preserve_grad:
            tx = tx.clone().detach().requires_grad_(True)

        t = tx[:, -1]
        x = tx[:, :-1]

        x = self.H(x)
        x = self.psi(t, x)
        x = self.H_inv(x)

        return {"model_in": tx, "model_out": x}

    def vector_field(self, x):
        Hx = self.H(x)
        FHx = self.psi.vector_field(Hx)
        with fwAD.dual_level():
            dual_input = fwAD.make_dual(Hx, FHx)
            dual_output = self.H_inv(dual_input)
            v = fwAD.unpack_dual(dual_output).tangent

        return v


class NeuralConjugate(nn.Module):
    """Constructs a conjugation from invertible networks.

    Parameters
    ---------
    layers: collection of layers
        List (or tuple) of layers.
    psi: str, optional
        Type of flow to use. Must be keys of `PSI_DICT`. Uses `matrix_exp` by
        default.
    """

    def __init__(self, layers, psi):
        super(NeuralConjugate, self).__init__()
        self.layers = nn.ModuleList(layers)

        # Choose the conjugate flow
        self.psi = psi

    def H(self, x):
        for k in self.layers:
            x = k(x)
        return x

    def H_inv(self, x):
        for k in reversed(self.layers):
            x = k.inverse(x)
        return x

    def forward(self, tx, preserve_grad=False):
        if not preserve_grad:
            tx = tx.clone().detach().requires_grad_(True)
        t = tx[:, -1]
        x = tx[:, :-1]

        x = self.H(x)
        x = self.psi(t, x)
        x = self.H_inv(x)

        return {"model_in": tx, "model_out": x}

    def vector_field(self, x):
        Hx = self.H(x)
        FHx = self.psi.vector_field(Hx)
        with fwAD.dual_level():
            dual_input = fwAD.make_dual(Hx, FHx)
            dual_output = self.H_inv(dual_input)
            v = fwAD.unpack_dual(dual_output).tangent

        return v

    @staticmethod
    def load_from_config(network_config):
        """
        Load a Neural Conjugate model from a given configuration.
        Parameters:
        network_config (dict): A dictionary containing the configuration for the network.
            Expected keys are:
            - "type" (str): The type of network. Must be "neural_conjugate".
            - "dim_repetitions" (int): The number of repetitions for the dimensions.
            - "zero_pads" (int): The number of zero paddings.
            - "n_coupling_layers" (int): The number of coupling layers.
            - "in_channels" (int): The number of input channels.
            - "hidden_layers" (int): The number of hidden layers for the SIREN model.

        Returns:
        NeuralConjugate: An instance of the NeuralConjugate model configured as per the provided configuration.
        Raises:
        ValueError: If the "type" in network_config is not "neural_conjugate".
        """

        if network_config.get("type", None) != "neural_conjugate":
            raise ValueError(
                "Only neural conjugate flows configs are supported for this"
                " model. Are you sure the provided config corresponds to a"
                " neural conjugate flow?"
            )

        repetitions = network_config["dim_repetitions"]
        zero_pads = network_config["zero_pads"]
        n_coupling_layers = network_config["n_coupling_layers"]
        in_channels = network_config["in_channels"]
        omega = network_config["omega_0"]
        internal_network_type = network_config.get("internal_network_type", "siren")

        dims = repetitions * in_channels + zero_pads
        models = []
        if internal_network_type == "siren":
            models = [
                SIREN(dims // 2, dims // 2, network_config["hidden_layers"], w0=omega)
                    for _ in range(n_coupling_layers * 2)]
        elif internal_network_type.startswith("mlp"):
            activation = internal_network_type.split("-")[-1]
            if activation.lower() not in MLP_ACTIVATIONS:
                raise ValueError(
                    "Unknown activation function for internal module: "
                    f" {activation.lower()} for NeuralODE. Known activations are: "
                    f" {list(MLP_ACTIVATIONS.keys())}"
                )

            models = [
                MLPv2(
                    dims // 2,
                    dims // 2,
                    activation=MLP_ACTIVATIONS[activation.lower()],
                    hidden_layer_config=network_config["hidden_layers"],
                ) for _ in range(n_coupling_layers * 2)]
        else:
            raise ValueError(
                f'Unknown internal module option: "{internal_network_type}"'
                f" Known activations are: {list(MLP_ACTIVATIONS.keys())}"
            )

        layers = [CopyPadder(repetitions)]

        for i in range(n_coupling_layers):
            layers.append(
                DoubleAdditiveCouplingLayer((models[2 * i], models[2 * i + 1]))
            )

        psi = AffineFlow(dim=dims, lie_group="general_linear")
        model = NeuralConjugate(layers=layers, psi=psi)
        return model

    def load_weights_from_model_path(self, model_path, device):
        """Loads weights from a file.

        # NOT IMPLEMENTED YET
        # TODO: Implement this method

        Parameters
        ----------
        model_path: str, PathLike

        device: str, torch.device

        Returns
        -------
        self: NeuralConjugate

        Raises
        ------
        FileNotFoundError if the file pointed by `model_path` is missing.
        """
        raise NotImplementedError()


if __name__ == "__main__":
    config = {
        "type": "neural_conjugate",
        "dim_repetitions": 2,
        "zero_pads": 0,
        "n_coupling_layers": 3,
        "in_channels": 2,
        "omega_0": 54,
        "internal_network_type": "siren",
        "hidden_layers": [64],
    }
    model = NeuralConjugate.load_from_config(config)
    print("Activation: sin", model)

    for opt, actopt in MLP_ACTIVATIONS.items():
        config["internal_network_type"] = f"mlp-{opt}"
        model = NeuralConjugate.load_from_config(config)
        print(f"Activation: {opt}", model)
