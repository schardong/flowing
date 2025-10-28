# coding: utf-8

import torch
import torch.nn as nn
import torchdiffeq
from ifmorph.fd_models import convert_to_fd
from ifmorph.model import SIREN, MLPv2


MLP_ACTIVATIONS = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class AutonomousModule(nn.Module):
    def __init__(self, module):
        super(AutonomousModule, self).__init__()
        self.module = module

    def forward(self, t, x):
        # Here we can use the time 't' in some way if needed
        return self.module(x)  # Example: add time to the output


class NeuralODE(nn.Module):
    def __init__(
        self, func: torch.nn.Module, method: str = "rk4", autonomous: bool = True
    ):
        super(NeuralODE, self).__init__()
        assert (
            autonomous
        ), "Neural ODEs are currently only implemented for autonomous systems."

        self.internal_module = func
        if autonomous:
            self.func = AutonomousModule(func)

        self.method = method

    def forward(self, t, x0):
        # Use torchdiffeq to solve the ODE
        out = torchdiffeq.odeint(self.func, x0, t, method=self.method)
        return out  # Return the final state

    @staticmethod
    def load_from_config(network_config):
        """
        Load a NeuralODE model from a given configuration.

        Parameters:
        network_config (dict): A dictionary containing the configuration for the network.
            Expected keys are:
            - "type" (str): The type of network. Must be "neural_ode".
            - "dim" (int): The input/output dimension of the function.
            - "width" (int): The width of the hidden layers.
            - "hidden_layers" (list or int): The number of hidden layers for the function.
            - "w0" (float): The frequency parameter for the SIREN model.
            - "method" (str): The ODE solver method (e.g., 'rk4').
            - "autonomous" (bool): Whether the system is autonomous.

        Returns:
        NeuralODE: An instance of the NeuralODE model configured as per the provided configuration.

        Raises:
        ValueError: If the "type" in network_config is not "neural_ode".
        """
        if network_config.get("type", None) != "neural_ode":
            raise ValueError(
                "Only neural ODE configs are supported for this model. Are"
                " you sure the provided config corresponds to a neural ODE?"
            )

        dim = network_config.get("dim", 2)
        width = network_config.get("width", 128)
        hidden_layers = network_config.get("hidden_layers", 2)
        w0 = network_config.get("w0", 44)
        method = network_config.get("method", "rk4")
        autonomous = network_config.get("autonomous", True)
        internal_network_type = network_config.get("internal_network_type", "siren")

        if internal_network_type == "siren":
            func = SIREN(dim, dim, [width] * hidden_layers, w0=w0, output_dict=False)
        elif internal_network_type.startswith("mlp-"):
            activation = internal_network_type.split("-")[1]
            if activation not in MLP_ACTIVATIONS:
                raise ValueError(
                    "Unknown activation function for internal module: "
                    f" {activation} for NeuralODE. Known activations are: "
                    f" {list(MLP_ACTIVATIONS.keys())}"
                )

            func = MLPv2(
                n_in_features=dim,
                n_out_features=dim,
                activation=MLP_ACTIVATIONS[activation],
                hidden_layer_config=[width] * hidden_layers,
            )
        else:
            raise ValueError(
                f"Unknown internal module: {internal_network_type} for NeuralODE"
            )
        func = convert_to_fd(func)
        return NeuralODE(func, method=method, autonomous=autonomous)

    def load_weights_from_model_path(self, model_path, device):
        """
        Load weights from a given model path.
        Parameters:
        model_path (str): The path to the model file containing the weights.
        device (torch.device): The device to load the model on.
        """
        weights = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(weights, NeuralODE):
            self.load_state_dict(weights.state_dict())
        elif isinstance(weights, dict):
            self.load_state_dict(weights)
        else:
            raise ValueError(
                "Can't load the model weights. The model weights must be"
                " either a NeuralConjugate or an OrderedDict."
            )

        # TODO: Check that by deleting the weights, we don't delete the model's
        # state_dict as well, since everything is a reference.
        del weights
