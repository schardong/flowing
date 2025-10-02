#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import linear
from ifmorph.model import SIREN, MLPv2, SineLayer
from ifmorph.neural_conjugate_flows import AdditiveCouplingLayer, DoubleAdditiveCouplingLayer, TripleCouplingLayer, AffineCouplingLayer, \
                                           CopyPadder, ZeroPadder, RectilinearFlow, LinearFlow, AffineFlow, NeuralConjugate

from warnings import warn


# Base FDModule Class
class FDModule:
    def __init__(self, *args, **kwargs):
        if not issubclass(self.__class__, nn.Module):
            warn("FDModule should be inherited together with a subclass of nn.Module")
        super().__init__(*args, **kwargs)
        self.apply(convert_to_fd)
        self.finish_conversion()

    def forward(self, x, dx=None, d2x=None):
        x, dx, d2x = self.fd_forward(x, dx, d2x)
        if dx is None:
            return x
        elif d2x is None:
            return x, dx
        return x, dx, d2x

    def fd_forward(self, x, dx=None, d2x=None):
        raise NotImplementedError

    @classmethod
    def convert_to_fd(cls, module):
        module.__class__ = cls
        module.apply(convert_to_fd)
        module.finish_conversion()
        return module

    def finish_conversion(self):
        pass

    def dx(self, x):
        batch, dim = x.shape
        dx = torch.eye(dim, device=x.device).unsqueeze(1).expand(dim, batch, dim)
        return dx

    def d2x(self, x):
        batch, dim = x.shape
        d2x = torch.zeros(dim, dim, batch, dim, device=x.device)
        return d2x

    def gradient(self, x):
        grad = self.fd_forward(x, self.dx(x))[1]
        return grad.permute(1, 2, 0)

    def forward_and_gradient(self, x):
        x, dx = self.fd_forward(x, self.dx(x))[:2]
        return x, dx.permute(1, 2, 0)

    def hessian(self, x):
        hess = self.fd_forward(x, self.dx(x), self.d2x(x))[2]
        return hess.permute(2, 3, 0, 1)

    def dxdxT(self, dx, d2x=None):
        if d2x is None or d2x.dim() == 4:
            return dx.unsqueeze(0) * dx.unsqueeze(1)
        else:
            return dx[:-1, :, :] * dx[-1:, :, :]


# Container Modules
class FDSequential(FDModule, nn.Sequential):
    def fd_forward(self, x, dx=None, d2x=None):
        for module in self:
            x, dx, d2x = module.fd_forward(x, dx, d2x)
        return x, dx, d2x


# Activation Modules
class FDActivation(FDModule, nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act

    def fd_forward(self, x, dx=None, d2x=None):
        fx = self.act(x)
        if dx is None:
            return fx, None, None
        fdx = torch.autograd.grad(fx, x, torch.ones_like(x), create_graph=True)[0]
        if d2x is None:
            return fx, fdx * dx, None
        fd2x = torch.autograd.grad(fdx, x, torch.ones_like(x), create_graph=True)[0]
        return fx, fdx * dx, fdx * d2x + fd2x * self.dxdxT(dx, d2x)

    def __repr__(self):
        return f"FDActivation({self.act.__name__})"


def convert_activation(act):
    if isinstance(act, FDModule):
        pass
    elif isinstance(act, nn.Module):
        try:
            act = convert_to_fd(act)
        except NotImplementedError:
            act = FDActivation(act)
    else:
        act = FDActivation(act)
    return act


class FDTanh(FDModule, nn.Tanh):
    def fd_forward(self, x, dx=None, d2x=None):
        fx = torch.tanh(x)
        if dx is None:
            return fx, None, None
        fdx = 1 - torch.square(fx)
        if d2x is None:
            return fx, fdx * dx, None
        fd2x = -2 * fx * fdx
        return fx, fdx * dx, fdx * d2x + fd2x * self.dxdxT(dx, d2x)


class FDSoftplus(FDModule, nn.Softplus):
    def fd_forward(self, x, dx=None, d2x=None):
        fx = torch.nn.functional.softplus(x, self.beta, self.threshold)
        if dx is None:
            return fx, None, None
        fdx = torch.exp(self.beta * (x - fx))
        if d2x is None:
            return fx, fdx * dx, None
        fd2x = self.beta * (fdx * torch.exp(-self.beta * fx))
        return fx, fdx * dx, fdx * d2x + fd2x * self.dxdxT(dx, d2x)


class FDReLU(FDModule, nn.ReLU):
    def fd_forward(self, x, dx=None, d2x=None):
        fx = torch.relu(x)
        if dx is None:
            return fx, None, None
        fdxdx = torch.where(x > 0, dx, torch.zeros_like(dx))
        if d2x is None:
            return fx, fdxdx, None
        fdxd2x = torch.where(x > 0, d2x, torch.zeros_like(d2x))
        return fx, fdxdx, fdxd2x


class FDLeakyReLU(FDModule, nn.LeakyReLU):
    def fd_forward(self, x, dx=None, d2x=None):
        fx = torch.nn.functional.leaky_relu(x, self.negative_slope, self.inplace)
        if dx is None:
            return fx, None, None
        fdxdx = torch.where(x > 0, dx, self.negative_slope * dx)
        if d2x is None:
            return fx, fdxdx, None
        fdxd2x = torch.where(x > 0, d2x, self.negative_slope * d2x)
        return fx, fdxdx, fdxd2x


class FDGELU(FDModule, nn.GELU):
    def fd_forward(self, x, dx=None, d2x=None):
        fx = torch.nn.functional.gelu(x, approximate=self.approximate)
        if dx is None:
            return fx, None, None
        if self.approximate == "none":
            Phix = (1 + torch.erf(x / np.sqrt(2))) / 2
            sqx = torch.square(x)
            phix = torch.exp(-sqx / 2) / np.sqrt(2*np.pi)
            fdx = x * phix + Phix
            if d2x is None:
                return fx, fdx * dx, None
            fd2x = (2 - sqx) * phix
            return fx, fdx * dx, fdx * d2x + fd2x * self.dxdxT(dx, d2x)
        else:
            raise NotImplementedError("Do you really need GELU approximation? It is not implemented yet")


class FDSELU(FDModule, nn.SELU):
    def fd_forward(self, x, dx=None, d2x=None):
        fx = torch.nn.functional.selu(x, self.inplace)
        if dx is None:
            return fx, None, None
        negx = 1.758099340847376859940217520812 * torch.exp(x)
        fdxdx = torch.where(x > 0, 1.0507009873554804934193349852946 * dx, negx * dx)
        if d2x is None:
            return fx, fdxdx, None
        fdxd2x = torch.where(x > 0, 1.0507009873554804934193349852946 * d2x, negx * (d2x + self.dxdxT(dx, d2x)))
        return fx, fdxdx, fdxd2x


class FDSineLayer(FDModule, SineLayer):
    def fd_forward(self, x, dx=None, d2x=None):
        w0x = self.w0 * x
        fx = torch.sin(w0x)
        if dx is None:
            return fx, None, None
        fdx = self.w0 * torch.cos(w0x)
        df = fdx * dx
        if d2x is None:
            return fx, df, None
        fd2x = (-self.w0 * self.w0) * fx
        d2f = fdx * d2x + fd2x * self.dxdxT(dx, d2x)
        return fx, df, d2f

    def __repr__(self):
        return f"FDSineLayer(w0={self.w0})"


class FDSigmoid(FDModule, nn.Sigmoid):
    def fd_forward(self, x, dx=None, d2x=None):
        fx = torch.sigmoid(x)
        if dx is None:
            return fx, None, None
        fdx = fx * (1 - fx)
        if d2x is None:
            return fx, fdx * dx, None
        fd2x = fdx * (1 - 2 * fx)
        return fx, fdx * dx, fdx * d2x + fd2x * self.dxdxT(dx, d2x)


# Linear Module
class FDLinear(FDModule, nn.Linear):
    def fd_forward(self, x, dx=None, d2x=None):
        lx = linear(x, self.weight, self.bias)
        if dx is None:
            return lx, None, None
        ldx = linear(dx, self.weight)
        if d2x is None:
            return lx, ldx, None
        ld2x = linear(d2x, self.weight)
        return lx, ldx, ld2x


# Model Modules
class FDMLP(FDModule, MLPv2):
    def fd_forward(self, x, dx=None, d2x=None):
        return self.net.fd_forward(x, dx, d2x)


class FDSiren(FDModule, SIREN):
    def fd_forward(self, x, dx=None, d2x=None):
        return self.net.fd_forward(x, dx, d2x)


# Coupling Layer Modules
def fd_chunk(x, dx=None, d2x=None):
    all_y, all_z = [], []
    for tensor in (x, dx, d2x):
        if tensor is None:
            all_y.append(None)
            all_z.append(None)
        else:
            y_i, z_i = tensor.chunk(2, dim=-1)
            all_y.append(y_i)
            all_z.append(z_i)
    return all_y, all_z


def fd_cat(all_y, all_z):
    out_tensors = []
    for i in range(3):
        if all_y[i] is None:
            out_tensors.append(None)
        else:
            out_tensors.append(torch.cat((all_y[i], all_z[i]), dim=-1))
    return out_tensors


class FDAdditiveCL(FDModule, AdditiveCouplingLayer):
    def fd_forward(self, x, dx=None, d2x=None):
        all_y, all_z = fd_chunk(x, dx, d2x)
        all_y, all_z = self.shuffle(all_y, all_z)

        all_delta = self.sub_model.fd_forward(*all_z)
        for i in range(3):
            if all_delta[i] is not None:
                all_y[i] = all_y[i] + self.initial_step_size * all_delta[i]

        all_y, all_z = self.shuffle(all_y, all_z)
        return fd_cat(all_y, all_z)

    def fd_inverse(self, x, dx=None, d2x=None):
        all_y, all_z = fd_chunk(x, dx, d2x)
        all_y, all_z = self.shuffle(all_y, all_z)

        all_delta = self.sub_model.fd_forward(*all_z)
        for i in range(3):
            if all_delta[i] is not None:
                all_y[i] = all_y[i] - self.initial_step_size * all_delta[i]

        all_y, all_z = self.shuffle(all_y, all_z)
        return fd_cat(all_y, all_z)


class FDDoubleAdditiveCL(FDModule, DoubleAdditiveCouplingLayer):
    def fd_forward(self, x, dx=None, d2x=None):
        all_y, all_z = fd_chunk(x, dx, d2x)

        all_delta0 = self.models[0].fd_forward(*all_z)
        for i in range(3):
            if all_delta0[i] is not None:
                all_y[i] = all_y[i] + self.gamma * all_delta0[i]

        all_delta1 = self.models[1].fd_forward(*all_y)
        for i in range(3):
            if all_delta1[i] is not None:
                all_z[i] = all_z[i] + self.gamma * all_delta1[i]

        return fd_cat(all_y, all_z)

    def fd_inverse(self, x, dx=None, d2x=None):
        all_y, all_z = fd_chunk(x, dx, d2x)

        all_delta1 = self.models[1].fd_forward(*all_y)
        for i in range(3):
            if all_delta1[i] is not None:
                all_z[i] = all_z[i] - self.gamma * all_delta1[i]

        all_delta0 = self.models[0].fd_forward(*all_z)
        for i in range(3):
            if all_delta0[i] is not None:
                all_y[i] = all_y[i] - self.gamma * all_delta0[i]

        return fd_cat(all_y, all_z)


class FDTripleCL(FDModule, TripleCouplingLayer):
    def fd_forward(self, x, dx=None, d2x=None):
        all_y, all_z = fd_chunk(x, dx, d2x)

        all_delta0 = self.models[0].fd_forward(*all_y)
        for i in range(3):
            if all_delta0[i] is not None:
                all_z[i] = all_z[i] + self.gamma * all_delta0[i]

        all_delta1 = self.models[1].fd_forward(*all_z)
        for i in range(3):
            if all_delta1[i] is not None:
                all_y[i] = all_y[i] + self.gamma * all_delta1[i]

        all_delta2 = self.models[2].fd_forward(*all_y)
        for i in range(3):
            if all_delta2[i] is not None:
                all_z[i] = all_z[i] + self.gamma * all_delta2[i]

        return fd_cat(all_y, all_z)

    def fd_inverse(self, x, dx=None, d2x=None):
        all_y, all_z = fd_chunk(x, dx, d2x)

        all_delta2 = self.models[2].fd_forward(*all_y)
        for i in range(3):
            if all_delta2[i] is not None:
                all_z[i] = all_z[i] - self.gamma * all_delta2[i]

        all_delta1 = self.models[1].fd_forward(*all_z)
        for i in range(3):
            if all_delta1[i] is not None:
                all_y[i] = all_y[i] - self.gamma * all_delta1[i]

        all_delta0 = self.models[0].fd_forward(*all_z)
        for i in range(3):
            if all_delta0[i] is not None:
                all_z[i] = all_z[i] - self.gamma * all_delta0[i]

        return fd_cat(all_y, all_z)


class FDAffineCL(FDModule, AffineCouplingLayer):
    def fd_forward(self, x, dx=None, d2x=None):
        all_y, all_z = fd_chunk(x, dx, d2x)
        all_y, all_z = self.shuffle(all_y, all_z)

        all_delta = self.sub_model_add.fd_forward(*all_z)
        all_mul = self.fd_mul(*all_z)
        # y = y * mul + delta
        all_y_cl = [all_y[0] * all_mul[0] + all_delta[0], None, None]
        if all_y[1] is not None:
            # dy = dy * mul + y dmul + ddelta
            all_y_cl[1] = all_y[1] * all_mul[0] + all_y[0] * all_mul[1] + all_delta[1]
        if all_y[2] is not None:
            # d2y = d2y * mul + 2 *dy * dmul +  y * d2mul
            if all_y[2].dim() == 4:
                dydmul = all_y[1].unsqueeze(0)*all_mul[1].unsqueeze(1)
                dydmul = dydmul + dydmul.transpose(0, 1)
            else:
                dydmul = all_y[1][:-1, :, :] * all_mul[1][-1:, :, :] + all_y[1][-1:, :, :] * all_mul[1][:-1, :, :]
            all_y_cl[2] = all_y[2] * all_mul[0] + dydmul + all_y[0] * all_mul[2]

        all_y, all_z = self.shuffle(all_y_cl, all_z)
        return fd_cat(all_y, all_z)

    def fd_inverse(self, x, dx=None, d2x=None):
        all_y, all_z = fd_chunk(x, dx, d2x)
        all_y, all_z = self.shuffle(all_y, all_z)

        all_delta = self.sub_model_add.fd_forward(*all_z)
        all_mul = self.fd_mul(*all_z, alpha=-self.mul_alpha)
        for i in range(3):
            if all_y[i] is not None:
                all_y[i] = all_y[i] - all_delta[i]
        # y = y * mul
        all_y_cl = [all_y[0] * all_mul[0] , None, None]
        if all_y[1] is not None:
            # dy = dy * mul + y dmul
            all_y_cl[1] = all_y[1] * all_mul[0] + all_y[0] * all_mul[1]
        if all_y[2] is not None:
            # d2y = d2y * mul + 2 *dy * dmul +  y * d2mul
            if all_y[2].dim() == 4:
                dydmul = all_y[1].unsqueeze(0)*all_mul[1].unsqueeze(1)
                dydmul = dydmul + dydmul.transpose(0, 1)
            else:
                dydmul = all_y[1][:-1, :, :] * all_mul[1][-1:, :, :] + all_y[1][-1:, :, :] * all_mul[1][:-1, :, :]
            all_y_cl[2] = all_y[2] * all_mul[0] + dydmul + all_y[0] * all_mul[2]

        all_y, all_z = self.shuffle(all_y_cl, all_z)
        return fd_cat(all_y, all_z)

    def fd_mul(self, x, dx=None, d2x=None, alpha=None):
        if alpha is None:
            alpha = self.mul_alpha
        x, dx, d2x = self.sub_model_mul.fd_forward(x, dx, d2x)
        ex = torch.exp(x * alpha)
        if dx is None:
            return ex, None, None
        edx = alpha * ex
        if d2x is None:
            return ex, edx * dx, None
        ed2x = (alpha * alpha) * ex
        return ex, edx * dx, edx * d2x + ed2x * self.dxdxT(dx, d2x)


# Padders
class FDCopyPadder(FDModule, CopyPadder):
    def fd_forward(self, x, dx=None, d2x=None):
        packed_x = [x, dx, d2x]
        for i, tensor in enumerate(packed_x):
            if tensor is not None:
                packed_x[i] = tensor.repeat(*([1]*(i+1)), self.n_copies)
        return packed_x

    def fd_inverse(self, x, dx=None, d2x=None):
        packed_x = [x, dx, d2x]
        if not self.training:
            for i, tensor in enumerate(packed_x):
                if tensor is not None:
                    packed_x[i] = tensor.reshape(*tensor.shape[:-1], self.n_copies, -1).mean(dim=-2)
        return packed_x


class FDZeroPadder(FDModule, ZeroPadder):
    def fd_forward(self, x, dx=None, d2x=None):
        packed_x = [x, dx, d2x]
        for i, tensor in enumerate(packed_x):
            if tensor is not None:
                packed_x[i] = nn.functional.pad(tensor, (0, self.n_zeroes), 'constant', 0.0)
        return packed_x

    def fd_inverse(self, x, dx=None, d2x=None):
        packed_x = [x, dx, d2x]
        if not self.training:
            for i, tensor in enumerate(packed_x):
                if tensor is not None:
                    packed_x[i] = tensor[..., :-self.n_zeroes]
        return packed_x


# Flow Modules
class FDFLow(FDModule):
    def fd_forward(self, t, x, dx=None, d2x=None, add_time=False):
        raise NotImplementedError

    def forward(self, t, x, dx=None, d2x=None, add_time=False):
        if dx is None:
            add_time = False
        f, df, d2f = self.fd_forward(t, x, dx, d2x, add_time)
        if df is None:
            return f
        elif d2f is None:
            return f, df
        return f, df, d2f

    def hessian(self, tx, add_time=True):
        t = tx[:, -1]
        x = tx[:, :-1]

        batch, dim = x.shape
        dx = torch.eye(dim, device=x.device).unsqueeze(1).expand(dim, batch, dim)
        d2x = torch.zeros(dim, dim, batch, dim, device=x.device)
        hess = self.fd_forward(t, x, dx, d2x, add_time)[2]
        return hess.permute(2, 3, 0, 1)

    def gradient(self, tx, add_time=True):
        t = tx[:, -1]
        x = tx[:, :-1]

        batch, dim = x.shape
        dx = torch.eye(dim, device=x.device).unsqueeze(1).expand(dim, batch, dim)
        grad = self.fd_forward(t, x, dx, None, add_time)[1]
        return grad.permute(1, 2, 0)


class FDRectilinearFlow(FDFLow, RectilinearFlow):
    def fd_forward(self, t, x, dx=None, d2x=None, add_time=False):
        t = t.view(-1, 1)
        f, df, d2f = x + t*self.b, dx, d2x
        if dx is not None and add_time:
            b = self.b.reshape(1, 1, -1).expand(1, x.size(0), -1)
            df = torch.cat((df, b) , dim=0)
            if d2x is not None:
                d2f = nn.functional.pad(d2f, (0, 0, 0, 0, 0, 1, 0, 1), 'constant', 0.0)
        return f, df, d2f


class FDLinearFlow(FDFLow, LinearFlow):
    def fd_forward(self, t, x, dx=None, d2x=None, add_time=False):
        A = self.lie_group(self.A)
        A = A.unsqueeze(0)
        t = t.view(-1, 1, 1)
        x = x.unsqueeze(-1)

        At = A*t
        ex = torch.matrix_exp(At)
        f = torch.matmul(ex, x).squeeze(-1)
        if dx is None:
            return f, None, None
        fdx = torch.matmul(ex, dx.unsqueeze(-1)).squeeze(-1)
        if add_time:
            Af = torch.matmul(A, f.unsqueeze(-1)).squeeze(-1)
            df = torch.cat((fdx, Af.unsqueeze(0)), dim=0)
        else:
            df = fdx
        if d2x is None:
            return f, df, None
        fd2x = torch.matmul(ex, d2x.unsqueeze(-1)).squeeze(-1)
        if add_time:
            fdxdt = torch.matmul(A, fdx.unsqueeze(-1)).squeeze(-1)
            fdtdt = torch.matmul(A, Af.unsqueeze(-1)).squeeze(-1) # fd2t = Af
            d2f0 = torch.cat((fd2x, fdxdt.unsqueeze(0)), dim=0)
            d2f1 = torch.cat((fdxdt, fdtdt.unsqueeze(0)), dim=0)
            d2f = torch.cat((d2f0, d2f1.unsqueeze(1)), dim=1)
        else:
            d2f = fd2x
        #fdxdt = torch.matmul(A, fdx.unsqueeze(-1)).squeeze(-1)
        #fdxdt = fdxdt.unsqueeze(0) # * dt.unsqueeze(1)
        #fdxdt = fdxdt + fdxdt.transpose(0, 1)
        #dtdt = dt.unsqueeze(0) * dt.unsqueeze(1)
        #fd2t = d2t * Af + dtdt * torch.matmul(A, Af.unsqueeze(-1)).squeeze(-1)
        #fd2t = torch.matmul(A, Af.unsqueeze(-1)).squeeze(-1)
        #d2f = fd2x + fdxdt + fd2t
        return f, df, d2f


class FDAffineFlow(FDFLow, AffineFlow):
    def fd_forward(self, t, x, dx=None, d2x=None, add_time=False):
        A = self.lie_group(self.A)
        b = self.b
        A_ext = torch.cat((A,b),dim=1)
        A_ext = torch.cat((A_ext,self.zeros),dim=0).unsqueeze(0)

        t = t.view(-1, 1, 1)

        x = nn.functional.pad(x, (0,1), 'constant', 1.0)
        x = x.unsqueeze(2)

        A_extt = A_ext*t
        ex_ext = torch.matrix_exp(A_extt)

        f = torch.matmul(ex_ext, x).squeeze(2)
        f = f[:, :-1]
        if dx is None:
            return f, None, None
        ex = ex_ext[..., :-1, :-1]
        fdx = torch.matmul(ex, dx.unsqueeze(-1)).squeeze(-1)
        if add_time:
            Afb = (torch.matmul(A, f.unsqueeze(-1)) + b).squeeze(-1)
            df = torch.cat((fdx, Afb.unsqueeze(0)), dim=0)
        else:
            df = fdx
        if d2x is None:
            return f, df, None
        fd2x = torch.matmul(ex, d2x.unsqueeze(-1)).squeeze(-1)
        if add_time:
            fdxdt = torch.matmul(A, fdx.unsqueeze(-1)).squeeze(-1)
            fdtdt = torch.matmul(A, Afb.unsqueeze(-1)).squeeze(-1) # fd2t = Afb
            d2f0 = torch.cat((fd2x, fdxdt.unsqueeze(0)), dim=0)
            d2f1 = torch.cat((fdxdt, fdtdt.unsqueeze(0)), dim=0)
            d2f = torch.cat((d2f0, d2f1.unsqueeze(1)), dim=1)
        else:
            d2f = fd2x
        return f, df, d2f


class FDNCF(FDFLow, NeuralConjugate):
    def __init__(self, layers, psi):
        super(NeuralConjugate, self).__init__()
        self.layers = nn.ModuleList([convert_to_fd(layer) for layer in layers])

        # Choose the conjugate flow
        self.psi = convert_to_fd(psi)

    def H(self, x, dx=None, d2x=None):
        for layer in self.layers:
            x, dx, d2x = layer.fd_forward(x, dx, d2x)
        return x, dx, d2x

    def H_inv(self, x, dx=None, d2x=None):
        for layer in reversed(self.layers):
            x, dx, d2x = layer.fd_inverse(x, dx, d2x)
        return x, dx, d2x

    def forward(self, tx, preserve_grad=None):
        t = tx[:, -1]
        x = tx[:, :-1]

        x = self.fd_forward(t, x)[0]

        return {"model_in": tx, "model_out": x}

    def fd_forward(self, t, x, dx=None, d2x=None, add_time=False):
        x, dx, d2x = self.H(x, dx, d2x)
        x, dx, d2x = self.psi.fd_forward(t, x, dx, d2x, add_time)
        x, dx, d2x = self.H_inv(x, dx, d2x)
        return x, dx, d2x

    def vector_field(self, x):
        Hx = self.H(x)[0]
        FHx = self.psi.vector_field(Hx)
        return self.H_inv(Hx, FHx.unsqueeze(0))[1].squeeze(0)

    def vector_field_jacobian(self, x):
        Hx, JHx, _ = self.H(x, self.dx(x))
        FHx = self.psi.vector_field(Hx)
        JFHx = torch.matmul(self.psi.vector_field_jacobian(Hx), JHx.permute(1, 2, 0)).permute(2, 0, 1)
        return self.H_inv(Hx,
                          torch.cat((JHx, FHx.unsqueeze(0)), dim=0),
                          JFHx)[2].permute(1, 2, 0)

    def load_weights_from_model_path(self, model_path, device):
        """
        Load weights from a given model path.
        Parameters:
        model_path (str): The path to the model file containing the weights.
        device (torch.device): The device to load the model on.
        """
        weights = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(weights, FDNCF):
            self.load_state_dict(weights.state_dict())
        elif isinstance(weights, dict):
            self.load_state_dict(weights)
        else:
            raise ValueError("Can't load the model weights. The model weights must be either a NeuralConjugate or an OrderedDict.")

        del weights


fd_modules_dict = {
    # Containers
    nn.Sequential: FDSequential,
    nn.ModuleList: None,
    nn.ModuleDict: None,
    nn.ParameterList: None,
    nn.ParameterDict: None,
    # Activations
    nn.Tanh: FDTanh,
    nn.ReLU: FDReLU,
    nn.LeakyReLU: FDLeakyReLU,
    nn.GELU: FDGELU,
    nn.SELU: FDSELU,
    nn.Softplus: FDSoftplus,
    SineLayer: FDSineLayer,
    nn.Sigmoid: FDSigmoid,
    # Linear
    nn.Linear: FDLinear,
    # Models
    MLPv2: FDMLP,
    SIREN: FDSiren,
    # Coupling Layers
    AdditiveCouplingLayer: FDAdditiveCL,
    DoubleAdditiveCouplingLayer: FDDoubleAdditiveCL,
    TripleCouplingLayer: FDTripleCL,
    AffineCouplingLayer: FDAffineCL,
    # Padders
    CopyPadder: FDCopyPadder,
    ZeroPadder: FDZeroPadder,
    # Flows
    RectilinearFlow: FDRectilinearFlow,
    LinearFlow: FDLinearFlow,
    AffineFlow: FDAffineFlow,
    NeuralConjugate: FDNCF,
}


def convert_to_fd(module, copy=False):
    if copy:
        module = deepcopy(module)
    if isinstance(module, FDModule):
        return module
    if type(module) in fd_modules_dict:
        fd_class = fd_modules_dict[type(module)]
        if fd_class is not None:
            module = fd_class.convert_to_fd(module)
        return module
    else:
        raise NotImplementedError(f"Conversion of {type(module)} to forward difference module is not implemented yet")


if __name__ == "__main__":
    from ifmorph.diff_operators import hessian
    from time import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print(device)

    ###### Test SIREN #####
    print('Testing forward difference SIREN')
    siren = SIREN(n_in_features=3,
                  n_out_features=2,
                  hidden_layer_config=[32,32],
                  w0=2, ww=3)

    siren_fd = convert_to_fd(siren, copy=True)

    x = torch.randn(500, 3)

    dictxy = siren(x)
    x, y = dictxy['model_in'], dictxy['model_out']
    start = time()
    n_tries = 1000
    for i in range(n_tries):
        hess = hessian(y, x).detach()
    print('Hessian time:', (time()-start)/n_tries)
    y = y.detach()

    y_fd = siren_fd(x.detach())
    start = time()
    for i in range(n_tries):
        hess_fd = siren_fd.hessian(x.detach())
    print('FD Hessian time:', (time()-start)/n_tries)
    print('y_rel_error:', (torch.linalg.norm(y_fd-y) / torch.linalg.norm(y)).item())
    print('hessian_rel_error:', (torch.linalg.norm(hess_fd-hess) / torch.linalg.norm(hess)).item())

    ###### Test NCF #####
    print('Testing forward difference NCF')

    dims = 4

    model_1 = MLPv2(n_in_features=dims//2, n_out_features=dims//2, hidden_layer_config=[12], activation=nn.Tanh)
    # model_1 = nn.init.xavier_uniform(model_1)
    model_2 = MLPv2(n_in_features=dims//2, n_out_features=dims//2, hidden_layer_config=[12], activation=torch.sigmoid)
    # model_2 = nn.init.xavier_uniform(model_2)

    #H = DoubleAdditiveCouplingLayer((model_1, model_2))
    #H = AdditiveCouplingLayer(model_1)
    H = AffineCouplingLayer(model_1, model_2)

    psi = AffineFlow(dim=dims)# lie_group='general_linear')

    # Construct model
    model = NeuralConjugate(layers=[H], psi=psi)
    fd_model = convert_to_fd(model, copy=True)

    x = torch.randn(500, dims)
    tx = torch.cat((x, torch.zeros(500, 1)), dim=1)
    old_jvf = fd_model.hessian(tx.detach())[:, :, :-1, -1]
    jvf = fd_model.vector_field_jacobian(x)
    print('jacobian_vector_field_rel_error:', (torch.linalg.norm(jvf-old_jvf) / torch.linalg.norm(old_jvf)).item())

    tx = torch.randn(500, dims + 1)
    dictxy = model(tx)
    tx, y = dictxy['model_in'], dictxy['model_out']
    start = time()
    n_tries = 100
    for i in range(n_tries):
        hess = hessian(y, tx).detach()
    print('Hessian time:', (time()-start)/n_tries)
    y = y.detach()

    y_fd = fd_model(tx.detach())['model_out']
    start = time()
    for i in range(n_tries):
        hess_fd = fd_model.hessian(tx.detach())
    print('FD Hessian time:', (time()-start)/n_tries)

    print('y_rel_error:', (torch.linalg.norm(y_fd-y) / torch.linalg.norm(y)).item())
    print('hessian_rel_error:', (torch.linalg.norm(hess_fd-hess) / torch.linalg.norm(hess)).item())

    fd_model_ = FDNCF(layers = [H], psi = psi)

    print('end')
