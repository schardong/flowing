# coding: utf-8

import torch
from ifmorph.diff_operators import hessian, jacobian
from ifmorph.fd_models import FDModule
from ifmorph.neural_odes import NeuralODE


DEFAULT_WEIGHTS_FMLOSS = {
    "data_constraint":  1e4,
    "identity_constraint":  1e3,
    "inv_constraint": 1e4,
    "diagoanl_flow_constraint": 1e3,
    "TPS_constraint": 1e3,
}


class WarpingLoss(torch.nn.Module):
    """Warping loss with feature matching between source and target.

    Parameters
    ----------
    warp_src_pts: torch.Tensor
        An Nx2 tensor with the feature locations in the source image. Note that
        these points must be normalized to the [-1, 1] range.

    warp_tgt_pts: torch.Tensor
        An Nx2 tensor with the feature locations in the target image. Note that
        these points must be normalized to the [-1, 1] range.

    intermediate_times: list, optional
        List of intermediate times where the data constraint will be fit. All
        values must be in range [0, 1]. By default is [0.25, 0.5, 0.75]

    constraint_weights: dict, optional
        The weight of each constraint in the final loss composition. By
        default, the weights are:
        {
            "data_constraint":  1e4,
            "identity_constraint":  1e3,
            "inv_constraint": 1e4,
            "TPS_constraint": 1e3,
        }
    """
    def __init__(
            self,
            warp_src_pts: torch.Tensor,
            warp_tgt_pts: torch.Tensor,
            intermediate_times: list = [0.25, 0.5, 0.75],
            constraint_weights: dict = DEFAULT_WEIGHTS_FMLOSS
    ):
        super(WarpingLoss, self).__init__()
        self.src = warp_src_pts
        self.tgt = warp_tgt_pts
        self.intermediate_times = intermediate_times
        self.constraint_weights = constraint_weights
        if intermediate_times is None or not len(intermediate_times):
            self.intermediate_times = [0.25, 0.5, 0.75]
        if constraint_weights is None or not len(constraint_weights):
            self.constraint_weights = DEFAULT_WEIGHTS_FMLOSS

        # Ensuring that all necessary weights are stored.
        for k, v in DEFAULT_WEIGHTS_FMLOSS.items():
            if k not in self.constraint_weights:
                self.constraint_weights[k] = v

    def forward(self, coords, model):
        """
        coords: torch.Tensor(shape=[N, 3])
        model: torch.nn.Module
        """
        M = model(coords)
        X = M["model_in"]
        Y = M["model_out"].squeeze()

        # thin plate spline energy
        hessian1 = hessian(Y, X)
        TPS_constraint = hessian1 ** 2

        # data fitting: f(src, 1)=tgt, f(tgt,-1)=src
        src = torch.cat((self.src, torch.ones_like(self.src[..., :1])), dim=1)
        y_src = model(src)['model_out']
        tgt = torch.cat((self.tgt, -torch.ones_like(self.tgt[..., :1])), dim=1)
        y_tgt = model(tgt)['model_out']
        data_constraint = (self.tgt - y_src)**2 + (self.src - y_tgt)**2
        data_constraint *= 1e2
        
        alignment_loss = "not calculated"

        # forcing the feature matching along time
        
        for t in self.intermediate_times:
            tgt_0 = torch.cat((self.tgt, (t-1)*torch.ones_like(self.tgt[..., :1])), dim=1)
            y_tgt_0 = model(tgt_0)['model_out']
            src_0 = torch.cat((self.src, t*torch.ones_like(self.src[..., :1])), dim=1)
            y_src_0 = model(src_0)['model_out']
            data_constraint += ((y_src_0 - y_tgt_0)**2)*5e1

            src_t = torch.cat((y_src_0, -t*torch.ones_like(self.src[..., :1])), dim=1)
            y_src_t = model(src_t)['model_out']
            data_constraint += ((y_src_t - self.src)**2)*2e1

            tgt_t = torch.cat((y_tgt_0, 1-t*torch.ones_like(self.tgt[..., :1])), dim=1)
            y_tgt_t = model(tgt_t)['model_out']
            data_constraint += ((y_tgt_t - self.tgt)**2)*2e1
            
            
            if t == 0.5:
                alignment_loss = (y_src_0 - y_tgt_0)**2
                alignment_loss = alignment_loss.mean()

        # identity constraint: f(p,0) = (p)
        diff_constraint = (Y - X[..., :2])**2
        identity_constraint = torch.where(torch.cat((coords[..., -1:], coords[..., -1:]), dim=-1) == 0, diff_constraint, torch.zeros_like(diff_constraint))

        # inverse constraint: f(f(p,t), -t) = p,  f(f(p,-t), t) = p
        Ys = torch.cat((Y, -X[..., -1:]), dim=1)
        model_Xs = model(Ys)
        Xs = model_Xs['model_out']

        # inverse constraint: f(f(p,t), 1-t) = f(p,1)
        Yt = torch.cat((Y, 1 - X[..., -1:]), dim=1)
        model_Xt = model(Yt)
        Xt = model_Xt['model_out']
        Y1 = torch.cat((X[...,0:2], torch.ones_like(X[..., -1:])), dim=1)
        X1 = model(Y1)['model_out']

        inv_constraint = (Xs - X[..., 0:2])**2 + (Xt - X1)**2
        inv_constraint = torch.where(torch.cat((coords[..., -1:], coords[..., -1:]), dim=-1) == 0, torch.zeros_like(inv_constraint), inv_constraint)

        return {
            "data_constraint": data_constraint.mean() * self.constraint_weights["data_constraint"],
            "identity_constraint": identity_constraint.mean() * self.constraint_weights["identity_constraint"],
            "inv_constraint": inv_constraint.mean() * self.constraint_weights["inv_constraint"],
            "TPS_constraint": TPS_constraint.mean() * self.constraint_weights["TPS_constraint"],
            "_0.5t_alignment_error": alignment_loss
        }




class Warping3DLoss(torch.nn.Module):
    """3D warping loss with feature matching between source and target.

    Parameters
    ----------
    warp_src_pts: torch.Tensor
        An Nx3 tensor with the feature locations in the source data. Note that
        these points must be normalized to the [-1, 1] range.

    warp_tgt_pts: torch.Tensor
        An Nx3 tensor with the feature locations in the target data. Note that
        these points must be normalized to the [-1, 1] range.

    intermediate_times: list, optional
        List of intermediate times where the data constraint will be fit. All
        values must be in range [0, 1]. By default is [0.25, 0.5, 0.75]

    constraint_weights: dict, optional
        The weight of each constraint in the final loss composition. By
        default, the weights are:
        {
            "data_constraint":  1e4,
            "identity_constraint":  1e3,
            "inv_constraint": 1e4,
            "TPS_constraint": 1e3,
        }
    """
    def __init__(
            self,
            warp_src_pts: torch.Tensor,
            warp_tgt_pts: torch.Tensor,
            intermediate_times: list = [0.25, 0.5, 0.75],
            constraint_weights: dict = DEFAULT_WEIGHTS_FMLOSS
    ):
        super(Warping3DLoss, self).__init__()
        self.src = warp_src_pts
        self.tgt = warp_tgt_pts
        self.intermediate_times = intermediate_times
        self.constraint_weights = constraint_weights
        if intermediate_times is None or not len(intermediate_times):
            self.intermediate_times = [0.25, 0.5, 0.75]
        if constraint_weights is None or not len(constraint_weights):
            self.constraint_weights = DEFAULT_WEIGHTS_FMLOSS

        # Ensuring that all necessary weights are stored.
        for k, v in DEFAULT_WEIGHTS_FMLOSS.items():
            if k not in self.constraint_weights:
                self.constraint_weights[k] = v

    def forward(self, coords, model):
        """
        coords: torch.Tensor(shape=[N, 3])
        model: torch.nn.Module
        """
        M = model(coords)
        X = M["model_in"]
        Y = M["model_out"].squeeze()

        # thin plate spline energy
        hessian1 = hessian(Y, X)
        TPS_constraint = hessian1 ** 2

        # data fitting: f(src, 1)=tgt, f(tgt,-1)=src
        src = torch.cat((self.src, torch.ones_like(self.src[..., :1])), dim=1)
        y_src = model(src)['model_out']
        tgt = torch.cat((self.tgt, -torch.ones_like(self.tgt[..., :1])), dim=1)
        y_tgt = model(tgt)['model_out']
        data_constraint = (self.tgt - y_src)**2 + (self.src - y_tgt)**2
        data_constraint *= 1e2

        # forcing the feature matching along time
        for t in self.intermediate_times:
            tgt_0 = torch.cat((self.tgt, (t-1)*torch.ones_like(self.tgt[..., :1])), dim=1)
            y_tgt_0 = model(tgt_0)['model_out']
            src_0 = torch.cat((self.src, t*torch.ones_like(self.src[..., :1])), dim=1)
            y_src_0 = model(src_0)['model_out']
            data_constraint += ((y_src_0 - y_tgt_0)**2)*5e1

            src_t = torch.cat((y_src_0, -t*torch.ones_like(self.src[..., :1])), dim=1)
            y_src_t = model(src_t)['model_out']
            data_constraint += ((y_src_t - self.src)**2)*2e1

            tgt_t = torch.cat((y_tgt_0, 1-t*torch.ones_like(self.tgt[..., :1])), dim=1)
            y_tgt_t = model(tgt_t)['model_out']
            data_constraint += ((y_tgt_t - self.tgt)**2)*2e1

        # identity constraint: f(p,0) = (p)
        diff_constraint = (Y - X[..., :3])**2
        identity_constraint = torch.where(torch.cat((coords[..., -1:], coords[..., -1:], coords[..., -1:]), dim=-1) == 0, diff_constraint, torch.zeros_like(diff_constraint))

        # inverse constraint: f(f(p,t), -t) = p,  f(f(p,-t), t) = p
        Ys = torch.cat((Y, -X[..., -1:]), dim=1)
        model_Xs = model(Ys)
        Xs = model_Xs['model_out']

        # inverse constraint: f(f(p,t), 1-t) = f(p,1)
        Yt = torch.cat((Y, 1 - X[..., -1:]), dim=1)
        model_Xt = model(Yt)
        Xt = model_Xt['model_out']
        Y1 = torch.cat((X[...,0:3], torch.ones_like(X[..., -1:])), dim=1)
        X1 = model(Y1)['model_out']

        inv_constraint = (Xs - X[..., 0:3])**2 + (Xt - X1)**2
        inv_constraint = torch.where(torch.cat((coords[..., -1:], coords[..., -1:], coords[..., -1:]), dim=-1) == 0, torch.zeros_like(inv_constraint), inv_constraint)

        return {
            "data_constraint": data_constraint.mean() * self.constraint_weights["data_constraint"],
            "identity_constraint": identity_constraint.mean() * self.constraint_weights["identity_constraint"],
            "inv_constraint": inv_constraint.mean() * self.constraint_weights["inv_constraint"],
            "TPS_constraint": TPS_constraint.mean() * self.constraint_weights["TPS_constraint"],
        }


class Warping3DLoss(torch.nn.Module):
    """3D warping loss with feature matching between source and target.

    Parameters
    ----------
    warp_src_pts: torch.Tensor
        An Nx3 tensor with the feature locations in the source data. Note that
        these points must be normalized to the [-1, 1] range.

    warp_tgt_pts: torch.Tensor
        An Nx3 tensor with the feature locations in the target data. Note that
        these points must be normalized to the [-1, 1] range.

    intermediate_times: list, optional
        List of intermediate times where the data constraint will be fit. All
        values must be in range [0, 1]. By default is [0.25, 0.5, 0.75]

    constraint_weights: dict, optional
        The weight of each constraint in the final loss composition. By
        default, the weights are:
        {
            "data_constraint":  1e4,
            "identity_constraint":  1e3,
            "inv_constraint": 1e4,
            "TPS_constraint": 1e3,
        }
    """
    def __init__(
            self,
            warp_src_pts: torch.Tensor,
            warp_tgt_pts: torch.Tensor,
            intermediate_times: list = [0.25, 0.5, 0.75],
            constraint_weights: dict = DEFAULT_WEIGHTS_FMLOSS
    ):
        super(Warping3DLoss, self).__init__()
        self.src = warp_src_pts
        self.tgt = warp_tgt_pts
        self.intermediate_times = intermediate_times
        self.constraint_weights = constraint_weights
        if intermediate_times is None or not len(intermediate_times):
            self.intermediate_times = [0.25, 0.5, 0.75]
        if constraint_weights is None or not len(constraint_weights):
            self.constraint_weights = DEFAULT_WEIGHTS_FMLOSS

        # Ensuring that all necessary weights are stored.
        for k, v in DEFAULT_WEIGHTS_FMLOSS.items():
            if k not in self.constraint_weights:
                self.constraint_weights[k] = v

    def forward(self, coords, model):
        """
        coords: torch.Tensor(shape=[N, 3])
        model: torch.nn.Module
        """
        M = model(coords)
        X = M["model_in"]
        Y = M["model_out"].squeeze()

        # thin plate spline energy
        hessian1 = hessian(Y, X)
        TPS_constraint = hessian1 ** 2

        # data fitting: f(src, 1)=tgt, f(tgt,-1)=src
        src = torch.cat((self.src, torch.ones_like(self.src[..., :1])), dim=1)
        y_src = model(src)['model_out']
        tgt = torch.cat((self.tgt, -torch.ones_like(self.tgt[..., :1])), dim=1)
        y_tgt = model(tgt)['model_out']
        data_constraint = (self.tgt - y_src)**2 + (self.src - y_tgt)**2
        data_constraint *= 1e2

        # forcing the feature matching along time
        for t in self.intermediate_times:
            tgt_0 = torch.cat((self.tgt, (t-1)*torch.ones_like(self.tgt[..., :1])), dim=1)
            y_tgt_0 = model(tgt_0)['model_out']
            src_0 = torch.cat((self.src, t*torch.ones_like(self.src[..., :1])), dim=1)
            y_src_0 = model(src_0)['model_out']
            data_constraint += ((y_src_0 - y_tgt_0)**2)*5e1

            src_t = torch.cat((y_src_0, -t*torch.ones_like(self.src[..., :1])), dim=1)
            y_src_t = model(src_t)['model_out']
            data_constraint += ((y_src_t - self.src)**2)*2e1

            tgt_t = torch.cat((y_tgt_0, 1-t*torch.ones_like(self.tgt[..., :1])), dim=1)
            y_tgt_t = model(tgt_t)['model_out']
            data_constraint += ((y_tgt_t - self.tgt)**2)*2e1

        # identity constraint: f(p,0) = (p)
        diff_constraint = (Y - X[..., :3])**2
        identity_constraint = torch.where(torch.cat((coords[..., -1:], coords[..., -1:], coords[..., -1:]), dim=-1) == 0, diff_constraint, torch.zeros_like(diff_constraint))

        # inverse constraint: f(f(p,t), -t) = p,  f(f(p,-t), t) = p
        Ys = torch.cat((Y, -X[..., -1:]), dim=1)
        model_Xs = model(Ys)
        Xs = model_Xs['model_out']

        # inverse constraint: f(f(p,t), 1-t) = f(p,1)
        Yt = torch.cat((Y, 1 - X[..., -1:]), dim=1)
        model_Xt = model(Yt)
        Xt = model_Xt['model_out']
        Y1 = torch.cat((X[...,0:3], torch.ones_like(X[..., -1:])), dim=1)
        X1 = model(Y1)['model_out']

        inv_constraint = (Xs - X[..., 0:3])**2 + (Xt - X1)**2
        inv_constraint = torch.where(torch.cat((coords[..., -1:], coords[..., -1:], coords[..., -1:]), dim=-1) == 0, torch.zeros_like(inv_constraint), inv_constraint)

        return {
            "data_constraint": data_constraint.mean() * self.constraint_weights["data_constraint"],
            "identity_constraint": identity_constraint.mean() * self.constraint_weights["identity_constraint"],
            "inv_constraint": inv_constraint.mean() * self.constraint_weights["inv_constraint"],
            "TPS_constraint": TPS_constraint.mean() * self.constraint_weights["TPS_constraint"],
        }

class FlowLoss(torch.nn.Module):
    """MSE loss with feature matching between source and target plus Hessian.

    Parameters
    ----------
    warp_src_pts: torch.Tensor
        An Nxk tensor with the feature locations in the source image. Note that
        these points must be normalized to the [-1, 1] range.

    warp_tgt_pts: torch.Tensor
        An Nxk tensor with the feature locations in the target image. Note that
        these points must be normalized to the [-1, 1] range.

    constraint_weights: dict, optional
        The weight of each constraint in the final loss composition. By
        default, the weights are:
        {
            "data_constraint":  1e4,
            "TPS_constraint": 1e3,
            "diagoanl_flow_constraint": 1e3,
        }
    """
    def __init__(
        self,
        constraint_weights: dict = DEFAULT_WEIGHTS_FMLOSS,
        loop=False,
        intermediate_times: list = [0.25, 0.5, 0.75, 1.0],
    ):
        super(FlowLoss, self).__init__()
        self.constraint_weights = constraint_weights
        self.intermediate_times = torch.tensor(intermediate_times)
        
        if constraint_weights is None or not len(constraint_weights):
            self.constraint_weights = DEFAULT_WEIGHTS_FMLOSS

        # Ensuring that all necessary weights are stored.
        for k, v in DEFAULT_WEIGHTS_FMLOSS.items():
            if k not in self.constraint_weights:
                self.constraint_weights[k] = v
                
        self.loop = loop

    def forward(self, coords, model:FDModule):
        """
        coords: torch.Tensor(shape=[states,N, dimenstion_size])
        model: torch.nn.Module
        """
        states = len(coords)
        n_points = len(coords[0])
        dimenstion_size = coords.shape[-1]
        delta_next_state = 1/(states-1)
        grid_points = torch.rand(1000, dimenstion_size+1, device=coords.device) * 2 - 1

        TPS_constraint = model.hessian(grid_points)
        flatten_coords = coords.view(-1, dimenstion_size)
        total_loss_evaluations =  (states - 1) * len(self.intermediate_times)
        self.intermediate_times = self.intermediate_times.to(flatten_coords.device)
        
        # data fitting: f(state_i, delta)= state_i+1
        ts = delta_next_state * self.intermediate_times.repeat(states*n_points).unsqueeze(-1)
        reversed_ts = ts - delta_next_state
        
        flatten_coords = torch.repeat_interleave(flatten_coords, len(self.intermediate_times), dim=0)
        delta_index_state_change = n_points*len(self.intermediate_times)
        
        src = torch.cat((flatten_coords[:-delta_index_state_change], ts[:-delta_index_state_change]), dim=-1)
        y_src = model(src)['model_out']
        
        tgt = torch.cat((flatten_coords[delta_index_state_change:], reversed_ts[delta_index_state_change:]), dim=-1)
        y_tgt = model(tgt)['model_out']
        diagoanl_flow_constraint = torch.tensor(0.0, device=coords.device)

        dim_repetitions = y_src.shape[-1]//dimenstion_size
        y_src = y_src.reshape(y_src.shape[0], dim_repetitions, -1)
        y_tgt = y_tgt.reshape(y_tgt.shape[0], dim_repetitions, -1)

        y_src_var, y_src = torch.var_mean(y_src, dim=1, keepdim=True)
        y_tgt_var, y_tgt = torch.var_mean(y_tgt, dim=1, keepdim=True)

        if dim_repetitions > 1:
            diagoanl_flow_constraint = y_src_var + y_tgt_var
            diagoanl_flow_constraint = diagoanl_flow_constraint/(total_loss_evaluations**2)


        if self.loop:
            raise NotImplementedError("Loop is not implemented yet")
        else:
            data_constraint = ((y_src - y_tgt)**2)/ (total_loss_evaluations**2)

        data_constraint *= 4e2
        TPS_constraint = (TPS_constraint**2)/(total_loss_evaluations**2)
        
        
        if delta_next_state/2 in ts:
            next_state_jump = ts[:-delta_index_state_change] == (delta_next_state/2)
            alignment_loss = (y_src[next_state_jump] - y_tgt[next_state_jump])**2
            alignment_loss = alignment_loss.mean()
        else:
            alignment_loss = "not calculated"
            


            
        

        return {
            "data_constraint": data_constraint.mean() * self.constraint_weights["data_constraint"],
            "TPS_constraint": TPS_constraint.mean() * self.constraint_weights["TPS_constraint"],
            "diagoanl_flow_constraint": diagoanl_flow_constraint.mean() * self.constraint_weights["diagoanl_flow_constraint"],
            "_0.5t_alignment_error": alignment_loss
        }


class NeuralODELoss(torch.nn.Module):
    """
    Loss function for Neural ODEs.

    Parameters
    ----------
    matching_loss_weight: float
        Weight for the matching loss term.

    jacobian_loss_weight: float
        Weight for the Jacobian loss term.

    timesteps: torch.Tensor
        A tensor of timesteps to use for evaluating the loss.
    """
    def __init__(self, matching_loss_weight: float=1024, jacobian_loss_weight: float=1, timesteps=[0.0,0.125,0.25,0.375,0.5]):
        super(NeuralODELoss, self).__init__()
        self.matching_loss_weight = matching_loss_weight
        self.jacobian_loss_weight = jacobian_loss_weight
        self.timesteps = torch.tensor(timesteps, dtype=torch.float32)

    def forward(self, node_model:NeuralODE, src:torch.Tensor, tgt:torch.Tensor, grid:torch.Tensor):
        """
        Compute the loss for the given Neural ODE, source, and target. Uses grid points for the jacobian loss.

        Parameters
        ----------
        node_model: NeuralODE
            The Neural ODE model.

        src: torch.Tensor
            Source points.

        tgt: torch.Tensor
            Target points.

        Returns
        -------
        dict
            A dictionary containing the matching loss and Jacobian loss.
        """
        self.timesteps = self.timesteps.to(src.device)
        
        # Forward and backward trajectories
        src_fwd = node_model(self.timesteps, src)[-1]
        tgt_bkwd = node_model(-self.timesteps, tgt)[-1]

        # Matching loss
        matching_loss = (src_fwd - tgt_bkwd).square().sum(dim=1).mean()

        # Jacobian loss
        samples = grid + torch.randn_like(grid) * 0.02
        jacobian = node_model.internal_module.gradient(samples)
        jacobian_loss = jacobian.square().sum(dim=[1, 2]).mean()

        return {
            "matching_loss": self.matching_loss_weight * matching_loss,
            "jacobian_loss": self.jacobian_loss_weight * jacobian_loss,
            "_0.5t_alignment_error": matching_loss/src_fwd.shape[-1],
        }
        
