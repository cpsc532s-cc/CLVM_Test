import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
from torch import FloatTensor as FT
from variational_methods import *
from decoders import *
from mnist_data import *
from viz import *
import time
import matplotlib.pyplot as plt

#G_STORE_DEVICE = torch.device('cpu')
G_COMP_DEVICE = torch.device('cpu')


class DiagGaussArrayLatentVars():
    def __init__(self, mean, log_var, device, requires_grad):
        if type(mean) is FT:
            self.mean = mean
        else:
            self.mean = FT(mean, device = device)

        if type(log_var) is FT:
            self.log_var = log_var
        else:
            self.log_var = FT(log_var, device = device)

        self._requires_grad = requires_grad
        self.mean.requires_grad = self._requires_grad
        self.log_var.requires_grad = self._requires_grad

    def params(self, log_var_reduction = 0):
        return (self.mean, self.log_var)

    def kl_divergence(self, other_latent):
        if not type(other_latent) is DiagGaussArrayLatentBatch:
            raise NotImplementedError()
        else:
            gauss_kl_div(self.params(), other_latent.params())

    def log_likelihood(self, data):
        gauss_log_p(self.params(),data)

    def sample(self, log_var_reduction = 0):
        return gauss_samp(self.params.params(log_var_reduction=log_var_reduction))

    @staticmethod
    def get_normal(dim, device, requires_grad):
        return DiagGaussArrayLatentVars(
                mean = t.zeros(dim, device = device),
                log_var = t.zeros(dim, device = device),
                requires_grad)


class DiagGaussArrayLatentStore:
    def __init__(self, dim, num):
        # Init to normal
        self.dim = dim
        self.mean = np.zeros(dim,np.float32)
        self.log_var = np.zeros(dim,np.float32)
        self.var_dict = {"mean": self.mean, "log_var": self.log_var}
        self._optimizer = None

    def load_batch(indices, device, requires_grad = False):
        # Loads batch of values for computation
        mean_batch = self.mean[indices]
        log_var_batch = self.log_var[indices]
        return DiagGaussArrayLatentBatch(self, mean_batch, log_var_batch,
                                     requires_grad = requires_grad)

    def register_optimizer(self, optimizer):
        self._optimizer = optimizer

    def optimizer(self):
        return self._optimizer


class AdamLatentOpt:
    def __init__(self, latent_var, params):
        self.latent_var = latent_var
        # Register self as optimizer
        self.latent_var.optimizer = self

        # Create m0 and m1 arrays
        self.params = params
        self.m0s = {}
        self.m1s = {}
        for k in latent_var.var_dict:
            shape = latent_var.var_dict[k].shape
            self.m0s[k] = np.zeros(shape,np.float32)
            self.m1s[k] = np.zeros(shape,np.float32)

    def step(indices, grad):
        lr = self.params["lr"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]
        e = self.params["e"]

        for k in latent_var.var_dict:
            self.m0s[k][indices] = self.m0s[k][indices]*b1+(1-b1)*grad[k]
            self.m1s[k][indices] = self.m1s[k][indices]*b2+(1-b2)*(grad[k]**2)
            b_m0 = self.m0s[k][indices]/(1-b1)
            b_m1 = self.m1s[k][indices]/(1-b2)
            self.latent_var.var_dict[k][indices] -= lr*b_m0/(np.sqrt(b_m1)+e)


class MiddleEdge:
    # Edge between two latents
    def __init__(self, model_class, model_params, output_latent_store):
        # Returns what the input_dim would have to be
        output_dim = output_latent_store.dim
        # Check if this is valid
        model = model_class(model_params)
        assert(model.is_valid_output_dim(output_dim))
        self.model = model
        # Model dictates what the input_latent has to be
        return self.model.get_required_input_dim(output_dim)

    def kl_div(input_latent_batch, output_latent_batch):
        # Corresponding batches of input and output latent variables
        # Sample z_in from q_in
        q_in_samp = input_latent_batch.sample()

        # Compute p(z_out|z_in)
        p_out = output_likelihood(q_in_samp)

        # Compute KL(p_out||q_out)
        q_out = output_latent_batch
        kl_q_p = p_out.kl_divergence(q_out)
        return kl_q_p

    def output_likelihood(input_sample):
        # Compute p(z_out|z_in)
        p_out_mean, p_out_log_var = self.model(input_sample)
        p_out = DiagGaussArrayLatentVars(p_out_mean, p_out_log_var)

        return p_out


class BottomEdge:
    # Edge between bottom latent and gt data
    def __init__(self, model_class, model_params, output_dim):
        # Returns what the input_dim would have to be
        # Check if this is valid
        model = model_class(model_params)
        assert(model.is_valid_output_dim(output_dim))
        self.model = model
        # Model dictates what the input_latent has to be
        return self.model.get_required_input_dim(output_dim)

    def lp_loss(input_latent_batch, output_data_batch):
        # Corresponding batches of input and output latent variables
        # Sample z_in from q_in
        q_in_samp = input_latent_batch.sample()

        # Compute likelihood p(x_out|z_in)
        p_out = output_likelihood(q_in_samp)

        # Compute log likelihood of the data
        lp_q_p = -p_out.log_likelihood(output_data_batch)

        return loss

    def output_likelihood(input_sample):
        # Compute p(z_out|z_in)
        p_out_mean, p_out_log_var = self.model(input_sample)
        p_out = DiagGaussArrayLatentVars(p_out_mean, p_out_log_var)

        return p_out


class CLVM_Stack:
    # Construct the stack starting from the bottom
    def __init__(self, data):
        self_.edges = []
        self_.latents = []
        self._edge_dep = {}
        self._latent_dep = {}

    def stack_latent(self, model_class, model_params)
        if len(self._edges) == 0:
            BottomEdge()
        else:
            MiddleEdge()

        # Create input latent for bottom edge
        edges[0].
        self._edge_dep[edges[0]] = []

    def update_latent(self, latent):
        ingoing_edge, outgoing_edge = self._latent_dep[latent]
        loss, mid_grad_kl = self.kl_loss(layer, index, extra)
        loss, mid_grad_lp = self.lp_loss(layer, index, extra)

    def update_edge(self, edge):
        input_latent, output_latent = self._edge_dep[edge]
        loss = self.lp_loss(layer, index, extra)

    def prior_loss(latent, prior):
        # Computes KL divergence
        q_z = latent
        p_z = prior
        return -q_z.kl_divergence(p_z)


class MNISTData():
    def __init__():


    def load_batch(indices, device, requires_grad = False):


def main():
    # MNIST Test
    edges = []
    bot_edge = BottomEdge()
    edges.append(bot_edge)
    edges.append(MiddleEdge)
    clvm = CLVM_Chain()


if __name__ == "__main__":
    main()

