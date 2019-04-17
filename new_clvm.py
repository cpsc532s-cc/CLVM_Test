import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
from torch import FloatTensor

from variational_methods import *
import time

import models as m

G_STORE_DEVICE = t.device('cpu')
#G_COMP_DEVICE = t.device('cuda:0')
G_COMP_DEVICE = t.device('cpu')

def FT(data, device = G_STORE_DEVICE, requires_grad = False):
    return t.tensor(data, device = device, requires_grad = requires_grad)

class DiagGaussArrayLatentVars():
    def __init__(self, mean, log_var, requires_grad = None, device = G_COMP_DEVICE):
        if type(mean) is t.Tensor:
            self.mean = mean
        else:
            #print(type(mean))
            self.mean = FT(mean, device = device)

        if type(log_var) is t.Tensor:
            self.log_var = log_var
        else:
            self.log_var = FT(log_var, device = device)

        if requires_grad is not None:
            self._requires_grad = requires_grad
            self.mean.requires_grad = self._requires_grad
            self.log_var.requires_grad = self._requires_grad

        self.fdims = self.mean.size()[1:]
        self.n = self.mean.size()[0]

    def params(self, log_var_reduction = 0):
        return (self.mean, self.log_var)

    def kl_divergence(self, other_latent):
        if not type(other_latent) is DiagGaussArrayLatentVars:
            raise NotImplementedError()
        else:
            return gauss_kl_div(self.params(), other_latent.params())

    def log_likelihood(self, data):
        return gauss_log_p(self.params(),data)

    def sample(self, log_var_reduction = 0):
        return gauss_samp(self.params(log_var_reduction=log_var_reduction))

    @staticmethod
    def get_normal(dim, requires_grad = False, device = G_COMP_DEVICE):
        return DiagGaussArrayLatentVars(
                mean = t.zeros(dim, device = device),
                log_var = t.zeros(dim, device = device),
                requires_grad=requires_grad, device = device)

    def grad_dict(self):
        return {"mean": self.mean.grad, "log_var": self.log_var.grad}


class DiagGaussArrayLatentStore:
    def __init__(self, fdims, n):
        # Init to normal
        self.fdims = fdims
        self.n = n
        dims = (n,) + fdims
        self.mean = np.zeros(dims,np.float32)
        self.log_var = np.zeros(dims,np.float32)
        self.var_dict = {"mean": self.mean, "log_var": self.log_var}
        self._optimizer = None
        self.cur_batch = None
        self.cur_indices = None

    def load_batch(self, indices, requires_grad = False, device = G_COMP_DEVICE):
        # Loads batch of values for computation
        mean_batch = self.mean[indices]
        #print("thing", self.mean.shape)
        log_var_batch = self.log_var[indices]
        self.cur_batch = DiagGaussArrayLatentVars(mean_batch, log_var_batch,
                                     requires_grad = requires_grad)
        self.cur_indices = indices
        return self.cur_batch

    def init_optimizer(self, optimizer_class, params):
        self._optimizer = optimizer_class(self, params)

    def optimizer(self):
        return self._optimizer


class AdamLatentOpt:
    def __init__(self, latent_store, params):
        self.latent_store = latent_store

        # Create m0 and m1 arrays
        self.params = params
        self.m0s = {}
        self.m1s = {}
        for k in latent_store.var_dict:
            shape = latent_store.var_dict[k].shape
            self.m0s[k] = np.zeros(shape,np.float32)
            self.m1s[k] = np.zeros(shape,np.float32)

    def step(self):
        grad = self.latent_store.cur_batch.grad_dict()
        indices = self.latent_store.cur_indices

        lr = self.params["lr"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]
        e = self.params["e"]

        for k in self.latent_store.var_dict:
            gradk = grad[k].cpu().numpy()
            self.m0s[k][indices] = self.m0s[k][indices]*b1+(1-b1)*gradk
            self.m1s[k][indices] = self.m1s[k][indices]*b2+(1-b2)*(gradk**2)
            b_m0 = self.m0s[k][indices]/(1-b1)
            b_m1 = self.m1s[k][indices]/(1-b2)
            self.latent_store.var_dict[k][indices] -= lr*b_m0/(np.sqrt(b_m1)+e)


class Edge:
    def __init__(self, model_class, model_configs, output_fdim, is_bot=False, lr=0.001):
        # Returns what the input_dim would have to be
        # Check if this is valid
        model = model_class(model_configs, output_fdim, is_bot)
        assert(model.is_valid_output_dim(output_fdim))
        self.model = model.to(G_COMP_DEVICE)
        self._optim = optim.Adam(self.model.parameters(),lr=lr)

    def get_required_input_dim(self, output_fdim):
        # Model dictates what the input_latent has to be
        return self.model.get_required_input_dim(output_fdim)

    def output_likelihood(self, input_sample):
        # Compute p(z_out|z_in)
        p_out_mean, p_out_log_var = self.model(input_sample)
        p_out = DiagGaussArrayLatentVars(p_out_mean, p_out_log_var)
        return p_out

    def step(self):
        self._optim.step()

    def zero_grad(self):
        self._optim.zero_grad()


class CLVM_Stack:
    # Construct the stack starting from the bottom
    def __init__(self, data, use_kl=True):
        self.use_kl = use_kl
        self._edges = []
        self._latents = []
        # Dependencies have structure  {obj: [in, out]}
        self._edge_dep = {}
        self._latent_dep = {}
        self._data = data

    def stack_latent(self, model_class, model_params, optimizer_class, optimizer_params, edge_lr):
        # New edge to be stacked
        cur_edge = None
        # Required dimensionality of new latent var
        req_in_fdims = None
        if len(self._latents) == 0:
            # First edge is bottom edge
            cur_edge = Edge(model_class, model_params, self._data.fdims, is_bot=True, lr=edge_lr)
            req_in_fdims = cur_edge.get_required_input_dim(self._data.fdims)

            # Register the new edge
            self._edges.append(cur_edge)
            self._edge_dep[cur_edge] = [None, self._data]
        else:
            # Stack new edge upon the last latent
            output_latent = self._latents[-1]
            cur_edge = Edge(model_class, model_params, output_latent.fdims, is_bot=False, lr=edge_lr)
            req_in_fdims = cur_edge.get_required_input_dim(output_latent.fdims)

            # Register the new ingoing edge to bottom latent
            self._latent_dep[output_latent][0] = cur_edge

            # Register the new edge
            self._edges.append(cur_edge)
            self._edge_dep[cur_edge] = [None, output_latent]

        # Generate a matching latent based on the inferred dimensions
        top_latent = DiagGaussArrayLatentStore(req_in_fdims, self._data.n)
        top_latent.init_optimizer(optimizer_class, optimizer_params)

        # Register the new latent
        self._latents.append(top_latent)
        self._latent_dep[top_latent] = [None, cur_edge]
        self._edge_dep[cur_edge][0] = top_latent

    def update_latent(self, latent, indices):
        #ingoing_edge, outgoing_edge = self._latent_dep[latent]
        latent_batch = latent.load_batch(indices, requires_grad=True)
        kl_loss = self.kl_loss(latent, latent_batch)
        if self.use_kl:
            kl_loss.backward()
        lp_loss = self.lp_loss(latent, latent_batch)
        lp_loss.backward()
        latent.optimizer().step()
        return np.asscalar(lp_loss.detach().cpu().numpy()), \
                 np.asscalar(kl_loss.detach().cpu().numpy())

    def update_edge(self, edge, indices):
        #print(indices)
        input_latent = self._edge_dep[edge][0]
        edge.zero_grad()
        lp_loss = self.lp_loss(input_latent, input_latent.load_batch(indices, requires_grad=False))
        lp_loss.backward()
        edge.step()
        return np.asscalar(lp_loss.detach().cpu().numpy())

    def kl_loss(self, latent, latent_batch):
        indices = latent.cur_indices
        ingoing_edge = self._latent_dep[latent][0]
        #latent_batch = latent.load_batch(indices, requires_grad=True)

        if ingoing_edge is None:
            # For top latent, kl with normal
            prior = DiagGaussArrayLatentVars.get_normal((latent_batch.n,)+(latent_batch.fdims))
            #print(t.sum(latent_batch.kl_divergence(prior)).detach().cpu().numpy())
        else:
            # For other dists kl based on previous latent
            prev_q = self._edge_dep[ingoing_edge][0].load_batch(indices)
            prior = ingoing_edge.output_likelihood(prev_q.sample())
        return t.mean(latent_batch.kl_divergence(prior))

    def lp_loss(self, input_latent, input_latent_batch):
        indices = input_latent.cur_indices
        edge = self._latent_dep[input_latent][1]
        _, output = self._edge_dep[edge]
        #input_latent_batch = input_latent.load_batch(indices, requires_grad=True)

        # Sample z_in from q_in
        #print(input_latent_batch.fdims)
        q_in_samp = input_latent_batch.sample()
        #print(q_in_samp.size())
        # Compute likelihood p(x_out|z_in)
        p_out = edge.output_likelihood(q_in_samp)
        #print(p_out)
        if output is self._data:
            # Compute log likelihood of the data
            output_data_batch = self._data.load_batch(indices)
            loss =  -t.mean(p_out.log_likelihood(output_data_batch))
            return loss
        else:
            # Compute kl wrt output
            output_latent_batch = output.load_batch(indices)
            return t.mean(p_out.kl_divergence(output_latent_batch))

    def print_rep(self):
        for latent in self._latents:
            print(latent.fdims)

    def update(self, indices, display=False, return_loss=False):
        np.set_printoptions(precision=4)
        if display:
            print("Latent:")
        lp_losses_l = []
        kl_losses_l = []
        for _ in range(5):
            lp_loss_l, kl_loss_l = self.update_latents(indices)
            lp_losses_l.append(lp_loss_l)
            kl_losses_l.append(kl_loss_l)
            if display:
                print(np.asarray([lp_loss_l, kl_loss_l]))
        lp_losses_e = self.update_edges(indices)
        if display:
            print("Edge:", np.asarray(lp_losses_e))
            print("-"*40)
        if return_loss:
            # Just return everything
            return lp_losses_l, kl_losses_l, lp_losses_e

    def update_edges(self, indices):
        lp_losses = []
        for edge in self._edges:
            lp_losses.append(self.update_edge(edge, indices))
        return lp_losses

    def update_latents(self, indices):
        lp_losses = []
        kl_losses = []
        for latent in self._latents:
            lp_loss, kl_loss = self.update_latent(latent, indices)
            lp_losses.append(lp_loss)
            kl_losses.append(kl_loss)
        return lp_losses, kl_losses

    def reconstruct(self, indices, latent_idx):
        latent = self._latents[latent_idx]
        latent_batch = latent.load_batch(indices)
        # Get sample
        latent_samps = latent_batch.sample()
        # Decode sample
        return self.decode_from_sample(latent_samps, latent_idx)

    def decode_from_sample(self, latent_samps, latent_idx):
        latent = self._latents[latent_idx]
        while True:
            # Get outgoing edge
            edge = self._latent_dep[latent][1]
            # Compute new latent batch with old sample
            latent_batch = edge.output_likelihood(latent_samps)
            latent = self._edge_dep[edge][1]
            if latent is self._data:
                break
            # Sample new latent batch
            latent_samps = latent_batch.sample()

        return latent_batch

    def sample(self, num):
        # Sample prior for the top layer
        top_fdims = self._latents[-1].fdims
        prior = DiagGaussArrayLatentVars.get_normal((num,)+(top_fdims))
        return self.decode_from_sample(prior.sample(), -1)

"""
def main():
    # MNIST Test
    data = MNISTData(128)
    opt_params={"lr":0.05, "b1":0.9, "b2":0.999, "e":1e-8}
    opt_class=AdamLatentOpt
    clvm = CLVM_Stack(data)
    clvm.stack_latent(m.Deconv2d, {"k": 5, "stride": 2, "i_pad":0, "i_chan": 4, "h_chan": 16}, opt_class, opt_params, 0.003)
    clvm.stack_latent(m.MLP, {"in_size": 128, "h_size": 256, "n_int":2}, opt_class, opt_params, 0.001)
    clvm.stack_latent(m.MLP, {"in_size": 32, "h_size": 64, "n_int":1}, opt_class, opt_params, 0.001)
    #clvm.stack_latent(m.Deconv2d, {"stride": 256, "h_size": 256, "n_int":3}, opt_class, opt_params, 0.003)
    #clvm.stack_latent(m.MLP, {"in_size": 128, "h_size": 256, "n_int":1}, opt_class, opt_params, 0.001)
    clvm.print_rep()

    ds = DisplayStream()
    for i in range(50000):
        indices = data.sample_indices()
        if i%50 == 0:
            print(i)
            lp_losses_l, kl_losses_l, lp_losses_e = clvm.update(indices, display=True, return_loss=True)

            recon = clvm.reconstruct(range(5), -1)
            recon_means = recon.sample().detach().cpu().numpy().reshape((-1,28,28)).squeeze()
            recon_tup = ()
            for recon_mean in recon_means:
                recon_tup += (recon_mean,)
            recon_img = np.hstack(recon_tup)

            trues = data.slice(range(5)).squeeze()
            true_tup = ()
            for true in trues:
                true_tup += (true,)
            true_img = np.hstack(true_tup)

            sample = clvm.sample(5)
            sample_means = sample.sample().detach().cpu().numpy().reshape((-1,28,28)).squeeze()
            sample_tup = ()
            for sample_mean in sample_means:
                sample_tup += (sample_mean,)
            sample_img = np.hstack(sample_tup)

            ds.show_img(np.vstack((recon_img, true_img, sample_img)))
        else:
            clvm.update(indices)

if __name__ == "__main__":
    main()

"""
