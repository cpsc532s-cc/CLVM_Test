import numpy as np
from mnist_data import *
from viz import *

import torch as t
from new_clvm import *

from PIL import Image
from tensorboardX import SummaryWriter

G_STORE_DEVICE = t.device('cpu')
G_COMP_DEVICE = t.device('cuda:0')
#G_COMP_DEVICE = t.device('cpu')

class MNISTData():
    def __init__(self, bs):
        self.bs = bs
        data = load_mnist()
        data = np.array(data.view((data.shape[0],)+(1,)+data.shape[1:]), dtype=np.float32)[::10]
        print(data.shape)
        self.fdims = data.shape[1:]
        self._data = (data-np.mean(data))/np.std(data)
        self.n = self._data.shape[0]

    def load_batch(self, indices, requires_grad = False, device = G_COMP_DEVICE):
        #print(self._data[indices].dtype, self._data[indices].shape)
        batch = FT(self._data[indices], device = device)
        batch.requires_grad = requires_grad
        return batch

    def sample_indices(self):
        #x = np.random.randint(0,self.n//self.bs)
        #return slice(x*self.bs,(x+1)*self.bs)
        return np.random.randint(0, self.n, size=self.bs)

    def slice(self, indices):
        return self._data[indices]#.reshape((-1,28,28))


def write_losses(writer, iter_n, lp_losses_l, kl_losses_l, lp_losses_e):
    #for i,  in enumerate(zip(lp_losses_l, kl_losses_l)):
    #    (lp_l, kl_l)
    # Just write the last and the first losses
    for i in range(len(lp_losses_l[0])):
        writer.add_scalars('loss/lp_loss{}'.format(i),
                {'start': lp_losses_l[0][i],
                    'end': lp_losses_l[-1][i]}, iter_n)
        writer.add_scalars('loss/kl_loss{}'.format(i),
                {'start': kl_losses_l[0][i],
                    'end': kl_losses_l[-1][i]}, iter_n)
    t_lp_loss = 0
    for i in range(len(lp_losses_l[0])):
        t_lp_loss += lp_losses_l[0][i]
    writer.add_scalar('loss/logp', -t_lp_loss, iter_n)


def run_clvm():
    # MNIST Test
    data = MNISTData(128)
    opt_params={"lr":0.05, "b1":0.9, "b2":0.999, "e":1e-8}
    opt_class=AdamLatentOpt
    clvm = CLVM_Stack(data)
    #clvm.stack_latent(m.Deconv2d, {"k": 5, "stride": 2, "i_pad":0, "i_chan": 4, "h_chan": 16}, opt_class, opt_params, 0.003)
    #clvm.stack_latent(m.MLP, {"in_size": 128, "h_size": 256, "n_int":2}, opt_class, opt_params, 0.001)
    clvm.stack_latent(m.ResNetBlock, {"k": 5, "i_chan": 4, "h_chan": 16, "n_h": 1}, opt_class, opt_params, 0.002)
    clvm.stack_latent(m.MLP, {"in_size": 256, "h_size": 384, "n_int":1}, opt_class, opt_params, 0.002)
    clvm.stack_latent(m.MLP, {"in_size": 128, "h_size": 256, "n_int":1}, opt_class, opt_params, 0.002)
    #clvm.stack_latent(m.Deconv2d, {"stride": 256, "h_size": 256, "n_int":3}, opt_class, opt_params, 0.003)
    #clvm.stack_latent(m.MLP, {"in_size": 128, "h_size": 256, "n_int":1}, opt_class, opt_params, 0.001)
    clvm.print_rep()

    writer = SummaryWriter()
    ds = DisplayStream()
    for i in range(50000):
        indices = data.sample_indices()
        if i%50 == 0:
            print(i)
            lp_losses_l, kl_losses_l, lp_losses_e = clvm.update(indices, display=True, return_loss=True)
            write_losses(writer, i, lp_losses_l, kl_losses_l, lp_losses_e)

            recon = clvm.reconstruct(range(5), -1)
            recon_means = recon.mean.detach().cpu().numpy().reshape((-1,28,28)).squeeze()
            recon_tup = ()
            recon_means = np.clip(recon_means, -1.0, 4)
            for recon_mean in recon_means:
                recon_tup += (recon_mean,)
            recon_img = np.hstack(recon_tup)

            trues = data.slice(range(5)).squeeze()
            true_tup = ()
            for true in trues:
                true_tup += (true,)
            true_img = np.hstack(true_tup)

            sample = clvm.sample(5)
            sample_means = sample.mean.detach().cpu().numpy().reshape((-1,28,28)).squeeze()
            sample_means = np.clip(sample_means, -1.0, 4)
            sample_tup = ()
            for sample_mean in sample_means:
                sample_tup += (sample_mean,)
            sample_img = np.hstack(sample_tup)

            ds.show_img(np.vstack((recon_img, true_img, sample_img)))
        else:
            clvm.update(indices)

run_clvm()
