import torch as t
from multi_timescale_clvm import *
from variational_methods import *

latent_var = LatentVar((5,))

latent_var.grad = True
for i in range(1000000):
    var = latent_var[1:4]
    loss = t.sum((gauss_samp(var)-3)**2)
    loss.backward()
    grad = var[0].grad.cpu().numpy(),var[1].grad.cpu().numpy()
    latent_var[1:4] = grad
    print(loss)


