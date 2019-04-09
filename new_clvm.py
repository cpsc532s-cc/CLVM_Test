
#G_STORE_DEVICE = torch.device('cpu')
G_COMP_DEVICE = torch.device('cpu')

class GaussArrayLatentStore:
    class GaussArrayLatentBatch():
        def __init__(self, latent_store, indices):
            self.mean = latent_store.mean[indices]
            self.log_var = latent_store.log_var[indices]

        def sample_

    def __init__(self, dim, num):
        self.device = device
        self.mean = np.zeros(dim,np.float32)
        self.log_var = np.zeros(dim,np.float32)
        self.var_dict = {"mean": self.mean, "log_var": self.log_var}

    def load_batch(indices, device):
        # Loads batch of values for computation
        mean = FT(self.mean[x], device = device)
        log_var = FT(self.log_var[x], device = device)
        mean.requires_grad=self.grad
        log_var.requires_grad=self.grad
        return (mean, log_var)

class AdamLatentOpt:
    def __init__(self, latent_var, params):
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

class LatentEdge:
    # Edge between two latents
    def __init__(self, output_latent):
        # Model dictates what the input_latent has to be

class BottomEdge:
    # Edge between bottom latent and gt data
    def __init__(self, output_data):

class CLVM_Chain:
    # Construct the chain starting from the bottom
    def __init__(self, edges):

    def update_layer():

