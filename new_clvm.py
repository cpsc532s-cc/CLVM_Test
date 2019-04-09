#G_STORE_DEVICE = torch.device('cpu')
G_COMP_DEVICE = torch.device('cpu')

class DiagGaussArrayLatentStore:
    class DiagGaussArrayLatentVars():
        def __init__(self, mean, log_var, requires_grad):
            if type(mean) is FT:
                self.mean = mean
            else:
                self.mean = FT(mean, device = device)

            if type(log_var) is FT:
                self.log_var = log_var
            else:
                self.log_var = FT(log_var[indices], device = device)

            self._requires_grad = requires_grad
            self.mean.requires_grad = self._requires_grad
            self.log_var.requires_grad = self._requires_grad

        def kl_divergence(self, other_latent):
            if not type(other_latent) is DiagGaussArrayLatentBatch:
                raise NotImplementedError()
            else:
                #TODO

        def sample(self, var_reduction):
            gauss_samp
            

    def __init__(self, dim, num):
        # Init to normal
        self.dim = dim
        self.mean = np.zeros(dim,np.float32)
        self.log_var = np.zeros(dim,np.float32)
        self.var_dict = {"mean": self.mean, "log_var": self.log_var}

    def load_batch(indices, device, requires_grad = False):
        # Loads batch of values for computation
        mean_batch = self.mean[indices]
        log_var_batch = self.log_var[indices]
        return DiagGaussArrayLatentBatch(self, mean_batch, log_var_batch, 
                                     requires_grad = requires_grad)


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

class MiddleEdge:
    # Edge between two latents
    def __init__(self, model_class, model_params, output_latent_store):
        # Returns what the input_dim would have to be
        # Model dictates what the input_latent has to be
        output_dim = output_latent_store.dim
        # Check if this is valid 
        model = model_class(model_params)
        assert(model.is_valid_output_dim(output_dim))
        self.model = model
        return self.model.get_required_input_dim(output_dim)

    def reconstruct_loss(input_latent_batch, output_latent_batch):
        # Corresponding batches of input and output latent variables
        
        # Sample z_in from q_in
        q_in_samp = input_latent_batch.sample()

        # Compute p(z_out|z_in)
        q_in_samp = input_latent_batch.sample()
        p_out_mean, p_out_log_var = self.model(q_in_samp)
        p_out = DiagGaussArrayLatentVars(p_out_mean, p_out_log_var)

        # Compute KL(p_out||q_out)
        q_out = output_latent_batch
        kl_q_p = p_out.kl_divergence(q_out)
        loss = -kl_q_p 

        return loss
        
        
class BottomEdge:
    # Edge between bottom latent and gt data
    def __init__(self, output_dim):


    def reconstruct_loss(input_latent, output_data):

class CLVM_Chain:
    # Construct the chain starting from the bottom
    def __init__(self, edges):

    def update_latent():


    def update_edge():

    def prior_loss(latent):
        # Computes KL divergence


