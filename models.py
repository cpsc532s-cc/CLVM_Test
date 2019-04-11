import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
from variational_methods import log_rect

class MLP(nn.Module):
    def __init__(self, model_configs, output_dim, is_bot=False):
        super(MLP, self).__init__()
        output_len = np.prod(output_dim)
        self.in_size = model_configs["in_size"]
        h_size = model_configs["h_size"]
        self.l1 = nn.Linear(self.in_size,h_size)
        self.mean = nn.Linear(h_size,output_len)
        self.log_var = nn.Linear(h_size,output_len)

    def forward(self, x):
        h = f.relu(self.l1(x))
        mean = self.mean(h)
        logvar = log_rect(self.log_var(h))
        return mean, logvar

    def is_valid_output_dim(self, output_dim):
        #TODO
        return True

    def get_required_input_dim(self, output_dim):
        return (self.in_size,)

