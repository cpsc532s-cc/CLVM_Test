import torch as t
from torch import nn
import torch.nn.functional as f
import numpy as np
from variational_methods import log_rect

class MLP(nn.Module):
    def __init__(self, model_configs, output_dim, is_bot=False):
        super(MLP, self).__init__()
        # Remember output dim so we can reshape to it later
        self.output_dim = output_dim
        output_len = np.prod(output_dim)
        self.in_size = model_configs["in_size"]
        h_size = model_configs["h_size"]
        n_int = model_configs["n_int"]

        self.ls = nn.ModuleList([nn.Linear(self.in_size,h_size)])
        for _ in range(n_int):
            self.ls.append(nn.Linear(h_size,h_size))
        self.mean = nn.Linear(h_size,output_len)
        self.log_var = nn.Linear(h_size,output_len)

    def forward(self, x):
        x = t.flatten(x, start_dim = 1)
        h = x
        for l in self.ls:
            h = f.leaky_relu(l(h))
        mean = self.mean(h)
        logvar = log_rect(self.log_var(h))
        return mean.reshape((-1,)+self.output_dim), logvar.reshape((-1,)+self.output_dim)

    def is_valid_output_dim(self, output_dim):
        #TODO
        return True

    def get_required_input_dim(self, output_dim):
        return (self.in_size,)

class ResNetBlock(nn.Module):
    def __init__(self, model_configs, output_dim, is_bot=False):
        super(ResNetBlock, self).__init__()
        # There is some ambiguity on the output_dim given an input_dim
        # pytorch represents as output_padding
        # Determine smallest o_pad to make things work:
        # d_in = (d_out-k-o_pad+2*i_pad)/stride + 1
        self.o_chan = output_dim[0]
        o_d_h = output_dim[1]
        o_d_w = output_dim[2]

        #self.stride = model_configs["stride"]
        self.k = model_configs["k"]
        #self.i_pad = model_configs["i_pad"]
        self.i_chan = model_configs["i_chan"]
        self.h_chan = model_configs["h_chan"]
        #self.n_lay = model_configs["n_lay"]
        self.n_h = model_configs["n_h"]

        # Padding required to be shape preserving
        pad = (self.k-1)//2

        self.start_block = nn.Conv2d(self.i_chan,self.h_chan,self.k,padding=pad)
        self.layers = nn.ModuleList([])
        for _ in range(self.n_h):
            self.layers.append(nn.Conv2d(self.h_chan,self.h_chan,self.k,padding=pad))

        # Output heads
        self.mean = nn.Conv2d(self.h_chan, self.o_chan, 3, padding=1)
        self.log_var = nn.Conv2d(self.h_chan, self.o_chan, 3, padding=1)

    def forward(self, x):
        h = f.leaky_relu(self.start_block(x))
        h2 = h
        for lay in self.layers:
            h2 = f.leaky_relu(lay(h2))
        h = h+h2
        mean = self.mean(h)
        logvar = log_rect(self.log_var(h))
        return mean, logvar

    def is_valid_output_dim(self, output_dim):
        #TODO
        # Needs to be c, w, h
        return len(output_dim) == 3

    def get_required_input_dim(self, output_dim):
        # Resnet block is shape preserving
        # Extended to allow converting channel umber though
        print(output_dim)
        return (self.i_chan,) + output_dim[1:]


class Deconv2d(nn.Module):
    def __init__(self, model_configs, output_dim, is_bot=False):
        super(Deconv2d, self).__init__()
        # There is some ambiguity on the output_dim given an input_dim
        # pytorch represents as output_padding
        # Determine smallest o_pad to make things work:
        # d_in = (d_out-k-o_pad+2*i_pad)/stride + 1
        self.o_chan = output_dim[0]
        o_d_h = output_dim[1]
        o_d_w = output_dim[2]

        self.stride = model_configs["stride"]
        self.k = model_configs["k"]
        self.i_pad = model_configs["i_pad"]
        self.i_chan = model_configs["i_chan"]
        self.h_chan = model_configs["h_chan"]
        #self.n_h = model_configs["n_h"]

        o_pad_h = (o_d_h-self.k+2*self.i_pad) % self.stride
        o_pad_w = (o_d_w-self.k+2*self.i_pad) % self.stride
        self.o_pad = (o_pad_h, o_pad_w)

        # Shape perserving
        #self.id_conv = nn.Conv2d(self.i_chan, self.h_chan, 3, padding=1)
        #self.id_conv_1 = nn.Conv2d(self.h_chan, self.h_chan, 3, padding=1)
        self.deconv = nn.ConvTranspose2d(self.i_chan, self.h_chan,
                self.k, stride=self.stride, padding=self.i_pad,
                output_padding=self.o_pad)

        # Output heads
        self.mean = nn.Conv2d(self.h_chan, self.o_chan, 3, padding=1)
        self.log_var = nn.Conv2d(self.h_chan, self.o_chan, 3, padding=1)

    def forward(self, x):
        h = f.leaky_relu(self.deconv(x))
        #h = f.leaky_relu(self.id_conv_1(h))
        mean = self.mean(h)
        logvar = log_rect(self.log_var(h))
        return mean, logvar

    def is_valid_output_dim(self, output_dim):
        #TODO
        return True

    def get_required_input_dim(self, output_dim):
        # Reference the pytorch docs:
        # d_in = (d_out-k-o_pad+2*i_pad)/stride + 1
        o_d_h = output_dim[1]
        o_d_w = output_dim[2]

        d_in_h = (o_d_h-self.k-self.o_pad[0]+2*self.i_pad)//self.stride + 1
        d_in_w = (o_d_w-self.k-self.o_pad[1]+2*self.i_pad)//self.stride + 1
        return (self.i_chan, d_in_h, d_in_w)

