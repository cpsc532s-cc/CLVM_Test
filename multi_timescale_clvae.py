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
import torch as t
from torch.FloatTensor as FT 
from torch.LongTensor as FT 

def text_to_indices(text):
    indices = []
    for i in text:
        indices.append(ord(i))
    return indices
        

def indices_to_text(text):
    text_array = []
    for i in text:
        text_array += 
    text = "".join(text_array)
    return text

def slice_tuple(x,sl)
    out = []
    for a in x:
        out.append(a[sl])
    return tuple(out)


class LatentVar(optimizer="adam")
    def __init__(dim,optimizer="adam",params={lr=0.1,b1=0.9,b2=0.999,e=1e-8}):
        self.optimizer = optimizer
        self.params = params
        if optimizer == "adam":
            self.mean = np.zeros(dim,np.float32)
            self.log_var = np.zeros(dim,np.float32)

            self.mean_m0 = np.zeros(dim,np.float32)
            self.log_var_m0 = np.zeros(dim,np.float32)

            self.mean_m1 = np.zeros(dim,np.float32)
            self.log_var_m1 = np.zeros(dim,np.float32)

    self.grad = False 
    self.shape = self.mean.shape

    def __getitem__(self,x):
        if self.grad:
            mean = FT(self.mean[x]).cuda()
            mean.requires_grad=self.grad
            log_var = FT(self.log_var[x]).cuda()
            mean.requires_grad=self.grad
            return FT(self.mean[x]).cuda(),FT(self.log_var[x]).cuda()
        else:
            return FT(self.mean[x]).cuda(),FT(self.log_var[x]).cuda()
            
    def __setitem__(self,x,grad):
        mean_grad,var_grad = grad  
        lr = self.params[lr]
        b1 = self.params[lr]
        b2 = self.params[lr]
        e = self.params[lr]

        self.mean_m1[x] = self.mean_m1[x]*b1-(1-b1)*grad[x]
        self.mean_m2[x] = self.mean_m2[x]*b2-(1-b2)*self.mean_m2[x]
        mean_m1 = self.mean_m1[x]/(1-b1)
        mean_m2 = self.mean_m2[x]/(1-b2)
        mean[x] = mean[x]-lr*grad/(np.sqrt(mean_m2)+e)


class MultiTimescaleCLVM(object):
    def __init__(self, data, embedding, bs, layers):
        self.data = data

        self.embedding = embedding

        self.layers = []
        curr_length = data.shape[0]
        for i, (downsampling, latent_size, window, model, opt) in enumerate(layers):
            curr_length = curr_lenght // downsampling
            latent_length = curr_lenght + window // down_sampling
            #Add zero padding to_data
            self.data = np.cat((np.zeros((window,),dtype=np.int8),data))
            if i == 0
                #Add zero padding to_data
                self.data = np.cat((np.zeros((window,),dtype=np.int8),data))
            latent_var = LatentVar((latent_lenght,latent_size))
            self.layers.append(latent_var, downsampling, window, model, opt)

    def update_layer(layer, index, extra):
        #What a fucking nightmare
        top_latent, downsampling, window, model, opt = self.layers[layer]
        if layer != 0:
            bot_latent, = self.layers[layer-1][0]
            bot_dist = latent_var[index-window:index+extra]
            bot_input = gauss_samp(bot_input)
        else:
            bot_input = self.embedding(self.data[index-window:index+extra]).cuda()



        #Upsample and prepare top layer
        length = bot_input.shape[0]
        width = top_latent.shape[1]
        offset = index % downsampling

        top_dist = top_latent[(index-window)//downsampling:(index+extra)//downsampling]
        top_input = t.zeros(length,width)
        top_input[downsampling-offset::downsampling,:] = gauss_samp(top_dist)


        #Prep input
        inputs = t.cat(top_input,bot_input,axis=1).unsqueeze(0)

        #Preform update
        opt.zero_grad()

        prediction = inputs(model)

        
        #Compute loss
        if layer != 0:
            target_dist = bot_latent[index:index+extra]
            targets = gauss_samp(target_dist)
            log_p = gauss_log_p(prediction,data)
            loss = -kl_div/extra

        if layer = 0:
            targets = LT(data[index:index+extra]).unsqueeze(0)
            log_p = -f.cross_entropy(inputs,targets,reduction=False)
            loss = -log_p/extra

        loss.backwards()

        opt.step()

    def update_latent(layer,index, extra):
        #Jesus christ why

        #TODO: Unfuck this code
        #if bottom layer
        if layer == 0:
            pass

        #if top layer
        elif layer == len(layers)-1:
            pass

        #
        else:
            top_latent, top_ds, top_window, top_model, _ = self.layers[layer+1]
            mid_latent, mid_ds, mid_window, mid_model, _ = self.layers[layer]
            bot_latent, _, _, _, _ = self.layers[layer-1]

            mid_latent.grad = True

            #Compute prior

            #Upsample and prepare top layer
            mid_dist = mid_latent[index-top_window:index+extra]
            mid_input = gauss_samp(mid_input)

            length = mid_input.shape[0]
            width = top_latent.shape[1]
            offset = index % downsampling

            top_dist = top_latent[(index-top_window)//top_ds:(index+extra)//top_ds]
            top_input = t.zeros(length,width)
            top_input[downsampling-offset::downsampling,:] = gauss_samp(top_dist)


            prior_inputs = t.cat(top_input,mid_input,axis=1).unsqueeze(0)
            prior = top_model(prior_inputs)

            #Compute log_kl term
            kl_div = gauss_kl_div(mid_dist,prior)


            #Upsample and prepare midlayer model 
            bot_dist = bot_latent[index-mid_window:index+extra*downsampling]
            bot_input = gauss_samp(bot_input)

            length = bot_input.shape[0]
            width = top_latent.shape[1]
            offset = offset

            mid_dist = mid_latent[index-bot_window//mid_ds:index+extra]
            top_input = t.zeros(length,width)
            top_input[downsampling-offset::downsampling,:] = gauss_samp(top_dist)

            #Compute log_p term

            targets  =

            #update parameters

            mid_latent.grad = True




















