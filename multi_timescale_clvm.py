import numpy as np
np.cat = np.concatenate
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
from torch import FloatTensor as FT 
from torch import LongTensor as LT 

def text_to_indices(text):
    indices = []
    for i in text:
        indices.append(ord(i))
    return indices
        

def indices_to_text(text):
    text_array = []
    for i in text:
        text_array += chr(i)
    text = "".join(text_array)
    return text

def slice_tuple(x,sl):
    out = []
    for a in x:
        out.append(a[sl])
    return tuple(out)

class LatentVar(object):
    def __init__(self, dim, offset=0, optimizer="adam", params={"lr":0.1, "b1":0.9, "b2":0.999, "e":1e-8}):
        self.optimizer = optimizer
        self.params = params
        self.offset = offset
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
        mean = FT(self.mean[x]).cuda()
        log_var = FT(self.log_var[x]).cuda()
        mean.requires_grad=self.grad
        log_var.requires_grad=self.grad

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
            curr_length = curr_length // downsampling
            latent_length = curr_length + window // downsampling
            #Add zero padding to_data
            self.data = np.concatenate((np.zeros((window,),dtype=np.int8),data))
            if i == 0:
                #Add zero padding to_data
                self.data = np.cat((np.zeros((window,),dtype=np.int8),data))
            latent_var = LatentVar((latent_length,latent_size),offset=window)
            self.layers.append((latent_var, downsampling, window, model, opt))

    def update_layer(self, layer, index, extra):
        #What a fucking nightmare
        top_latent, ds, window, model, opt = self.layers[layer]
        if layer != 0:
            bot_latent = self.layers[layer-1][0]
            bot_dist = bot_latent[index:index+extra+window+1]
            bot_input = gauss_samp(bot_dist)[:-1]
        else:
            bot_input = self.embedding(self.data[index:index+extra+window]).cuda()

        #Upsample and prepare top layer
        assert bot_input.shape[0] == extra+window
        length = bot_input.shape[0]
        width = top_latent.shape[1]
        offset = (index-window) % ds

        top_dist = top_latent[(index)//ds:(index+extra+window)//ds]
        #assert  top_dist[0].shape[0] == length
        top_input = t.zeros(length,width).cuda()
        top_input[(ds-offset)%ds::ds,:] = gauss_samp(top_dist)


        #Prep input
        inputs = t.cat((top_input,bot_input),1).unsqueeze(0)

        #Preform update
        opt.zero_grad()

        prediction = model(inputs)
        
        #Compute loss
        if layer != 0:
            targets = bot_input[-extra:]
            log_p = gauss_log_p(prediction,targets)
            loss = mean_sum(-log_p)

        if layer == 0:
            targets = LT(data[index:index+extra]).unsqueeze(0)
            log_p = -f.cross_entropy(inputs,targets,reduction=False)
            loss = -log_p/extra

        loss.backwards()

        opt.step()

    def update_latent(self,layer,index, extra):
        #Jesus christ why
        if layer < 0 or layer >= len(self.layers):
            msg = "invalid layer, recived{} expected values between 0 and {}"
            raise Exception(msg.format(layer,len(self.layers)))


        #TODO: Unfuck this code
        #if bottom layer
        if layer == 0:
            pass

        #if top layer
        elif layer == len(self.layers)-1:
            pass

        else:
            top_latent, top_ds, top_window, top_model, _ = self.layers[layer+1]
            mid_latent, mid_ds, mid_window, mid_model, _ = self.layers[layer]
            bot_latent, _, _, _, _ = self.layers[layer-1]

            mid_latent.grad = True

            #Compute prior

            #Upsample and prepare top layer
            mid_dist = mid_latent[index-top_window:index+extra+1]
            mid_input = gauss_samp(mid_dist)[:-1]

            length = mid_input.shape[0]
            width = top_latent.shape[1]
            offset = (index-top_window) % top_ds

            top_dist = top_latent[(index-top_window)//top_ds:(index+extra)//top_ds]
            top_input = t.zeros(length,width)
            top_input[top_ds-offset::top_ds,:] = gauss_samp(top_dist)


            prior_inputs = t.cat(top_input,mid_input,axis=1).unsqueeze(0)
            prior = top_model(prior_inputs)

            #Compute log_kl term
            kl_div = gauss_kl_div(mid_dist[-extra:],prior)

            #Upsample and prepare midlayer model 
            bot_dist = bot_latent[index-mid_window:index+extra*mid_ds+1]
            bot_input = gauss_samp(bot_input)[:-1]

            length = bot_input.shape[0]
            width = mid_latent.shape[1]
            offset = 0

            mid_dist = mid_latent[index-bot_window:index+extra]
            top_input = t.zeros(length,width)
            mid_input[mid_ds-offset::mid_ds,:] = mid_input

            prediciton_inputs = t.cat(mid_input,bot_input,axis=1).unsqueeze(0)
            predictions = mid_model(prediciton_inputs)

            #Compute log_p term
            targets = bot_input[-extra:]
            log_p = gauss_log_p(prediction,targets)

            loss = mean_sum(kl_div - log_p)
            loss.backwards()

            #TODO: make a dist class
            mean_grad = mid_input.grad.cpu().numpy()[-extra:]
            log_var_grad = mid_input.grad.cpu().numpy()[-extra:]
            mid_latent[index:index+extra+1] = (mean_grad,log_var_grad)

            #update parameters
            mid_latent.grad = False

class TopMiniConv(nn.Module):
    def __init__(self,top_ch,bot_ch):
        int_ch = 15
        super(TopMiniConv, self).__init__()
        self.l1 = nn.Conv1d(top_ch+bot_ch,int_ch,5,dilation=2,)
        self.l2 = nn.Conv1d(int_ch,int_ch,2,dilation=1,)
        self.mean = nn.Conv1d(int_ch,bot_ch,5,dilation=1,)
        self.log_var = nn.Conv1d(int_ch,bot_ch,5,dilation=1,)

    def forward(self,x):
        x = x.permute(0,2,1)
        h1 = f.relu(self.l1(x))
        h2 = f.relu(self.l2(h1))
        mean = self.mean(h2).permute(0,2,1)
        log_var = self.log_var(h2).permute(0,2,1)
        return mean,log_var

class BotMiniConv(nn.Module):
    def __init__(self,top_ch,bot_ch):
        int_ch = 15
        super(BotMiniConv, self).__init__()
        self.l1 = nn.Conv1d(top_ch+bot_ch,int_ch,5,dilation=2,)
        self.l2 = nn.Conv1d(int_ch,int_ch,2,dilation=1,)
        self.dist = nn.Conv1d(int_ch,bot_ch,5,dilation=1,)

    def forward(self,x):
        x = x.permute(0,2,1)
        h1 = f.relu(self.l1(x))
        h2 = f.relu(self.l2(h1))
        dist = self.dist(h2)
        dist = dist.permute(0,2,1)
        return dist


def main():
    data = np.array(text_to_indices("fdjkldfs.asasfj;ajafkdasfkljdfaskdafsfas;jasfd;ja;jasfd;"))
    mt = TopMiniConv(1,1).cuda()
    mm = TopMiniConv(1,1).cuda()
    mb = BotMiniConv(1,1).cuda()

    l3 = (2, 1, 15, mt, optim.Adam(mt.parameters(),lr=0.0001))
    l2 = (2, 1, 15, mm, optim.Adam(mm.parameters(),lr=0.0001))
    l1 = (2, 1, 15, mb, optim.Adam(mb.parameters(),lr=0.0001))
    
    layers = [l1,l2,l3]

    embedding = lambda x: FT(np.eye(256)[x]).cuda()

    clvm = MultiTimescaleCLVM(data, embedding, 1, layers)
    #clvm.update_latent(1,0,10)
    clvm.update_layer(1,0,10)

if __name__ == "__main__":
    main()











