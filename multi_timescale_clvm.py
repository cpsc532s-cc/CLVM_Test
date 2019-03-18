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
        y = mean,log_var
        return y
            
    def __setitem__(self,x,grad):
        mean_grad,log_var_grad = grad  
        lr = self.params["lr"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]
        e = self.params["e"]

        self.mean_m0[x] = self.mean_m0[x]*b1-(1-b1)*mean_grad[x]
        self.mean_m1[x] = self.mean_m1[x]*b2-(1-b2)*self.mean_m1[x]
        mean_m0 = self.mean_m0[x]/(1-b1)
        mean_m1 = self.mean_m1[x]/(1-b2)
        self.mean[x] -= lr*mean_grad[x]/(np.sqrt(mean_m1)+e)

        self.log_var_m0[x] = self.log_var_m0[x]*b1-(1-b1)*log_var_grad[x]
        self.log_var_m1[x] = self.log_var_m1[x]*b2-(1-b2)*self.log_var_m1[x]
        log_var_m0 = self.log_var_m0[x]/(1-b1)
        log_var_m1 = self.log_var_m1[x]/(1-b2)
        self.log_var[x] -= self.log_var[x]-lr*log_var_grad[x]/(np.sqrt(log_var_m1)+e)

def get_top_slice(window, index, extra, ds):
    offset = (index-window) % ds
    start = (index+offset)//ds
    l = len(range((ds-offset)%ds,extra+window,ds))
    stop = start + l
    step = ds
    return slice(start,stop)


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

    def lp_loss(self, layer, index, extra, compute_grad=True):
        top_latent, ds, window, model, opt = self.layers[layer] 
        if compute_grad:
            top_latent.grad = True
        if layer != 0:
            bot_latent = self.layers[layer-1][0]
            bot_dist = bot_latent[index:index+extra+window+1]
            bot_vals = gauss_samp(bot_dist)
            bot_input = bot_vals[:-1]
        else:
            bot_input = self.embedding(self.data[index:index+extra+window]).cuda()

        #Upsample and prepare top layer
        length = bot_input.shape[0]
        width = top_latent.shape[1]

        top_dist = top_latent[get_top_slice(window, index, extra, ds)]
        top_input = t.zeros(length,width).cuda()
        offset = (index-window) % ds
        top_input[(ds-offset)%ds::ds,:] = gauss_samp(top_dist)

        #Prep input
        inputs = t.cat((top_input,bot_input),1).unsqueeze(0)

        prediction = model(inputs)
        #Compute loss
        if layer != 0:
            targets = bot_vals[-(1+extra):]
            log_p = gauss_log_p(prediction,targets)
            loss = -t.sum(log_p)

        if layer == 0:
            targets = LT(self.data[index:index+extra+1]).cuda().unsqueeze(0)
            print(targets.shape)
            print(prediction.shape)
            log_p = -f.cross_entropy(prediction,targets,reduction="none")
            loss = -t.sum(log_p)
        if compute_grad:
            loss.backward()
            top_latent.grad = False 
            return loss, top_dist

        return loss
    
    def kl_loss(self, layer, index, extra,compute_grad=True):
        top_latent, top_ds, top_window, top_model, _ = self.layers[layer+1]
        mid_latent, mid_ds, mid_window, _, _ = self.layers[layer]
        if compute_grad:
            mid_latent.grad = True

        #Upsample and prepare top layer
        mid_dist = mid_latent[index:index+extra+top_window+1]
        mid_vals = gauss_samp(mid_dist)
        mid_input = mid_vals[:-1]

        length = mid_input.shape[0]
        assert length == extra+top_window
        width = top_latent.shape[1]
        offset = (index-top_window) % top_ds

        top_dist = top_latent[get_top_slice(top_window, index, extra, top_ds)]
        top_input = t.zeros(length,width).cuda()
        top_input[(top_ds-offset)%top_ds::top_ds,:] = gauss_samp(top_dist)

        #Compute prior

        prior_inputs = t.cat((top_input,mid_input),1).unsqueeze(0)
        prior = top_model(prior_inputs)

        #Compute log_kl term
        sub_mid_dist = mid_dist[0][index:index+extra+1],mid_dist[0][index:index+extra+1]
        kl_div = gauss_kl_div(sub_mid_dist,prior)
        loss = t.sum(kl_div)

        if compute_grad:
            loss.backward()
            mid_latent.grad = False 
            return loss, mid_dist

        return loss

    def update_layer(self, layer, index, extra): 
        opt = self.layers[layer][4]
        opt.zero_grad()
        loss = mean_lp_loss(self, layer, index, extra, compute_grad=False)
        loss.backward()

        opt.step()

    def update_latent(self,layer,index, extra):
        if layer < 0 or layer >= len(self.layers):
            msg = "invalid layer, recived{} expected values between 0 and {}"
            raise Exception(msg.format(layer,len(self.layers)))

        mid_latent, mid_ds, mid_window, mid_model, _ = self.layers[layer]
        if index < 0:
            msg = "index must be greater than 0"
            raise Exception(msg)

        if index+extra+1 >= mid_latent.shape[0]:
            msg = "invalid index and extra parameters"
            raise Exception(msg)

        #if bottom layer

        #if top layer
        if layer == len(self.layers)-1:
            bot_index = index*mid_ds
            bot_extra = (extra+1)*mid_ds

            loss, mid_dist_lp = self.lp_loss(layer, bot_index, bot_extra)
            mean_grad = mid_dist_lp[0].grad.cpu().numpy()
            log_var_grad = mid_dist_lp[1].grad.cpu().numpy()
            min_len = mean_grad.shape[0]
        else:
            loss, mid_dist_kl =  self.kl_loss(layer, index, extra)

            bot_index = index*mid_ds
            bot_extra = (extra+1)*mid_ds

            loss, mid_dist_lp = self.lp_loss(layer, bot_index, bot_extra)

            kl_len = mid_dist_kl[0].shape[0]
            lp_len = mid_dist_lp[0].shape[0]
            if kl_len > lp_len:
                kl_grad = mid_dist_kl[0].grad[-lp_len:].cpu().numpy(), mid_dist_kl[1].grad[-lp_len:].cpu().numpy()
                lp_grad = mid_dist_lp[0].grad.cpu().numpy(), mid_dist_lp[1].grad.cpu().numpy()
                min_len = lp_len
            else:
                lp_grad = mid_dist_lp[0].grad[-kl_len:].cpu().numpy(), mid_dist_lp[1].grad[-kl_len:].cpu().numpy()
                kl_grad = mid_dist_kl[0].grad.cpu().numpy(), mid_dist_kl[1].grad.cpu().numpy()
                min_len = kl_len

            mean_grad = kl_grad[0]+lp_grad[0]
            log_var_grad = kl_grad[1]+lp_grad[1]

        mid_latent[index+extra+1-min_len:index+extra+1] = (mean_grad,log_var_grad)


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
    def __init__(self,top_ch,bot_ch,bot_out):
        int_ch = 15
        super(BotMiniConv, self).__init__()
        self.l1 = nn.Conv1d(top_ch+bot_ch,int_ch,5,dilation=2,)
        self.l2 = nn.Conv1d(int_ch,int_ch,2,dilation=1,)
        self.dist = nn.Conv1d(int_ch,bot_out,5,dilation=1,)

    def forward(self,x):
        x = x.permute(0,2,1)
        h1 = f.relu(self.l1(x))
        h2 = f.relu(self.l2(h1))
        dist = self.dist(h2)
        #dist = dist.permute(0,2,1)
        return dist


def main():
    data = np.array(text_to_indices("fdjkldfs.asasfj;ajafkdasfkljdfaskdafsfas;jasfd;ja;jasfd;"))
    mt = TopMiniConv(1,1).cuda()
    mm = TopMiniConv(1,1).cuda()
    mb = BotMiniConv(1,256,256).cuda()

    l3 = (2, 1, 14, mt, optim.Adam(mt.parameters(),lr=0.0001))
    l2 = (2, 1, 14, mm, optim.Adam(mm.parameters(),lr=0.0001))
    l1 = (2, 1, 14, mb, optim.Adam(mb.parameters(),lr=0.0001))
    
    layers = [l1,l2,l3]

    embedding = lambda x: FT(np.eye(256)[x]).cuda()

    clvm = MultiTimescaleCLVM(data, embedding, 1, layers)
    clvm.update_latent(0,1,0)
    #clvm.update_layer(1,3,10)

if __name__ == "__main__":
    main()











