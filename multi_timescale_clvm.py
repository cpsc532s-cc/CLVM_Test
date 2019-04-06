import numpy as np
np.cat = np.concatenate
np.random.seed(100)
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

        self.mean_m0[x] = self.mean_m0[x]*b1+(1-b1)*mean_grad
        self.mean_m1[x] = self.mean_m1[x]*b2+(1-b2)*(mean_grad**2)
        mean_m0 = self.mean_m0[x]/(1-b1)
        mean_m1 = self.mean_m1[x]/(1-b2)
        self.mean[x] -= lr*mean_m0/(np.sqrt(mean_m1)+e)

        self.log_var_m0[x] = self.log_var_m0[x]*b1+(1-b1)*log_var_grad
        self.log_var_m1[x] = self.log_var_m1[x]*b2+(1-b2)*(log_var_grad**2)
        log_var_m0 = self.log_var_m0[x]/(1-b1)
        log_var_m1 = self.log_var_m1[x]/(1-b2)
        self.log_var[x] -= lr*log_var_m0/(np.sqrt(log_var_m1)+e)

def get_top_slice(true_offset, window, index, extra, ds):
    imputed_offset = window // ds
    delta = true_offset - imputed_offset
    offset = (index-window) % ds
    start = (index+offset) // ds + delta
    l = len(range((ds-offset)%ds,extra+2*window,ds))
    stop = start + l 
    step = ds
    return slice(start,stop)

def get_bot_slice(true_offset, window, index, extra):
    start = index+true_offset-window
    stop = start + 2*window+extra+1
    return slice(start,stop)

class MTCLVMLayer(object):
    #TODO: Finish implementing
    def __init__(self, latent_var, downsampling, max_window, model, opt):
        self.max_window = max_window
        self.downsampling = downsampling 
        self.latent_var = latent_var

class MTCLVMManager(object):
    def __init__(self, data_plural, embedding, layers, update_sizes):
        self.mtclvms = []
        self.layers = layers
        self.update_sizes = update_sizes
        weights = [] 
        for i,data in enumerate(data_plural):
            weights.append(data.shape[0]) #weight updates based on data size
            self.mtclvms.append(MultiTimescaleCLVM(data, embedding, layers))
        self.weights = np.array(weights,dtype=np.float32)

    def update_model(self,model_update_prob = 0.05):
        total_weight = np.sum(self.weights)
        data_probs = self.weights / total_weight
        mtclvm_index = np.random.choice(np.arange(len(self.mtclvms)),p=data_probs)
        mtclvm = self.mtclvms[mtclvm_index]
        layer_probs = np.array([2**(-i) for i in range(len(self.layers))],dtype=np.float32)
        layer_index = np.random.choice(np.arange(len(self.layers)),p=layer_probs/np.sum(layer_probs))
        extra = self.update_sizes[layer_index]
        length =  mtclvm.layers[layer_index][1]
        index = np.random.random_integers(0, length-extra-1)
        if np.random.rand() < model_update_prob:
            mtclvm.update_layer(layer_index, index, extra)
        else:
            mtclvm.update_latent(layer_index, index, extra)

class MultiTimescaleCLVM(object):
    def __init__(self, data, embedding, layers):
        self.data = data
        self.embedding = embedding
        self.layers = []
        self.tl = 0
        curr_length = data.shape[0]
        #Generate latent variables
        for i, (downsampling, latent_size, window, model, opt) in enumerate(layers):
            curr_length = curr_length // downsampling
            if (i < len(layers)-1):
                next_window = layers[i+1][2]
                offset = max(window // downsampling, next_window)
                padding = max(window // downsampling, next_window)
                latent_length = offset + curr_length + padding
            else:
                offset = window // downsampling
                padding = window // downsampling
                latent_length = offset + curr_length + padding 

            if i == 0:
                pad = np.zeros((window,),dtype=np.int32)
                self.data = np.cat((pad, data, pad))
            latent_var = LatentVar((latent_length,latent_size),offset=window)
            self.layers.append((offset, curr_length, latent_var, downsampling, window, model, opt))

    def lp_loss(self, layer, index, extra, compute_grad=True):
        self.val_index_extra(layer, index, extra)

        top_offset, _, top_latent, ds, window, model, opt = self.layers[layer] 
        bot_index = index*ds
        bot_extra = (extra+1)*ds-1
        if compute_grad:
            top_latent.grad = True
        if layer != 0:
            bot_latent = self.layers[layer-1][2]
            bot_offset = self.layers[layer-1][0]
            sl_bot = get_bot_slice(bot_offset, window, bot_index, bot_extra)
            bot_dist = bot_latent[sl_bot]
            bot_vals = gauss_samp(bot_dist)
            bot_input = bot_vals[:-1]
        else:
            sl_bot = get_bot_slice(window, window, bot_index, bot_extra)
            bot_input = self.embedding(self.data[sl_bot.start:sl_bot.stop-1]).cuda()

        #Upsample and prepare top layer
        length = bot_input.shape[0]
        assert length == bot_extra+2*window
        width = top_latent.shape[1]

        sl_top = get_top_slice(top_offset ,window, bot_index, bot_extra, ds)
        top_dist = top_latent[sl_top]
        top_input = t.zeros(length,width).cuda()
        offset = (bot_index-window) % ds
        #TODO elimante artifical variance reduction
        top_input[(ds-offset)%ds::ds,:] = gauss_samp((top_dist[0],top_dist[1]-4))

        #Prep input
        inputs = t.cat((top_input,bot_input),1).unsqueeze(0)

        prediction = model(inputs)
        assert prediction[0].shape[1] == bot_extra+window+1
        #Compute loss
        if layer != 0:
            targets = bot_vals[window:]
            log_p = gauss_log_p(prediction,targets)
            loss = -t.sum(log_p)
        else:
            a = window+index
            b = a+bot_extra+window+1
            targets = LT(self.data[a:b]).cuda().unsqueeze(0)
            log_p = -f.cross_entropy(prediction,targets,reduction="none")
            loss = -t.sum(log_p)

        if compute_grad:
            loss.backward()
            sl_top.start
            a = window // ds
            b = a+extra+1
            top_grad = top_dist[0].grad[a:b],top_dist[1].grad[a:b]
            top_latent.grad = False 
            return loss, top_grad

        return loss
    
    def kl_loss(self, layer, index, extra, compute_grad=True):
        self.val_index_extra(layer, index, extra)

        #when using the top layer, compute kl w.r.t unit variance zero mean gaussian
        mid_offset, _, mid_latent, mid_ds, mid_window, _, _ = self.layers[layer]
        if compute_grad:
            mid_latent.grad = True

        if layer == len(self.layers)-1:
            mid_dist = mid_latent[mid_offset+index:mid_offset+index+extra+1]
            length = mid_dist[0].shape[0]
            width  = mid_dist[0].shape[1]
            prior = t.zeros(length,width).cuda(),t.zeros(length,width).cuda()
            kl_div = gauss_kl_div(mid_dist,prior)
            loss = t.sum(kl_div)

            if compute_grad:
                loss.backward()
                mid_grad = mid_dist[0].grad, mid_dist[1].grad

        else:
            top_offset, _, top_latent, top_ds, top_window, top_model, _ = self.layers[layer+1]

            #Upsample and prepare top layer
            mid_dist = mid_latent[get_bot_slice(mid_offset, top_window, index, extra)]
            mid_vals = gauss_samp(mid_dist)
            mid_input = mid_vals[:-1]

            length = mid_input.shape[0]
            assert length == extra+2*top_window
            width = top_latent.shape[1]
            offset = (index-top_window) % top_ds

            top_dist = top_latent[get_top_slice(top_offset ,top_window, index, extra, top_ds)]
            top_input = t.zeros(length, width).cuda()
            top_input[(top_ds-offset)%top_ds::top_ds,:] = gauss_samp(top_dist)

            #Compute prior
            prior_inputs = t.cat((top_input,mid_input),1).unsqueeze(0)
            prior = top_model(prior_inputs)

            #Compute log_kl term
            sub_mid_dist = mid_dist[0][top_window:],mid_dist[0][top_window:]
            kl_div = gauss_kl_div(sub_mid_dist,prior)

            loss = t.sum(kl_div)

            if compute_grad:
                loss.backward()
                a = top_window
                b = a+extra+1
                mid_grad = mid_dist[0].grad[a:b],mid_dist[1].grad[a:b]

        if compute_grad:
            return loss, mid_grad
        else:
            return loss

    def val_index_extra(self, layer, index, extra):
        if layer < 0 or layer >= len(self.layers):
            msg = "invalid layer, recived{} expected values between 0 and {}"
            raise Exception(msg.format(layer,len(self.layers)-1))

        true_offset, length, latent, _, _, _, _ = self.layers[layer]
        if index < 0:
            msg = "invalid index, recived {} index must be greater than 0"
            raise Exception(msg.format(index))
        if index + extra + 1 > length:
            msg = "invalid index and extra parameters, recived {} index + extra must be <= {}"
            raise Exception(msg.format(index+extra,length - 1))

    def update_layer(self, layer, index, extra): 
        #print("LY",layer,index,extra)
        opt = self.layers[layer][6]
        opt.zero_grad()
        loss = self.lp_loss(layer, index, extra, compute_grad=False)
        loss.backward()
        val = loss.detach().cpu().numpy()
        print(self.tl,"\t   \t",val)
        self.tl = self.tl*0.99 + 0.01*val
        opt.step()

    def update_latent(self, layer, index, extra):
        #print("LT",layer,index,extra)
        offset, _, mid_latent, mid_ds, mid_window, mid_model, _ = self.layers[layer]

        loss, mid_grad_kl = self.kl_loss(layer, index, extra)
        loss, mid_grad_lp = self.lp_loss(layer, index, extra)
        kl_grad = mid_grad_kl[0].cpu().numpy(), mid_grad_kl[1].cpu().numpy()
        lp_grad =  mid_grad_lp[0].cpu().numpy(), mid_grad_lp[1].cpu().numpy()

        #mean_grad = kl_grad[0]+lp_grad[0]
        #log_var_grad = kl_grad[1]+lp_grad[1]
        #mean_grad = kl_grad[0]
        #log_var_grad = kl_grad[1]
        mean_grad = lp_grad[0]
        log_var_grad = lp_grad[1]
        mid_latent[offset+index:offset+index+extra+1] = (mean_grad,log_var_grad)


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
        int_ch = 2
        super(BotMiniConv, self).__init__()
        self.l1 = nn.Conv1d(top_ch+bot_ch,int_ch,1,dilation=1,)
        self.l2 = nn.Conv1d(int_ch,int_ch,1,dilation=1,)
        self.dist = nn.Conv1d(int_ch,bot_out,1,dilation=1,)

    def forward(self,x):
        x = x.permute(0,2,1)
        h1 = f.relu(self.l1(x))
        h2 = f.relu(self.l2(h1))
        dist = self.dist(h2)
        #dist = dist.permute(0,2,1)
        return dist


def main():
    #text = "8501787865267716952377698607604837129164588734274390137438570245109617199589971983609906988907556575046253680228421466195655905777961449219611022508194193039358796606341136549054584764220810765696329568031236546736476748253341722371862956806941842624579522592797827605713515211614355488629591219459333575087580797555422588087985002034296322138357571638919133209077969973060273564"
    #data = np.array(text_to_indices(text))
    data = np.array(text_to_indices("850178"))
    mt = TopMiniConv(1,1).cuda()
    mm = TopMiniConv(1,1).cuda()
    mb = BotMiniConv(1,256,256).cuda()

    l3 = (2, 1, 14, mt, optim.Adam(mt.parameters(),lr=0.0001))
    l2 = (2, 1, 14, mm, optim.Adam(mm.parameters(),lr=0.0001))
    l1 = (2, 1, 1, mb, optim.Adam(mb.parameters(),lr=0.01))
    
    layers = [l1]#,l2,l3]
    embedding = lambda x: FT(np.eye(256)[x]).cuda()
    update_sizes = [1]



    #clvm = MultiTimescaleCLVM(data, embedding, layers)
    mtclvmm = MTCLVMManager([data], embedding, layers, update_sizes)
    for i in range(500000):
        mtclvmm.update_model()

if __name__ == "__main__":
    main()











