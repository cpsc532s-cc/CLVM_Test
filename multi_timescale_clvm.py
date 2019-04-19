import numpy as np
import math
import pdb
np.cat = np.concatenate
np.random.seed(100)
import torch as t 
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
import os.path
from torch import FloatTensor as FT
from variational_methods import *
from gutenberg_data import *
from decoders import * 
from mnist_data import *
from viz import *
import time
import matplotlib.pyplot as plt
import torch as t
import pickle
from torch import FloatTensor as FT 
from torch import LongTensor as LT 

from scipy.io import wavfile
import numpy as np
from scipy.signal import butter, lfilter

MODEL_PATH = "model"

def text_to_indices(text):
    indices = []
    for i in text:
        indices.append(ord(i))
    return indices
        
def shitty_text_to_indices(text):
    indices = []
    for i in text:
        x = ord(i)
        if (x == 8216 or x == 8217):
                x = 39
        if (x == 8220 or x == 8221):
                x = 34
        if (x > 255):
            continue
        indices.append(x)
    return indices

def indices_to_text(indicies):
    #TODO: Make more efficient
    text_array = []
    for i in indicies:
        text_array += chr(i)
    text = "".join(text_array)
    return text

def slice_tuple(x,sl):
    out = []
    for a in x:
        out.append(a[sl])
    return tuple(out)

class LatentVar(object):
    def __init__(self, dim, offset=0, optimizer="adam", params={"lr":0.01, "b1":0.9, "b2":0.999, "e":1e-8}):
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
            
            #TODO: COme up with a less degenerate way of doing this
            self.parameters = ["mean", "log_var", "mean_m0", "log_var_m0", "mean_m1", "log_var_m1"]

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

    def save(self, name, directory):
        for i in range(len(self.parameters)):
            f = name + str(i) + ".npy"
            path = os.path.join(directory,f)
            arr = self.__dict__[self.parameters[i]]
            np.save(path,arr)


    def load(self, name, directory):
        for i in range(len(self.parameters)):
            f = name + str(i) + ".npy"
            path = os.path.join(directory,f)
            val = np.load(path)
            self.__dict__[self.parameters[i]] = val


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
        self.embedding = embedding
        weights = [] 
        for i,data in enumerate(data_plural):
            weights.append(data.shape[0]) #weight updates based on data size
            self.mtclvms.append(MultiTimescaleCLVM(data, embedding, layers))
        self.weights = np.array(weights,dtype=np.float32)

    def save(self, directory):
        for i in range(len(self.layers)):
            layer = self.layers[i]

            f = "layer_param" + str(i) + ".t"
            path = os.path.join(directory,f)
            state_dict = layer[3].state_dict()
            torch.save(state_dict,path)

            f = "layer_opt" + str(i) + ".t"
            path = os.path.join(directory,f)
            state_dict = layer[4].state_dict()
            torch.save(state_dict,path)

        for i in range(len(self.mtclvms)):
            name = "mtclvm" + str(i)
            self.mtclvms[i].save(name, directory)


    def load(self, directory):
        for i in range(len(self.layers)):
            try:
                layer = self.layers[i]

                f = "layer_param" + str(i) + ".t"
                path = os.path.join(directory,f)
                state_dict = torch.load(path)
                layer[3].load_state_dict(state_dict)


                f = "layer_opt" + str(i) + ".t"
                path = os.path.join(directory,f)
                state_dict = torch.load(path)
                layer[4].load_state_dict(state_dict)
            except:
                pass

        for i in range(len(self.mtclvms)):
            try:
                name = "mtclvm" + str(i)
                self.mtclvms[i].load(name, directory)
            except:
                pass

    def update_model(self,layer_index,latent_update_count,kl_lambda=1):
        extra = self.update_sizes[layer_index]
        total_weight = np.sum(self.weights)
        data_probs = self.weights / total_weight

        for i in range(latent_update_count):
            mtclvm_index = np.random.choice(np.arange(len(self.mtclvms)),p=data_probs)
            mtclvm = self.mtclvms[mtclvm_index]
            length =  mtclvm.layers[layer_index][1]
            index = np.random.random_integers(0, length-extra-1)
            loss2 = mtclvm.update_latent(layer_index, index, extra, kl_lambda=kl_lambda)

        mtclvm_index = np.random.choice(np.arange(len(self.mtclvms)),p=data_probs)
        mtclvm = self.mtclvms[mtclvm_index]
        length =  mtclvm.layers[layer_index][1]
        index = np.random.random_integers(0, length-extra-1)
        loss1 = mtclvm.update_layer(layer_index, index, extra)

        if latent_update_count == 0:
            return loss1
        else:
            return loss2

    def cheng_update_model(self,layer_index,latent_update_count,kl_lambda=1):
        total_weight = np.sum(self.weights)
        data_probs = self.weights / total_weight
        mtclvm_index = np.random.choice(np.arange(len(self.mtclvms)),p=data_probs)
        mtclvm = self.mtclvms[mtclvm_index]

        extra = self.update_sizes[layer_index]
        length =  mtclvm.layers[layer_index][1]
        index = np.random.random_integers(0, length-extra-1)

        for i in range(latent_update_count):
            loss2 = mtclvm.update_latent(layer_index, index, extra, kl_lambda=kl_lambda)
        loss1= mtclvm.update_layer(layer_index, index, extra)

        if latent_update_count == 0:
            return loss1
        else:
            return loss2


    def generate(self, top_len):
        #initialize top layer
        top_model = self.layers[-1][3]
        top_window = self.layers[-1][2]
        top_size = self.layers[-1][1]
        top_ds = self.layers[-1][0]
        top_offset = top_window//top_ds
        top_latent_dist = t.zeros((top_len + top_offset,top_size)),t.zeros((top_len + top_offset,top_size))
        top_latent = gauss_samp(top_latent_dist)

        #generate_intermediate layers
        for i in list(range(len(self.layers)-1))[::-1]:
            bot_model = self.layers[i][3]
            bot_window = self.layers[i][2]
            bot_size = self.layers[i][1]
            bot_ds = self.layers[i][0]

            bot_offset = max(top_window, bot_window//bot_ds)
            bot_len = top_len * top_ds 
            top_input = t.zeros((top_window+bot_len,top_size)).cuda()
            top_input[top_window % top_ds::top_ds,:] = top_latent[-(top_window // top_ds+top_len):]
            bot_latent = t.zeros((bot_len + bot_offset,bot_size)).cuda()
            bot_latent_offset_dist = t.zeros((bot_offset,bot_size)).cuda(),t.zeros((bot_offset,bot_size)).cuda()

            bot_latent[:bot_offset,:] = gauss_samp(bot_latent_offset_dist)

            bot_delta = bot_offset - top_window
            for j in range(bot_len):
                top_input_subset = top_input[j:j+top_window]
                bot_input_subset = bot_latent[bot_delta+j:bot_delta+j+top_window]

                model_input = t.cat((top_input_subset,bot_input_subset), 1).unsqueeze(0)
                model_output = top_model(model_input)
                bot_latent[j+bot_offset,:] = gauss_samp(model_output)

            top_len = bot_len
            top_model = bot_model
            top_window = bot_window
            top_offset = bot_offset
            top_ds = bot_ds
            top_latent = bot_latent
            top_size = bot_size

        #generate intermediate layers
        samp_len = top_len*top_ds
        padded_samp = t.zeros((samp_len + top_window,),dtype=t.int32)

        #TODO: Verify
        top_input = t.zeros(top_window+samp_len,top_size).cuda()
        top_input[top_window % top_ds::top_ds] = top_latent[-(top_window // top_ds+top_len):]

        for j in range(samp_len):
            top_input_subset = top_input[j:j+top_window]
            bot_input_subset = self.embedding(padded_samp[j:top_window+j])
            model_input = t.cat((top_input_subset,bot_input_subset), 1).unsqueeze(0)
            model_output = top_model(model_input)
            sample = t.multinomial(t.squeeze(f.softmax(model_output,1)),1)
            padded_samp[top_window+j] = sample

        output = padded_samp[top_window:]
        return output




class MultiTimescaleCLVM(object):
    def __init__(self, data, embedding, layers):
        self.data = data
        self.embedding = embedding
        self.layers = []
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

    def save(self, name, directory):
        for i in range(len(self.layers)):
            f = name +"layer" + str(i) 
            self.layers[i][2].save(f, directory)

    def load(self, name, directory):
        for i in range(len(self.layers)):
            f = name +"layer" + str(i) 
            self.layers[i][2].load(f, directory)

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
        top_input[(ds-offset)%ds::ds,:] = gauss_samp((top_dist[0],top_dist[1]))

        #Prep input
        inputs = t.cat((top_input,bot_input),1).unsqueeze(0)

        prediction = model(inputs)
        #print(index,prediction[0,:2,:7],prediction[0,:2,:7])
        #print(prediction[0].shape[1],bot_extra+window+1)
        assert prediction[0].shape[1] == bot_extra+window+1
        #Compute loss
        if layer != 0:
            #targets = bot_vals[window:]
            #log_p = gauss_log_p(prediction,targets)
            #loss = -t.sum(log_p)
            
            targets = bot_dist[0][window:],bot_dist[1][window:]
            exp_neg_log_p = gauss_kl_div(targets,prediction)
            loss = t.sum(exp_neg_log_p)
        else:
            a = window+bot_index
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
            sub_mid_dist = mid_dist[0][top_window:],mid_dist[1][top_window:]
            kl_div = gauss_kl_div(sub_mid_dist,prior)
            #print(t.std(sub_mid_dist[0])/t.exp(t.mean(sub_mid_dist[1])/2))

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
        opt = self.layers[layer][6]
        opt.zero_grad()
        loss = self.lp_loss(layer, index, extra, compute_grad=False)
        loss.backward()
        loss = loss.detach().cpu().numpy()
        opt.step()
        return loss
        

    def update_latent(self, layer, index, extra, kl_lambda=1):
        #print("LT",layer,index,extra)
        offset, _, mid_latent, mid_ds, mid_window, mid_model, _ = self.layers[layer]

        kl_loss, mid_grad_kl = self.kl_loss(layer, index, extra)
        lp_loss, mid_grad_lp = self.lp_loss(layer, index, extra)
        kl_grad = mid_grad_kl[0].cpu().numpy(), mid_grad_kl[1].cpu().numpy()
        lp_grad =  mid_grad_lp[0].cpu().numpy(), mid_grad_lp[1].cpu().numpy()

        mean_grad = kl_grad[0]*kl_lambda+lp_grad[0]
        log_var_grad = kl_grad[1]*kl_lambda+lp_grad[1]
        #mean_grad = kl_grad[0]
        #log_var_grad = kl_grad[1]
        #mean_grad = lp_grad[0]
        #log_var_grad = lp_grad[1]
        mid_latent[offset+index:offset+index+extra+1] = (mean_grad,log_var_grad)
        return kl_loss.detach().cpu().numpy(), lp_loss.detach().cpu().numpy()


class TopMiniConv(nn.Module):
    def __init__(self,top_ch,bot_ch,ar=True):
        self.ar = ar
        self.bot_ch = bot_ch
        int_ch = 512
        super(TopMiniConv, self).__init__()
        self.l1 = nn.Conv1d(top_ch+bot_ch,int_ch,5,dilation=2,)
        self.l2 = nn.Conv1d(int_ch,int_ch,2,dilation=1,)
        self.l3 = nn.Conv1d(int_ch,int_ch,1,dilation=1,)
        self.l4 = nn.Conv1d(int_ch,int_ch,1,dilation=1,)
        self.l5 = nn.Conv1d(int_ch,int_ch,1,dilation=1,)
        self.l6 = nn.Conv1d(int_ch,int_ch,1,dilation=1,)
        self.mean = nn.Conv1d(int_ch,bot_ch,5,dilation=1,)
        self.log_var = nn.Conv1d(int_ch,bot_ch,5,dilation=1,)

    def forward(self,x):
        if not self.ar:
            x[:,:,-self.bot_ch:] = 0
        x = x.permute(0,2,1)
        h1 = f.leaky_relu(self.l1(x))
        h2 = f.leaky_relu(self.l2(h1))
        h3 = f.leaky_relu(self.l4(h2))+h2
        h4 = f.leaky_relu(self.l4(h3))+h3
        h5 = f.leaky_relu(self.l5(h4))+h4
        h6 = f.leaky_relu(self.l6(h5))+h5
        mean = self.mean(h2).permute(0,2,1)
        log_var = self.log_var(h2).permute(0,2,1)
        #print(mean.shape)
        #print(mean)
        return mean,log_var

class BotMiniConv(nn.Module):
    def __init__(self,top_ch,bot_ch,bot_out,ar=True):
        self.ar = ar
        self.bot_ch = bot_ch
        int_ch = 512
        super(BotMiniConv, self).__init__()
        self.l1 = nn.Conv1d(top_ch+bot_ch,int_ch,4,dilation=1,)
        self.l2 = nn.Conv1d(int_ch,int_ch,4,dilation=1,)
        self.l3 = nn.Conv1d(int_ch,int_ch,4,dilation=1,)
        self.l4 = nn.Conv1d(int_ch,int_ch,1,dilation=1,)
        self.l5 = nn.Conv1d(int_ch,int_ch,1,dilation=1,)
        self.l6 = nn.Conv1d(int_ch,int_ch,1,dilation=1,)
        self.dist = nn.Conv1d(int_ch,bot_out,4,dilation=1,)

    def forward(self,x):
        if not self.ar:
            x[:,:,-self.bot_ch:] = 0
        x = x.permute(0,2,1)
        h1 = f.relu(self.l1(x))
        h2 = f.relu(self.l2(h1))
        h3 = f.relu(self.l3(h2))
        h4 = f.relu(self.l4(h3))+h3
        h5 = f.relu(self.l5(h4))+h4
        h6 = f.relu(self.l6(h5))+h5
        dist = self.dist(h6)
        #dist = dist.permute(0,2,1)
        return dist

#https://stackoverflow.com/questions/48393608/pytorch-network-parameter-calculation
def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param

def main():
    pre_text = "629256083683113848749365415435049699408567335747273975038007434615314954522374078191171949141418830801581429434637224555511728401165397825357655622115124820378852506676560199186630"
    #text = ""
    #for x in pre_text:
    #    text = "".join([text]+[x]*20)
    #text = str(text)
    #print(text)
    #text = get_moby_dick()

    #print(sorted(list(set(text))))
    #print(text_to_indices(sorted(list(set(text)))))
    books = get_texts(10)
    data = []
    total_characters = 0
    for text in books:
        if len(shitty_text_to_indices(text)) > 10000:
            data.append(np.array(shitty_text_to_indices(text)))
            total_characters += len(text)
            print(len(text))
    print("Total Characters:", total_characters)
    mt = TopMiniConv(1,2,ar=False).cuda()
    mm = TopMiniConv(2,3,ar=False).cuda()
    mb = BotMiniConv(3,256,256,ar=False).cuda()

    print("MM Parameters:", count_parameters(mm))

    print("MB Parameters:", count_parameters(mb))

    l3 = (2, 1, 14, mt, optim.Adam(mt.parameters(),lr=0.0001))
    l2 = (2, 2, 14, mm, optim.Adam(mm.parameters(),lr=0.0001))

    l1 = (2, 3, 13, mb, optim.Adam(mb.parameters(),lr=0.0001))
    
    layers = [l1,l2,l3]
    embedding = lambda x: FT(np.eye(256)[x]).cuda()
    #embedding = lambda x: FT(np.arange(256)[x,np.newaxis]).cuda()
    update_sizes = [1024,1024,1024]



    #clvm = MultiTimescaleCLVM(data, embedding, layers)
    mtclvmm = MTCLVMManager(data, embedding, layers, update_sizes)

    mtclvmm.load("model")
    losses0 = []
    losses1 = []
    losses2 = []
    losses3 = []
    for i in range(15000):
        if i < 40000:
            kl_lambda = 0.0
            #kl_lambda = math.cos(math.pi*i/15000)**2
        else:
            kl_lambda = 1

        kl_loss_0,lp_loss_0 = mtclvmm.cheng_update_model(0,5,kl_lambda=kl_lambda)
        kl_loss_1,lp_loss_1 = mtclvmm.cheng_update_model(1,5,kl_lambda=kl_lambda)
        kl_loss_2,lp_loss_2 = mtclvmm.cheng_update_model(2,3,kl_lambda=kl_lambda)


        string = "{} {:15.2f} {:15.2f} {:15.2f} {:15.2f}".format(i,lp_loss_0,kl_loss_0,kl_loss_1,kl_loss_2)
        #string = "{} {:15.2f} {:15.2f} {:15.2f}".format(i,lp_loss_0,kl_loss_0,kl_loss_1)
        #string = "{} {:15.2f}".format(i, kl_loss_1)
        #string = "{} {:15.2f} {:15.2f}".format(i,lp_loss_0,kl_loss_0)
        print(string)
        if i % 1 == 0:
            losses0.append(lp_loss_0)
            losses1.append(kl_loss_0)
            losses2.append(kl_loss_1)
            losses3.append(kl_loss_2)
            pass
    np.save("loss0.npy", np.array(losses0,dtype=np.float32))
    np.save("loss1.npy", np.array(losses1,dtype=np.float32))
    np.save("loss2.npy", np.array(losses2,dtype=np.float32))
    np.save("loss3.npy", np.array(losses3,dtype=np.float32))

    mtclvmm.save("model")


    #print("##############################################")
    #for i in range(20000):
        #mtclvmm.update_model(layer_index = 1)

    print("A")
    for i in range(5):
        sample = indices_to_text(mtclvmm.generate(500).detach().cpu().numpy())
        print(sample)
        print("######################################################")
    print("B")

if __name__ == "__main__":
    main()











