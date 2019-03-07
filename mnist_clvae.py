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

DATA_SIZE = 784

L1_SIZE = 32

L2_SIZE = 100

class TopDecoder(nn.Module):
    def __init__(self):
        super(TopDecoder, self).__init__()
        self.l1 = nn.Linear(32,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,256)
        self.mean = nn.Linear(256,100)
        self.log_var = nn.Linear(256,100)

    def forward(self,x):
        h1 = f.relu(self.l1(x))
        h2 = f.relu(self.l2(h1))
        h3 = f.relu(self.l3(h2))
        mean = self.mean(h3)
        logvar = log_rect(self.log_var(h3))
        return mean,logvar

class BottomDecoder(nn.Module):
    def __init__(self):
        super(BottomDecoder, self).__init__()
        self.l1 = nn.Linear(100,256)
        self.mean = nn.Linear(256,784)
        self.log_var = nn.Linear(256,784)

    def forward(self,x):
        h = f.relu(self.l1(x))
        mean = self.mean(h)
        logvar = log_rect(self.log_var(h))
        return mean,logvar

class CLVD(object):
    def __init__(self, data, bs, top_decoder, bottom_decoder):
        self.data = FT(data).cuda()
        self.n = data.shape[0]
        self.bs = bs

        self.td = top_decoder
        self.bd = bottom_decoder
        
        l1_zeros = np.zeros((self.n, L1_SIZE))
        l2_zeros = np.zeros((self.n, L2_SIZE))
        self.l1 = t.tensor(l1_zeros,dtype=t.float32, requires_grad=True, device="cuda"), t.tensor(l1_zeros, dtype=t.float32, requires_grad=True, device="cuda")
        self.l2 = t.tensor(l2_zeros,dtype=t.float32,requires_grad=True, device="cuda"), t.tensor(l2_zeros,dtype=t.float32,requires_grad=True, device="cuda")

        self.td_opt = opt.Adam(self.td.parameters(), lr=0.001)
        self.bd_opt = opt.Adam(self.bd.parameters(), lr=0.003)
        self.l1_opt = opt.Adam(self.l1, lr=0.1)
        self.l2_opt = opt.Adam(self.l2, lr=0.1)

    def sample_index(self):
        x = np.random.randint(0,self.n//self.bs)
        return slice(x*self.bs,(x+1)*self.bs)

    def update_l1_td(self):
        self.l1_opt.zero_grad()
        self.td_opt.zero_grad()

        index = self.sample_index()
        prior = t.zeros_like(self.l1[0][index]), t.zeros_like(self.l1[0][index])
        l1 = self.l1[0][index], self.l1[1][index]
        l2 = self.l2[0][index], self.l2[1][index]
        l1_samp = gauss_samp(l1)
        l2_samp = gauss_samp(l2)
        prediction = self.td(l1_samp)

        kl_div = mean_sum(gauss_kl_div(l1,prior))
        log_p = mean_sum(gauss_log_p(prediction,l2_samp))

        loss = kl_div-log_p
        loss.backward()

        self.l1_opt.step()
        self.td_opt.step()

        return loss

    def update_l2(self):
        self.l2_opt.zero_grad()

        index = self.sample_index()
        l1 = self.l1[0][index], self.l1[1][index]
        l2 = self.l2[0][index], self.l2[1][index]
        l1_samp = gauss_samp(l1)
        l2_samp = gauss_samp(l2)
        data = self.data[index]
        prior = self.td(l1_samp)
        prediction = self.bd(l2_samp)

        kl_div = mean_sum(gauss_kl_div(l2,prior))
        log_p = mean_sum(gauss_log_p(prediction,data))

        loss = kl_div-log_p
        loss.backward()

        self.l2_opt.step()

        return loss

    def update_bd(self):
        self.bd_opt.zero_grad()
        
        index = self.sample_index()
        l2 = self.l2[0][index], self.l2[1][index]
        l2_samp = gauss_samp(l2)
        data = self.data[index]
        prediction = self.bd(l2_samp)

        kl_div = 0
        log_p = mean_sum(gauss_log_p(prediction,data))

        loss = kl_div-log_p
        loss.backward()

        self.bd_opt.step()

        return loss

def train():
    data = np.array(load_mnist().view(-1,784), dtype=np.float32)[::50]
    data = (data-np.mean(data))/np.std(data)
    td = TopDecoder().cuda()
    #bd = BottomDecoder().cuda()
    #td = MNISTDeconvDecoder1().cuda()
    bd = MNISTDeconvDecoder2().cuda()
    clvd = CLVD(data, 256, td, bd)

    ds = DisplayStream()
    ds2 = viz.ImageSaver("./figs/run1")


    for i in range(100000):
        try:
            for _ in range(5):
                loss1 = clvd.update_l1_td()
            loss2 = clvd.update_l2()
            loss3 = clvd.update_bd()
            if i % 50 == 0:
                n_image = 4
                img = clvd.data[0:n_image].view((n_image,28,28)).cpu().numpy()
                l2_samp = gauss_samp((clvd.l2[0][0:n_image],clvd.l2[1][0:n_image]))
                rec_mean  = clvd.bd(clvd.l2[0][0:n_image])[0].view((n_image,28,28)).cpu().detach().numpy()
                rec = gauss_samp(clvd.bd(l2_samp)).view((n_image,28,28)).cpu().detach().numpy()
                images = []

                l1_prior = (clvd.l1[0][0:n_image]*0,clvd.l1[1][0:n_image]*0)
                l2_prior = clvd.td(gauss_samp(l1_prior))
                samp_dist = clvd.bd(gauss_samp(l2_prior))
                samps = gauss_samp(samp_dist).view((n_image,28,28)).cpu().detach().numpy()
                #samps = samp_dist[0].view((n_image,28,28)).cpu().detach().numpy()

                for i in range(n_image):
                    images.append(np.hstack((img[i],rec[i],rec_mean[i],samps[i])))
                out_img = np.vstack(images)
                ds.show_img(out_img)
                ds2.save_img(out_img)

                print("L1:{:10.2f}, L2:{:10.2f}, L3:{:10.2f}".format(loss1,loss2,loss3))
        except:
            break
            pass
    ds2.make_gif(50)
    

def main():
    train()

if __name__ == "__main__":
    main()





