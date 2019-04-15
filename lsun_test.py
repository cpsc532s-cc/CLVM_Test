import numpy as np
from viz import *
import time

import models as m
from new_clvm import *
from torchvision import datasets, transforms
import torch as t

import glob
from scipy import misc
import matplotlib.pyplot as plt

from PIL import Image
from tensorboardX import SummaryWriter

writer = SummaryWriter()

G_STORE_DEVICE = t.device('cpu')
G_COMP_DEVICE = t.device('cuda:0')

class LSUNData():
    def __init__(self, bs):
        self.bs = bs
        #bedroom_train_set = datasets.LSUN(root='./data', classes=['bedroom_train'], transform=transforms.ToTensor())
        imgs = []
        for i in range(4):
            imgs += glob.glob("./data/lsun1/bedroom/0/{}/**/*.jpg".format(i), recursive=True)
        print("Loaded data set of size: {}".format(len(imgs)))
        imgs_array = np.zeros((len(imgs), 3, 64, 64),dtype = np.float32)
        for i, img_path in enumerate(imgs):
            img = Image.open(img_path)
            # Find smaller dim and crop to it
            width, height = img.size   # Get dimensions
            mindim = min(width, height)

            left = (width - mindim)//2
            top = (height - mindim)//2
            right = (width + mindim)//2
            bottom = (height + mindim)//2

            img = img.crop(box = (left, top, right, bottom))
            img = img.resize((64,64))

            img_arr = np.asarray(img, dtype=np.float32)
            imgs_array[i]=img_arr.transpose((2,0,1))

        #print(bedroom_train_set.size())
        data = imgs_array
        #data = np.array(bedroom_train_set, dtype=np.float32)[::20]
        self.fdims = data.shape[1:]
        self.mean = np.mean(data)
        self.std = np.std(data)
        self._data = (data-np.mean(data))/np.std(data)
        self.n = self._data.shape[0]

    def load_batch(self, indices, requires_grad = False, device = G_COMP_DEVICE):
        #print(self._data[indices].dtype, self._data[indices].shape)
        batch = FT(self._data[indices], device = device)
        batch.requires_grad = requires_grad
        return batch

    def sample_indices(self):
        x = np.random.randint(0,self.n//self.bs)
        return slice(x*self.bs,(x+1)*self.bs)

    def slice(self, indices):
        return self._data[indices]#.reshape((-1,28,28))

    def collate_images(self, image_batch):
        # Unconvert
        image_batch = image_batch.transpose((0,2,3,1))*self.std+self.mean
        # hstack
        image_tup = ()
        for image in image_batch:
            image_tup += (image,)
        return np.clip(np.concatenate(image_tup, axis=1)/255, 0, 1)


def main():
    data = LSUNData(64)
    opt_params={"lr":0.02, "b1":0.9, "b2":0.999, "e":1e-8}
    opt_class=AdamLatentOpt
    clvm = CLVM_Stack(data, use_kl = False)
    clvm.stack_latent(m.ResNetBlock, {"k": 5, "i_chan": 8, "h_chan": 16, "n_h": 2}, opt_class, opt_params, 0.002)
    clvm.stack_latent(m.Deconv2d, {"k": 5, "stride": 2, "i_pad":0, "i_chan": 8, "h_chan": 32}, opt_class, opt_params, 0.002)
    clvm.stack_latent(m.ResNetBlock, {"k": 5, "i_chan": 8, "h_chan": 16, "n_h": 2}, opt_class, opt_params, 0.002)
    clvm.stack_latent(m.Deconv2d, {"k": 5, "stride": 2, "i_pad":0, "i_chan": 8, "h_chan": 32}, opt_class, opt_params, 0.002)
    clvm.stack_latent(m.ResNetBlock, {"k": 5, "i_chan": 8, "h_chan": 16, "n_h": 2}, opt_class, opt_params, 0.002)
    clvm.stack_latent(m.Deconv2d, {"k": 5, "stride": 2, "i_pad":0, "i_chan": 8, "h_chan": 32}, opt_class, opt_params, 0.002)
    clvm.stack_latent(m.MLP, {"in_size": 128, "h_size": 256, "n_int":2}, opt_class, opt_params, 0.002)
    #clvm.stack_latent(m.Deconv2d, {"k": 5, "stride": 1, "i_pad":0, "i_chan": 8, "h_chan": 16}, opt_class, opt_params, 0.01)
    #clvm.stack_latent(m.Deconv2d, {"k": 5, "stride": 2, "i_pad":0, "i_chan": 8, "h_chan": 16}, opt_class, opt_params, 0.01)
    #clvm.stack_latent(m.Deconv2d, {"k": 5, "stride": 2, "i_pad":0, "i_chan": 16, "h_chan": 32}, opt_class, opt_params, 0.01)
    #clvm.stack_latent(m.MLP, {"in_size": 128, "h_size": 256, "n_int":2}, opt_class, opt_params, 0.01)
    #clvm.stack_latent(m.MLP, {"in_size": 32, "h_size": 64, "n_int":1}, opt_class, opt_params, 0.01)
    #clvm.stack_latent(m.MLP, {"in_size": 8, "h_size": 16}, opt_class, opt_params, 0.001)
    clvm.print_rep()

    ds = DisplayStream()
    for i in range(50000):
        indices = data.sample_indices()
        if i%20 == 0:
            clvm.update(indices, display = True)
            recon = clvm.reconstruct(range(5), -1)
            recon_means = recon.sample().detach().cpu().numpy().reshape((-1,3,64,64)).squeeze()
            recon_img = data.collate_images(recon_means)

            trues = data.slice(range(5)).squeeze()
            true_img = data.collate_images(trues)

            sample = clvm.sample(5)
            samples = sample.sample().detach().cpu().numpy().reshape((-1,3,64,64)).squeeze()
            sample_img = data.collate_images(samples)

            ds.show_img(np.concatenate((recon_img, true_img, sample_img),axis=0))
        else:
            clvm.update(indices)

if __name__ == "__main__":
    main()

