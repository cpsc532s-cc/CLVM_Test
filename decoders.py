import torch
from torch import nn
import numpy as np
from variational_methods import log_rect

class MNISTDeconvDecoder1(nn.Module):
    def __init__(self):
        super(MNISTDeconvDecoder1, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(8, 4, 3)
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(4, 4, 3)
        self.relu = nn.ReLU()
        self.means_conv = nn.Conv2d(4, 4, 1)
        self.vars_conv = nn.Conv2d(4, 4, 1)

    def input_size(self):
        return 8

    def output_size(self):
        return 5*5*4

    def forward(self, input_latent):
        input_latent = input_latent.view(-1,8,1,1)
        feature1 = self.relu(self.deconv1(input_latent))
        feature2 = self.relu(self.deconv2(feature1))
        means = self.means_conv(feature2)
        vars = self.vars_conv(feature2)
        return means.view(-1, self.output_size()), log_rect(vars.view(-1, self.output_size()))

class MNISTDeconvDecoder2(nn.Module):
    def __init__(self):
        super(MNISTDeconvDecoder2, self).__init__()
        self.relu = nn.ReLU()
        self.deconvs = []
        # input 5
        self.deconvs.append(nn.ConvTranspose2d(4, 4, 3, stride=2))
        # 11
        self.deconvs.append(nn.ConvTranspose2d(4, 2, 3, stride=2))
        self.deconvs.append(nn.ConvTranspose2d(2, 2, 4))
        # 25
        self.deconvs.append(nn.ConvTranspose2d(2, 2, 5))
        # 29
        self.means_conv = nn.Conv2d(2, 1, 3)
        self.vars_conv = nn.Conv2d(2, 1, 3)
        # 28

    def input_size(self):
        return 4*5*5

    def output_size(self):
        return 1*28*28

    def forward(self, input_latent):
        feature = input_latent.view(-1,4,5,5)
        for deconv in self.deconvs:
            feature = self.relu(deconv(feature))
        means = self.means_conv(feature)
        vars = self.vars_conv(feature)
        return means.view(-1, self.output_size()), log_rect(vars.view(-1, self.output_size()))

if __name__ == "__main__":
    input_t = torch.FloatTensor(np.arange(8)).view(1,8,1,1)
    output_1 = (MNISTDeconvDecoder1())(input_t)
    output_2 = (MNISTDeconvDecoder2())(output_1[0])
    print(output_2[0].size(), output_2[1].size())

