import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from models.mlpgan import MLPGenerator, MLPDiscriminator
from gan import GAN
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

def test_gaussian(mean, std, num_data):
    num_iters = 1000
    noise_size = 1
    sample_size = 1
    batch_size = 10
    # Sample some data
    data = torch.randn(num_data, sample_size) * std + mean
    data = data.resize_(num_data, sample_size)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    data_iter = iter(dataloader)
    # Construct a GAN
    gen = MLPGenerator(noise_size, sample_size)
    dis = MLPDiscriminator(sample_size)
    gan = GAN(gen, dis, data_iter)

    plt.ion()
    for ii in range(num_iters):
        gan.train(2, batch_size)
        # Sample and visualize

        x = data.numpy().reshape(num_data)
        # x = np.random.randn(20000) * std + mean
        y, bin_edges = np.histogram(x, bins=200, density=True)
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        plt.plot(bin_centers,y,'-')

        x = gan.sample_gen(20000).data.numpy()
        y, bin_edges = np.histogram(x, bins=200, density=True)
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        plt.plot(bin_centers,y,'-')

        plt.savefig('./figs/pic' + str(ii))

        plt.pause(0.5)
        plt.cla()

