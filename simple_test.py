import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from models.mlpgan import MLPGenerator, MLPDiscriminator
from gan import GAN
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import subprocess

def test_gaussian(mean, std, num_data, make_gif=False):
    num_iters = 30
    noise_size = 1
    sample_size = 1
    batch_size = 100
    # Sample some data
    data = torch.randn(num_data, sample_size) * std + mean
    data = data.resize_(num_data, sample_size)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    # data_iter = iter(data_loader)
    # Construct a GAN
    gen = MLPGenerator(noise_size, sample_size)
    dis = MLPDiscriminator(sample_size)
    gan = GAN(gen, dis, data_loader)

    plt.ion()
    for ii in range(num_iters):
        gan.train(20, batch_size)
        # Sample and visualize

        # True Distribution
        x = data.numpy().reshape(num_data)
        # x = np.random.randn(20000) * std + mean
        y, bin_edges = np.histogram(x, bins=200, density=True)
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        plt.plot(bin_centers,y,'-')

        # Generator Approximation
        x = gan.sample_gen(20000).data.numpy()
        y, bin_edges = np.histogram(x, bins=200, density=True)
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        plt.plot(bin_centers,y,'-')

        # Discriminator Probability
        axes = plt.gca()
        print(axes.get_xlim())
        x_lim = axes.get_xlim()
        x = torch.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 200).resize_(200, sample_size)
        y = dis.forward(Variable(x))
        plt.plot(x.numpy(), y.data.numpy(),'-')

        if make_gif:
            plt.savefig('./figs/pic' + str(ii).zfill(3))

        plt.pause(0.5)
        plt.cla()

    if make_gif:
        subprocess.call([ 'convert', '-loop', '0', '-delay', '50', './figs/pic*.png', './figs/output.gif'])

