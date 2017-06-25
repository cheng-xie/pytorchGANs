import numpy as np
import sys
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from models.mlpgan import MLPGenerator, MLPDiscriminator
from gan import GAN
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import subprocess

def test_gaussian(mean, std, num_data, make_gif=False, use_gpu=True):
    num_iters = 30
    noise_size = 1
    sample_size = 1
    batch_size = 512 
    # Sample some data
    data = torch.randn(num_data, sample_size) * std + mean
    data = data.resize_(num_data, sample_size)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=use_gpu, num_workers=4, drop_last=True)
    # data_iter = iter(data_loader)
    # Construct a GAN
    gen = MLPGenerator(noise_size, sample_size)
    dis = MLPDiscriminator(sample_size)
    gan = GAN(gen, dis, data_loader, use_gpu=use_gpu)

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
        x = gan.sample_gen(20000).data.cpu().numpy()
        y, bin_edges = np.histogram(x, bins=200, density=True)
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        plt.plot(bin_centers,y,'-')

        # Discriminator Probability
        axes = plt.gca()
        x_lim = axes.get_xlim()
        x = torch.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 200).resize_(200, sample_size)
        if use_gpu:
            x = x.cuda()
        y = dis.forward(Variable(x))
        plt.plot(x.cpu().numpy(), y.data.cpu().numpy(),'-')

        if make_gif:
            plt.savefig('./figs/pic' + str(ii).zfill(3))

        plt.pause(0.01)
        plt.cla()

    if make_gif:
        subprocess.call([ 'convert', '-loop', '0', '-delay', '50', './figs/pic*.png', './figs/output.gif'])

def main(argv):
    use_gpu = False 
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--gif'  , action='store_true', help='generate gif')
    args = parser.parse_args()

    use_gpu = args.cuda
    if not torch.cuda.is_available():
        print('CUDA not detected using CPU') 
        use_gpu = False 

    make_gif = args.gif

    test_gaussian(1,2,100000, make_gif=True, use_gpu=use_gpu)

if __name__ == '__main__':
    main(sys.argv)
