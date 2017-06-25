import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

class GAN:
    def __init__(self, generator, discriminator, data_load, use_gpu=True):
        self.use_gpu = use_gpu

        self.gen = generator
        self.dis = discriminator

        self.data_load = data_load

        # Dimensions of samples
        self.sample_size = iter(data_load).next()[0].size()
        # Dimensions of noise
        self.noise_size = generator.input_size()

        self.criterion = nn.BCELoss()

        if use_gpu:
            self.gen.cuda()
            self.dis.cuda()
            self.criterion.cuda()

        self.gen_optim = optim.Adam(self.gen.parameters())
        self.dis_optim = optim.Adam(self.dis.parameters())

        self.dis_steps = 30
        self.gen_steps = 1

        self.data_iter = iter(data_load)

    def train(self, iters, batch_size):
        ''' Trains the GAN for some number of steps
        '''
        self.target_ones = Variable(torch.ones(batch_size))
        self.target_zeros = Variable(torch.zeros(batch_size))
        if self.use_gpu:
            self.target_ones = self.target_ones.cuda()
            self.target_zeros = self.target_zeros.cuda()

        tot_loss_real = 0
        tot_loss_fake = 0
        tot_loss_gen = 0

        for _ in range(iters):
            # Train Discriminator
            for _ in range(self.dis_steps):
                self.dis.zero_grad()

                # Backpropogate on true data
                data_samples = Variable(self._sample_data(batch_size))
                prediction = self.dis.forward(data_samples)
                loss_real = self.criterion(prediction, self.target_ones)
                loss_real.backward()

                # Backpropogate on fake data
                data_samples = self.sample_gen(batch_size)
                # Don't propogate through generator
                data_samples = data_samples.detach()
                prediction = self.dis.forward(data_samples)
                loss_fake = self.criterion(prediction, self.target_zeros)
                loss_fake.backward()

                self.dis_optim.step()
                tot_loss_fake += loss_fake.data[0]
                tot_loss_real += loss_real.data[0]


            # Train Generator
            for _ in range(self.gen_steps):
                self.gen.zero_grad()

                # Try to trick the Discriminator
                data_samples = self.sample_gen(batch_size)
                # Do propogate through generator
                prediction = self.dis.forward(data_samples)
                loss = self.criterion(prediction, self.target_ones)
                loss.backward()
                tot_loss_gen += loss.data[0]

                self.gen_optim.step()

            print('Disc Ave Loss: Fake:', tot_loss_fake/self.dis_steps/iters, ' Real: ', tot_loss_real/self.dis_steps/iters)
            print('Gen Ave Loss: ', tot_loss_gen/self.gen_steps/iters)

    def sample_gen(self, num_samples):
        ''' Draws num_samples samples from the generator
        '''
        # Generate random noise variable from Normal dist
        noise = torch.randn(num_samples, self.noise_size)

        if self.use_gpu:
            noise = noise.cuda()

        noise = Variable(noise)
        fake_samples = self.gen.forward(noise)
        return fake_samples

    def _sample_data(self, num_samples):
        ''' Draws num_samples samples from the data
        '''
        try:
            samples = self.data_iter.next()
        except:
            # start a new iterator at the end of the episode
            self.data_iter = iter(self.data_load)
            samples = self.data_iter.next()

        if self.use_gpu:
            # copy onto the GPU
            samples = samples.cuda()

        return samples

