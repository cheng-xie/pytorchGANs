import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

class GAN:
    def __init__(self, generator, discriminator, data):
        self.gen = generator
        self.dis = discriminator
        self.data = data

        # Dimensions of samples
        self.sample_size = data.next()[0].size()
        # Dimensions of noise
        self.noise_size = generator.input_size()

        self.criterion = nn.BCELoss()

        self.gen_optim = optim.Adam(self.gen.parameters())
        self.dis_optim = optim.Adam(self.dis.parameters())

        self.dis_steps = 10
        self.gen_steps = 1

    def train(self, iters, batch_size):
        ''' Trains the GAN for some number of steps
        '''
        for _ in range(iters):
            # Train Discriminator
            for _ in range(self.dis_steps):
                self.dis.zero_grad()

                # Backpropogate on true data
                target = Variable(torch.ones(batch_size))
                data_samples = Variable(self.sample_data(batch_size))
                prediction = self.dis.forward(data_samples)
                loss_real = self.criterion(prediction, target)
                loss_real.backward()

                # Backpropogate on fake data
                target = Variable(torch.zeros(batch_size))
                data_samples = self.sample_gen(batch_size)
                # Don't propogate through generator
                data_samples = data_samples.detach()
                prediction = self.dis.forward(data_samples)
                loss_fake = self.criterion(prediction, target)
                loss_fake.backward()

                self.dis_optim.step()

                print('Disc', loss_fake.data[0], ' ', loss_real.data[0])

            # Train Generator
            for _ in range(self.gen_steps):
                self.gen.zero_grad()

                # Try to trick the Discriminator
                target = Variable(torch.ones(batch_size))
                data_samples = self.sample_gen(batch_size)
                # Do propogate through generator
                prediction = self.dis.forward(data_samples)
                loss = self.criterion(prediction, target)
                loss.backward()

                print('Gen', loss.data[0])

                self.gen_optim.step()

    def sample_gen(self, num_samples):
        ''' Draws num_samples samples from the generator
        '''
        # Generate random noise variable from Normal dist
        noise = torch.randn(num_samples, self.noise_size)
        noise = Variable(noise)
        fake_samples = self.gen.forward(noise)
        return fake_samples

    def sample_data(self, num_samples):
        ''' Draws num_samples samples from the data
        '''
        return self.data.next()

