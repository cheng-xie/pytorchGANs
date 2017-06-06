import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class MLPGenerator(nn.Module):
    def __init__(self, d_in, d_out):
        super(MLPGenerator, self).__init__()
        H1 = 10
        self._d_in = d_in
        self._d_out = d_out

        # noise -> hidden
        self.l1 = nn.Linear(d_in, H1)
        # hidden -> sample
        self.l2 = nn.Linear(H1, d_out)

    def forward(self, input):
        x = self.l1(input)
        x = F.elu(x)
        x = self.l2(x)
        # x = F.tanh(x)
        return x

    def input_size(self):
        return self._d_in

class MLPDiscriminator(nn.Module):
    def __init__(self, d_in):
        super(MLPDiscriminator, self).__init__()
        H1 = 10
        self.d_in = d_in

        # sample -> hidden
        self.l1 = nn.Linear(d_in, H1)
        # hidden -> probability
        self.l2 = nn.Linear(H1, 1)

    def forward(self, input):
        x = self.l1(input)
        x = F.elu(x)
        x = self.l2(x)
        x = F.sigmoid(x)
        return x

