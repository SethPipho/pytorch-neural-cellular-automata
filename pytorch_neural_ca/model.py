import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import imageio
import random

class NeuralCA(nn.Module):
    def __init__(self, channels=16, device=torch.device('cpu')):
        super(NeuralCA, self).__init__()

        self.channels = channels
        self.device = device

        self.conv1 = nn.Conv2d(self.channels * 3, 128, 1)
        self.conv2 = nn.Conv2d(128, self.channels, 1)
        self.conv2.weight.data.fill_(0.0)

        self.to(self.device)

    def perception(self, state):
        sobol_x_kernal = torch.tensor([[1., 0, -1.],
                                       [2., 0, -2.],
                                       [1., 0, -1.]], device=self.device)

        sobol_y_kernel = torch.tensor([[ 1.,  2.,  1.],
                                       [ 0.,  0.,  0.],
                                       [-1., -2., -1.]],device=self.device)

        sobol_x_kernal = sobol_x_kernal.view(1,1,3,3).repeat(self.channels, 1, 1, 1)
        sobol_y_kernel = sobol_y_kernel.view(1,1,3,3).repeat(self.channels, 1, 1, 1)

        sobol_x = F.conv2d(state, sobol_x_kernal, groups=self.channels, padding=1)
        sobol_y = F.conv2d(state, sobol_y_kernel, groups=self.channels, padding=1)

        return torch.cat((state, sobol_x, sobol_y), 1)

    def forward(self, state):
        perception = self.perception(state)
        ds = F.relu(self.conv1(perception))
        ds = self.conv2(ds)
        ds = ds * (torch.rand((state.shape[2], state.shape[3]), device=self.device) < .5) #simulate async update
        alive = nn.functional.max_pool2d(state[:,3,:,:], 3, stride=1, padding=1) > .1 #alive masking
        next_state = state + ds
        next_state = next_state * alive
        return next_state


  