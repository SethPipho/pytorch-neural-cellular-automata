import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import imageio
import random
import math

from pytorch_neural_ca.util import reset_conv2d

class NeuralCA(nn.Module):
    def __init__(self, channels=16, device=torch.device('cpu')):
        super(NeuralCA, self).__init__()

        self.channels = channels
        self.device = device

        self.conv1 = nn.Conv2d(self.channels * 3, 128, 1)
        self.conv2 = nn.Conv2d(128, self.channels, 1)

        sobol_x_kernel = torch.tensor([[1., 0, -1.],[2., 0, -2.],[1., 0, -1.]], device=self.device)
        sobol_y_kernel = torch.tensor([[1.,2., 1.],[ 0.,0.,0.], [-1.,-2.,-1.]],device=self.device)
        self.sobol_x_kernel = sobol_x_kernel.view(1,1,3,3).repeat(self.channels, 1, 1, 1)
        self.sobol_y_kernel = sobol_y_kernel.view(1,1,3,3).repeat(self.channels, 1, 1, 1)

        self.reset_weights()
        self.to(self.device)

    def reset_weights(self):
      reset_conv2d(self.conv1)
      reset_conv2d(self.conv2)
      self.conv2.weight.data.fill_(0.0)

    def perception(self, state):
        sobol_x = F.conv2d(state, self.sobol_x_kernel, groups=self.channels, padding=1)
        sobol_y = F.conv2d(state, self.sobol_y_kernel, groups=self.channels, padding=1)
        return torch.cat((state, sobol_x, sobol_y), 1)

    def forward(self, state):
        perception = self.perception(state)
        ds = F.relu(self.conv1(perception))
        ds = self.conv2(ds)
        ds = ds * (torch.rand((state.shape[0], 1, state.shape[2], state.shape[3]), device=self.device) < .5) #simulate async update
        alive = nn.functional.max_pool2d(state[:,3,:,:], 3, stride=1, padding=1) > .1 #alive masking
        alive = torch.unsqueeze(alive, 1)
        next_state = state + ds
        next_state = next_state * alive
        return next_state


  