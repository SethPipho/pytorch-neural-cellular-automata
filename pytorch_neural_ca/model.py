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
    def __init__(self, dims =(32,32), channels=16):
        super(NeuralCA, self).__init__()

        self.dims = dims
        self.channels = channels

        self.conv1 = nn.Conv2d(self.channels * 3, 128, 1)
        self.conv2 = nn.Conv2d(128, self.channels, 1)

        self.conv2.weight.data.fill_(0.0)



    def perception(self, state):
        sobol_x_kernal = torch.tensor([[1., 0, -1.],
                                       [2., 0, -2.],
                                       [1., 0, -1.]])

        sobol_y_kernel = torch.tensor([[ 1.,  2.,  1.],
                                       [ 0.,  0.,  0.],
                                       [-1., -2., -1.]])

        sobol_x_kernal = sobol_x_kernal.view(1,1,3,3).repeat(self.channels, 1, 1, 1)
        sobol_y_kernel = sobol_y_kernel.view(1,1,3,3).repeat(self.channels, 1, 1, 1)

        sobol_x = F.conv2d(state, sobol_x_kernal, groups=self.channels, padding=1)
        sobol_y = F.conv2d(state, sobol_y_kernel, groups=self.channels, padding=1)

        return torch.cat((state, sobol_x, sobol_y), 1)

  
        

    def forward(self, state):
        ''' Given state, return the change in state (ds) for the next step '''
        perception = self.perception(state)
        
        ds = F.relu(self.conv1(perception))
        ds = self.conv2(ds)
        
        rand = torch.rand(self.dims) < .5
        ds = ds * rand
       
        alive = nn.functional.max_pool2d(state[:,3,:,:], 3, stride=1, padding=1) > .1
        
        
        next_state = state + ds

      
        next_state = next_state * alive
        return next_state



def plot_torch_image(t):
    plt.imshow(t.permute(1,2,0)[:,:, :4])


def initial_state(width, height, channels):
    state = torch.zeros((1, channels, height, width))
    state[:, 3:, height // 2, width // 2] = 1.0

    return state
    


def state_to_frame(state):
    frame = state.detach()
    frame = F.upsample(frame, size=(512,512))
    frame = frame[0]
    
    frame = frame.permute(1,2,0)
    frame = frame[:,:,:4]
    frame = torch.clamp(frame, 0.0, 1.0) * 255.0
    frame = frame.numpy().astype(np.uint8)
    return frame




if __name__ == "__main__":

    width = 64
    height = 64
    channels = 16
    target_file = 'emoji.png'
    epochs = 2000
    step_range = [96, 128]

    target = Image.open(target_file)
    target = target.resize((width, height))
    target = transforms.ToTensor()(target)
    target = torch.unsqueeze(target, 0)

    model = NeuralCA(dims=(height, width) ,channels=channels)
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum=.9)
    loss_fn = nn.MSELoss()


    for epoch in range(epochs):
        

        optimizer.zero_grad()
        state = initial_state(width, height, channels)
        
        for step in range(random.randint(step_range[0], step_range[1])):
            state = model(state)
           
        loss = loss_fn(state[:, :4, :, :], target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
       
        print('{}/{} loss:{}'.format(epoch, epochs, loss.detach().numpy()))

        


    state = initial_state(width, height, channels)

    plt.figure()
    plot_torch_image(state.detach()[0])

    writer = imageio.get_writer("test.mp4", fps=10)
    resize = transforms.Resize((512,512))

    for step in range(128 * 2):
        state = model(state)
        frame = state_to_frame(state)
        writer.append_data(frame)



    writer.close()
  
    plt.figure()
    plot_torch_image(state.detach()[0])
    #plt.show()


  