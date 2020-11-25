import sys
import math

import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm
import cmapy

import torch
import torch.nn.functional as F


def channel_to_image(state, channel):
    img = state.clone().numpy()[0,channel, :, :]
    if channel == 3:
        img[img < .1] = 0.0
        img[img >= .1] = 1.0
    img = (np.arctan(img) / (3.141592 / 2) / 2 + .5)
    img = img * 255.0
    img = img.astype('uint8')
    img = cmapy.colorize(img, 'RdBu')
    return img
    

def reset_conv2d(layer):
    n = layer.in_channels
    for k in layer.kernel_size:
        n *= k
    stdv = 1. / math.sqrt(n)
    layer.weight.data.uniform_(-stdv, stdv)
    if layer.bias is not None:
        layer.bias.data.uniform_(-stdv, stdv)


def resize_and_pad(image, width, padding):
    im_w, img_h = image.size
    aspect = im_w / img_h
    height = int(width / aspect)
    image = image.resize((width - padding * 2, height - padding * 2), Image.BICUBIC)
    target = Image.new('RGBA', (width, height))
    target.paste(image, (padding, padding), image)
    return target

def generate_initial_state(width, height, channels, device=torch.device('cpu')):
    state = torch.zeros((1, channels, height, width), device=device)
    state[:, 3:, height // 2, width // 2] = 1.0
    return state

def value_noise(dims=(64, 64), batch_size=1, scale=1, mode='bicubic', device=torch.device('cpu')):
    x = torch.rand((batch_size, 1, dims[0]//scale, dims[1]//scale), device=device)
    x = F.interpolate(x, size=[dims[0], dims[1]], mode=mode, align_corners=False)
    return x

def state_to_image(state, target_size=(128,128)) -> np.ndarray:
    frame = state.detach()
    frame = F.interpolate(frame, size=target_size, mode='bicubic', align_corners=False)
  
    frame = frame.permute(0,2,3,1)
    frame = frame[:,:,:,:4]
    frame = torch.clamp(frame, 0.0, 1.0) 
    frame = frame.cpu().numpy()
    return frame


def render_test_video(model, output, width, height, steps=1000, verbose=True):
    
    state = generate_initial_state(width, height, model.channels, device=model.device)
    writer = imageio.get_writer(output, fps=24)
    
    with tqdm(total=steps, disable=not verbose) as pbar, torch.no_grad():
        for step in range(steps):

            if step % 200 == 0 and step != 0:
                #damage
                noise = value_noise(dims=(height, width), batch_size=1, scale=4, device=model.device)
                mask = noise < .25
                state = state * mask

            state = model(state)
            frame = state_to_image(state)[0]
            frame = np.uint8(frame * 255.0)
            writer.append_data(frame)
            pbar.update(1)
    
    writer.close()


def alpha_over(a, b):
    ''' Alpha composites a over b https://en.wikipedia.org/wiki/Alpha_compositing '''
    color_a = a[:,:,:3]
    color_b = b[:,:,:3]
    
    alpha_a = a[:,:,-1:]
    alpha_b = b[:,:,-1:]

    return ((color_a * alpha_a) + (color_b * alpha_b) * (1. - alpha_a))/(alpha_a + alpha_b * (1 - alpha_a))