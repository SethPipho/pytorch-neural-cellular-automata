
from pathlib import Path
import sys

import click
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
import cmapy

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from pytorch_neural_ca.model import NeuralCA
from pytorch_neural_ca.train import train_ca
from pytorch_neural_ca.util import generate_initial_state, state_to_image, render_test_video, alpha_over, resize_and_pad, channel_to_image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", device)

@click.group()
def cli():
    pass

    
@cli.command()
@click.option("--image", help="Path to target image")
@click.option("--output", help="Directory to output model and training data")
@click.option("--width", help="Width to resize image (preserves aspect ratio) (default: 64)", default=64)
@click.option("--padding", help="Amount to pad image edges (default: 12)", default=12)
@click.option("--batch-size", help="Batch size (default: 8)", default=8)
@click.option("--epochs", help="Number of training iterations (default: 8000)", default=8000)
@click.option("--lr", help="learning rate (default: 1e-3)", default=1e-3, type=float)
@click.option("--step-range", help="min and max steps to grow ca (default: 64 96)", nargs=2, default=[64, 96], type=int)
@click.option("--grad-clip-val", help="max norm value for gradient clipping (default: .1)", default=.1)
@click.option("--sample-every", help="Output pool samples and demo videos every n epochs (default: 1000)", default=1000)
@click.option("--channels", help="Number of hidden state channels in model (default: 12)", default=12)
def train(image:str, 
          output:str, 
          width:int, 
          padding:int, 
          epochs:int,  
          batch_size:int, 
          step_range, 
          grad_clip_val:float, 
          lr:float, 
          sample_every:int,
          channels:int):

    image = Image.open(image).convert('RGBA')
    target = resize_and_pad(image, width, padding)
    target = transforms.ToTensor()(target)
    target = target.to(device)

    model = NeuralCA(channels=channels + 4, device=device)
    train_ca(model, 
             target, 
             output, 
             epochs=epochs, 
             lr=lr,
             step_range=step_range, 
             batch_size=batch_size, 
             grad_clip_val=grad_clip_val,
             sample_every=sample_every
             )

    torch.save(model, Path(output, 'model.pk'))


@cli.command()
@click.option("--model", help="path to model")
@click.option("--output", help="path to output video", default="demo.mp4")
@click.option("--size", help="size of grid", default=32)
@click.option("--steps", help="number of steps to run ca", default=1000)
def render_video(model:str, output:str, size:int, steps:int):

    model = torch.load(model)
    model.device = device
    model.to(device)
    render_ca_video(model, output, size=size, steps=steps)

@cli.command()
@click.option("--model", help="path to model")
@click.option("--size", help="size of grid (default: 64)", default=64)
@click.option("--window-size", help="size of window (default: 256)", default=256)
@click.option("--show-channels", help="vizualize channels (default false", default=False, is_flag=True)
def demo(model:str, size:int, window_size:int, show_channels:bool):

    import cv2

    print('Left Mouse to erase')
    print('Right Click to place seed')
    print("'R' key to clear")

    window_scale = window_size / size

    model = torch.load(model, map_location=device)
    model.device = device
    model.to(device)
    model.eval()
    
    seed  = generate_initial_state(size,size, model.channels, device=device)
    state = seed
    
    mask = np.zeros((size,size), np.uint8)
    mask.fill(255)
    background = np.ones((size, size, 4), np.float)


    drawing = False 
    def mouse_callback(event,x,y,flags,param):
        nonlocal drawing, state

        x = int(x/window_scale)
        y = int(y/window_scale)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_RBUTTONUP:
            with torch.no_grad():
                state[:,:3,y,x] = 0.0
                state[:,3:,y,x] = 1.0
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.circle(mask,(x,y), 5, 0,-1)
                with torch.no_grad():
                    torch_mask = torch.from_numpy(mask).float() / 255.0
                    state = state * torch_mask
                    
                    
                    
    
    cv2.namedWindow('demo')
    cv2.setMouseCallback('demo', mouse_callback)

    if show_channels:
        cv2.namedWindow('channels')

    while cv2.getWindowProperty('demo', 0) >= 0:
        with torch.no_grad():
            state = model(state)
            
            
            

        img = state_to_image(state,target_size=(size,size))[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img = alpha_over(img, background)
        img = cv2.resize(img, (window_size, window_size))
        
        cviz_plot_size = 128
        cviz_cols = 4
        cviz_rows = model.channels // cviz_cols
        cviz_w = cviz_cols * cviz_plot_size
        cviz_h = cviz_rows * cviz_plot_size

        chan_viz = np.zeros( (cviz_h, cviz_w, 3), dtype=np.uint8)


        for i in range(model.channels):
            row = i // cviz_cols
            col = i % cviz_cols
            row_offset = row * cviz_plot_size
            col_offset = col * cviz_plot_size

            chan = channel_to_image(state, i)
            chan = cv2.resize(chan, (cviz_plot_size, cviz_plot_size))
            chan_viz[row_offset: row_offset + cviz_plot_size, col_offset: col_offset + cviz_plot_size] = chan

        cv2.imshow('demo', img)
        if show_channels:
            cv2.imshow('channels', chan_viz)
        mask.fill(255)

        key = cv2.waitKey(10)
        if key == 27:
            break
        if key == ord('r'):
            state *= 0.0
           
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    cli()