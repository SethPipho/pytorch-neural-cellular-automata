
from pathlib import Path
import sys

import click
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from pytorch_neural_ca.model import NeuralCA
from pytorch_neural_ca.train import train_ca
from pytorch_neural_ca.util import generate_initial_state, state_to_image, render_ca_video

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@click.group()
def cli():
    pass

@cli.command()
@click.option("--target", help="path to target image")
@click.option("--output", help="directory to output model and training stats")
@click.option("--size", help="size of grid", default=64)
@click.option("--epochs", help="number of training iterations", default=30000)
@click.option("--step-range", help="min and max steps to grow ca", nargs=2, default=[64, 96], type=int)
@click.option("--batch-size", help="batch size", default=8)
@click.option("--grad-clip-val", help="max norm value for gradient clipping", default=.1)
def train(target:str, output:str, size:int, epochs:int, step_range, batch_size:int, grad_clip_val:float):

    
    width = size
    height = size
    channels = 16

    target = Image.open(target)
    target = target.resize((width, height))
    target = transforms.ToTensor()(target)
    target = torch.unsqueeze(target, 0)
    target = target.to(device)

    model = NeuralCA(channels=channels, device=device)
    train_ca(model, target, output, width=width, height=height, epochs=epochs, step_range=step_range, batch_size=batch_size, grad_clip_val=grad_clip_val)

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
    
    
  
if __name__ == "__main__":
    cli()