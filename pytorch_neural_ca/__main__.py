
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
from pytorch_neural_ca.util import generate_initial_state, state_to_image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@click.group()
def cli():
    pass

@cli.command()
@click.option("--target", help="path to target image")
@click.option("--output", help="directory to output model and training stats")
@click.option("--size", help="size of grid", default=32)
@click.option("--epochs", help="number of training iterations", default=30000)
@click.option("--step-range", help="min and max steps to grow ca", nargs=2, default=[64, 96], type=int)
def train(target:str, output:str, size:int, epochs:int, step_range):

    Path(output).mkdir(parents=True, exist_ok=True)
    
    width = size
    height = size
    channels = 16

    target = Image.open(target)
    target = target.resize((width, height))
    target = transforms.ToTensor()(target)
    target = torch.unsqueeze(target, 0)
    target = target.to(device)

    model = NeuralCA(channels=channels, device=device)
    train_ca(model, target, output, width=width, height=height, epochs=epochs, step_range=step_range)

    torch.save(model, Path(output, 'model.pk'))


@cli.command()
@click.option("--model", help="path to model")
@click.option("--output", help="path to output video", default="demo.mp4")
@click.option("--size", help="size of grid", default=32)
@click.option("--steps", help="number of steps to run ca", default=200)
def render_video(model:str, output:str, size:int, steps:int):

    width = size
    height = size
 
    model = torch.load(model)
    model.device = device
    model.to(device)
    
    state = generate_initial_state(width, height, model.channels, device=device)
  
    writer = imageio.get_writer(output, fps=24)
    
    with tqdm(total=steps, file=sys.stdout) as pbar, torch.no_grad():
        for step in range(steps):
            state = model(state)
            frame = state_to_image(state)
            writer.append_data(frame)
            pbar.update(1)
        writer.close()
  
if __name__ == "__main__":
    cli()