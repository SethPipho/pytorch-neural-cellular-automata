
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
from pytorch_neural_ca.train import train_growing
from pytorch_neural_ca.util import generate_initial_state, state_to_image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@click.group()
def cli():
    pass

@cli.command()
@click.option("--target", help="path to target image")
@click.option("--output", help="directory to output model and training stats")
def train(target:str, output:str):

    Path(output).mkdir(parents=True, exist_ok=True)
    
    width = 32
    height = 32
    channels = 16

    target = Image.open(target)
    target = target.resize((width, height))
    target = transforms.ToTensor()(target)
    target = torch.unsqueeze(target, 0)

    model = NeuralCA(channels=channels, device=device)
    train_growing(model, target, width=width, height=height, epochs=5000)

    torch.save(model, Path(output, 'model.pk'))


@cli.command()
@click.option("--model", help="path to model")
@click.option("--output", help="path to output video", default="demo.mp4")
def render_video(model:str, output:str):

    width = 64
    height = 64
    steps = 200

    model = torch.load(model)
    state = generate_initial_state(width, height, model.channels)

    writer = imageio.get_writer(output, fps=24)
    
    with tqdm(total=steps, file=sys.stdout) as pbar:
        for step in range(steps):
            state = model(state)
            frame = state_to_image(state)
            writer.append_data(frame)
            pbar.update(1)
        writer.close()
  
if __name__ == "__main__":
    cli()