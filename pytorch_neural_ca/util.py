import torch
import torch.nn.functional as F
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm
import sys



def generate_initial_state(width, height, channels, device=torch.device('cpu')):
    state = torch.zeros((1, channels, height, width), device=device)
    state[:, 3:, height // 2, width // 2] = 1.0
    return state

def state_to_image(state, target_size=(128,128)) -> np.ndarray:
    frame = state.detach()
    frame = F.interpolate(frame, size=target_size)
  
    frame = frame.permute(0,2,3,1)
    frame = frame[:,:,:,:4]
    frame = torch.clamp(frame, 0.0, 1.0) * 255.0
    frame = frame.cpu().numpy().astype(np.uint8)
    return frame


def render_ca_video(model, output, size=32, steps=1000, verbose=True):
    
    state = generate_initial_state(size, size, model.channels, device=model.device)
    writer = imageio.get_writer(output, fps=24)
    
    with tqdm(total=steps, disable=not verbose) as pbar, torch.no_grad():
        for step in range(steps):
            state = model(state)
            frame = state_to_image(state)[0]
            writer.append_data(frame)
            pbar.update(1)
    
    writer.close()