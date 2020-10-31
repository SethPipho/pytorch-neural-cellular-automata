import torch
import torch.nn.functional as F
import numpy as np



def generate_initial_state(width, height, channels, device=torch.device('cpu')):
    state = torch.zeros((1, channels, height, width), device=device)
    state[:, 3:, height // 2, width // 2] = 1.0
    return state

def state_to_image(state):
    frame = state.detach()
    frame = F.interpolate(frame, size=(512,512))
    frame = frame[0]
    
    frame = frame.permute(1,2,0)
    frame = frame[:,:,:4]
    frame = torch.clamp(frame, 0.0, 1.0) * 255.0
    frame = frame.cpu().numpy().astype(np.uint8)
    return frame
