import torch
import random
from tqdm import tqdm
import sys

from pytorch_neural_ca.util import generate_initial_state

def train_growing(model, target, epochs=1000, step_range=[64,96]):
    ''' Train NeuralCA model to grow into target image'''
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    loss_fn = torch.nn.MSELoss()

    with tqdm(total=epochs, file=sys.stdout) as pbar:
        for epoch in range(epochs):
            optimizer.zero_grad()
            state = generate_initial_state(model.dims[1], model.dims[0], model.channels)
            
            n_steps = random.randint(step_range[0], step_range[1])
            for step in range(n_steps):
                state = model(state)
            
            loss = loss_fn(state[:, :4, :, :], target)
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pbar.update(1)
            loss_str = str(loss.detach().numpy())[:10]
            pbar.set_postfix({'loss':loss_str})