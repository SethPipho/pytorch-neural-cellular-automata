import torch
import random
from tqdm import tqdm
import sys

from pytorch_neural_ca.util import generate_initial_state

def train_ca(model, target, width=32, height=32, pool_size=1024, batch_size=8, epochs=1000, step_range=[64,96]):
    ''' Train NeuralCA model to grow into target image'''
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    loss_fn = torch.nn.MSELoss()

    target = target.repeat(batch_size, 1, 1, 1)
    pool = generate_initial_state(width, height, model.channels, device=model.device)
    pool = pool.repeat((pool_size, 1, 1, 1))
   
    with tqdm(total=epochs, file=sys.stdout) as pbar:
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            #sample pool
            with torch.no_grad():
                idx = (torch.rand(batch_size, device=device) * pool_size).long()
                state = pool[idx]

           
            n_steps = random.randint(step_range[0], step_range[1])
            for step in range(n_steps):
                state = model(state)
            
            sq_error = (state[:, :4, :, :] - target) ** 2
            per_sample_loss = torch.mean(sq_error, [1,2,3])
            loss = torch.mean(per_sample_loss)
            loss.backward()

            #return new states to pool
            with torch.no_grad():
                pool[idx] = state
                worst = torch.argsort(per_sample_loss, descending=True)[0]
                pool[worst] = generate_initial_state(width, height, model.channels, device=model.device)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pbar.update(1)
            loss_str = str(loss.detach().cpu().numpy())[:10]
            pbar.set_postfix({'loss':loss_str})