import torch
import random
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from pathlib import Path

from pytorch_neural_ca.util import generate_initial_state, state_to_image


def train_ca(model, target, output_dir, width=32, height=32, pool_size=1024, batch_size=8, epochs=1000, step_range=[64,96]):
    ''' Train NeuralCA model to grow into target image'''
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    #schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5000], gamma=0.5)

    target = target.repeat(batch_size, 1, 1, 1)
    seed = generate_initial_state(width, height, model.channels, device=model.device)
    pool = seed.repeat((pool_size, 1, 1, 1))
   
    with tqdm(total=epochs, file=sys.stdout) as pbar:
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            #sample pool
            with torch.no_grad():
                idx = (torch.rand(batch_size, device=model.device) * pool_size).long()
                state = pool[idx]
                sq_error = (state[:, :4, :, :] - target) ** 2
                per_sample_loss = torch.mean(sq_error, [1,2,3])
                worst = torch.argsort(per_sample_loss, descending=True)
                state[worst[:2]] = seed

                 #remove dead states
                dead = torch.max(torch.max(state[:,4,:,:], 2)[0], 1)[0] < .1
                dead_idx = torch.nonzero(torch.where(dead, torch.tensor(1., device=model.device), torch.tensor(0., device=model.device)))
                state[dead_idx] = seed
           
            n_steps = random.randint(step_range[0], step_range[1])
            for step in range(n_steps):
                state = model(state)
            
            sq_error = (state[:, :4, :, :] - target) ** 2
            per_sample_loss = torch.mean(sq_error, [1,2,3])
            loss = torch.mean(per_sample_loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()
            #schedule.step()

            #return new states to pool
            with torch.no_grad():
                pool[idx] = state
                
               


            

            pbar.update(1)
            loss_str = str(loss.detach().cpu().numpy())[:10]
            pbar.set_postfix({'loss':loss_str})

            if epoch % 1000 == 0:
                #sample pool
                with torch.no_grad():
                    idx = (torch.rand(16, device=model.device) * pool_size).long()
                    state = pool[idx]                
                    w=10
                    h=10
                    fig=plt.figure(figsize=(20, 20))
                    columns = 4
                    rows = 4
                    for i in range(1, columns*rows +1):
                        img = state_to_image(state[i-1:i])
                        fig.add_subplot(rows, columns, i)
                        plt.imshow(img)
                    plt.savefig(Path(output_dir, 'pool-{}.png'.format(epoch)))

                    

