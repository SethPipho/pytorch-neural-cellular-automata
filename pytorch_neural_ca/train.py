import torch
import random
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path

from pytorch_neural_ca.util import generate_initial_state, state_to_image, render_ca_video, value_noise


def train_ca(model, target, output_dir, width=64, height=64, pool_size=1024, batch_size=8, epochs=1000, step_range=[64,96], sample_every=1000, grad_clip_val=.1):
    ''' Train NeuralCA model to grow into target image'''

    Path(output_dir, 'samples').mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    #schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5000], gamma=0.5)

    target = target.repeat(batch_size, 1, 1, 1)
    seed = generate_initial_state(width, height, model.channels, device=model.device)
    pool = seed.repeat((pool_size, 1, 1, 1))
   
    with tqdm(total=epochs, file=sys.stdout) as pbar:
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            with torch.no_grad():
                #sample pool
                idx = (torch.rand(batch_size, device=model.device) * pool_size).long()
                state = pool[idx]

                #replace worst state with seed
                sq_error = (state[:, :4, :, :] - target) ** 2
                per_sample_loss = torch.mean(sq_error, [1,2,3])
                sorted_idx = torch.argsort(per_sample_loss)
                state[sorted_idx[-1]] = seed

                #damage
                noise = value_noise(dims=(width,height), batch_size=2, scale=4, device=model.device)
                mask = noise < .5
                mask[:,:, width//2, height//2] = True #avoid killing intial seed
                state[sorted_idx[1:3]] = state[sorted_idx[1:3]] * mask

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            optimizer.step()
            #schedule.step()

            #return new states to pool
            with torch.no_grad():
                pool[idx] = state
                
               
            pbar.update(1)
            loss_str = str(loss.detach().cpu().numpy())[:10]
            pbar.set_postfix({'loss':loss_str})

            if epoch % sample_every == 0 and epoch != 0:
                #sample pool
                with torch.no_grad():
                    idx = (torch.rand(16, device=model.device) * pool_size).long()
                    state = pool[idx]
                    imgs = state_to_image(state)
                
                #https://matplotlib.org/3.1.1/gallery/axes_grid1/simple_axesgrid.html
                fig = plt.figure(figsize=(10., 10.))
                grid = ImageGrid(fig, 111, nrows_ncols=(4, 4),  axes_pad=0.1)
                for ax, im in zip(grid, imgs):
                    ax.imshow(im)
                
                path = Path(output_dir, 'samples', 'pool-{}.png'.format(epoch))
                plt.savefig(path)

                path = Path(output_dir, 'samples', 'video-{}.mp4'.format(epoch))
                render_ca_video(model, path, width, height, verbose=False)

                    

