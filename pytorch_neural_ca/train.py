import torch
import random
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path

from pytorch_neural_ca.util import generate_initial_state, state_to_image, render_ca_video, value_noise


def train_ca(model, 
             target, 
             output_dir, 
             width=64, 
             height=64, 
             batch_size=8, 
             epochs=1000,
             lr=1e-3,
             step_range=[64,96],
             pool_size=1024,
             grad_clip_val=.1 ,
             sample_every=1000):
    ''' Train regenerating NeuralCA model'''

    Path(output_dir, 'samples').mkdir(parents=True, exist_ok=True)

    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  
    target = target.repeat(batch_size, 1, 1, 1)
    seed = generate_initial_state(width, height, model.channels, device=model.device)
    pool = seed.repeat((pool_size, 1, 1, 1))

    pbar = tqdm(total=epochs, file=sys.stdout)
    
    epoch = 0
    while epoch < epochs:
        epoch += 1 
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
            noise = value_noise(dims=(height, width), batch_size=2, scale=4, device=model.device)
            mask = noise < .5
            mask[:,:, width//2, height//2] = True #avoid killing intial seed
            state[sorted_idx[1:3]] = state[sorted_idx[1:3]] * mask

          
        
        n_steps = random.randint(step_range[0], step_range[1])
        for step in range(n_steps):
            state = model(state)
        



        
        
        with torch.no_grad():
          #reset training loop is found dead state
          max_alive_cell_per_sample = torch.amax(state[:,4,:,:], (1,2))
          is_dead_state = torch.amax(max_alive_cell_per_sample) < .1
          
          if is_dead_state:
            print("Found dead state, reseting training")
            model.reset_weights()
            pool = seed.repeat((pool_size, 1, 1, 1))
            epoch = 0
            pbar.reset()
            continue



        sq_error = (state[:, :4, :, :] - target) ** 2
        loss = torch.mean(sq_error)
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

                    

