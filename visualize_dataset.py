import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from PIL import Image
import os
import numpy as np

register_codecs()

def read_dataset(dataset_path):
    with zarr.ZipStore(dataset_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )
    return replay_buffer

task_name = 'pour_water' # or 'arrange_mouse' 'fold_towel' 'unplug_charger'
dataset_dir = f'data/dataset/{task_name}'
save_dir = f'data/data_vis/{task_name}'
os.makedirs(save_dir, exist_ok=True)

replay_buffer = read_dataset(os.path.join(dataset_dir, 'dataset.zarr.zip'))
with open(os.path.join(dataset_dir, 'count.txt'), 'r') as f:
    counts = f.readlines()
counts = [int(count[:-1]) if count[-1] == '\n' else int(count) for count in counts]
counts = [0] + list(np.cumsum(counts))[:-1]

episode_ends = replay_buffer.episode_ends[:]
episode_start = [0] + list(episode_ends)

save_img_idx = [episode_start[count] for count in counts]
for i, single_episode in enumerate(save_img_idx):
    episode_img = replay_buffer.data.camera0_rgb[single_episode]
    episode_img = Image.fromarray(episode_img)
    episode_img.save(os.path.join(save_dir, f'{i}.png'))
