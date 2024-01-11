import skvideo.io
import os
from glob import glob
import h5py
from tqdm import tqdm
import numpy as np
from robohive.utils.prompt_utils import prompt, Prompt
import click
import time
import os
import data_checker
from numba import jit, cuda, vectorize
from timeit import default_timer as timer

def flatten(l):
    return[item for sublist in l for item in sublist]

@jit(target_backend='cuda:1', forceobj=True)		
def make_all_video(path):
    for task in glob(path + "*"):
        task_name = task.split('/')[-1]
        print(task_name)
        first_frame_top = []
        total_frames = []
        frames_left = []
        
        for i in tqdm(glob(task + "*.h5"),  colour="red"):
            if "roboset" in i:
                h5 = h5py.File(i, 'r')
                t = h5['Trial0/data/time'].shape[0]
                print(i)
                for _,trials in enumerate(tqdm(h5,  colour="green")):
                    total_frames.append(np.dstack((h5[f'{trials}/data/rgb_left'], h5[f'{trials}/data/rgb_right'], h5[f'{trials}/data/rgb_top'], h5[f'{trials}/data/rgb_wrist'])))
                    first_frame_top.append(h5[f'{trials}/data/rgb_top'][0])

        frames_left = flatten(total_frames)


        prompt(f"ALL FRAMES: {len(frames_left)}", color="green", type=Prompt.INFO)
        prompt(f"ALL FRAMES: {len(first_frame_top)}", color="red", type=Prompt.INFO)

        skvideo.io.vwrite(f"{task_name}.mp4", np.asarray(frames_left))
        skvideo.io.vwrite(f"{task_name}_dist.mp4", np.asarray(first_frame_top))

        time.sleep(2)
        
        os.system(f'ffmpeg -i {task_name}.mp4 -vf "setpts=0.01*PTS" {task_name}_flash.mp4')
        os.remove(f'{task_name}.mp4')

@jit(target_backend='cuda', forceobj=True)	
def make_video(path):
    task_name = path.split('/')[-2]
    first_frame_top = []
    total_frames = []

    for i in tqdm(glob(path + "*.h5"),  colour="red"):
        h5 = h5py.File(i, 'r')
        t = h5['Trial0/data/time'].shape[0]
        print(i)
        for _,trials in enumerate(tqdm(h5,  colour="green")):
            total_frames.append(np.dstack((h5[f'{trials}/data/rgb_left'], h5[f'{trials}/data/rgb_right'], h5[f'{trials}/data/rgb_top'], h5[f'{trials}/data/rgb_wrist'])))
            first_frame_top.append(h5[f'{trials}/data/rgb_top'][0])

    frames_left = flatten(total_frames)


    prompt(f"ALL FRAMES: {len(frames_left)}", color="green", type=Prompt.INFO)
    prompt(f"ALL FRAMES: {len(first_frame_top)}", color="red", type=Prompt.INFO)

    skvideo.io.vwrite(f"{task_name}.mp4", np.asarray(frames_left))
    skvideo.io.vwrite(f"{task_name}_dist.mp4", np.asarray(first_frame_top))

    time.sleep(2)
    
    os.system(f'ffmpeg -i {task_name}.mp4 -vf "setpts=0.01*PTS" {task_name}_flash.mp4')
    os.remove(f'{task_name}.mp4')

@click.command(help="efhekfhew")
@click.option('-p','--path', type=str, help='directory path')
@click.option('-d', '--dir', type=bool, help="is task or dir")
def utils(path, dir):
    if dir:
        start = timer()
        make_all_video(path=path)
        print("Start - ", timer()-start)
    else:
        start = timer()
        make_video(path=path)
        print("Start - ", timer()-start)

if __name__ == '__main__':
    utils()