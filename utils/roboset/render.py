import os
from glob import glob
import skvideo.io
import h5py
import numpy as np
import click
from natsort import natsorted 
def render(path):
#path = '/home/jadv/Documents/RoboSet/test/'
    for tasks in natsorted(glob(path + "*")):
        print("> Task: ", tasks)
        taskname = tasks.split('/')[-1]
        if len(taskname)>5 and not taskname == "blurredv0.3":
            f = glob(tasks + "/*.h5")[0]
            h5 = h5py.File(f)
            rgb_left = np.asarray(h5['Trial0/data/rgb_right'])
            skvideo.io.vwrite(f"videos/{taskname}.gif", rgb_left)

@click.command(help="efhekfhew")
@click.option('-p','--path', type=str, help='directory path')
def utils(path):
    render(path)

if __name__ == '__main__':
    utils()

