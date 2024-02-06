import pickle
import cv2
import skvideo.io
import numpy as np
import os

def flatten(l):
    return[item for sublist in l for item in sublist]


def render(path:str):
    ob = open(path, 'rb+')
    obj = pickle.load(ob)
    print(obj.keys())
    rgb = obj['rgb']
    steps = len(rgb)
    frames=[]
    for i in range(steps):
        frames.append(rgb[i].numpy())
        # cv2.imwrite(f"{i}.png", np.asarray(rgb[i])[:,:,::-1])


    skvideo.io.vwrite("test.mp4", np.asarray(frames))

    os.system(f'ffmpeg -i test.mp4 -vf "setpts=50*PTS" test_flash.mp4')
    os.remove(f'test.mp4')

# path = "/home/jaydv/Documents/home-robot/data/hw_exps/spot/2024-01-23-15-05-59/spot_output_2024-01-23-15-05-59.pkl"
# ob = open(path, 'rb+')
# obj = pickle.load(ob)
# print(obj.keys())
# rgb = obj['rgb']
# steps = len(rgb)
# frames=[]
# for i in range(steps):
#     frames.append(rgb[i].numpy())
#     cv2.imwrite(f"{i}.png", np.asarray(rgb[i])[:,:,::-1])


# skvideo.io.vwrite("test.mp4", np.asarray(frames))

# os.system(f'ffmpeg -i test.mp4 -vf "setpts=4*PTS" test_flash.gif')
# os.remove(f'test.mp4')