import pickle
import cv2
import skvideo.io
import numpy as np
import os

def flatten(l):
    return[item for sublist in l for item in sublist]

path = "/checkpoint/jayvakil/RoboSet_code/datasets/spot_data/spot_output_2023-12-08-20-19-34.pkl"
ob = open(path, 'rb+')
obj = pickle.load(ob)
print(obj.keys())
rgb = obj['rgb']
steps = len(rgb)
frames=[]
breakpoint()
for i in range(steps):
    frames.append(rgb[i].numpy())
    cv2.imwrite(f"{i}.png", np.asarray(rgb[i])[:,:,::-1])


skvideo.io.vwrite("test.mp4", np.asarray(frames))

os.system(f'ffmpeg -i test.mp4 -vf "setpts=4*PTS" test_flash.gif')
os.remove(f'test.mp4')