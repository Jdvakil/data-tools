from logger import Robo_logger
import h5py
import cv2 
import numpy as np
from glob import glob
from tqdm import tqdm
import shutil
import os
import skvideo.io
import click
from numba import jit, cuda, vectorize
from timeit import default_timer as timer
from natsort import natsorted

def flatten(l):
    return [item for sublist in l for item in sublist]

@jit(target_backend ="cuda", forceobj=True)
def blur(dir_path):
    task_name = os.path.basename(dir_path[:-1])
    #task_name = name.replace(name[:-5], "pick_ketchup_from_table_place_on_plate")
    if not os.path.isdir(os.path.join(os.getcwd(),task_name)):
        os.mkdir(task_name)
        print(f"made dir -- {task_name}")
    else:
        shutil.rmtree(os.path.join(os.getcwd(),task_name))
        os.mkdir(task_name)
        print(f"removed and remade dir -- {task_name}")
    file_dir = os.path.join(os.getcwd(),task_name)

    radius = 50
    intensity = 11


    # left_cereal = [215,10]
    # #left_tea = [305,30]

    # top_cereal=[830-424,30]
    # #top_tea = [805-424, 180]

    right_me = [45,25]
    #right_tea = [865-(424*2), 95]

    #Keep appending 
    # left_coords = [left_cereal] #left_tea]
    # top_coords = [top_cereal]#, top_tea] 
    right_coords = [right_me] # , right_tea]
    #wrist_coords = [wrist_butter]

    rgb_frames = []
    for path in natsorted(glob(dir_path + "*.h5")):
        name = os.path.basename(path[:-4])
        filename=os.path.join(file_dir, name + "_blurred")
        h5 = h5py.File(path, 'r')
        
        rgb_left = []
        rgb_top = []
        rgb_right = []
        rgb_wrist = []

        cams = {}
        der = {}
        conf = {}

        trace = Robo_logger(name=filename)
        for key, value in enumerate(h5):
            print(f"RoboSet:> {value}")
            trace.create_group(f"{value}")
            cams['time']      = h5[f'{value}/data/time']
            cams['d_left']    = h5[f'{value}/data/d_left']
            cams['d_right']   = h5[f'{value}/data/d_right']
            cams['d_top']     = h5[f'{value}/data/d_top']
            cams['d_wrist']   = h5[f'{value}/data/d_wrist']
            cams['qp_arm']    = h5[f'{value}/data/qp_arm']
            cams['qp_ee']     = h5[f'{value}/data/qp_ee']
            cams['qv_ee']     = h5[f'{value}/data/qv_ee']
            cams['qv_arm']    = h5[f'{value}/data/qv_arm']
            cams['ctrl_arm']  = h5[f'{value}/data/ctrl_arm']
            cams['ctrl_ee']   = h5[f'{value}/data/ctrl_ee']
            cams['rgb_left'] = h5[f'{value}/data/rgb_left']
            cams['rgb_top'] = h5[f'{value}/data/rgb_top']
            #cams['rgb_right'] = h5[f'{value}/data/rgb_right']
            cams['rgb_wrist'] = h5[f'{value}/data/rgb_wrist']
            
            if 'derived' in list(h5[value].keys()):
                derived_keys = list(h5[f'{value}/derived'].keys())
                for derived in derived_keys:
                    der[derived] = h5[f'{value}/derived/{derived}']
            # if 'config' in list(h5[value].keys()):
            #     config_keys = list(h5[f'{value}/config'].keys())
            #     for config in config_keys:
            #         conf[config] = h5[f'{value}/config/{config}']

            horizon = h5[f'{value}/data/time'].shape[0]
            for i in tqdm(range(horizon)):
                img_left = h5[f'{value}/data/rgb_left'][i]
                img_top =  h5[f'{value}/data/rgb_top'][i]
                img_right =  h5[f'{value}/data/rgb_right'][i]
                img_wrist =  h5[f'{value}/data/rgb_wrist'][i]

                blurred_img_left = cv2.GaussianBlur(img_left, (intensity, intensity), 0)
                blurred_img_top = cv2.GaussianBlur(img_top, (intensity, intensity), 0)
                blurred_img_right = cv2.GaussianBlur(img_right, (intensity, intensity), 0)
                blurred_img_wrist = cv2.GaussianBlur(img_wrist, (intensity, intensity), 0)
                
                mask_left = np.ones((240, 424, 3), dtype=np.uint8)
                mask_top = np.ones((240, 424, 3), dtype=np.uint8)
                mask_right = np.ones((240, 424, 3), dtype=np.uint8)
                mask_wrist = np.ones((240, 424, 3), dtype=np.uint8)
                
                # for l in left_coords:
                #     mask_left = cv2.circle(mask_left, (l[0],l[1]), radius, (100, 100, 100), -1)
                #     out_left = np.where(mask_left==(1, 1, 1), img_left, blurred_img_left)
                # for t in top_coords:
                #     mask_top = cv2.circle(mask_top, (t[0],t[1]), radius, (100, 100, 100), -1)
                #     out_top = np.where(mask_top==(1, 1, 1), img_top, blurred_img_top)
                for r in right_coords:
                    mask_right = cv2.circle(mask_right, (r[0],r[1]), radius, (100, 100, 100), -1)
                    out_right = np.where(mask_right==(1, 1, 1), img_right, blurred_img_right)
                # for w in wrist_coords:
                #     mask_wrist = cv2.circle(mask_wrist, (w[0],w[1]), radius, (100, 100, 100), -1)
                #     out_wrist = np.where(mask_wrist==(1, 1, 1), img_wrist, blurred_img_wrist)

                # rgb_left.append(np.asarray(out_left))
                # rgb_top.append(np.asarray(out_top))
                rgb_right.append(np.asarray(out_right))
                #rgb_wrist.append(np.asarray(out_wrist))

            # cams['rgb_left'] = np.asarray(rgb_left)
            # cams['rgb_top'] = np.asarray(rgb_top)
            cams['rgb_right'] = np.asarray(rgb_right)
            #cams['rgb_wrist'] = np.asarray(rgb_wrist) 

            rgb_frames.append(np.dstack((cams['rgb_left'], rgb_right, cams['rgb_top'], cams['rgb_wrist'])))
        
            trace.append_datum_post_process(group_key=f'{value}', dataset_key='data', dataset_val=cams)
            if 'derived' in list(h5[value].keys()):
                trace.append_datum_post_process(group_key=f'{value}', dataset_key='derived', dataset_val=der)
            # if 'config' in list(h5[value].keys()):
            #     trace.append_datum_post_process(group_key=f'{value}', dataset_key='config', dataset_val=conf)
            
            rgb_left = []
            rgb_top = []
            rgb_right = []
            rgb_wrist = []
        trace.save(f"{filename}.h5", verify_length=True)
        print(len(rgb_frames))

    filename_mp4 = os.path.join(file_dir,task_name) + "_rgb.mp4" 
    frames = flatten(rgb_frames)
    print(len(frames))
    skvideo.io.vwrite(f"{filename_mp4}", np.asarray(frames))
    os.system(f'ffmpeg -i {filename_mp4} -vf "setpts=0.045*PTS" {task_name}_flash.mp4')
    #os.remove(f"{filename_mp4}")
    print(f"Saving {filename_mp4}... size: {np.asarray(frames).shape}")

@click.command(help="efhekfhew")
@click.option('-p','--path', type=str, help='directory path')
def utils(path):
    start = timer()
    blur(path)
    print("Start - ", timer()-start)

if __name__ == '__main__':
    utils()