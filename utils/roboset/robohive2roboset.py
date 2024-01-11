import h5py
from logger import Robo_logger as Trace
import os
import numpy as np
import click
import time
import glob
from tqdm import tqdm
from robohive.utils.paths_utils import render
import numba
from numba import jit, cuda
import pickle
from robohive.logger.grouped_datasets import Trace as T
import rename

def check_file_extension(file_path, extension):
    _, file_ext = os.path.splitext(file_path)
    return file_ext.lower() == extension

#Take a h5 file with robohive format and save it as roboset
#TODO: @jdvakil add config/derived for datasets 
def robohive2roboset(rollout_path, output_dir=None, max_paths=1e6):
    """
    rollout_path : path of the h5 file
    output_dir   : directory to save the new roboset h5
    max_paths    : number of rollouts to convert 
    """
    #check if file exists
    if not os.path.isfile(rollout_path):
        raise TypeError("File doesn't exist") 
    obj = h5py.File(rollout_path, "r")
    print(" > Now doing: ", rollout_path)
    #check if file is of h5 format 
    if not isinstance(obj, h5py.Group):
        raise TypeError("File type not supported")
    if 'env_infos' not in obj[list(enumerate(obj))[0][1]].keys():
        raise TypeError("Format is not RoboHive")
    if output_dir == None:
        output_dir = os.path.dirname(rollout_path)
    rollout_name = os.path.split(rollout_path)[-1]
    file_name, file_type = os.path.splitext(rollout_name)
    output_name = os.path.join(output_dir, (file_name + "_roboset"))
    trace = Trace('roboset')
    datum = {}
    derived = {}
    count = 0
    for trial, value in obj.items():
        trace.create_group(trial)
        datum = {
            "time"      : value['time'],
            "rgb_left"  : value['env_infos/visual_dict/rgb:left_cam:240x424:2d'],
            "rgb_right" : value['env_infos']['visual_dict']['rgb:right_cam:240x424:2d'],
            "rgb_top"   : value['env_infos']['visual_dict']['rgb:top_cam:240x424:2d'], 
            "rgb_wrist" : value['env_infos']['visual_dict']['rgb:Franka_wrist_cam:240x424:2d'],
            "d_left"    : value['env_infos']['visual_dict']['d:left_cam:240x424:2d'],
            "d_right"   : value['env_infos']['visual_dict']['d:right_cam:240x424:2d'],
            "d_top"     : value['env_infos']['visual_dict']['d:top_cam:240x424:2d'],
            "d_wrist"   : value['env_infos']['visual_dict']['d:Franka_wrist_cam:240x424:2d'],
            # "qp_arm"    : value['env_infos']['obs_dict']['qp_arm'],
            # "qp_ee"     : value['env_infos']['obs_dict']['qp_ee'],
            # "qv_arm"    : value['env_infos']['obs_dict']['qv_arm'],
            # "qv_ee"     : value['env_infos']['obs_dict']['qv_ee'],
            # "ctrl_arm"  : value['env_infos']['obs_dict']['ctrl_arm'],
            # "ctrl_ee"   : value['env_infos']['obs_dict']['ctrl_ee'],
        }

        if 'pos_ee' in value['env_infos/obs_dict'].keys():
            derived['pos_ee'] = value["env_infos/obs_dict/pos_ee"]
        if 'rot_ee' in value['env_infos/obs_dict'].keys():
            derived['rot_ee'] = value["env_infos/obs_dict/rot_ee"]

        trace.append_datum_post_process(group_key=trial, dataset_key='derived', dataset_val=derived)
        trace.append_datum_post_process(group_key=f'{trial}', dataset_key='data', dataset_val=datum)
        
        if 'user_cmt' in value.keys():
            for _, comment in enumerate(value['user_cmt']):
                trace.create_dataset(group_key=trial, dataset_key='config/solved', dataset_val=np.float16(comment))
        count = count + 1
        if count >= max_paths:
            break
    trace.flatten()
    trace.save(trace_name=f"{output_name}.h5")

def robohive2roboset_sim_data(rollout_path, output_dir=None, max_paths=1e6):
    """
    rollout_path : path of the h5 file
    output_dir   : directory to save the new roboset h5
    max_paths    : number of rollouts to convert 
    """
    
    #check if file exists
    if not os.path.isfile(rollout_path):
        raise TypeError("File doesn't exist")
    if check_file_extension(path, ".h5"):
        obj = h5py.File(rollout_path, "r")
    if check_file_extension(rollout_path, ".pickle"):
        with open(path, 'rb') as f:
            obj = pickle.load(f)   
    #check if file is of h5 format 
    if not isinstance(obj, h5py.Group):
        raise TypeError("File type not supported")
    if 'env_infos' not in obj[list(enumerate(obj))[0][1]].keys():
        raise TypeError("Format is not RoboHive")
    if output_dir == None:
        output_dir = os.path.dirname(rollout_path)
    rollout_name = os.path.split(rollout_path)[-1]
    file_name, file_type = os.path.splitext(rollout_name)
    output_name = os.path.join(output_dir, (file_name + "_roboset"))
    trace = Trace('roboset')
    datum = {}
    derived = {}
    count = 0
    for trial, value in obj.items():
        trace.create_group(trial)
        datum = {
            "rgb_right" :value['env_infos/visual_dict/rgb:right_cam:240x424:2d'],
            "rgb_left" :value['env_infos/visual_dict/rgb:left_cam:240x424:2d'],
            "rgb_top" :value['env_infos/visual_dict/rgb:top_cam:240x424:2d'],
            "qp_arm"    : value['env_infos']['obs_dict']['qp_arm'],
            "qp_ee"     : value['env_infos']['obs_dict']['qp_ee'],
            "qv_arm"    : value['env_infos']['obs_dict']['qv_arm'],
            "qv_ee"     : value['env_infos']['obs_dict']['qv_ee'],
            "ctrl_arm"  : value['ctrl_arm'],
            "ctrl_ee"   : value['ctrl_ee'], 
            "time"      : value["time"],
        }

        trace.append_datum_post_process(group_key=trial, dataset_key='derived', dataset_val=derived)
        trace.append_datum_post_process(group_key=f'{trial}', dataset_key='data', dataset_val=datum)
        if 'user_cmt' in value.keys():
            for _, comment in enumerate(value['user_cmt']):
                trace.create_dataset(group_key=trial, dataset_key='config/solved', dataset_val=np.float16(comment))
        count = count + 1
        if count >= max_paths:
            break
    trace.flatten()
    trace.save(trace_name=f"{output_name}.h5")

def pick_hive_2_set(rollout_path, output_dir=None, max_paths=1e6):
    """
    rollout_path : path of the h5 file
    output_dir   : directory to save the new roboset h5
    max_paths    : number of rollouts to convert 
    """
    #check if file exists
    if not os.path.isfile(rollout_path):
        raise TypeError("File doesn't exist") 
    if check_file_extension(rollout_path, ".h5"):
        obj = h5py.File(rollout_path, "r")
    if check_file_extension(rollout_path, ".pickle"):
        f = open(rollout_path, 'rb')
        obj = pickle.load(f) 
        if isinstance(obj, T):
            obj = obj[list(obj.trace.keys())[0]]
        else:
            obj = obj[list(obj.keys())[0]]
            if len(obj.keys()) == 1:
                print("object is a single trial dict")
                obj = obj['Trial0']
    print(" > Now doing: ", rollout_path)
    #check if file is of h5 format 
    # if not isinstance(obj, h5py.Group):
    #     raise TypeError("File type not supported")
    # if 'env_infos' not in obj[list(enumerate(obj))[0][1]].keys():
    #     raise TypeError("Format is not RoboHive")
    if output_dir == None:
        output_dir = os.path.dirname(rollout_path)

    rollout_name = os.path.split(rollout_path)[-1]
    file_name, file_type = os.path.splitext(rollout_name)
    output_name = os.path.join(output_dir, (file_name + "_roboset"))
    trace = Trace('roboset')
    datum = {}
    derived = {}
    count = 0
    qp_arm = []
    qp_ee = []
    qv_arm = []
    qv_ee = []
    if isinstance(obj, dict):
        trial='Trial0'
        # for trial, value in obj.items():
        trace.create_group(trial)
        datum = {
            "time"      : obj['time'],
            "rgb_left"  : obj['env_infos/visual_dict/rgb:left_cam:240x424:2d'],
            "rgb_right" : obj['env_infos/visual_dict/rgb:right_cam:240x424:2d'],
            "rgb_top"   : obj['env_infos/visual_dict/rgb:top_cam:240x424:2d'], 
            #"rgb_wrist" : obj['env_infos/visual_dict/rgb:Franka_wrist_cam:240x424:2d'],
            "d_left"    : obj['env_infos/visual_dict/d:left_cam:240x424:2d'],
            "d_right"   : obj['env_infos/visual_dict/d:right_cam:240x424:2d'],
            "d_top"     : obj['env_infos/visual_dict/d:top_cam:240x424:2d'],
            #"d_wrist"   : obj['env_infos/visual_dict/d:Franka_wrist_cam:240x424:2d'],
            "ctrl_arm"  : obj['actions']
        }
        for step in range(obj['time'].shape[0]):
            qp_arm.append(obj['env_infos/proprio_dict/qp'][step][:7])
            qp_ee.append(obj['env_infos/proprio_dict/qp'][step][7])
            qv_arm.append(obj['env_infos/proprio_dict/qv'][step][:7])
            qv_ee.append(obj['env_infos/proprio_dict/qv'][step][7])

        datum['qp_arm'] = np.asarray(qp_arm)
        datum['qp_ee'] = np.asarray(qp_ee)
        datum['qv_arm'] = np.asarray(qv_arm)
        datum['qv_ee'] = np.asarray(qv_ee)
        datum['ctrl_ee'] = np.asarray(qp_ee)

        if 'pos_ee' in obj.keys():
            derived['pos_ee'] = obj["env_infos/obs_dict/pos_ee"]
        if 'rot_ee' in obj.keys():
            derived['rot_ee'] = obj["env_infos/obs_dict/rot_ee"]

        trace.append_datum_post_process(group_key=trial, dataset_key='derived', dataset_val=derived)
        trace.append_datum_post_process(group_key=f'{trial}', dataset_key='data', dataset_val=datum)
        
        if 'user_cmt' in obj.keys():
            for _, comment in enumerate(obj['user_cmt']):
                trace.create_dataset(group_key=trial, dataset_key='config/solved', dataset_val=np.float16(comment))
        count = count + 1
        # if count >= max_paths:
        #     break
    else:
        for trial, value in obj.items():
            trace.create_group(trial)
            datum = {
                "time"      : value['time'],
                "rgb_left"  : value['env_infos/visual_dict/rgb:left_cam:240x424:2d'],
                "rgb_right" : value['env_infos/visual_dict/rgb:right_cam:240x424:2d'],
                "rgb_top"   : value['env_infos/visual_dict/rgb:top_cam:240x424:2d'], 
                #"rgb_wrist" : value['env_infos/visual_dict/rgb:Franka_wrist_cam:240x424:2d'],
                "d_left"    : value['env_infos/visual_dict/d:left_cam:240x424:2d'],
                "d_right"   : value['env_infos/visual_dict/d:right_cam:240x424:2d'],
                "d_top"     : value['env_infos/visual_dict/d:top_cam:240x424:2d'],
                #"d_wrist"   : value['env_infos/visual_dict/d:Franka_wrist_cam:240x424:2d'],
                "ctrl_arm"  : value['actions']
            }
            for step in range(value['time'].shape[0]):
                qp_arm.append(value['env_infos/proprio_dict/qp'][step][:7])
                qp_ee.append(value['env_infos/proprio_dict/qp'][step][7])
                qv_arm.append(value['env_infos/proprio_dict/qv'][step][:7])
                qv_ee.append(value['env_infos/proprio_dict/qv'][step][7])

            datum['qp_arm'] = np.asarray(qp_arm)
            datum['qp_ee'] = np.asarray(qp_ee)
            datum['qv_arm'] = np.asarray(qv_arm)
            datum['qv_ee'] = np.asarray(qv_ee)
            datum['ctrl_ee'] = np.asarray(qp_ee)

            # if 'pos_ee' in value['env_infos/obs_dict'].keys():
            #     derived['pos_ee'] = value["env_infos/obs_dict/pos_ee"]
            # if 'rot_ee' in value['env_infos/obs_dict'].keys():
            #     derived['rot_ee'] = value["env_infos/obs_dict/rot_ee"]

            trace.append_datum_post_process(group_key=trial, dataset_key='derived', dataset_val=derived)
            trace.append_datum_post_process(group_key=f'{trial}', dataset_key='data', dataset_val=datum)
            
            if 'user_cmt' in value.keys():
                for _, comment in enumerate(value['user_cmt']):
                    trace.create_dataset(group_key=trial, dataset_key='config/solved', dataset_val=np.float16(comment))
            count = count + 1
            if count >= max_paths:
                break
    trace.flatten()
    trace.save(trace_name=f"{output_name}.h5")
    time.sleep(1)
    render(rollout_path=f"{output_name}.h5", render_format="mp4")


def main_pickle(path:str, output_dir:str):
    #path = '/mnt/raid5/data/plancaster/robohive_base/demonstrations/franka-FrankaBinPickReal_v2d-better-yaw/'
    # threads_per_block = 128
    # blocks_per_grid =  1 #(input_data.shape[0] + threads_per_block - 1) // threads_per_block
    ite = glob.glob(path + "*/")
    for i in ite:
        output_path = i.replace("plancaster", "jaydv")
        foldername=f'{output_path.split("/")[-3]}_{output_path.split("/")[-2]}'
        if not os.path.exists(output_path):
            # If it doesn't exist, create it
            os.makedirs(output_path)
            print("Making dir -- ", output_path)
        for f in tqdm(glob.glob(i + "/*.pickle")):
            new_file_name = f.replace("plancaster", "jaydv").replace(".pickle", f"_roboset_{foldername}.h5")
            if "roboset" not in f:
                if not os.path.exists(new_file_name):
                    #pick_hive_2_set(rollout_path=f, output_dir=f"{path}/{path.split('/')[-2]}_roboset/")
                    pick_hive_2_set(rollout_path=f, output_dir=output_path)
                else:
                    print(" > skipping", new_file_name)

def main_normal(path, output_dir):
    for i in tqdm(glob.glob(path + "/*.h5")):
        new_file_name = i.replace("rp05", "rp05_roboset").replace(".h5", "_roboset.h5")
        if "roboset" not in i:
            if not os.path.exists(new_file_name):
                pick_hive_2_set(rollout_path=i, output_dir=output_dir)
            else:
                print(" > skipping", new_file_name)

@click.command(help="efhekfhew")
@click.option('-p','--path', type=str, help='directory path')
@click.option('-d','--data', type=str, help='p for pickle, s for sim, n for normal', default='n')
@click.option('-od','--output_dir', type=str, help='directory path')
def utils(path, data, output_dir):
    if data == 'p':
        main_pickle(path, output_dir)
    if data == 'n':
        main_normal(path, output_dir)

if __name__ == '__main__':
    utils()