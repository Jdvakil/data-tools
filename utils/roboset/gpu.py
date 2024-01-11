import os
import h5py
import numpy as np
from numba import cuda
from logger import Robo_logger as Trace
import click
import time
import glob
from tqdm import tqdm
from robohive.utils.paths_utils import render
from numba import cuda
trace = Trace('roboset')

@cuda.jit
def pick_hive_2_set_gpu(rollout_path, output_name, max_paths):
    # check if file exists
    if not os.path.isfile(rollout_path):
        raise TypeError("File doesn't exist")

    # Open the H5 file
    obj = h5py.File(rollout_path, "r")

    # Initialize data containers
    datum = {}
    derived = {}
    count = 0
    qp_arm = []
    qp_ee = []
    qv_arm = []
    qv_ee = []

    # Iterate through trials
    for trial, value in obj.items():
        trace.create_group(trial)

        datum = {
            "time": value['time'],
            "rgb_left": value['env_infos/visual_dict/rgb:left_cam:240x424:2d'],
            "rgb_right": value['env_infos']['visual_dict']['rgb:right_cam:240x424:2d'],
            "rgb_top": value['env_infos']['visual_dict']['rgb:top_cam:240x424:2d'],
            "rgb_wrist": value['env_infos']['visual_dict']['rgb:Franka_wrist_cam:240x424:2d'],
            "d_left": value['env_infos']['visual_dict']['d:left_cam:240x424:2d'],
            "d_right": value['env_infos']['visual_dict']['d:right_cam:240x424:2d'],
            "d_top": value['env_infos']['visual_dict']['d:top_cam:240x424:2d'],
            "d_wrist": value['env_infos']['visual_dict']['d:Franka_wrist_cam:240x424:2d'],
            "ctrl_arm": value['actions']
        }

        for step in range(value['time'].shape[0]):
            qp_arm.append(value['env_infos/proprio_dict']['qp'][step][:7])
            qp_ee.append(value['env_infos/proprio_dict']['qp'][step][7])
            qv_arm.append(value['env_infos/proprio_dict']['qv'][step][:7])
            qv_ee.append(value['env_infos/proprio_dict']['qv'][step][7])

        datum['qp_arm'] = np.asarray(qp_arm)
        datum['qp_ee'] = np.asarray(qp_ee)
        datum['qv_arm'] = np.asarray(qv_arm)
        datum['qv_ee'] = np.asarray(qv_ee)

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

# Define a wrapper function for GPU execution
def pick_hive_2_set(rollout_path, output_dir=None, max_paths=1e6):
    if output_dir is None:
        output_dir = os.path.dirname(rollout_path)

    rollout_name = os.path.split(rollout_path)[-1]
    file_name, _ = os.path.splitext(rollout_name)
    output_name = os.path.join(output_dir, (file_name + "_roboset"))

    # Launch the GPU kernel
    threadsperblock = 256  # You can adjust this based on your specific problem and GPU
    blockspergrid = 1
    pick_hive_2_set_gpu[blockspergrid, threadsperblock](rollout_path, output_name, max_paths)

    # Call the render function as needed (assuming it's not GPU-accelerated)
    time.sleep(1)
    render(rollout_path=f"{output_name}.h5", render_format="mp4")

path = '/mnt/raid5/data/jaydv/autonomous_bin_pick_1420/'
for f in tqdm(glob.glob(path + "*.h5")):
    if "roboset" not in f:
        pick_hive_2_set(rollout_path=f, output_dir=f"{path}/{path.split('/')[-2]}_roboset/")
