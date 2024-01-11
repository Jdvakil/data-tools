# #!/bin/bash

# for task in "/mnt/raid5/data/roboset/v0.3_1/*"
# do
#     for file in $task"/*"
#     do
#         echo $file
#         $CONDA_PREFIX/bin/python /home/jaydv/Documents/robohive/robohive/utils/paths_utils.py -p $file -u pickle2h5
#     done
# done

import os
from glob import glob
from tqdm import tqdm 
from robohive.utils.paths_utils import pickle2h5

# path = "/mnt/raid5/data/roboset/v0.3_1/"
# for task in glob(path + "/*"):
#     for pick in tqdm(glob(task + "/*")):
#         print(pick)
#         pickle2h5(rollout_path=pick, compress_path=True)

import pickle

p = "/mnt/raid5/data/roboset/v0.3_1/toaster_open/toaster_open_data_2_20221108-203856_paths_25_.h5"
obj = open(p, 'rb+')
print(pickle.load(obj))