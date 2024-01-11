#tar a directory with a string path
#tar -czvf name-of-archive.tar.gz /path/to/directory-or-file
from glob import glob
import os
from numba import cuda, jit
import numba
import tarfile
from tqdm import tqdm
# use numba to run this function in parallel 
def tar():
    path = "/mnt/raid5/data/roboset/v0.4/"
    for task in glob((path + "*/*/*")):
        if "tar.gz" not in task:
            name = task.split('/')[-1]   
            cmd = f"tar -czvf {name}.tar.gz {task}"
            print(cmd)
            os.system(cmd)

def move():
    path="/mnt/raid5/data/jaydv/robohive_base/episodes/franka-FrankaPlanarPushReal_v2d/set_17_plannar_push_eval/"
    dirname = "/mnt/raid5/data/jaydv/data"
    count = 1
    test = []
    findex = 0
    n = 455 
    for files in glob(path + "/*.h5"):
        if count < n:
            test.append(files)
        if count == n:
            tar_filename = f"Autonomous_RoboSet_Set_17_Plannar_Push_Eval_{n}_{findex}.tar.gz"
            with tarfile.open(f"{dirname}/{tar_filename}", 'w') as tar:
                # Add each file from the list to the archive
                for file_to_tar in test:
                    tar.add(file_to_tar)
            print(f'Tar archive {tar_filename} has been created.')
            count = 0
            test = []
            findex += 1
        count += 1


if __name__ == "__main__":
    move()
