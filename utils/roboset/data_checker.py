from robohive.utils.paths_utils import print_h5_schema
from robohive.utils.prompt_utils import prompt, Prompt
from glob import glob
import h5py
import click 
from tqdm import tqdm
from natsort import natsorted

def logger(text, file):
    f = open(file, 'a')
    if 'fail' in file:
        prompt(f"Writing to file: {file}", color="red", type=Prompt.INFO)
    if 'pass' in file:
        prompt(f"Writing to file: {file}", color="green", type=Prompt.INFO)
    f.write(text)
    f.close()


def check_all_data(path):
    #print("A") if a > b else print("B")
    data_keys = ['ctrl_arm', 'ctrl_ee', 'd_left', 'd_right', 'd_top', 'd_wrist', 'qp_arm', 'qp_ee', 'qv_arm', 'qv_ee', 'rgb_left', 'rgb_right', 'rgb_top', 'rgb_wrist', 'time']
    for taskname in tqdm(glob(path + "*")):
        task = taskname.split('/')[-1]
        fail = f"{task}_fail.txt"
        success = f"{task}_pass.txt"
        for i in natsorted(glob(taskname + "/*.h5")):
            logger(i + "\n" , fail)
            logger(i + "\n" , success)
            h5 = h5py.File(i, 'r')
            if len(list(h5.keys())) != 25:
                logger(f"> {i} has less than 25 trials", fail)
            for key, value in enumerate(h5):
                logger("> Data: Success\n", success) if 'data' in h5[value].keys() else logger ("> Data: Key missing\n", fail)
                logger("> Derived: Success\n", success) if 'derived' in h5[value].keys() else logger ("> Derived: Key missing\n", fail)
                logger("> Data/ctrl_ee: Success\n", success) if 'data' in h5[value].keys() else logger ("> Data: Key missing\n", fail)
                curr_keys = list(h5[value]['data'].keys())
                if len(data_keys) == len(curr_keys):
                    for keys in data_keys:
                        if h5[value]['data'][keys].shape[0] != 150:
                            logger(f"> {keys} of {i} is not of the right length -- {h5[value]['data'][keys].shape}\n", fail)
                        
                        if 'arm' in keys:
                            if h5[value]['data'][keys].shape[1] != 7:
                                logger(f"> {keys} of {i} is arm but len not correct -- {h5[value]['data'][keys].shape}\n", fail)
                        
                        if 'rgb' in keys:
                            if h5[value]['data'][keys].shape != (150, 240, 424, 3):
                                logger(f"> {keys} of {i} is rgb but len not correct -- {h5[value]['data'][keys].shape}\n", fail)
                        
                        if 'd_' in keys:
                            if h5[value]['data'][keys].shape != (150, 240, 424):
                                logger(f"> {keys} of {i} is depth but len not correct -- {h5[value]['data'][keys].shape}\n", fail)
                        
                        else:
                            logger(f"> All keys correct\n", success)
                for keys in data_keys:
                    if keys not in curr_keys:
                        logger(f"> {keys} missing from {i}\n", fail)

def check_data(path):
    #print("A") if a > b else print("B")
    data_keys = ['ctrl_arm', 'ctrl_ee', 'd_left', 'd_right', 'd_top', 'd_wrist', 'qp_arm', 'qp_ee', 'qv_arm', 'qv_ee', 'rgb_left', 'rgb_right', 'rgb_top', 'rgb_wrist', 'time']

    task = path.split('/')[-2]
    fail = f"{task}_fail.txt"
    success = f"{task}_pass.txt"
    for i in natsorted(glob(path + "*.h5")):
        logger(i + "\n" , fail)
        logger(i + "\n" , success)
        h5 = h5py.File(i, 'r')
        if len(list(h5.keys())) != 25:
            logger(f"> {i} has less than 25 trials", fail)
        for key, value in enumerate(h5):
            logger("> Data: Success\n", success) if 'data' in h5[value].keys() else logger ("> Data: Key missing\n", fail)
            logger("> Derived: Success\n", success) if 'derived' in h5[value].keys() else logger ("> Derived: Key missing\n", fail)
            logger("> Data/ctrl_ee: Success\n", success) if 'data' in h5[value].keys() else logger ("> Data: Key missing\n", fail)
            curr_keys = list(h5[value]['data'].keys())
            if len(data_keys) == len(curr_keys):
                for keys in data_keys:
                    if h5[value]['data'][keys].shape[0] != 150:
                        logger(f"> {keys} of {i} is not of the right length -- {h5[value]['data'][keys].shape}\n", fail)
                    
                    if 'arm' in keys:
                        if h5[value]['data'][keys].shape[1] != 7:
                            logger(f"> {keys} of {i} is arm but len not correct -- {h5[value]['data'][keys].shape}\n", fail)
                    
                    if 'rgb' in keys:
                        if h5[value]['data'][keys].shape != (150, 240, 424, 3):
                            logger(f"> {keys} of {i} is rgb but len not correct -- {h5[value]['data'][keys].shape}\n", fail)
                    
                    if 'd_' in keys:
                        if h5[value]['data'][keys].shape != (150, 240, 424):
                            logger(f"> {keys} of {i} is depth but len not correct -- {h5[value]['data'][keys].shape}\n", fail)
                    
                    else:
                        logger(f"> All keys correct\n", success)
            for keys in data_keys:
                if keys not in curr_keys:
                    logger(f"> {keys} missing from {i}\n", fail)

@click.command(help="efhekfhew")
@click.option('-p','--path', type=str, help='directory path')
@click.option('-d', '--dir', type=bool, help="is task or dir")
def utils(path, dir):
    if dir:
        check_all_data(path=path)
    else:
        check_data(path=path)

if __name__ == '__main__':
    utils()