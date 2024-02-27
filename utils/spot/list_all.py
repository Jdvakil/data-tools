import pickle
import skvideo.io
import numpy as np
import os
import json 
from glob import glob

import home_robot

def list_all(path : str):
    """
    List queries for a given SPOT data file.
    """
    ob = open(path, 'rb+')
    obj = pickle.load(ob)
    print(obj)
    ob.close()

def read_pickle_and_write_to_text(input_file, output_file):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    with open(output_file, 'w') as f:
        f.write(str(data))



if __name__ == "__main__":
    read_pickle_and_write_to_text("/large_experiments/cortex/jayvakil/spot_data/exploration/annotated/spot_output_2023-12-13-16-19-13_queries.pkl", "output.txt")