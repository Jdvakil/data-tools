import pickle
import skvideo.io
import numpy as np
import os
import json 
from glob import glob
from natsort import natsorted

import home_robot

CATEGORIES = [
    "object",
    "attribute",
    "object_state",
    "object_localization",
    "spatial_reasoning",
    "functional_reasoning",
    "world_knowledge"
]


def render(path : str):
    """
    Given path of spot data, render out a mp4 video of the rgb frames.
    """
    ob = open(path, 'rb+')
    obj = pickle.load(ob)
    filename = path.split("/")[-1].replace(".pkl", ".mp4")
    print(obj.keys())
    rgb = obj['rgb']
    steps = len(rgb)
    frames = []
    for i in range(steps):
        frames.append(rgb[i].numpy())

    skvideo.io.vwrite(filename, np.asarray(frames))
    newfile = filename.replace(".mp4", "_fast.mp4")
    os.system(f'ffmpeg -i {filename} -vf "setpts=50*PTS" {newfile}')
    os.remove(filename)


def annotate(path : str):
    """
    Annotation tool for SPOT data.
    """
    filename = path.split("/")[-1].replace(".pkl", "_queries.pkl")
    filepath = "/home/jaydv/Documents/data-tools/datasets/spot_data/exploration/annotated/" + filename
    ob = open(path, 'rb+')
    obj = pickle.load(ob)
    render(path)
    queries = {}
    qlist = []
    for cat in (CATEGORIES):
        print(f"Category: {cat}")
        ask = input("Do you want to annotate this category? (y/n): ")
        if ask == 'y':
            query = input(f"Enter query for {cat}: ")
            answer = input(f"Enter the ground truth answer for {cat}: ")
            while query != "":
                qlist.append((query, answer))
                print(f"Query added to list. {len(qlist)} Q/A in list.")   
                query = input(f"Enter query for {cat}: ")
                answer = input(f"Enter the ground truth answer for {cat}: ")
            print("Query cannot be empty. Skipping...")
            queries[cat] = qlist
            qlist = []
            continue
    obj['questions'] = queries
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
        print(f"Queries saved to {filepath}")
        f.close()

    ob.close()

def list_queries(path : str):
    """
    List queries for a given SPOT data file.
    """
    ob = open(path, 'rb+')
    obj = pickle.load(ob)
    print(json.dumps(obj['questions'],  sort_keys=True, indent=4))
    ob.close()

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

def main():
    """
    Main function.
    """
    path = "/home/jaydv/Documents/data-tools/datasets/spot_data/exploration/"
    files = glob(path + "*.pkl")
    for file in files:
        print(f"File: {file}")
        ann_file = file.replace("exploration/", "exploration/annotated/").replace(".pkl", "_queries.pkl")
        if not os.path.exists(ann_file):
            ask = input(f"Annotate {file}? (y/n): ")
            if ask == 'y':
                annotate(file)
            else:
                continue
        else:
            print(f"Annotations already exist for {file}. Skipping...")
            continue


if __name__ == "__main__":
    # main()
    list_queries("/home/jaydv/Documents/data-tools/datasets/spot_data/exploration/annotated/spot_output_2023-12-13-16-19-13_queries.pkl")