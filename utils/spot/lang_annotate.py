import pickle
import skvideo.io
import numpy as np
import os

from glob import glob

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
    filepath = "/home/jaydv/Documents/data-tools/datasets/spot_data/" + filename
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
            while query != "":
                qlist.append(query)
                print(f"Query added to list. {len(qlist)} queries in list.")   
                query = input(f"Enter query for {cat}: ")
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
