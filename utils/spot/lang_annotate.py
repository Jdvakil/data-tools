import pickle
import cv2
import skvideo.io
import numpy as np
import os
import render

import home_robot

CATEGORIES = ["object", "attribute", "object_state", "object_localization", "spatial_reasoning", "functional_reasoning", "world_knowledge"]

dataset = "/home/jaydv/Documents/data-tools/datasets/spot_data/spot_output_2023-12-06-19-19-50.pkl"
filename = dataset.split("/")[-1].replace(".pkl", "_queries.pkl")
filepath = "/home/jaydv/Documents/data-tools/datasets/spot_data/" + filename
ob = open(dataset, 'rb+')
obj = pickle.load(ob)


render.render(dataset)

queries = {}
qlist = []

for cat in (CATEGORIES):
    print(f"Category: {cat}")
    ask = input("Do you want to annotate this category? (y/n): ")
    if ask =='y':
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