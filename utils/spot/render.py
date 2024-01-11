import pickle
import cv2
import skvideo.io

path="/checkpoint/jayvakil/RoboSet_code/datasets/spot_data/spot_output_2023-12-08-19-54-23.pkl"
ob = open(path, 'rb+')
obj = pickle.load(ob)
print(obj.keys())