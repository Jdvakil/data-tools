import robohive.utils.paths_utils as pu
import glob
path = "/mnt/raid5/data/jaydv/autonomous_bin_pick/"
for data in glob.glob(path + "*.h5"):
    if "roboset" in data:
        print(data)
        pu.render(rollout_path=data, render_format="mp4", cam_names=['left'])