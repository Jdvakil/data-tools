from glob import glob
import os
from loguru import logger

def old():
    path = "/mnt/raid5/data/roboset/v0.4/baking/scene_2/baking_close_oven_scene_2"

    old_task = path.split("/")[-2][:-5]
    new_task = "flap_close_toaster_oven"

    print(path.replace(old_task, new_task))
    os.rename(path, path.replace(old_task, new_task))
    path = path.replace(old_task, new_task)
    print("New path: ", path)
    for i in glob(path + "/*.h5"):
        print(i.replace(old_task, new_task))
        os.rename(i, i.replace(old_task, new_task))

    os.system(f"tar -cvzf {new_task}.tar.gz {path}")

def new():
    skills = {
        "close_oven": "flap_close_oven",
        "open_oven": "flap_open_oven",
        "close_drawer": "slide_close_drawer",
        "open_drawer": "slide_open_drawer",
        }

    tasks = {
        "setting_table": "baking_prep",
        "making_brownies": "serve_soup",
        # "baking": "heat_soup", 
        "making_tea": "make_tea",
        "making_toast": "make_toast",
        "cleaning_up": "clean_kitchen",
        "putting_away_dishes": "stow_bowl"
    }
    path = "/mnt/raid5/data/roboset/v0.4/"
    #walk through a directory using os.walk 
    for root, dirs, files in os.walk(path):
        for d1 in (dirs):
            task_path = os.path.join(root, d1)
            for d2 in glob(task_path +"/*/*.h5"):
                # os.rename(os.path.split(d2)[0], os.path.split(d2)[0].replace(keys, tasks[keys]).replace(k1, skills[k1]))
                par_dir, task_name = os.path.split(d2)
                scene = d2.split("/")[-2]
                task = d2.split("/")[-1]
                for keys, values in tasks.items():
                    if keys in d2:
                        for k1, v1 in skills.items():
                            if keys in par_dir and keys != 'baking':
                                new_par_dir = par_dir.replace(keys, tasks[keys]).replace(k1, skills[k1])
                                logger.info(f"{par_dir} -> {new_par_dir}")
                                if os.path.exists(par_dir):
                                    breakpoint()
                                    os.rename(par_dir, new_par_dir)
                                    d2 = os.path.join(new_par_dir, task_name)
                            if k1 in d2:
                                new_name = d2.replace(keys, tasks[keys]) #.replace(k1, skills[k1])
                                if skills[k1] not in new_name:
                                    new_name = new_name.replace(k1, skills[k1])
                                logger.info(f"{d2.split('/')[-1]} ->  {new_name.split('/')[-1]}")
                                os.rename(d2, new_name)

def curr(path: str):
#glob(glob(path + "*/*")[0] + "/*")[0].replace(".mp4", f'_{glob(path + "*/*")[0].split("/")[-2]}_{glob(path + "*/*")[0].split("/")[-1]}.mp4')
    for i in glob(path + "*/*"):
        for files in glob(i + "/*.h5"):
            foldername = f'{i.split("/")[-2]}_{i.split("/")[-1]}'
            newfile = files.replace(".h5", f"_{foldername}.h5")
            logger.info(f"Renaming {files} -> {newfile} ")
            os.rename(files, newfile)

path = "/mnt/raid5/data/jaydv/robohive_base/episodes/franka-FrankaPlanarPushReal_v2d/"
curr(path=path)
