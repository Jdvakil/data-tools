import tarfile
import os
import glob
from tqdm import tqdm

path = "/mnt/raid5/data/jaydv/bin_pick_data/"
count = 0
test = []
findex = 0
for files in glob.glob(path + "/*.h5"):
    if count < 1000:
        test.append(files)
    if count == 1000:
        tar_filename = f"bin_pick_data_{findex}.tar.gz"
        with tarfile.open(tar_filename, 'w') as tar:
            # Add each file from the list to the archive
            for file_to_tar in test:
                tar.add(file_to_tar)
        print(f'Tar archive {tar_filename} has been created.')
        count = 0
        test = []
        findex += 1
    count += 1
