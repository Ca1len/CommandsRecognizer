import os 
import numpy as np
import torch
import json
import shutil
import time


LOAD_DELAY = 60


def normalize(array):
    mn = np.min(array)
    mx = np.max(array)
    return (array - mn) / (mx - mn)


def standartize(array):
    std = np.std(array)
    mean = np.mean(array)
    return (array - mean) / std


def read_file(file):
    l = []
    with open(file, 'r') as file:
        for line in file.readlines():
            l.append(int(line))
        
    return standartize(np.array(l, dtype=np.float32))


def read_txt_data(dir_path):

    files = os.listdir(dir_path)

    data = {}

    for file in files:
        p = os.path.join(dir_path, file)
        with open(p, 'r') as read_f:
            key = file[:-4]
            data[key] = []
            for line in read_f.readlines():
                data[key].append(int(line))
            
            data[key] = standartize(np.array(data[key], dtype=np.float32))

    return data


def read_json(file):
    try:
        start = time.time()
        if not os.path.isfile(file) and (time.time() - start) > LOAD_DELAY:
            raise Exception("Cannot find file")

        file = open(file, 'r')
        data = json.load(file)
        file.close()
        array = data.get("data")
        if array is None:
            raise Exception("Data in json is not Array like")
        return np.array(array, dtype=np.float32)
    except Exception as e:
        raise Exception("Invalid data")


def remove_file(path):
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))
