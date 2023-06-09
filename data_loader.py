import os 
import numpy as np
import torch


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


def read_data():
    path = r"/home/cailen/Data/Fylhtq/"

    files = os.listdir(path)

    data = {}

    for file in files:
        p = os.path.join(path, file)
        with open(p, 'r') as read_f:
            key = file[:-4]
            data[key] = []
            for line in read_f.readlines():
                data[key].append(int(line))
            
            data[key] = standartize(np.array(data[key], dtype=np.float32))

    return data