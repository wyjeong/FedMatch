import os
import pdb
import json
import torch
import random
import numpy as np
from datetime import datetime

def np_save(base_dir, filename, data):
    if os.path.isdir(base_dir) == False:
        os.makedirs(base_dir)
    np.save(os.path.join(base_dir, filename), data)

def np_load(base_dir, filename):
    return np.load(os.path.join(base_dir, filename), allow_pickle=True).item()

def write_file(filepath, filename, data):
    if os.path.isdir(filepath) == False:
        os.makedirs(filepath)
    with open(os.path.join(filepath, filename), 'w+') as outfile:
        json.dump(data, outfile)

def debugger():
    pdb.set_trace()

