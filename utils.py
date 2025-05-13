import logging
from collections import defaultdict

import torch
import random
import numpy as np
import os


def filter_and_keep_first_duplicate_j(coordinates):
    seen_j = set()  
    result = []  

    for coord in coordinates:
        i, j = coord
        if j not in seen_j:
            result.append(coord)  
            seen_j.add(j)  

    return result


def filter_and_keep_random_duplicate_j(coordinates):
    j_to_coords = defaultdict(list)  


    for coord in coordinates:
        i, j = coord
        j_to_coords[j].append(coord)

    result = []


    for j, coords in j_to_coords.items():
        result.append(random.choice(coords))  

    return result



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def set_logger(args):

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    file_handler = logging.FileHandler(args.output_dir + "logs.log")
    file_handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger = logging.getLogger(args.model_type)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    return logger
