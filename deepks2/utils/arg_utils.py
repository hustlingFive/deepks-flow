import imp
import os
from glob import glob
import numpy as np
from collections.abc import Mapping
from itertools import chain
from deepks2.utils.file_utils import load_array

# below are argument chekcing utils

def check_list(arg, nullable=True):
    # make sure the argument is a list
    if arg is None:
        if nullable:
            return []
        else:
            raise TypeError("arg cannot be None")
    if not isinstance(arg, (list, tuple, np.ndarray)):
        return [arg]
    return arg


def check_array(arr, nullable=True):
    if arr is None:
        if nullable:
            return arr
        else:
            raise TypeError("arg cannot be None")
    if isinstance(arr, str):
        return load_array(arr)
    else:
        return np.array(arr)


def flat_file_list(file_list, filter_func=lambda p: True, sort=True):
    # make sure file list contains desired files
    # flat all wildcards and files contains other files (once)
    # if no satisfied files, return empty list
    file_list = check_list(file_list)
    if sort:
        file_list = sorted(sum([glob(p) for p in file_list], []))
    else:
        file_list = sum([glob(p) for p in file_list], [])
    new_list = []
    for p in file_list:
        if filter_func(p):
            new_list.append(p)
        else:
            with open(p) as f:
                sub_list = f.read().splitlines()
                if sort:
                    sub_list = sorted(sum([glob(p) for p in sub_list], []))
                else: 
                    sub_list = sum([glob(p) for p in sub_list], [])
                new_list.extend(sub_list)
    return new_list


def load_dirs(path_list):
    return flat_file_list(path_list, os.path.isdir)

def load_xyz_files(file_list):
    return flat_file_list(file_list, is_xyz)

def load_sys_paths(sys_list):
    return flat_file_list(sys_list, lambda p: os.path.isdir(p) or is_xyz(p))

def is_xyz(p):
    return os.path.splitext(p)[1] == '.xyz'


def deep_update(o, u=(), **f):
    """Recursively update a dict.

    Subdict's won't be overwritten but also updated.
    """
    if not isinstance(o, Mapping):
        return u
    kvlst = chain(u.items() if isinstance(u, Mapping) else u, 
                  f.items())
    for k, v in kvlst:
        if isinstance(v, Mapping):
            o[k] = deep_update(o.get(k, {}), v)
        else:
            o[k] = v
    return o







