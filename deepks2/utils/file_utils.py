import os
import yaml
import numpy as np
# below are file loading utils

def load_yaml(file_path):
    with open(file_path, 'r') as fp:
        res = yaml.safe_load(fp)
    return res


def save_yaml(data, file_path):
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(file_path, 'w') as fp:
        yaml.safe_dump(data, fp)


def load_array(file):
    ext = os.path.splitext(file)[-1]
    if "npy" in ext:
        return np.load(file)
    elif "npz" in ext:
        raise NotImplementedError
    else:
        try:
            arr = np.loadtxt(file)
        except ValueError:
            arr = np.loadtxt(file, dtype=str)
        return arr


def parse_xyz(filename):
    with open(filename) as fp:
        natom = int(fp.readline())
        comments = fp.readline().strip()
        atom_str = fp.readlines()
    atom_list = [a.split() for a in atom_str]
    elements = [a[0] for a in atom_list]
    coords = np.array([a[1:] for a in atom_list], dtype=float)
    return natom, comments, elements, coords

