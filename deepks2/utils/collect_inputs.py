import os, sys, shutil
from deepks2.utils.path_utils import copy_file, copy_dir
from deepks2.utils.file_utils import load_yaml, save_yaml
from deepks2.utils.arg_utils import load_sys_paths
from pathlib import Path
from deepks2.constants import *

def assert_exist(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No required file or directory: {path}")


def check_input_folder(data, name, input_folder="share"):
    # save data to share_folder/name. 
    # if data is None or False, do nothing, return None
    # otherwise, return name, and do one of the following:
    #   if data is True, check the existence in share.
    #   if data is a file name, copy it to share.
    #   if data is a dict, save it as an yaml file in share.
    #   otherwise, throw an error
    if not data:
        return None
    dst_name = os.path.join(input_folder, name)
    if data is True:
        assert_exist(dst_name)
        return name
    elif isinstance(data, str) and os.path.exists(data):
        copy_file(data, dst_name)
        return name
    elif isinstance(data, dict):
        save_yaml(data, dst_name)
        return name
    else:
        raise ValueError(f"Invalid argument: {data}")


def check_arg_dict(data, default, strict=True):
    if data is None:
        data = {}
    if isinstance(data, str):
        data = load_yaml(data)
    allowed = {k:v for k,v in data.items() if k in default}
    outside = {k:v for k,v in data.items() if k not in default}
    if outside:
        print(f"following ars are not in the default list: {list(outside.keys())}"
              +"and would be discarded" if strict else "but kept", file=sys.stderr)
    if strict:
        return {**default, **allowed}
    else:
        return {**default, **data}


def collect_systems(systems, folder=None):
    # check all systems have different basename
    # if there's duplicate, concat its dirname into the basename sep by a "."
    # then collect all systems into `folder` by symlink
    sys_list = [os.path.abspath(s) for s in load_sys_paths(systems)]
    parents, bases = map(list, zip(*[os.path.split(s.rstrip(os.path.sep)) 
                                        for s in sys_list]))
    dups = range(len(sys_list))
    while True:
        count_dict = {bases[i]:[] for i in dups}
        for i in dups:
            count_dict[bases[i]].append(i)
        dup_dict = {k:v for k,v in count_dict.items() if len(v)>1}
        if not dup_dict:
            break
        dups = sum(dup_dict.values(), [])
        if all(parents[i] in ("/", "") for i in dups):
            print("System list have duplicated terms, index:", dups, file=sys.stderr)
            break
        for di in dups:
            if parents[di] in ("/", ""):
                continue
            newp, newb = os.path.split(parents[di])
            parents[di] = newp
            bases[di] = f"{newb}.{bases[di]}"
    if folder is None:
        return bases
    targets = [os.path.join(folder, b) for b in bases]
    for s, t in zip(sys_list, targets):
        copy_dir(s, t, use_abs=True)
    return targets

def collect_inputs(workdir=".",share_folder = "share",input_folder = "input",
                systems_train = None, systems_test = None,
                train_input=None, init_train=None,
                scf_input=None, init_scf=None):
    # os.mkdir(input_folder)
    workdir = Path(workdir)
    this_share = workdir/share_folder
    this_input = workdir/input_folder

    # check share folder contains required data
    # and collect the systems into share folder
    systems_train = collect_systems(systems_train, os.path.join(this_share, SYS_TRAIN))
    # check test systems 
    systems_test = collect_systems(systems_test, os.path.join(this_share, SYS_TEST))

    check_input_folder(train_input,TRN_ARGS_NAME,this_input)
    check_input_folder(init_train,INIT_TRN_NAME,this_input)
    check_input_folder(scf_input,SCF_ARGS_NAME_ABACUS,this_input)
    check_input_folder(init_scf,INIT_SCF_NAME_ABACUS,this_input)

    if scf_input is not None :
        orb_files = scf_input["orb_files"]
        proj_file = scf_input["proj_file"]
        pp_files = scf_input["pp_files"]
        for file in orb_files:
            shutil.copy(workdir/file, this_input)
        for file in pp_files:
            shutil.copy(workdir/file, this_input)
        for file in proj_file:
            shutil.copy(workdir/file, this_input)

    return this_share, this_input