import os, sys, shutil
from tkinter.messagebox import NO
from deepks2.utils.path_utils import copy_file, link_file
from deepks2.utils.file_utils import load_yaml, save_yaml
from deepks2.utils.arg_utils import load_sys_paths
from pathlib import Path
from deepks2.constants import (DATA_TEST,DATA_TRAIN, DEFAULT_TRAIN_ARGS,
DEFAULT_SCF_ARGS_ABACUS,SYS_TRAIN,SYS_TEST,TRN_ARGS_NAME,INIT_TRN_NAME,SCF_ARGS_NAME,INIT_SCF_NAME)

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
        link_file(s, t, use_abs=True)
    return targets

def collect_inputs(workdir=".",stru_file = "stru_file",share_folder = "share",
                config_file="config_file", strict = True, init_model = False,
                systems_train = None, systems_test = None,
                train_only = False,no_test = False,
                train_conf = None, init_train_conf = None,
                scf_conf = None, init_scf_conf = None,
                default_train_args = DEFAULT_TRAIN_ARGS, default_scf_args_abacus = DEFAULT_SCF_ARGS_ABACUS):
    # make folders
    workdir = Path(workdir)
    this_share = workdir/share_folder
    this_stru = workdir/stru_file
    this_conf = workdir/config_file

    # check share_folder, systems 
    if not train_only:
        if systems_train is not None:
            systems_train = collect_systems(systems_train, os.path.join(this_share, SYS_TRAIN))
        elif not os.path.exists(this_share/SYS_TRAIN):
            os.makedirs(this_share/SYS_TRAIN)
        if systems_test is not None:
            systems_test = collect_systems(systems_test, os.path.join(this_share, SYS_TEST))
        elif not os.path.exists(this_share/SYS_TEST):
            os.makedirs(this_share/SYS_TEST)
    else:
        if not os.path.exists(this_share/DATA_TRAIN):
            raise RuntimeError('train data should in : ', str(this_share/DATA_TRAIN))
        if not os.path.exists(this_share/DATA_TEST) and not no_test:
            raise RuntimeError('train data should in : ', str(this_share/DATA_TEST))

    orb_scf, train_tag = None, None
    if train_conf is not None:
        train_conf = check_arg_dict(train_conf, default_train_args, strict)
        check_input_folder(train_conf,TRN_ARGS_NAME,this_conf)
        train_tag = train_conf
    if init_train_conf is not None:
        init_train_conf = check_arg_dict(init_train_conf, default_train_args, strict)
        check_input_folder(init_train_conf,INIT_TRN_NAME,this_conf)
        if not init_model:
            train_tag = init_train_conf
    if scf_conf is not None:
        scf_conf = check_arg_dict(scf_conf, default_scf_args_abacus, strict)
        check_input_folder(scf_conf,SCF_ARGS_NAME,this_conf)
        orb_scf = scf_conf
    if init_scf_conf is not None:
        init_scf_conf = check_arg_dict(init_scf_conf, default_scf_args_abacus, strict)
        check_input_folder(init_scf_conf,INIT_SCF_NAME,this_conf)
        if not init_model:
            orb_scf = init_scf_conf

    if orb_scf is not None :
        for file in orb_scf["orb_files"]:
            link_file(workdir/file, this_stru/file)
        for file in orb_scf["pp_files"]:
            link_file(workdir/file, this_stru/file)
        for file in orb_scf["proj_file"]:
            link_file(workdir/file, this_stru/file)
    else:
        print("Run without scf input.")
    
    if train_tag is None :
        print("Run without train input.")

    return this_share, this_stru, this_conf