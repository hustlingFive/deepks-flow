import os
import sys
import shutil
from tkinter.messagebox import NO
from deepks2.utils.path_utils import copy_file, link_file
from deepks2.utils.file_utils import load_yaml, save_yaml
from deepks2.utils.arg_utils import load_sys_paths
from pathlib import Path
from deepks2.constants import (DATA_TEST, DATA_TRAIN, DEFAULT_TRAIN_ARGS,
                               DEFAULT_SCF_ARGS_ABACUS, SYS_TRAIN, SYS_TEST, TRN_ARGS_NAME, INIT_TRN_NAME, SCF_ARGS_NAME, INIT_SCF_NAME)


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
    allowed = {k: v for k, v in data.items() if k in default}
    outside = {k: v for k, v in data.items() if k not in default}
    if outside:
        print(f"following ars are not in the default list: {list(outside.keys())}"
              + "and would be discarded" if strict else "but kept", file=sys.stderr)
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
        count_dict = {bases[i]: [] for i in dups}
        for i in dups:
            count_dict[bases[i]].append(i)
        dup_dict = {k: v for k, v in count_dict.items() if len(v) > 1}
        if not dup_dict:
            break
        dups = sum(dup_dict.values(), [])
        if all(parents[i] in ("/", "") for i in dups):
            print("System list have duplicated terms, index:",
                  dups, file=sys.stderr)
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


def collect_sys_dir(sys, workdir, dir, name):
    workdir,target =Path(workdir), Path(dir)/name
    if sys is not None:
        collect_systems([str(workdir/s) for s in sys], target)
    elif not os.path.exists(target):
        os.makedirs(target)

def collect_conf_dir(conf,dir,yaml_name,template,strict):
    if conf is not None:
        conf = check_arg_dict(conf, template, strict)
        check_input_folder(conf, yaml_name, dir)
        return conf
    else:
        return None

def collect_stru_dir(orb,workdir,targetdir):
    workdir,targetdir = Path(workdir),Path(targetdir)
    for file in orb["orb_files"]:
        link_file(workdir/file, targetdir/file)
    for file in orb["pp_files"]:
        link_file(workdir/file, targetdir/file)
    for file in orb["proj_file"]:
        link_file(workdir/file, targetdir/file)

def collect_inputs(workdir=".", stru_file="stru_file", share_folder="share",
                   config_file="config_file", strict=True, init_model=False,
                   systems_train=None, systems_test=None,
                   train_only=False, no_test=False,
                   train_conf=None, init_train_conf=None,
                   scf_conf=None, init_scf_conf=None,
                   default_train_args=DEFAULT_TRAIN_ARGS, default_scf_args=DEFAULT_SCF_ARGS_ABACUS):
    # make folders
    workdir = Path(workdir)
    this_share = workdir/share_folder
    this_stru = workdir/stru_file
    this_conf = workdir/config_file

    # check share_folder, systems
    if not train_only:
        collect_sys_dir(systems_train,workdir,this_share,SYS_TRAIN)
        collect_sys_dir(systems_test,workdir,this_share,SYS_TEST)
    else:
        if not os.path.exists(this_share/DATA_TRAIN):
            raise RuntimeError('train data should in : ',
                               str(this_share/DATA_TRAIN))
        if not os.path.exists(this_share/DATA_TEST) and not no_test:
            raise RuntimeError('test data should in : ',
                               str(this_share/DATA_TEST))
    if not train_only:
        scf_conf = collect_conf_dir(scf_conf,this_conf,SCF_ARGS_NAME,default_scf_args,strict)
        init_scf_conf = collect_conf_dir(init_scf_conf,this_conf,INIT_SCF_NAME,default_scf_args,strict)
        print("Collected scf config in %s" % str(this_conf))

        if not init_model and init_scf_conf is not None:
            collect_stru_dir(init_scf_conf,workdir,this_stru)
        elif init_model and scf_conf is not None:
            collect_stru_dir(scf_conf,workdir,this_stru)
        else:
            raise RuntimeError('Should have scf or initscf input')
        print("Collected stru files in %s" % str(this_stru))

    train_conf = collect_conf_dir(train_conf,this_conf,TRN_ARGS_NAME,default_train_args,strict)
    init_train_conf = collect_conf_dir(init_train_conf,this_conf,INIT_TRN_NAME,default_train_args,strict)

    if train_conf is not None or init_train_conf is not None:
        print("Collected train config in %s" % str(this_conf))

    return this_share, this_stru, this_conf
