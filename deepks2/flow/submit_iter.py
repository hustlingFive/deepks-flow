import glob, os, pickle, sys, shutil
from pathlib import Path
from dflow import (
    InputParameter,
    OutputParameter,
    Inputs,
    InputArtifact,
    Outputs,
    OutputArtifact,
    Workflow,
    Step,
    Steps,
    upload_artifact,
    download_artifact,
    S3Artifact,
    argo_range,
)
from dflow.python import (
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    upload_packages,
    FatalError,
    TransientError,
)

from deepks2.utils.path_utils import copy_file, copy_dir
from deepks2.utils.file_utils import load_yaml, save_yaml
from deepks2.utils.arg_utils import load_sys_paths
from deepks2.utils.basis_utils import load_basis, save_basis
from deepks2.utils.step_config import normalize as normalize_step_dict


from deepks2.op.iter_op import (
    ConvertScfAbacus,
    GatherStatsScfAbacus,
    PrepScfAbacus,
    RunScfAbacus,
    PrepTrain,
    RunTrain
)
from deepks2.step.iter_step import (
    PrepRunTrain,
    PrepRunScfAbacus,
    MakeIterBlock,
    Iterate,
)
from deepks2.constants import *

# from deepks2.utils.step_config import normalize as normalize_step_dict



    

def assert_exist(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No required file or directory: {path}")


def check_share_folder(data, name, share_folder="share"):
    # save data to share_folder/name. 
    # if data is None or False, do nothing, return None
    # otherwise, return name, and do one of the following:
    #   if data is True, check the existence in share.
    #   if data is a file name, copy it to share.
    #   if data is a dict, save it as an yaml file in share.
    #   otherwise, throw an error
    if not data:
        return None
    dst_name = os.path.join(share_folder, name)
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




def workflow_iterate(systems_train=None, systems_test=None,n_iter = 0, 
                 *, proj_basis=None, workdir=".", share_folder="share",
                 scf_input=True, scf_machine=None,
                 train_input=True, train_machine=None,
                 init_model=False, init_scf=True, init_train=True,
                 init_scf_machine=None, init_train_machine=None,
                 cleanup=False, strict=True, 
                 use_abacus=False, scf_abacus=None, init_scf_abacus=None,#caoyu add 2021-07-22
                 dflow_config=None, lebesgue_context_config=None, upload_python_package = None,
                 default_config = None, converge=None):
    # check share folder contains required data
    # and collect the systems into share folder
    if systems_train is None: # load default training systems
        default_train = os.path.join(share_folder, DEFAULT_TRAIN)
        assert_exist(default_train) # must have training systems.
        systems_train = default_train
    systems_train = collect_systems(systems_train, os.path.join(share_folder, SYS_TRAIN))
    # check test systems 
    if systems_test is None: # try to load default testing systems
        default_test = os.path.join(share_folder, DEFAULT_TEST)
        if os.path.exists(default_test): # if exists then use it
            systems_test = default_test
        else: # if empty use last one of training system
            systems_test = systems_train[-1]
    systems_test = collect_systems(systems_test, os.path.join(share_folder, SYS_TEST))

    # check share folder contains required yaml file
    # scf_args_name = check_share_folder(scf_input, SCF_ARGS_NAME, share_folder)
    # check required machine parameters
    scf_machine = check_arg_dict(scf_machine, DEFAULT_SCF_MACHINE, strict)
    train_machine = check_arg_dict(train_machine, DEFAULT_TRN_MACHINE, strict)
    scf_resource = scf_machine.pop("resources")
    default_config = normalize_step_dict(default_config)
    # n_iter_max = converge.pop("n_iter_max")


    # make tasks
    if use_abacus:  #caoyu add 2021-07-22
        scf_abacus = check_arg_dict(scf_abacus, DEFAULT_SCF_ARGS_ABACUS, strict)
        scf_config = dict(scf_abacus, **scf_resource)
        print("1")
        # no_model False
        scf_machine = normalize_step_dict(scf_machine)
        scf_op = PrepRunScfAbacus(
            "prep-run-scf-abacus",
            PrepScfAbacus,
            RunScfAbacus,
            prep_config = default_config,
            run_config = scf_machine,
            upload_python_package = upload_python_package,
        )
        
    else:
        raise RuntimeError('unknown input of use_abacus: ', use_abacus)
    print("2")
    train_config = dict(train_input, **converge)

    # restart True
    train_machine=normalize_step_dict(train_machine)
    train_op = PrepRunTrain(
        "prep-run-dp-train",
        PrepTrain,
        RunTrain,
        prep_config = default_config,
        run_config = train_machine,
        upload_python_package = upload_python_package,
    )
    
    
    print("3")
    iter_block_op = MakeIterBlock(
        "iter",
        ConvertScfAbacus,
        scf_op,
        GatherStatsScfAbacus,
        train_op,
        convert_config = default_config,
        gather_config = default_config,
        upload_python_package=upload_python_package,
    )
    iter_config = dict(scf_config, **train_config)

    # make init
    if init_model: # if set true or give str, check share/init/model.pth
        raise RuntimeError('unknown input of init_model: ', init_model)
    elif init_scf or init_train: # otherwise, make an init iteration to train the first model
        init_scf_machine = (check_arg_dict(init_scf_machine, DEFAULT_SCF_MACHINE, strict)
            if init_scf_machine is not None else scf_machine)
        if use_abacus:  #caoyu add 2021-07-22
            init_scf_abacus = check_arg_dict(init_scf_abacus, DEFAULT_SCF_ARGS_ABACUS, strict)
            init_scf_config = dict(init_scf_abacus,**scf_resource)
            print("2.1")
            init_scf_machine = normalize_step_dict(init_scf_machine)
            init_scf_op = PrepRunScfAbacus(
                "init-prep-run-scf-abacus",
                PrepScfAbacus,
                RunScfAbacus,
                prep_config = default_config,
                run_config = init_scf_machine,
                upload_python_package = upload_python_package,
            )
        else:
            raise RuntimeError('unknown input of use_abacus: ', use_abacus)
            
        init_train_machine = (check_arg_dict(init_train_machine, DEFAULT_SCF_MACHINE, strict)
            if init_train_machine is not None else train_machine)
        
        init_train_config = dict(init_train, **converge)
        print("2.2")
        # restart False
        init_train_machine=normalize_step_dict(init_train_machine)
        init_train_op = PrepRunTrain(
            "init-prep-run-dp-train",
            PrepTrain,
            RunTrain,
            prep_config = default_config,
            run_config = init_train_machine,
            upload_python_package = upload_python_package,
        )

        print("2.3")
        init_iter_block_op = MakeIterBlock(
            "init-iter",
            ConvertScfAbacus,
            init_scf_op,
            GatherStatsScfAbacus,
            init_train_op,
            convert_config = default_config,
            gather_config = default_config,
            upload_python_package=upload_python_package,
        )
        init_iter_config = dict(init_scf_config,**init_train_config)


    # deepks
    iter_op = Iterate(
        "deepks-iter", 
        init_iter_block_op,
        iter_block_op,
        upload_python_package = upload_python_package,
        step_config = default_config, # config for block_id op
    )

    stru_file = "stru_file"
    os.mkdir(stru_file)

    if init_scf_abacus is not None :
        orb_files=init_scf_abacus["orb_files"]
        proj_file=init_scf_abacus["proj_file"]
        pp_files=init_scf_abacus["pp_files"]
        for file in orb_files:
            shutil.copy(file, stru_file)
        for file in pp_files:
            shutil.copy(file, stru_file)
        for file in proj_file:
            shutil.copy(file, stru_file)
    
    # return iter_op
    iter = Step(
        'deepks-iter',
        template = iter_op,
        parameters = {
            "block_id" : "iter.init",
            "n_iter" : n_iter,
            "init_iter_config" : init_iter_config,
            "iter_config" : iter_config,
        },
        artifacts = {
            "stru_file" : upload_artifact(Path(stru_file)),
            "system" : upload_artifact(Path(share_folder)),
        },
    )

    return iter


def submit_iterate(*args, **kwargs):
    # set global config
    from dflow import config, s3_config
    dflow_config = kwargs.get('dflow_config', None)
    if dflow_config is not None :
        config["host"] = dflow_config.get('host', None)
        s3_config["endpoint"] = dflow_config.get('s3_endpoint', None)
        config["k8s_api_server"] = dflow_config.get('k8s_api_server', None)
        config["token"] = dflow_config.get('token', None)    

    # lebesgue context
    from dflow.plugins.lebesgue import LebesgueContext
    lb_context_config = kwargs.get("lebesgue_context_config", None)
    if lb_context_config:
        lebesgue_context = LebesgueContext(
            **lb_context_config,
        )
    else :
        lebesgue_context = None

    # print('config:', config)
    # print('s3_config:',s3_config)
    # print('lebsque context:', lb_context_config)
    # print(wf_config)

    deepks_iter = workflow_iterate(*args, **kwargs)

    wf = Workflow(name="deepks-iter", context=lebesgue_context)
    wf.add(deepks_iter)

    # reuse steps for resubmit
    reuse_step = None

    wf.submit(reuse_step=reuse_step)

    return wf



