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

from deepks2.utils.file_utils import load_yaml
from deepks2.utils.arg_utils import deep_update
from deepks2.utils.collect_inputs import check_arg_dict, collect_inputs
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


def workflow_mixiter(systems_train=None, systems_test=None,n_iter = 0, 
                 *, proj_basis=None, workdir=".", share_folder="share",
                 scf_input=True, scf_machine=None,
                 train_input=True, train_machine=None,
                 init_model=False, init_scf=True, init_train=True,
                 init_scf_machine=None, init_train_machine=None,
                 cleanup=False, strict=True, 
                 use_abacus=False, scf_abacus=None, init_scf_abacus=None,#caoyu add 2021-07-22
                 dflow_config=None, lebesgue_context_config=None, upload_python_package = None,
                 default_config = None, converge=None, compounds=None,
                 input_system = "systems.yaml",input_stru_abacus = "scf_abacus.yaml"):
    
    # prepare compounds
    shares = []
    inputs = []

    for workdir in compounds:
        stru_abacus = load_yaml(workdir/input_stru_abacus)
        system = load_yaml(workdir/input_system)

        systems_train=system.pop("systems_train", None)
        systems_test=system.pop("systems_test", None)

        if use_abacus:
            this_scf = stru_abacus.pop("scf_abacus", None)
            this_init_scf = stru_abacus.pop("init_scf_abacus", None)
            if this_init_scf is None :
                this_init_scf = this_scf
            
            this_scf = check_arg_dict(deep_update(this_scf, scf_abacus), DEFAULT_SCF_ARGS_ABACUS, strict)
            this_init_scf = check_arg_dict(deep_update(this_init_scf,init_scf_abacus), DEFAULT_SCF_ARGS_ABACUS, strict)
            
        else:
            raise RuntimeError('unknown input of use_abacus: ', use_abacus)
        
        this_share, this_input = collect_inputs(workdir=workdir,share_folder=share_folder,input_folder="input",
                                                systems_train = systems_train, systems_test = systems_test,
                                                train_input=train_input,init_train=init_train,
                                                scf_input=this_scf,init_scf=this_init_scf)
        shares.append(this_share)
        inputs.append(this_input)
        
    # check input_file folder contains required yaml file
    # scf_args_name = check_share_folder(scf_input, SCF_ARGS_NAME, input_file)
    # check required machine parameters
    scf_machine = check_arg_dict(scf_machine, DEFAULT_SCF_MACHINE, strict)
    train_machine = check_arg_dict(train_machine, DEFAULT_TRN_MACHINE, strict)
    scf_resource = scf_machine.pop("resources")
    default_config = normalize_step_dict(default_config)
    # n_iter_max = converge.pop("n_iter_max")


    # make tasks
    if use_abacus:
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
    # iter_config = dict(scf_config, **train_config)

    # # make init
    # if init_model: # if set true or give str, check share/init/model.pth
    #     raise RuntimeError('unknown input of init_model: ', init_model)
    # elif init_scf or init_train: # otherwise, make an init iteration to train the first model
    #     init_scf_machine = (check_arg_dict(init_scf_machine, DEFAULT_SCF_MACHINE, strict)
    #         if init_scf_machine is not None else scf_machine)
    #     if use_abacus:  #caoyu add 2021-07-22
    #         init_scf_abacus = check_arg_dict(init_scf_abacus, DEFAULT_SCF_ARGS_ABACUS, strict)
    #         init_scf_config = dict(init_scf_abacus,**scf_resource)
    #         print("2.1")
    #         init_scf_machine = normalize_step_dict(init_scf_machine)
    #         init_scf_op = PrepRunScfAbacus(
    #             "init-prep-run-scf-abacus",
    #             PrepScfAbacus,
    #             RunScfAbacus,
    #             prep_config = default_config,
    #             run_config = init_scf_machine,
    #             upload_python_package = upload_python_package,
    #         )
    #     else:
    #         raise RuntimeError('unknown input of use_abacus: ', use_abacus)
            
    #     init_train_machine = (check_arg_dict(init_train_machine, DEFAULT_SCF_MACHINE, strict)
    #         if init_train_machine is not None else train_machine)
        
    #     init_train_config = dict(init_train, **converge)
    #     print("2.2")
    #     # restart False
    #     init_train_machine=normalize_step_dict(init_train_machine)
    #     init_train_op = PrepRunTrain(
    #         "init-prep-run-dp-train",
    #         PrepTrain,
    #         RunTrain,
    #         prep_config = default_config,
    #         run_config = init_train_machine,
    #         upload_python_package = upload_python_package,
    #     )

    #     print("2.3")
    #     init_iter_block_op = MakeIterBlock(
    #         "init-iter",
    #         ConvertScfAbacus,
    #         init_scf_op,
    #         GatherStatsScfAbacus,
    #         init_train_op,
    #         convert_config = default_config,
    #         gather_config = default_config,
    #         upload_python_package=upload_python_package,
    #     )
    #     init_iter_config = dict(init_scf_config,**init_train_config)


    # # deepks
    # iter_op = Iterate(
    #     "deepks-iter", 
    #     init_iter_block_op,
    #     iter_block_op,
    #     upload_python_package = upload_python_package,
    #     step_config = default_config, # config for block_id op
    # )

    # # return iter_op
    # iter = Step(
    #     'deepks-mixiter',
    #     template = iter_op,
    #     parameters = {
    #         "block_id" : "iter.init",
    #         "n_iter" : n_iter,
    #         "init_iter_config" : init_iter_config,
    #         "iter_config" : iter_config,
    #     },
    #     artifacts = {
    #         "input_file" : upload_artifact(Path(input_file)), # input_files
    #         "system" : upload_artifact(Path(share_folder)),  # systems
    #     },
    # )

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

    deepks_iter = workflow_mixiter(*args, **kwargs)

    wf = Workflow(name="deepks-mixiter", context=lebesgue_context)
    wf.add(deepks_iter)

    # reuse steps for resubmit
    reuse_step = None

    wf.submit(reuse_step=reuse_step)

    return wf



