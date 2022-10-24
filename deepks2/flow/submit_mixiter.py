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

from deepks2.utils.step_config import normalize as normalize_step_dict
from deepks2.utils.collect_inputs import collect_inputs, check_arg_dict


from deepks2.op.iter_op.convert_scf_op import ConvertScfAbacus
from deepks2.op.iter_op.gather_scf_op import GatherStatsScfAbacus
from deepks2.op.iter_op.scf_abacus_op import PrepScfAbacus,RunScfAbacus
from deepks2.op.iter_op.train_op import PrepTrain,RunTrain
from deepks2.step.iter_step import (
    PrepRunTrain,
    ScfAbacus,
    DeepksAbacusIter,
)
from deepks2.constants import (INIT_SCF_NAME,INIT_TRN_NAME,DEFAULT_SCF_MACHINE,DEFAULT_TRN_MACHINE,DEFAULT_TRAIN_ARGS,DEFAULT_SCF_ARGS_ABACUS)


def workflow_iterate(systems_train=None, systems_test=None,n_iter = 0, 
                 *, proj_basis=None, workdir=".", share_folder="share",
                 scf_input=None, scf_machine=None,
                 train_input=None, train_machine=None,
                 init_model=False, init_scf=None, init_train=None,
                 init_scf_machine=None, init_train_machine=None,
                 cleanup=False, strict=True, no_test=False,
                 use_abacus=False, scf_abacus=None, init_scf_abacus=None,#caoyu add 2021-07-22
                 dflow_config=None, lebesgue_context_config=None, upload_python_package = None,
                 default_config = None,compound = None, **kwargs):
    train_only = False
    scf_yaml_name = INIT_SCF_NAME
    train_yaml_name = INIT_TRN_NAME

    # check model.pth
    model_file = None
    init_model = False

    # generate n_iter
    n_now, n_max = 0, n_iter
    n_iter = "-".join([str(n_now).zfill(2),str(n_max).zfill(2)])

    scf_machine = check_arg_dict(scf_machine, DEFAULT_SCF_MACHINE, strict)
    train_machine = check_arg_dict(train_machine, DEFAULT_TRN_MACHINE, strict)

    default_config = normalize_step_dict(default_config)
    scf_machine = normalize_step_dict(scf_machine)
    train_machine = normalize_step_dict(train_machine)

    if use_abacus:
        this_share, this_stru, this_conf = collect_inputs(workdir = workdir,stru_file = "stru_file",share_folder = share_folder,
                                            config_file = "config_file",strict=strict, init_model=init_model,
                                            systems_train = systems_train, systems_test = systems_test,
                                            train_only=train_only,no_test=no_test,
                                            train_conf = train_input, init_train_conf = init_train,
                                            scf_conf = scf_abacus, init_scf_conf = init_scf_abacus,
                                            default_train_args = DEFAULT_TRAIN_ARGS, default_scf_args_abacus = DEFAULT_SCF_ARGS_ABACUS)
        
        scf_op = ScfAbacus(
            "prep-run-scf-abacus",
            ConvertScfAbacus,
            PrepScfAbacus,
            RunScfAbacus,
            GatherStatsScfAbacus,
            assistant_config = default_config,
            run_config = scf_machine,
            upload_python_package = upload_python_package,
        )
    else:
        raise RuntimeError('unknown input of use_abacus: ', use_abacus)


    # restart True
    train_op = PrepRunTrain(
        "prep-run-dp-train",
        PrepTrain,
        RunTrain,
        assistant_config = default_config,
        run_config = train_machine,
        upload_python_package = upload_python_package,
    )
    
    iter_op = DeepksAbacusIter(
        "iter",
        scf_op,
        train_op,
        assistant_config = default_config,
        upload_python_package=upload_python_package,
    )
    
    # return iter_op
    iter = Step(
        'deepks-iter',
        template = iter_op,
        parameters = {
            "block_id" : "iter.init",
            "n_iter" : n_iter,
            "scf_yaml_name" : scf_yaml_name,
            "train_yaml_name" : train_yaml_name,
            "no_model": not init_model,
            "no_test":no_test,
        },
        artifacts = {
            "model" : upload_artifact(model_file),
            "system" : upload_artifact(this_share),
            "config_file" : upload_artifact(this_conf),
            "stru_file" : upload_artifact(this_stru),
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



