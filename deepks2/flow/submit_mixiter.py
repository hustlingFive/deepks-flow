import glob, os, pickle, sys, shutil
from pathlib import Path
from dflow import (
    Workflow,
    Step,
    Steps,
    upload_artifact,
)

from deepks2.utils.step_config import normalize as normalize_step_dict
from deepks2.utils.collect_inputs import collect_inputs, check_arg_dict,collect_conf_dir
from deepks2.utils.arg_utils import deep_update
from deepks2.utils.file_utils import load_yaml
from deepks2.utils.bohrium_config import bohrium_config_from_dict


from deepks2.op.iter_op.convert_scf_op import ConvertScfAbacus
from deepks2.op.iter_op.gather_scf_op import GatherStatsScfAbacus
from deepks2.op.iter_op.scf_abacus_op import PrepScfAbacus,RunScfAbacus
from deepks2.op.iter_op.train_op import PrepTrain,RunTrain
from deepks2.step.iter_step import (
    PrepRunTrain,
    ScfAbacus,
    DeepksAbacusMixIter,
)
from deepks2.constants import (INIT_SCF_NAME,INIT_TRN_NAME,DEFAULT_SCF_MACHINE,DEFAULT_TRN_MACHINE,TRN_ARGS_NAME,DEFAULT_SCF_ARGS_ABACUS,DEFAULT_TRAIN_ARGS)
from copy import deepcopy

def workflow_mixiter(n_iter = 0, 
                *, proj_basis=None, workdir=".", share_folder="share",
                scf_machine=None, train_machine=None, init_model=False,
                init_train=None, train_input=None,
                strict=True, no_test=False, use_abacus=False, upload_python_package = None,
                default_config = None,compounds = None,compounds_yaml=None, **kwargs):
    train_only = False
    scf_yaml_name = INIT_SCF_NAME
    train_yaml_name = INIT_TRN_NAME
    mixed_num = len(compounds)

    # check model.pth
    model_file = None
    init_model = False

    # generate n_iter
    n_now, n_max = 0, n_iter
    n_iter = "-".join([str(n_now).zfill(2),str(n_max).zfill(2)])

    this_scf_conf,this_stru,this_share = [],[],[]

    scf_machine = check_arg_dict(scf_machine, DEFAULT_SCF_MACHINE, strict)
    train_machine = check_arg_dict(train_machine, DEFAULT_TRN_MACHINE, strict)

    default_config = normalize_step_dict(default_config)
    scf_machine = normalize_step_dict(scf_machine)
    train_machine = normalize_step_dict(train_machine)

    
    for compound in compounds:
        temp_kwargs = deepcopy(kwargs)
        for yaml in compounds_yaml:
            temp_kwargs = deep_update(temp_kwargs, load_yaml(Path(compound)/yaml))
        systems_train = temp_kwargs.pop("systems_train")
        systems_test = temp_kwargs.pop("systems_test",None)
        print("Collecting compound : %s"%str(compound))
        if use_abacus:
            default_scf_args = DEFAULT_SCF_ARGS_ABACUS
            init_scf_conf = temp_kwargs.pop("init_scf_abacus")
            scf_conf = temp_kwargs.pop("scf_abacus")
        else:
            raise RuntimeError('unknown input of use_abacus: ', use_abacus)

        temp_share, temp_stru, temp_conf = collect_inputs(workdir = compound,stru_file = "stru_file",share_folder = share_folder,
                                            config_file = "config_file",strict=strict, init_model=init_model,
                                            systems_train = systems_train, systems_test = systems_test,
                                            train_only=train_only,no_test=no_test,
                                            scf_conf = scf_conf, init_scf_conf = init_scf_conf,
                                            default_scf_args = default_scf_args)
        this_share.append(temp_share)
        this_scf_conf.append(temp_conf)
        this_stru.append(temp_stru)
    
    this_train_conf = Path(workdir)/"config_file"
    collect_conf_dir(train_input,this_train_conf,TRN_ARGS_NAME,DEFAULT_TRAIN_ARGS,strict)
    collect_conf_dir(init_train,this_train_conf,INIT_TRN_NAME,DEFAULT_TRAIN_ARGS,strict)

    scf_op = ScfAbacus(
        "prep-run-scf-abacus",
        True,
        ConvertScfAbacus,
        PrepScfAbacus,
        RunScfAbacus,
        GatherStatsScfAbacus,
        assistant_config = default_config,
        run_config = scf_machine,
        upload_python_package = upload_python_package,
    )

    # restart True
    train_op = PrepRunTrain(
        "prep-run-dp-train",
        PrepTrain,
        RunTrain,
        assistant_config = default_config,
        run_config = train_machine,
        upload_python_package = upload_python_package,
    )
    
    mixiter_op = DeepksAbacusMixIter(
        "iter",
        scf_op,
        train_op,
        mixed_num,
        assistant_config = default_config,
        upload_python_package=upload_python_package,
    )
    
    # return iter_op
    mixiter = Step(
        'deepks-mixiter',
        template = mixiter_op,
        parameters = {
            "block_id" : "iter.init",
            "n_iter" : n_iter,
            "scf_yaml_name" : scf_yaml_name,
            "train_yaml_name" : train_yaml_name,
            "no_model": not init_model,
            "no_test":no_test,
        },
        artifacts = {
            "model" : None,
            "systems" : upload_artifact(this_share),
            "scf_config_files" : upload_artifact(this_scf_conf),
            "train_config_files" : upload_artifact(this_train_conf),
            "stru_files" : upload_artifact(this_stru),
        },
    )
    return mixiter


def submit_mixiter(*args, **kwargs):
    # set global config
    # from dflow import config, s3_config
    # dflow_config = kwargs.get('dflow_config', None)
    # if dflow_config is not None :
    #     config["host"] = dflow_config.get('host', None)
    #     s3_config["endpoint"] = dflow_config.get('s3_endpoint', None)
    #     config["k8s_api_server"] = dflow_config.get('k8s_api_server', None)
    #     config["token"] = dflow_config.get('token', None)    

    bohrium_config = kwargs.get('bohrium_config', None)
    if bohrium_config is not None:
        bohrium_config_from_dict(bohrium_config)

    # bohrium context
    from dflow.plugins.bohrium import BohriumContext
    bh_context_config = kwargs.pop("bohrium_context_config", None)
    if bh_context_config:
        bohrium_context = BohriumContext(
            **bh_context_config,
        )
    else:
        bohrium_context = None

    # print('config:', config)
    # print('s3_config:',s3_config)
    # print('lebsque context:', lb_context_config)
    # print(wf_config)

    deepks_mixiter = workflow_mixiter(*args, **kwargs)

    wf = Workflow(name="deepks-mixiter", context=bohrium_context)
    wf.add(deepks_mixiter)

    # reuse steps for resubmit
    reuse_step = None

    wf.submit(reuse_step=reuse_step)

    return wf



