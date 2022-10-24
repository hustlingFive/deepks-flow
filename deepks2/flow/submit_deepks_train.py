import glob
import os
import pickle
from symbol import not_test
import sys
import shutil
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
from deepks2.utils.collect_inputs import check_arg_dict, collect_inputs, collect_systems
from deepks2.utils.step_config import normalize as normalize_step_dict

from deepks2.op.iter_op.train_op import PrepTrain,RunTrain
from deepks2.step.iter_step import (
    PrepRunTrain,
)
from deepks2.constants import (DEFAULT_SCF_ARGS_ABACUS,
                               DEFAULT_SCF_MACHINE,
                               DEFAULT_TRN_MACHINE,
                               INIT_TRN_NAME,
                               TRN_ARGS_NAME,
                               SYS_TRAIN, SYS_TEST)


def workflow_deepks_train(systems_train=None, systems_test=None,
                          *, workdir=".", share_folder="share",
                          scf_input=None, train_machine=None,
                          init_model=False, init_scf=None,
                          init_scf_machine=None, init_train=None, train_input=None,
                          cleanup=False, strict=True,train_folder=None,
                          use_abacus=False, scf_abacus=None, init_scf_abacus=None,
                          dflow_config=None, upload_python_package=None,
                          default_config=None,no_test=False, **kwargs):
    if not os.path.exists(train_folder):
        raise RuntimeError('unknown input of train_folder: ', train_folder)
    # check model.pth
    if init_model:
        model_file = Path(init_model)
        init_model = True
        yaml_name = TRN_ARGS_NAME
    else:
        model_file = None
        init_model = False
        yaml_name = INIT_TRN_NAME

    this_train, _, this_conf = collect_inputs(workdir=workdir, share_folder=train_folder,train_only=True,
                                            config_file="config_file", stru_file="stru_file", strict=strict,
                                            systems_train=None, systems_test=None,no_test=no_test,
                                            train_conf=train_input, init_train_conf=init_train,
                                            scf_conf=None, init_scf_conf=None)

    train_machine = check_arg_dict(train_machine, DEFAULT_TRN_MACHINE, strict)

    default_config = normalize_step_dict(default_config)
    train_machine = normalize_step_dict(train_machine)

    train_op = PrepRunTrain(
        "prep-run-deepks-train",
        PrepTrain,
        RunTrain,
        assistant_config=default_config,
        run_config=train_machine,
        upload_python_package=upload_python_package,
    )

    # return train_op
    train = Step(
        'deepks-scf',
        template=train_op,
        parameters={
            "block_id": "train.deepks",
            "no_model": not init_model,
            "no_test": no_test,
            "yaml_name": yaml_name,
        },
        artifacts={
            "00_scf": upload_artifact(this_train),
            "model": upload_artifact(model_file),
            "config_file": upload_artifact(this_conf),
        },
    )

    return train


def submit_deepks_train(*args, **kwargs):
    # set global config
    from dflow import config, s3_config
    dflow_config = kwargs.get('dflow_config', None)
    if dflow_config is not None:
        config["host"] = dflow_config.get('host', None)
        s3_config["endpoint"] = dflow_config.get('s3_endpoint', None)
        config["k8s_api_server"] = dflow_config.get('k8s_api_server', None)
        config["token"] = dflow_config.get('token', None)

    # lebesgue context
    from dflow.plugins.lebesgue import LebesgueContext
    lb_context_config = kwargs.pop("lebesgue_context_config", None)
    if lb_context_config:
        lebesgue_context = LebesgueContext(
            **lb_context_config,
        )
    else:
        lebesgue_context = None

    # print('config:', config)
    # print('s3_config:',s3_config)
    # print('lebsque context:', lb_context_config)
    # print(wf_config)

    deepks_scf = workflow_deepks_train(*args, **kwargs)

    wf = Workflow(name="deepks-train", context=lebesgue_context)
    wf.add(deepks_scf)

    # reuse steps for resubmit
    reuse_step = None

    wf.submit(reuse_step=reuse_step)

    return wf
