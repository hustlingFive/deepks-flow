import glob, os, pickle, sys, shutil
from pathlib import Path
from dflow import (
    Workflow,
    Step,
    upload_artifact,
)

from deepks2.utils.collect_inputs import check_arg_dict, collect_inputs
from deepks2.utils.step_config import normalize as normalize_step_dict
from deepks2.utils.bohrium_config import bohrium_config_from_dict


from deepks2.op.iter_op.convert_scf_op import ConvertScfAbacus
from deepks2.op.iter_op.gather_scf_op import GatherStatsScfAbacus
from deepks2.op.iter_op.scf_abacus_op import PrepScfAbacus,RunScfAbacus
from deepks2.step.iter_step import (
    ScfAbacus,
)
from deepks2.constants import (DEFAULT_SCF_ARGS_ABACUS,
                               DEFAULT_TRAIN_ARGS,
                               DEFAULT_SCF_MACHINE,
                               DEFAULT_TRN_MACHINE,
                               INIT_SCF_NAME,
                               SCF_ARGS_NAME,
                               SYS_TRAIN, SYS_TEST)

def workflow_scf_abacus(systems_train = None, systems_test = None,
                 *, workdir=".", share_folder="share",
                 scf_input=True, scf_machine=None,
                 init_model=False, init_scf=True, 
                 init_scf_machine=None,
                 cleanup=False, strict=True, 
                 use_abacus=False, scf_abacus=None, init_scf_abacus=None,
                 dflow_config=None, upload_python_package = None,
                 default_config = None, **kwargs):
    if use_abacus:
        # check model.pth
        if init_model:
            model_file = Path(init_model)
            init_model = True
            yaml_name = SCF_ARGS_NAME
        else:
            model_file = None
            init_model = False
            yaml_name = INIT_SCF_NAME

        this_share, this_stru, this_conf = collect_inputs(workdir = workdir,share_folder = share_folder,init_model=init_model,
                                            config_file = "config_file", stru_file = "stru_file",strict=strict,
                                            systems_train = systems_train, systems_test = systems_test,
                                            train_conf = None, init_train_conf = None,
                                            scf_conf = scf_abacus, init_scf_conf = init_scf_abacus,
                                            default_train_args=DEFAULT_TRAIN_ARGS,default_scf_args = DEFAULT_SCF_ARGS_ABACUS)
        
        
        scf_machine = check_arg_dict(scf_machine, DEFAULT_SCF_MACHINE, strict)

        default_config = normalize_step_dict(default_config)
        scf_machine = normalize_step_dict(scf_machine)

        scf_op = ScfAbacus(
            "prep-run-scf-abacus",
            False,
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

    # return iter_op
    scf = Step(
        'deepks-scf',
        template = scf_op,
        parameters = {
            "block_id" : "scf.abacus",
            "yaml_name" : yaml_name,
            "no_model" : not init_model,
        },
        artifacts = {
            "model" : upload_artifact(model_file),
            "system" : upload_artifact(this_share),
            "config_file" : upload_artifact(this_conf),
            "stru_file" : upload_artifact(this_stru),
        },
    )

    return scf


def submit_scf_abacus(*args, **kwargs):
  
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

    deepks_scf = workflow_scf_abacus(*args, **kwargs)

    wf = Workflow(name="deepks-scf-abacus", context=bohrium_context)
    wf.add(deepks_scf)

    # reuse steps for resubmit
    reuse_step = None

    wf.submit(reuse_step=reuse_step)

    return wf



