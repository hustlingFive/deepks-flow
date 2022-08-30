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
    argo_range,
    argo_len,
    argo_sequence,
)
from dflow.python import(
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    Slices,
)
# from deepks2.constants import (
#     train_index_pattern,
# )
from deepks2.utils.step_config import normalize as normalize_step_dict
from deepks2.utils.step_config import init_executor

from deepks2.constants import scf_abacus_index_pattern

import os
from typing import Set, List
from pathlib import Path
from copy import deepcopy


class PrepRunScfAbacus(Steps):
    def __init__(
            self,
            name : str,
            prep_scf_abacus_op : OP,
            run_scf_abacus_op : OP,
            prep_config : dict = normalize_step_dict({}),
            run_config : dict = normalize_step_dict({}),
            upload_python_package : str = None,
    ):
        self._input_parameters = {
            "block_id" : InputParameter(type=str, value=""),
            # "numb_models": InputParameter(type=int),
            "scf_abacus_config" : InputParameter(),
        }        
        self._input_artifacts = {
            # "init_models" : InputArtifact(optional=True),
            "system" : InputArtifact(),
            "stru_file" : InputArtifact(),
            # "iter_data" : InputArtifact(),
        }
        self._output_parameters = {
            # "task_names":OutputParameter()
        }
        self._output_artifacts = {
            # "errs": OutputArtifact(),
            # "log_datas": OutputArtifact(),
            # "log_scfs": OutputArtifact(),
            "task_paths": OutputArtifact(),
            # "lcurves": OutputArtifact(),
        }

        super().__init__(        
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                # parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )
        
        self._keys = ['prep-scf-abacus', 'run-scf-abacus']
        self.step_keys = {}
        ii = 'prep-scf-abacus'
        self.step_keys[ii] = '--'.join(
            ["%s"%self.inputs.parameters["block_id"], ii]
        )
        ii = 'run-scf-abacus'
        self.step_keys[ii] = '--'.join(
            ["%s"%self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )

        print("preprun:",run_config)

        self = _prep_run_scf_abacus(
            self, 
            self.step_keys,
            prep_scf_abacus_op,
            run_scf_abacus_op,
            prep_config = prep_config,
            run_config = run_config,
            upload_python_package = upload_python_package,
        )            

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def keys(self):
        return self._keys
    

def _prep_run_scf_abacus(
        scf_abacus_steps,
        step_keys,
        prep_scf_abacus_op : OP,
        run_scf_abacus_op : OP,
        prep_config : dict = normalize_step_dict({}),
        run_config : dict = normalize_step_dict({}),
        upload_python_package : str = None,
):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop('template_config')
    run_template_config = run_config.pop('template_config')
    prep_executor = init_executor(prep_config.pop('executor'))
    run_executor = init_executor(run_config.pop('executor'))

    print("preprun_inner:",run_config)
    print("preprun_inner:",run_template_config)
    print("preprun_inner:",run_executor)

    # prepare for parallel scf abacus
    prep_scf_abacus = Step(
        'prep-scf-abacus',
        template=PythonOPTemplate(
            prep_scf_abacus_op,
            # output_artifact_archive={
            #     "task_paths": None
            # },
            python_packages = upload_python_package,
            **prep_template_config,
        ),
        parameters={
            "scf_abacus_config": scf_abacus_steps.inputs.parameters['scf_abacus_config'],
            # "template_script": scf_abacus_steps.inputs.parameters['template_script'],
        },
        artifacts={
            "system":scf_abacus_steps.inputs.artifacts['system'],
            # "stru_file":scf_abacus_steps.inputs.artifacts['stru_file']
        },
        key = step_keys['prep-scf-abacus'],
        executor = prep_executor,
        **prep_config,
    )
    scf_abacus_steps.add(prep_scf_abacus)

    run_scf_abacus = Step(
        'run-scf-abacus',
        template=PythonOPTemplate(
            run_scf_abacus_op,
            slices = Slices(
                "{{item}}",
                # input_parameter = ["task_name"],
                input_artifact = ["task_path"],
                # output_parameter=["task_name"],
                output_artifact = ["task_path"],
            ),
            python_packages = upload_python_package,
            **run_template_config,
        ),
        parameters={
            "scf_abacus_config" : scf_abacus_steps.inputs.parameters['scf_abacus_config'],
            # "task_name" : prep_scf_abacus.outputs.parameters["task_names"],
        },
        artifacts={
            'task_path' : prep_scf_abacus.outputs.artifacts['task_paths'],
            "stru_file" : scf_abacus_steps.inputs.artifacts["stru_file"],
        },
        # with_sequence=argo_sequence(argo_len(prep_scf_abacus.outputs.parameters["task_names"]), format=scf_abacus_index_pattern),
        # with_param=argo_range(scf_abacus_steps.inputs.parameters["numb_models"]),
        with_param=argo_range(argo_len(prep_scf_abacus.outputs.parameters["task_names"])),
        key = step_keys['run-scf-abacus'],
        executor = run_executor,
        util_image="base_dflow_deepks",
        **run_config,
    )
    scf_abacus_steps.add(run_scf_abacus)

    scf_abacus_steps.outputs.artifacts["task_paths"]._from = run_scf_abacus.outputs.artifacts["task_path"]
    # scf_abacus_steps.outputs.parameters["task_names"] = prep_scf_abacus.outputs.parameters["task_names"]
    # scf_abacus_steps.outputs.artifacts["models"]._from = run_scf_abacus.outputs.artifacts["model"]
    # scf_abacus_steps.outputs.artifacts["log_datas"]._from = run_scf_abacus.outputs.artifacts["log_data"]
    # scf_abacus_steps.outputs.artifacts["log_scfs"]._from = run_scf_abacus.outputs.artifacts["log_scf"]

    return scf_abacus_steps


