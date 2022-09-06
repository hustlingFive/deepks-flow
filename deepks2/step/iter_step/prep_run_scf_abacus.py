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
            "n_iter": InputParameter(type=int),
            "scf_abacus_config" : InputParameter(),
        }        
        self._input_artifacts = {
            "model" : InputArtifact(optional=True),
            "system" : InputArtifact(),
            "stru_file" : InputArtifact(),
        }
        self._output_parameters = {
        }
        self._output_artifacts = {
            "task_paths": OutputArtifact(),
        }

        super().__init__(        
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
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

    # prepare for parallel scf abacus
    prep_scf_abacus = Step(
        'prep-scf-abacus',
        template=PythonOPTemplate(
            prep_scf_abacus_op,
            python_packages = upload_python_package,
            **prep_template_config,
        ),
        parameters={
            "scf_abacus_config": scf_abacus_steps.inputs.parameters['scf_abacus_config'],
        },
        artifacts={
            "system":scf_abacus_steps.inputs.artifacts['system'],
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
                input_artifact = ["task_path"],
                output_artifact = ["task_path"],
            ),
            python_packages = upload_python_package,
            **run_template_config,
        ),
        parameters={
            "scf_abacus_config" : scf_abacus_steps.inputs.parameters['scf_abacus_config'],
            "n_iter" : scf_abacus_steps.inputs.parameters["n_iter"],
        },
        artifacts={
            'task_path' : prep_scf_abacus.outputs.artifacts['task_paths'],
            "stru_file" : scf_abacus_steps.inputs.artifacts["stru_file"],
            "model" : scf_abacus_steps.inputs.artifacts["model"],
        },
        with_param=argo_range(argo_len(prep_scf_abacus.outputs.parameters["task_names"])),
        key = step_keys['run-scf-abacus'],
        executor = run_executor,
        util_image="base_dflow_deepks_v2",
        **run_config,
    )
    scf_abacus_steps.add(run_scf_abacus)

    scf_abacus_steps.outputs.artifacts["task_paths"]._from = run_scf_abacus.outputs.artifacts["task_path"]


    return scf_abacus_steps


