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
from deepks2.utils.arg_utils import deep_update
from deepks2.utils.step_config import normalize as normalize_step_dict
from deepks2.utils.step_config import init_executor
from deepks2.constants import (BOHRIUM_DEEPKS_TRAIN)
import os
from typing import Set, List
from pathlib import Path
from copy import deepcopy



# prepare train
class PrepRunTrain(Steps):
    def __init__(
            self,
            name : str,
            prep_train_op : OP,
            run_train_op : OP,
            assistant_config : dict = normalize_step_dict({}),
            run_config : dict = normalize_step_dict({}),
            upload_python_package : str = None,
    ):
        self._input_parameters = {
            "block_id" : InputParameter(type=str, value=""),
            "no_model" : InputParameter(type=bool),
            "no_test" : InputParameter(type=bool),
            "yaml_name" : InputParameter(),
        }
        self._input_artifacts = {
            "00_scf" : InputArtifact(),
            "model":InputArtifact(optional=True),
            "config_file":InputArtifact(),
        }
        self._output_parameters = {
        }
        self._output_artifacts = {
            "01_train": OutputArtifact(),
            "model": OutputArtifact(),
        }

        super().__init__(        
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )
        
        self._keys = ["prep-train", "run-train"]
        self.step_keys = {}
        ii = "prep-train"
        self.step_keys[ii] = "--".join(
            ["%s"%self.inputs.parameters["block_id"], ii]
        )
        ii = "run-train"
        self.step_keys[ii] = "--".join(
            # ["%s"%self.inputs.parameters["block_id"], ii + "-{{item}}"]
            ["%s"%self.inputs.parameters["block_id"], ii]
        )

        self = _prep_run_train(
            self, 
            self.step_keys,
            prep_train_op,
            run_train_op,
            assistant_config = assistant_config,
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
    

def _prep_run_train(
        train_steps,
        step_keys,
        prep_train_op : OP,
        run_train_op : OP,
        assistant_config : dict = normalize_step_dict({}),
        run_config : dict = normalize_step_dict({}),
        upload_python_package : str = None,
):
    assistant_config = deepcopy(assistant_config)
    run_config = deepcopy(run_config)
    assistant_template_config = assistant_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    assistant_executor = assistant_config.pop("executor")
    run_executor = run_config.pop("executor")
    run_executor = init_executor(deep_update(
        run_executor, BOHRIUM_DEEPKS_TRAIN))
    assistant_executor = init_executor(assistant_executor)

    prep_train = Step(
        "prep-train",
        template=PythonOPTemplate(
            prep_train_op,
            output_artifact_archive={
                "01_train": None ,
            },
            python_packages = upload_python_package,
            **assistant_template_config,
        ),
        parameters={
            "yaml_name": train_steps.inputs.parameters["yaml_name"],
            "no_model": train_steps.inputs.parameters["no_model"]
        },
        artifacts={
            "00_scf": train_steps.inputs.artifacts["00_scf"],
            "model": train_steps.inputs.artifacts["model"],
            "config_file": train_steps.inputs.artifacts["config_file"],
        },
        key = step_keys["prep-train"],
        executor = assistant_executor,
        **assistant_config,
    )
    train_steps.add(prep_train)

    run_train = Step(
        "run-train",
        template=PythonOPTemplate(
            run_train_op,
            output_artifact_archive={
                "01_train": None ,
            },
            python_packages = upload_python_package,
            **run_template_config,
        ),
        parameters={
            "yaml_name": train_steps.inputs.parameters["yaml_name"],
            "no_model": train_steps.inputs.parameters["no_model"],
            "no_test": train_steps.inputs.parameters["no_test"],
        },
        artifacts={
            "01_train" : prep_train.outputs.artifacts["01_train"],
            "config_file": train_steps.inputs.artifacts["config_file"],
        },
        key = step_keys["run-train"],
        executor = run_executor,
        **run_config,
    )
    train_steps.add(run_train)

    train_steps.outputs.artifacts["model"]._from= run_train.outputs.artifacts["model"]
    train_steps.outputs.artifacts["01_train"]._from= run_train.outputs.artifacts["01_train"]
    # train_steps.outputs.parameters["stop_or_converge"].value_from_parameter= run_train.outputs.parameters["stop_or_converge"]
    

    return train_steps


