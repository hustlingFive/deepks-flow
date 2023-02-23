import imp
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

from copy import deepcopy


class ScfAbacus(Steps):
    def __init__(
            self,
            name: str,
            for_mix_scf: bool,
            convert_scf_op: OP,
            prep_scf_abacus_op: OP,
            run_scf_abacus_op: OP,
            gather_stats_scf_op: OP,
            assistant_config: dict = normalize_step_dict({}),
            run_config: dict = normalize_step_dict({}),
            upload_python_package: str = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
            "yaml_name": InputParameter(type=str, value=""),
            "no_model": InputParameter(type=bool, value=""),
        }
        self._input_artifacts = {
            "model": InputArtifact(optional=True),
            "system": InputArtifact(),
            "config_file": InputArtifact(),
            "stru_file": InputArtifact(),
        }
        self._output_parameters = {
        }
        self._output_artifacts = {
            "00_scf": OutputArtifact(),
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

        self._keys = ['convert-scf', 'prep-scf-abacus',
                      'run-scf-abacus', 'gather-scf']
        self.step_keys = {}
        for ii in self._keys:
            self.step_keys[ii] = '--'.join(
                ["%s" % self.inputs.parameters["block_id"], ii]
            )
        ii = 'run-scf-abacus'
        self.step_keys[ii] = '--'.join(
            ["%s" % self.inputs.parameters["block_id"], ii + ("-{{inputs.parameters.dflow_key}}-{{item}}" if for_mix_scf else "-{{item}}")]
        )

        self = _scf_abacus(
            self,
            self.step_keys,
            name,
            convert_scf_op,
            prep_scf_abacus_op,
            run_scf_abacus_op,
            gather_stats_scf_op,
            assistant_config=assistant_config,
            run_config=run_config,
            upload_python_package=upload_python_package,
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


def _scf_abacus(
        scf_abacus_steps,
        step_keys,
        name,
        convert_scf_op: OP,
        prep_scf_abacus_op: OP,
        run_scf_abacus_op: OP,
        gather_stats_scf_op: OP,
        assistant_config: dict = normalize_step_dict({}),
        run_config: dict = normalize_step_dict({}),
        upload_python_package: str = None,
):
    assistant_config = deepcopy(assistant_config)
    run_config = deepcopy(run_config)
    assistant_template_config = assistant_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    assistant_executor = assistant_config.pop("executor")
    run_executor = run_config.pop("executor")
    run_executor = init_executor(run_executor)
    assistant_executor = init_executor(assistant_executor)

    convert_scf = Step(
        name=name + "-convert-scf",
        template=PythonOPTemplate(
            convert_scf_op,
            output_artifact_archive={
                "tasks": None,
            },
            python_packages=upload_python_package,
            **assistant_template_config,
        ),
        parameters={
            "no_model": scf_abacus_steps.inputs.parameters["no_model"],
            "yaml_name": scf_abacus_steps.inputs.parameters["yaml_name"],
        },
        artifacts={
            "config_file": scf_abacus_steps.inputs.artifacts["config_file"],
            "system": scf_abacus_steps.inputs.artifacts["system"],
            "model": scf_abacus_steps.inputs.artifacts["model"],
        },
        key=step_keys["convert-scf"],
        executor=assistant_executor,
        **assistant_config,
    )
    scf_abacus_steps.add(convert_scf)

    # prepare for parallel scf abacus
    prep_scf_abacus = Step(
        name=name + "-prep-scf-abacus",
        template=PythonOPTemplate(
            prep_scf_abacus_op,
            python_packages=upload_python_package,
            **assistant_template_config,
        ),
        parameters={
            "yaml_name": scf_abacus_steps.inputs.parameters["yaml_name"],
        },
        artifacts={
            "config_file": scf_abacus_steps.inputs.artifacts["config_file"],
            "tasks": convert_scf.outputs.artifacts["tasks"],
        },
        key=step_keys["prep-scf-abacus"],
        executor=assistant_executor,
        **assistant_config,
    )
    scf_abacus_steps.add(prep_scf_abacus)

    run_scf_abacus = Step(
        name=name + "-run-scf-abacus",
        template=PythonOPTemplate(
            run_scf_abacus_op,
            slices=Slices(
                "{{item}}",
                input_artifact=["task_path"],
                output_artifact=["task_path"],
            ),
            python_packages=upload_python_package,
            **run_template_config,
        ),
        parameters={
            "no_model": scf_abacus_steps.inputs.parameters["no_model"],
            "yaml_name": scf_abacus_steps.inputs.parameters["yaml_name"],
        },
        artifacts={
            "config_file": scf_abacus_steps.inputs.artifacts["config_file"],
            "stru_file": scf_abacus_steps.inputs.artifacts["stru_file"],
            "task_path": prep_scf_abacus.outputs.artifacts["task_paths"],
            "model": convert_scf.outputs.artifacts["model"],
        },
        with_param=argo_range(
            argo_len(prep_scf_abacus.outputs.parameters["task_names"])),
        key=step_keys["run-scf-abacus"],
        executor=run_executor,
        util_image=assistant_template_config.get("image"),
        **run_config,
    )
    scf_abacus_steps.add(run_scf_abacus)

    # scf_abacus_steps.outputs.artifacts["task_paths"]._from = run_scf_abacus.outputs.artifacts["task_path"]

    gather_stats_scf = Step(
        name=name + "-gather-stats-scf",
        template=PythonOPTemplate(
            gather_stats_scf_op,
            output_artifact_archive={
                "00_scf": None
            },
            python_packages=upload_python_package,
            **assistant_template_config,
        ),
        parameters={
            "yaml_name": scf_abacus_steps.inputs.parameters["yaml_name"],
        },
        artifacts={
            "config_file": scf_abacus_steps.inputs.artifacts["config_file"],
            "system": scf_abacus_steps.inputs.artifacts["system"],
            "task_paths": run_scf_abacus.outputs.artifacts["task_path"],
        },
        key=step_keys["gather-scf"],
        executor=assistant_executor,
        **assistant_config,
    )
    scf_abacus_steps.add(gather_stats_scf)

    scf_abacus_steps.outputs.artifacts["00_scf"]._from = \
        gather_stats_scf.outputs.artifacts["00_scf"]

    return scf_abacus_steps
