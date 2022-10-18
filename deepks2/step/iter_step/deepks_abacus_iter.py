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
    if_expression,
)
from dflow.python import(
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    Slices,
    BigParameter,
)
import pickle
import jsonpickle
import os
from typing import (
    List
)
from pathlib import Path
from deepks2.utils.step_config import normalize as normalize_step_dict
from deepks2.utils.step_config import init_executor
from deepks2.constants import (SCF_ARGS_NAME,INIT_SCF_NAME,TRN_ARGS_NAME,INIT_TRN_NAME)

from copy import deepcopy

class MakeBlockId(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "n_iter": str,
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "block_id": str,
            "scf_yaml_name":str,
            "train_yaml_name":str,
            "n_iter":str,
            "no_model":bool,
            "go_ahead":bool,
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
    ) -> OPIO:
        n_iter = ip["n_iter"]

        n_iter = n_iter.split("-")
        n_now, n_max = int(n_iter[0]), int(n_iter[1])
        name = MakeBlockId.get_name(n_now+1)
        if n_now < n_max:
            n_iter[0]= str(n_now+1).zfill(2)
            go_ahead = True
        else:
            go_ahead = False
        n_iter="-".join(n_iter)

        return OPIO({
            "block_id": name[0],
            "scf_yaml_name": name[1],
            "train_yaml_name": name[2],
            "n_iter": n_iter,
            "go_ahead":go_ahead,
            "no_model":False,

        })
    
    @staticmethod
    def get_name(n_iter_now):
        if n_iter_now == 0:
            name = "iter.init"
            scf_name = INIT_SCF_NAME # for ABACUS only
            train_name = INIT_TRN_NAME
        else:
            name = "iter.%s" % str(n_iter_now-1).zfill(2)
            scf_name = SCF_ARGS_NAME # for ABACUS only
            train_name = TRN_ARGS_NAME
        return name, scf_name, train_name

class DeepksAbacusIter(Steps):
    def __init__(
            self,
            name : str,
            scf_op : OP,
            train_op : OP,
            assistant_config : dict = normalize_step_dict({}),
            upload_python_package : str = None,
    ):
        self._input_parameters={
            "n_iter": InputParameter(type=str),
            "block_id" : InputParameter(type=str),
            "scf_yaml_name" : InputParameter(type=str),
            "train_yaml_name" : InputParameter(type=str),
            "no_model" : InputParameter(type=bool),
            "no_test" : InputParameter(type=bool),
        }
        self._input_artifacts={
            "model": InputArtifact(optional=True),
            "system" : InputArtifact(),
            "config_file": InputArtifact(),
            "stru_file" : InputArtifact(),
        }
        self._output_parameters={
        }
        self._output_artifacts={
            "model": OutputArtifact(),
            "00_scf" : OutputArtifact(),
            "01_train" : OutputArtifact(),
        }
        
        super().__init__(
            name = name,
            inputs = Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )

        # self._my_keys = ["convert-scf","gather-scf"]
        # self._keys = \
        #     self._my_keys[:1] + \
        #     scf_op.keys + \
        #     self._my_keys[1:2]+ \
        #     train_op.keys 
        #     # self._my_keys[0]


        # self.step_keys = {}
        # for ii in self._my_keys:
        #     self.step_keys[ii] = "--".join(
        #         ["%s"%self.inputs.parameters["block_id"], ii]
        #     )
        self._my_keys = ["id"]
        # self._my_keys = ["iter"]
        self._keys = \
            scf_op.keys + \
            train_op.keys + \
            self._my_keys[:1]
        self.iter_key = "loop"
        self.step_keys = {}
        for ii in self._my_keys:
            self.step_keys[ii] = "--".join(
                ["%s" % self.inputs.parameters["block_id"], ii]
            )

        self = _iter(
            self,
            self.step_keys,
            name,
            scf_op,
            train_op,
            assistant_config=assistant_config,
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

    @property
    def iter_keys(self):
        return [self.iter_key] + self.iter.keys


def _iter(
        steps : Steps,
        step_keys : List[str],
        name : str,
        scf_op : OP,
        train_op : OP,
        assistant_config: dict = normalize_step_dict({}),
        upload_python_package: str = None
):
    assistant_config = deepcopy(assistant_config)
    assistant_template_config = assistant_config.pop("template_config")
    assistant_executor = assistant_config.pop("executor")
    assistant_executor = init_executor(assistant_executor)
    # convert_scf = Step(
    #     name = name + "-convert-scf",
    #     template=PythonOPTemplate(
    #         convert_scf_op,
    #         output_artifact_archive={
    #             "system": None,
    #         },
    #         python_packages = upload_python_package,
    #         **convert_scf_template_config,
    #     ),
    #     parameters={
    #         "n_iter": block_steps.inputs.parameters["n_iter"],
    #         "scf_config" : block_steps.inputs.parameters["config"],
    #     },
    #     artifacts={
    #         "stru_file" : block_steps.inputs.artifacts["stru_file"],
    #         "system" : block_steps.inputs.artifacts["system"],
    #         "model" : block_steps.inputs.artifacts["model"],
    #     },
    #     key = step_keys["convert-scf"],
    #     executor = convert_scf_executor,
    #     **convert_config,
    # )
    # block_steps.add(convert_scf)
        
    scf = Step(
        name = name + "-scf",
        template = scf_op,
        parameters={
            "block_id" : steps.inputs.parameters["block_id"],
            "yaml_name" : steps.inputs.parameters["scf_yaml_name"],
            "no_model": steps.inputs.parameters["no_model"],
        },
        artifacts={
            "model" : steps.inputs.artifacts["model"],
            "system": steps.inputs.artifacts["system"], 
            "config_file": steps.inputs.artifacts["config_file"], 
            "stru_file" : steps.inputs.artifacts["stru_file"]        
        },
        key = "--".join(["%s"%steps.inputs.parameters["block_id"], "scf"]),
    )
    steps.add(scf)

    # gather_stats_scf = Step(
    #     name = name + "-gather-stats-scf",
    #     template=PythonOPTemplate(
    #         gather_stats_scf_op,
    #         output_artifact_archive={
    #             "00_scf": None
    #         },
    #         python_packages = upload_python_package,
    #         **gather_scf_template_config,
    #     ),
    #     parameters={
    #         "gather_config": block_steps.inputs.parameters["config"],
    #     },
    #     artifacts={
    #         "task_paths" : prep_run_scf.outputs.artifacts["task_paths"],
    #         "system": block_steps.inputs.artifacts["system"]
    #     },
    #     key = step_keys["gather-scf"],
    #     executor = gather_scf_executor,
    #     **gather_config,
    # )
    # block_steps.add(gather_stats_scf)

    train = Step(
        name = name + "-train",
        template = train_op,
        parameters={
            "block_id" : steps.inputs.parameters["block_id"],
            "yaml_name" : steps.inputs.parameters["train_yaml_name"],
            "no_model": steps.inputs.parameters["no_model"],
            "no_test": steps.inputs.parameters["no_test"],
        },
        artifacts={
            "00_scf": scf.outputs.artifacts["00_scf"], 
            "model": steps.inputs.artifacts["model"], 
            "config_file": steps.inputs.artifacts["config_file"], 
        },
        key = "--".join(["%s"%steps.inputs.parameters["block_id"], "train"]),
    )
    steps.add(train)

    block_id_step = Step(
        name=name + "-iter-id",
        template=PythonOPTemplate(
            MakeBlockId,
            python_packages=upload_python_package,
            **assistant_template_config,
        ),
        parameters={
            "n_iter":steps.inputs.parameters["n_iter"]
        },
        artifacts={
        },
        key=step_keys["id"],
        executor=assistant_executor,
        **assistant_config,
    )
    steps.add(block_id_step)

    next_step = Step(
        name = name + "-next",
        template=steps,
        parameters={
            "block_id": block_id_step.outputs.parameters["block_id"],
            "n_iter": block_id_step.outputs.parameters["n_iter"],
            "scf_yaml_name": block_id_step.outputs.parameters["scf_yaml_name"],
            "train_yaml_name": block_id_step.outputs.parameters["train_yaml_name"],
            "no_model": block_id_step.outputs.parameters["no_model"],
            "no_test" : steps.inputs.parameters["no_test"],
        },
        artifacts={
            "model": train.outputs.artifacts["model"],
            "system": steps.inputs.artifacts["system"],
            "config_file": steps.inputs.artifacts["config_file"],
            "stru_file": steps.inputs.artifacts["stru_file"],
        },
        when="%s == true" % (block_id_step.outputs.parameters["go_ahead"]),
    )
    steps.add(next_step)

    steps.outputs.artifacts["model"].from_expression = \
        if_expression(
            _if = (block_id_step.outputs.parameters["go_ahead"] == False),
            _then = train.outputs.artifacts["model"],
            _else = next_step.outputs.artifacts["model"],
        )
    steps.outputs.artifacts["00_scf"].from_expression = \
        if_expression(
            _if = (block_id_step.outputs.parameters["go_ahead"] == False),
            _then = scf.outputs.artifacts["00_scf"],
            _else = next_step.outputs.artifacts["00_scf"],
        )
    steps.outputs.artifacts["01_train"].from_expression = \
        if_expression(
            _if = (block_id_step.outputs.parameters["go_ahead"] == False),
            _then = train.outputs.artifacts["01_train"],
            _else = next_step.outputs.artifacts["01_train"],
        )

    return steps


