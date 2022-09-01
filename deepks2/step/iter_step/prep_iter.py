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
# from dpgen2.exploration.scheduler import ExplorationScheduler
# from dpgen2.exploration.report import ExplorationReport
# from dpgen2.exploration.task import ExplorationTaskGroup
# from dpgen2.exploration.selector import ConfSelector
# from dpgen2.superop.block import ConcurrentLearningBlock

from deepks2.utils.step_config import normalize as normalize_step_dict
from deepks2.utils.step_config import init_executor

from copy import deepcopy



class MakeIterBlock(Steps):
    def __init__(
            self,
            name : str,
            convert_scf_op : OP,
            scf_op : OP,
            gather_stats_scf_op :OP,
            train_op : OP,
            convert_config : dict = normalize_step_dict({}),
            gather_config : dict = normalize_step_dict({}),
            upload_python_package : str = None,
    ):
        self._input_parameters={
            "block_id" : InputParameter(),
            "n_iter": InputParameter(type=int),
            "config" : InputParameter(),
        }
        self._input_artifacts={
            "stru_file" : InputArtifact(),
            "model" : InputArtifact(optional=True),
            "system" : InputArtifact(),
        }
        self._output_parameters={
            "stop_or_converge": OutputParameter(type=bool),
            "n_iter": OutputParameter(type=int),
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

        self._my_keys = ['convert-scf','gather-scf']
        self._keys = \
            self._my_keys[:1] + \
            scf_op.keys + \
            self._my_keys[1:2]+ \
            train_op.keys 
            # self._my_keys[0]


        self.step_keys = {}
        for ii in self._my_keys:
            self.step_keys[ii] = '--'.join(
                ["%s"%self.inputs.parameters["block_id"], ii]
            )

        self = _iterblock(
            self,
            self.step_keys,
            name,
            convert_scf_op,
            scf_op,
            gather_stats_scf_op,
            train_op,
            # collect_data_op,
            convert_config = convert_config,
            gather_config = gather_config,
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


def _iterblock(
        block_steps : Steps,
        step_keys : List[str],
        name : str,
        convert_scf_op : OP,
        scf_op : OP,
        gather_stats_scf_op : OP,
        train_op : OP,
        convert_config : dict = normalize_step_dict({}),
        gather_config : dict = normalize_step_dict({}),
        upload_python_package : str = None,
):
    convert_config = deepcopy(convert_config)
    gather_config = deepcopy(gather_config)
    convert_scf_template_config = convert_config.pop('template_config')
    gather_scf_template_config = gather_config.pop('template_config')
    convert_scf_executor = init_executor(convert_config.pop('executor'))
    gather_scf_executor = init_executor(gather_config.pop('executor'))
        
    convert_scf = Step(
        name = name + '-convert-scf',
        template=PythonOPTemplate(
            convert_scf_op,
            output_artifact_archive={
                "system": None,
                # "convert_log": None
            },
            python_packages = upload_python_package,
            **convert_scf_template_config,
        ),
        parameters={
            "n_iter": block_steps.inputs.parameters['n_iter'],
            "scf_config" : block_steps.inputs.parameters['config'],
            # "type_map": block_steps.inputs.parameters["type_map"],
            # "traj_fmt": 'lammps/dump',
        },
        artifacts={
            "stru_file" : block_steps.inputs.artifacts['stru_file'],
            "system" : block_steps.inputs.artifacts['system'],
            "model" : block_steps.inputs.artifacts['model'],
            # "model_devis" : prep_run_lmp.outputs.artifacts['model_devis'],
        },
        key = step_keys['convert-scf'],
        executor = convert_scf_executor,
        **convert_config,
    )
    block_steps.add(convert_scf)
        
    prep_run_scf = Step(
        name = name + '-scf',
        template = scf_op,
        parameters={
            "block_id" : block_steps.inputs.parameters['block_id'],
            "scf_abacus_config" : block_steps.inputs.parameters['config'],
            # "fp_config": block_steps.inputs.parameters['fp_config'],
            # "type_map": block_steps.inputs.parameters["type_map"],
        },
        artifacts={
            "system": convert_scf.outputs.artifacts['system'], 
            "stru_file" : block_steps.inputs.artifacts['stru_file']        
            # "confs" : select_confs.outputs.artifacts['confs'],
        },
        key = '--'.join(["%s"%block_steps.inputs.parameters["block_id"], "prep-run-scf"]),
    )
    block_steps.add(prep_run_scf)

    gather_stats_scf = Step(
        name = name + '-gather-stats-scf',
        template=PythonOPTemplate(
            gather_stats_scf_op,
            output_artifact_archive={
                "00_scf": None
            },
            python_packages = upload_python_package,
            **gather_scf_template_config,
        ),
        parameters={
            "gather_config": block_steps.inputs.parameters['config'],
            # "task_paths" : prep_run_scf.outputs.parameters["task_paths"],

            # "type_map": block_steps.inputs.parameters["type_map"],
            # "traj_fmt": 'lammps/dump',
        },
        artifacts={
            "task_paths" : prep_run_scf.outputs.artifacts['task_paths'],
            "system": block_steps.inputs.artifacts['system']
        },
        key = step_keys['gather-scf'],
        executor = gather_scf_executor,
        **gather_config,
    )
    block_steps.add(gather_stats_scf)

    prep_run_train = Step(
        name = name + '-train',
        template = train_op,
        parameters={
            "block_id" : block_steps.inputs.parameters['block_id'],
            "n_iter" : block_steps.inputs.parameters['n_iter'],
            "train_config" : block_steps.inputs.parameters["config"],
            # "fp_config": block_steps.inputs.parameters['fp_config'], 
            # "type_map": block_steps.inputs.parameters["type_map"],
        },
        artifacts={
            "00_scf": gather_stats_scf.outputs.artifacts['00_scf'], 
            "model": block_steps.outputs.artifacts['model'], 
            # "confs" : select_confs.outputs.artifacts['confs'],
        },
        key = '--'.join(["%s"%block_steps.inputs.parameters["block_id"], "prep-run-train"]),
    )
    block_steps.add(prep_run_train)
    

    block_steps.outputs.parameters["n_iter"].value_from_parameter= \
        block_steps.inputs.parameters["n_iter"]
    block_steps.outputs.parameters["stop_or_converge"].value_from_parameter= \
        prep_run_train.outputs.parameters["stop_or_converge"]
    block_steps.outputs.artifacts["model"]._from= \
        prep_run_train.outputs.artifacts["model"]
    block_steps.outputs.artifacts["00_scf"]._from= \
        gather_stats_scf.outputs.artifacts["00_scf"]
    block_steps.outputs.artifacts["01_train"]._from= \
        prep_run_train.outputs.artifacts["01_train"]
    

    return block_steps


