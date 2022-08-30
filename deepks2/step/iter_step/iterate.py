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
from typing import (
    List
)


from deepks2.utils.step_config import normalize as normalize_step_dict
from deepks2.utils.step_config import init_executor



from copy import deepcopy



class MakeBlockId(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "n_iter": int,
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "block_id": str,
            "n_iter":int
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
    ) -> OPIO:
        n_iter = ip["n_iter"]+1

        return OPIO({
            "n_iter": n_iter,
            "block_id": MakeBlockId.get_block_id(n_iter)
        })
    
    @staticmethod
    def get_block_id(n_iter_now):
            if n_iter_now == 0:
                dir = "iter.init"
            else:
                dir = "iter.%s" % str(n_iter_now-1).zfill(2)
            return dir


class ManageBlock(Steps):
    def __init__(
            self,
            name: str,
            iter_block_op: Steps,
            step_config: dict = normalize_step_dict({}),
            upload_python_package: str = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(),
            "n_iter": InputParameter(type=int),
            "config" : InputParameter(),
            # "iter_config" : steps.inputs.parameters["iter_config"],
            # "n_iter_max": InputParameter(type=int),
            # "template_script": InputParameter(),
            # "train_config": InputParameter(),
            # "lmp_config": InputParameter(),
            # "conf_selector": InputParameter(),
            # "fp_config": InputParameter(),
        }
        self._input_artifacts = {
            "stru_file" : InputArtifact(),
            "model": InputArtifact(optional=True),
            "system": InputArtifact(),
            # "fp_inputs": InputArtifact(),
        }
        self._output_parameters = {
            "n_iter": OutputParameter(type=int)
        }
        self._output_artifacts = {
            # "exploration_scheduler": OutputArtifact(),
            "model": OutputArtifact(),
            # "iter_data": OutputArtifact(),
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

        self._my_keys = ['id']
        self._keys = \
            iter_block_op.keys + \
            self._my_keys[:1]
        self.step_keys = {}
        for ii in self._my_keys:
            self.step_keys[ii] = '--'.join(
                ["%s" % self.inputs.parameters["block_id"], ii]
            )

        self = _block(
            self,
            self.step_keys,
            name,
            iter_block_op,
            step_config=step_config,
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


# id
class Iterate(Steps):
    def __init__(
            self,
            name: str,
            init_iter_block_op: Steps,
            iter_block_op: Steps,
            step_config: dict = normalize_step_dict({}),
            upload_python_package: str = None,
    ):
        self.iter = ManageBlock(
            name+'-iter',
            iter_block_op,
            step_config=step_config,
            upload_python_package=upload_python_package,
        )

        self._input_parameters = {
            "block_id": InputParameter(),
            "n_iter": InputParameter(type=int),
            "init_iter_config" : InputParameter(),
            "iter_config" : InputParameter(),
            # "n_iter_max": InputParameter(type=int),
            # "train_config": InputParameter(),
            # "lmp_config": InputParameter(),
            # "fp_config": InputParameter(),
        }
        self._input_artifacts = {
            "stru_file" : InputArtifact(),
            "model": InputArtifact(optional=True),
            "system": InputArtifact(),
        }
        self._output_parameters = {
            "n_iter_next": OutputParameter(),
        }
        self._output_artifacts = {
            "model": OutputArtifact(),
            # "models": OutputArtifact(),
            # "iter_data": OutputArtifact(),
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

        self._init_keys = ['id']
        # self._my_keys = ['iter']
        self._keys = \
            init_iter_block_op.keys + \
            self._init_keys[:1]
        self.iter_key = 'loop'
        self.step_keys = {}
        for ii in self._init_keys:
            self.step_keys[ii] = '--'.join(['init', ii])

        self = _iter(
            self,
            self.step_keys,
            name,
            init_iter_block_op,
            self.iter,
            self.iter_key,
            step_config=step_config,
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
    def init_keys(self):
        return self._init_keys

    @property
    def iter_keys(self):
        return [self.iter_key] + self.iter.keys



def _block(
        steps,
        step_keys,
        name: str,
        iter_block_op: OP,
        step_config: dict = normalize_step_dict({}),
        upload_python_package: str = None,
):
    step_config = deepcopy(step_config)
    step_template_config = step_config.pop('template_config')
    step_executor = init_executor(step_config.pop('executor',None))


    iter_block_step = Step(
        name=name + '-iter-block',
        template=iter_block_op,
        parameters={
            "block_id": steps.inputs.parameters['block_id'],
            "n_iter": steps.inputs.parameters['n_iter'],
            "config": steps.inputs.parameters["config"],

            # "template_script": steps.inputs.parameters["template_script"],
            # "train_config": steps.inputs.parameters["train_config"],
            # "lmp_config": steps.inputs.parameters["lmp_config"],
            # "conf_selector": steps.inputs.parameters["conf_selector"],
            # "fp_config": steps.inputs.parameters["fp_config"],
        },
        artifacts={
            "stru_file": steps.inputs.artifacts["stru_file"],
            "system": steps.inputs.artifacts["system"],
            "model": steps.inputs.artifacts["model"]
        },
        # key=step_keys['iter'],
    )
    steps.add(iter_block_step)

    block_id_step = Step(
        name=name + '-iter-id',
        template=PythonOPTemplate(
            MakeBlockId,
            python_packages=upload_python_package,
            **step_template_config,
        ),
        parameters={
            "n_iter":iter_block_step.outputs.parameters["n_iter"]
        },
        artifacts={
        },
        key=step_keys['id'],
        executor=step_executor,
        **step_config,
    )
    steps.add(block_id_step)

    next_step = Step(
        name=name+'-next',
        template=steps,
        parameters={
            "block_id": block_id_step.outputs.parameters['block_id'],
            "n_iter": block_id_step.outputs.parameters["n_iter"],
            "config": steps.inputs.parameters["config"],
        },
        artifacts={
            "stru_file": steps.inputs.artifacts["stru_file"],
            "system": steps.inputs.artifacts['system'],
            "model": iter_block_step.outputs.artifacts['model'],
        },
        when="%s == false" % (iter_block_step.outputs.parameters['stop_or_converge']),
    )
    steps.add(next_step)


    steps.outputs.parameters['n_iter'].value_from_expression = \
        if_expression(
            _if = (iter_block_step.outputs.parameters['stop_or_converge'] == True),
            _then = block_id_step.outputs.parameters['n_iter'],
            _else = next_step.outputs.parameters['n_iter'],
        )
    steps.outputs.artifacts['model'].from_expression = \
        if_expression(
            _if = (iter_block_step.outputs.parameters['stop_or_converge'] == True),
            _then = iter_block_step.outputs.artifacts['model'],
            _else = next_step.outputs.artifacts['model'],
        )

    return steps


def _iter(
        steps,
        step_keys,
        name,
        init_iter_block_op,
        iter_op,
        iter_key,
        step_config: dict = normalize_step_dict({}),
        upload_python_package: str = None
):
    step_config = deepcopy(step_config)
    step_template_config = step_config.pop('template_config')
    step_executor = init_executor(step_config.pop('executor',None))



    # link_step = Step(
    #     name=name + '-link-iter',
    #     template=PythonOPTemplate(
    #         MakeBlockLink,
    #         python_packages=upload_python_package,
    #         **step_template_config,
    #     ),
    #     parameters={
    #         "n_iter": 0
    #     },
    #     artifacts={
    #     },
    #     key=step_keys['link'],
    #     executor=step_executor,
    #     **step_config,
    # )
    # steps.add(link_step)

    init_iter_step = Step(
        name=name + '-init-iter-block',
        template=init_iter_block_op,
        parameters={
            "block_id": steps.inputs.parameters["block_id"],
            "n_iter": steps.inputs.parameters["n_iter"],
            "config" : steps.inputs.parameters["init_iter_config"],
            # "iter_config" : steps.inputs.parameters["iter_config"],
            # "n_iter_max": steps.inputs.parameters["n_iter_max"],
            # "template_script": steps.inputs.parameters["template_script"],
            # "train_config": steps.inputs.parameters["train_config"],
            # "lmp_config": steps.inputs.parameters["lmp_config"],
            # "conf_selector": steps.inputs.parameters["conf_selector"],
            # "fp_config": steps.inputs.parameters["fp_config"],
        },
        artifacts={
            "stru_file": steps.inputs.artifacts["stru_file"],
            "system": steps.inputs.artifacts["system"],
            # "iter_data": steps.inputs.artifacts["iter_data"],
        },
        # key=step_keys['inititerblock'],
    )
    steps.add(init_iter_step)

    block_id_step = Step(
        name=name + '-iter-id',
        template=PythonOPTemplate(
            MakeBlockId,
            python_packages=upload_python_package,
            **step_template_config,
        ),
        parameters={
            "n_iter":init_iter_step.outputs.parameters["n_iter"]
        },
        artifacts={
        },
        key=step_keys['id'],
        executor=step_executor,
        **step_config,
    )
    steps.add(block_id_step)

    iter_step = Step(
        name=name + '-iter',
        template=iter_op,
        parameters={
            "block_id": block_id_step.outputs.parameters['block_id'],
            "n_iter": block_id_step.outputs.parameters['n_iter'],
            # "init_iter_config" : steps.inputs.parameters["init_iter_config"],
            "config" : steps.inputs.parameters["iter_config"],
            # "n_iter_max": steps.inputs.parameters['n_iter_max'],
            # "template_script": steps.inputs.parameters['template_script'],
            # "train_config": steps.inputs.parameters['train_config'],
            # "conf_selector": scheduler_step.outputs.parameters['conf_selector'],
            # "lmp_config": steps.inputs.parameters['lmp_config'],
            # "fp_config": steps.inputs.parameters['fp_config'],
        },
        artifacts={
            "stru_file": steps.inputs.artifacts["stru_file"],
            "system": steps.inputs.artifacts['system'],
            "model": init_iter_step.outputs.artifacts['model'],
        },
        key='--'.join(["%s" %
                      block_id_step.outputs.parameters['block_id'], iter_key]),
    )
    steps.add(iter_step)

    steps.outputs.parameters['n_iter_next'].value_from_expression = \
        if_expression(
            _if = (init_iter_step.outputs.parameters['stop_or_converge'] == True),
            _then = block_id_step.outputs.parameters['n_iter'],
            _else = iter_step.outputs.parameters['n_iter'],
        )
    steps.outputs.artifacts['model'].from_expression = \
        if_expression(
            _if = (init_iter_step.outputs.parameters['stop_or_converge'] == True),
            _then = init_iter_step.outputs.artifacts['model'],
            _else = iter_step.outputs.artifacts['model'],
        )

    return steps