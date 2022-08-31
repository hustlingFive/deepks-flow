from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
import os ,shutil
from typing import Tuple, List, Set, Union
from pathlib import Path

class ConvertScfAbacus(OP):


    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "scf_config" : Union[dict, List[dict]],
            "n_iter": int,
            "stru_file" : Artifact(Path),
            "model" : Artifact(Path, optional=True),
            "system" : Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "system" : Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        convert_scf_config = ip["scf_config"]
        n_iter = ip["n_iter"]
        system = ip["system"]
        model_file = ip["model"]
        stru_file = ip["stru_file"]

        cwd = os.getcwd()
        os.chdir(system)

        CMODEL_FILE = "model.ptg"
        SYS_TRAIN = "systems_train"
        SYS_TEST = "systems_test"

        sys_train = system / SYS_TRAIN
        sys_test = system / SYS_TEST
        systems_train = []
        systems_test = []
        for root,dirs,files in os.walk(sys_train):
            for dir in dirs:
                systems_train.append(os.path.join(root,dir))
        for root,dirs,files in os.walk(sys_test):
            for dir in dirs:
                systems_test.append(os.path.join(root,dir))

        no_model = True if n_iter == 0 else False

        orb_files=convert_scf_config["orb_files"]
        pp_files=convert_scf_config["pp_files"]
        proj_file=convert_scf_config["proj_file"]
        orb_files=["../../"+str(os.path.basename(s)) for s in orb_files]
        pp_files=["../../"+str(os.path.basename(s)) for s in pp_files]
        proj_file=["../../"+str(os.path.basename(s)) for s in proj_file]
        if not no_model:
            convert_scf_config["model_file"]="../../"+CMODEL_FILE


        from deepks.iterate.template_abacus import convert_data

        convert_scf_config.update(
            systems_train=systems_train, 
            systems_test=systems_test,
            model_file=model_file,
            no_model=no_model, 
            orb_files=orb_files,
            pp_files=pp_files,
            proj_file=proj_file,
            )
            
        convert_data(**convert_scf_config)

        os.chdir(cwd)

        return OPIO({
            "system" : system,
        })
            

