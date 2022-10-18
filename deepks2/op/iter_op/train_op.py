import imp
import os, sys
import yaml
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, List, Set, Union
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
)
from deepks2.constants import (DEFAULT_TRAIN_ARGS, 
                                MODEL_FILE, 
                                TRN_ARGS_NAME, 
                                RESTART_MODEL, 
                                DATA_TRAIN, 
                                DATA_TEST, 
                                TRN_STEP_DIR,
                                LOG_TRAIN)
from deepks2.utils.file_utils import load_yaml

class PrepTrain(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "yaml_name":str,
            "no_model": bool,
            "00_scf": Artifact(Path),
            "model": Artifact(Path, optional=True),
            "config_file":Artifact(Path)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "01_train": Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
    ) -> OPIO:
        # OP input
        restart = not ip["no_model"]
        scf = ip["00_scf"]
        old_model = ip["model"]

        if restart:
            shutil.copy(old_model, scf/RESTART_MODEL)

        os.mkdir(TRN_STEP_DIR)
        train = Path(TRN_STEP_DIR)
        shutil.copytree(scf, train, dirs_exist_ok = True)


        op = OPIO({
            "01_train": train,
        })
        return op

class RunTrain(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "no_model": bool,
            "no_test": bool,
            "yaml_name" : str,
            "01_train": Artifact(Path),
            "config_file": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "01_train": Artifact(Path),
            "model" : Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
    ) -> OPIO:
        cwd=os.getcwd()
        # OP input
        train = ip["01_train"]
        restart = not ip["no_model"]
        no_test = ip["no_test"]
        yaml_name = ip["yaml_name"]
        config_file = ip["config_file"]

        os.chdir(train)
        group_data = False

        train_config = load_yaml(config_file/yaml_name)
        train_config.update(
            train_paths = DATA_TRAIN + ("" if group_data else "/*"), 
            test_paths = (DATA_TEST + ("" if group_data else "/*")) if not no_test else None,
            restart = RESTART_MODEL if restart else None, 
            ckpt_file = MODEL_FILE,
        )
        
        from deepks.model.train import main as deepks_train
        deepks_train(**train_config)

        # # for auto converge judge
        # n_iter_max = train_config.pop("n_iter_max",3)
        # train_curve_wave_max = train_config.pop("train_curve_wave_max", 0.2)
        # wave = train_curve_wave(train/log_train)
        # print("Current train_curve_wave is %.2f %%" % (wave*100))
        # stop_or_converge = True if wave <= train_curve_wave_max else False

        os.chdir(cwd)
        return OPIO({
            "01_train": train,
            "model": train/MODEL_FILE
        })

# # for the auto trn_err wave detact
    # @staticmethod
    # def train_curve_wave(path):
    #     err = []
    #     with open(path, 'r') as f:
    #         for line in f:
    #             if line.startswith('#'):
    #                 continue
    #             else:
    #                 a = line.split()
    #                 err.append(a)
    #     n = len(err)
    #     ave = 0.
    #     for i in range(n):
    #         ave += float(err[i][1])/n
    #     max_wave_rate = 0.
    #     for i in range(n):
    #         if float(err[i][1]) > ave:
    #             rate = (float(err[i][1])-ave)/ave
    #         else:
    #             rate = (ave-float(err[i][1]))/ave
    #         if max_wave_rate < rate:
    #             max_wave_rate = rate
    #     return max_wave_rate
