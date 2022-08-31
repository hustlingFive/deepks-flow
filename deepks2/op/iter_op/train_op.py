import os
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


class PrepTrain(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "train_config": Union[dict, List[dict]],
            "00_scf": Artifact(Path),
            "model": Artifact(Path, optional=True),
            "n_iter": int
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "01_train": Artifact(Path),
            "command": str,
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
    ) -> OPIO:

        scf = ip["00_scf"]
        old_model = ip["model"]
        train_config = ip["train_config"]
        n_iter = ip["n_iter"]
        restart = False if n_iter == 0 else True

        source_arg = "train_input.yaml"
        restart_model = "old_model.pth"

        PrepTrain.save_yaml(train_config, scf/source_arg)
        if restart:
            shutil.copy(old_model, scf/restart_model)

        os.mkdir("01.train")
        train = Path("01.train")
        shutil.copytree(scf, train, dirs_exist_ok = True)

        # make the command
        TRN_CMD = " ".join([
            "{python} -u",
            "-m deepks.model.train"
            # os.path.join(QCDIR, "train/train.py") # this is the backup choice
        ])

        arg_file = "train_input.yaml"
        save_model = "model.pth"
        data_train = "data_train"
        data_test = "data_test"
        group_data = False

        # set up basic args
        command = TRN_CMD.format(python="python")
        # set up optional args
        if arg_file:
            command += f" {arg_file}"
        if restart:
            command += f" -r {restart_model}"
        if data_train:
            command += f" -d {data_train}" + ("" if group_data else "/*")
        if data_test:
            command += f" -t {data_test}" + ("" if group_data else "/*")
        if save_model:
            command += f" -o {save_model}"

        op = OPIO({
            "01_train": train,
            "command": command,
        })
        return op

    @staticmethod
    def save_yaml(data, file_path):
        dirname = os.path.dirname(file_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(file_path, 'w') as fp:
            yaml.safe_dump(data, fp)


class RunTrain(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "01_train": Artifact(Path),
            "train_config": Union[dict, List[dict]],
            "command": str,
            "n_iter" : int,
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "01_train": Artifact(Path),
            "stop_or_converge": bool,
            "model" : Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
    ) -> OPIO:
        def train_curve_wave(path):
            err = []
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    else:
                        a = line.split()
                        err.append(a)
            n = len(err)
            ave = 0.
            for i in range(n):
                ave += float(err[i][1])/n
            max_wave_rate = 0.
            for i in range(n):
                if float(err[i][1]) > ave:
                    rate = (float(err[i][1])-ave)/ave
                else:
                    rate = (ave-float(err[i][1]))/ave
                if max_wave_rate < rate:
                    max_wave_rate = rate
            return max_wave_rate

        cwd=os.getcwd()

        train = ip["01_train"]
        n_iter = ip["n_iter"]
        train_config = ip["train_config"]
        n_iter_max = train_config.pop("n_iter_max",3)
        command = ip["command"]
        train_curve_wave_max = train_config.pop("train_curve_wave_max", 0.2)

        log_train = "log.train"
        save_model = "model.pth"

        os.chdir(train)
        f=open(log_train,"wb")
        p=subprocess.Popen(command,stdout=subprocess.PIPE,shell=True)
        f.write(p.stdout.read())
        p.wait()
        f.close()
        


        wave = train_curve_wave(train/log_train)
        print("Current train_curve_wave is %.2f %%" % (wave*100))

        stop_or_converge = True if wave <= train_curve_wave_max else False

        
        os.chdir(cwd)
        return OPIO({
            "01_train": train,
            "stop_or_converge": stop_or_converge or n_iter>=n_iter_max,
            "model": train/save_model
        })
