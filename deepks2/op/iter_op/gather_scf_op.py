from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
import os, shutil
from typing import List
from pathlib import Path
from deepks2.utils.file_utils import load_yaml
from deepks2.utils.path_utils import copy_dir
from deepks2.constants import DATA_TRAIN, DATA_TEST, SYS_TRAIN, SYS_TEST, SCF_STEP_DIR

class GatherStatsScfAbacus(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "yaml_name" : str,
            "config_file": Artifact(Path),
            "system":Artifact(Path),
            "task_paths" : Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "00_scf" : Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        cwd = os.getcwd()
        # OP input
        yaml_name = ip["yaml_name"]
        config_file = ip["config_file"]
        system = ip["system"]
        task_paths = ip["task_paths"]

        os.chdir(system)

        os.mkdir(SCF_STEP_DIR)
        scf_dir = Path(SCF_STEP_DIR)
        train_dump = str(scf_dir/DATA_TRAIN)
        test_dump = str(scf_dir/DATA_TEST)
        gather_scf_config = load_yaml(config_file/yaml_name)

        systems_train = [SYS_TRAIN+"/"+s for s in os.listdir(SYS_TRAIN)] if os.path.exists(SYS_TRAIN) else []
        systems_test = [SYS_TEST+"/"+s for s in os.listdir(SYS_TEST)] if os.path.exists(SYS_TEST) else []
        
        for path in task_paths:
            sys = SYS_TRAIN if os.path.basename(path).startswith("train") else SYS_TEST
            for ii in os.listdir(path/"ABACUS"):
                id = int(ii[-4:])
                GatherStatsScfAbacus.updateFile(path/"ABACUS"/ii/"conv", ii, str(id))
                for dir in os.listdir(sys):
                    if dir.endswith(ii[:-4]):
                        group = dir
                        break
                shutil.copytree(path/"ABACUS"/ii,system/sys/group/"ABACUS"/str(id))


        from deepks.iterate.template_abacus import gather_stats_abacus

        gather_scf_config.update(
            systems_train = systems_train,
            systems_test = systems_test,
            train_dump = train_dump,
            test_dump = test_dump,
        )
        
        gather_stats_abacus(**gather_scf_config)

        # copy atom.npy
        sys_train_paths = [s for s in os.listdir(SYS_TRAIN)] if os.path.exists(SYS_TRAIN) else []
        sys_test_paths = [s for s in os.listdir(SYS_TEST)] if os.path.exists(SYS_TEST) else []
        for path in sys_train_paths:
            shutil.copy2(f"{SYS_TRAIN}/{path}/atom.npy",f"{train_dump}/{path}/atom.npy")
        for path in sys_test_paths:
            shutil.copy2(f"{SYS_TEST}/{path}/atom.npy",f"{test_dump}/{path}/atom.npy")

        os.chdir(cwd)
        return OPIO({
            '00_scf' : system/SCF_STEP_DIR,
        })

    @staticmethod
    def updateFile(file,old_str,new_str):
        file_data = ""
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if old_str in line:
                    line = line.replace(old_str,new_str)
                file_data += line
        with open(file,"w",encoding="utf-8") as f:
            f.write(file_data)
                

class GatherMixScfAbacus(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "00_scfs" : Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "00_scf" : Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        # OP input
        mixscf = ip["00_scfs"]

        os.mkdir(SCF_STEP_DIR)
        scf_dir = Path(SCF_STEP_DIR)

        for data in [DATA_TRAIN,DATA_TEST]:
            os.mkdir(scf_dir/data)
            for path in mixscf:
                for group in os.listdir(path/data):
                    copy_dir(path/data/group, scf_dir/data/group)

        return OPIO({
            '00_scf' : scf_dir,
        })
