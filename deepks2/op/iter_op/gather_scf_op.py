from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
import os, shutil
import numpy as np
from typing import Tuple, List, Set,Union
from pathlib import Path

class GatherStatsScfAbacus(OP):
    

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "gather_config": Union[dict, List[dict]],
            "task_paths" : Artifact(List[Path]),
            "system":Artifact(Path)
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
        task_paths = ip["task_paths"]
        gather_config = ip["gather_config"]
        system = ip["system"]

        os.chdir(system)

        os.mkdir("00.scf")
        scf_dir = Path("00.scf")
        train_dump = str(scf_dir/"data_train")
        test_dump = str(scf_dir/"data_test")
        SYS_TRAIN = "systems_train"
        SYS_TEST = "systems_test"


        systems_train = [SYS_TRAIN+"/"+s for s in os.listdir(SYS_TRAIN)]
        systems_test = [SYS_TEST+"/"+s for s in os.listdir(SYS_TEST)]
        
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

        gather_config.update(
            systems_train = systems_train,
            systems_test = systems_test,
            train_dump = train_dump,
            test_dump = test_dump,
            **gather_config
        )
        
        gather_stats_abacus(**gather_config)

        scf_dir = scf_dir.resolve()
        
        os.chdir(cwd)
        return OPIO({
            '00_scf' : scf_dir,
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
                


