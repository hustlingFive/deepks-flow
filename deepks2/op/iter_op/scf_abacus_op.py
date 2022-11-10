import os, shutil,subprocess
from pathlib import Path
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    TransientError,
    FatalError,
)
from typing import (
    Tuple, 
    List, 
)
from typing import Tuple, List, Union
from deepks2.constants import(
    SYS_TRAIN,
    SYS_TEST
)
from deepks2.utils.file_utils import load_yaml
from deepks2.constants import SCF_OUT_LOG, SCF_ERR_LOG

class PrepScfAbacus(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "yaml_name" : str,
            "config_file" : Artifact(Path),
            "tasks" : Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_names" : List[str],
            "task_paths" : Artifact(List[Path]),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        # OP input
        yaml_name = ip["yaml_name"]
        config_file = ip["config_file"]
        system = ip["tasks"]

        prep_scf_config = load_yaml(config_file/yaml_name)

        group_size = prep_scf_config.pop("group_size")

        system_train = []
        system_test = []
        train_parent = system/SYS_TRAIN
        test_parent = system/SYS_TEST
        if os.path.exists(train_parent):
            for group in os.listdir(train_parent):
                system_train.append(train_parent/group)
        if os.path.exists(test_parent):
            for group in os.listdir(test_parent):
                system_test.append(test_parent/group)

        sys_train_name = [os.path.basename(s) for s in system_train]
        sys_test_name = [os.path.basename(s) for s in system_test]

        task_names = []
        task_paths = []
        counter = 0
        group_num = -1

        for group in sys_train_name:
            dir=train_parent / group / "ABACUS"
            list=os.listdir(dir)
            for this in list:
                if ((counter+1)/group_size)>(group_num+1):
                    group_num+=1
                    group_name = "train.%s"% str(group_num).zfill(2)
                    task_names.append(group_name)
                    task_paths.append(train_parent/group_name)
                shutil.copytree((Path(dir)/this).resolve(),(train_parent/group_name/"ABACUS"/(str(group)+str(this).zfill(4))))
                counter+=1

        group_num = -1
        counter = 0
        for group in sys_test_name:
            dir=test_parent / group / "ABACUS"
            list=os.listdir(dir)
            for this in list:
                if ((counter+1)/group_size)>(group_num+1):
                    group_num+=1
                    group_name = "test.%s"% str(group_num).zfill(2)
                    task_names.append(group_name)
                    task_paths.append(test_parent/group_name)
                shutil.copytree((Path(dir)/this).resolve(),(test_parent/group_name/"ABACUS"/(str(group)+str(this).zfill(4))))
                counter+=1


        op = OPIO({
            "task_names" : task_names,
            "task_paths" : task_paths,
        })
        return op




class RunScfAbacus(OP):
   

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "no_model" : bool,
            "yaml_name" : str,
            "config_file" : Artifact(Path),
            "stru_file" : Artifact(Path),
            "task_path" : Artifact(Path),
            "model" : Artifact(Path, optional=True),
        })
    
    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_path" : Artifact(Path),
            # "err" : Artifact(Path),
            # "log_scf" : Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        cwd = os.getcwd()
        # OP input
        no_model = ip["no_model"]
        yaml_name = ip["yaml_name"]
        config_file = ip["config_file"]
        stru_file = ip["stru_file"]
        task_path = ip["task_path"]
        model = ip["model"]

        run_scf_config = load_yaml(config_file/yaml_name)

        run_cmd = run_scf_config.pop("run_cmd")
        abacus_path = run_scf_config.pop("abacus_path")
        cpus_per_task = run_scf_config.pop("cpus_per_task")

        shutil.copytree(stru_file, task_path,dirs_exist_ok = True)
        if not no_model:
            shutil.copy(model, task_path)
        
        os.chdir(task_path)

        for f in os.listdir(task_path/"ABACUS"):
            print(f)
            cmd=str(f"ulimit -s unlimited && \
                . /opt/intel/oneapi/setvars.sh && \
                cd ABACUS/{f}/ &&  \
                {run_cmd} -n {cpus_per_task} {abacus_path} > {SCF_OUT_LOG} 2>{SCF_ERR_LOG}  &&  \
                echo {f}`grep convergence ./OUT.ABACUS/running_scf.log` > conv  &&  \
                echo {f}`grep convergence ./OUT.ABACUS/running_scf.log`")
            p = subprocess.Popen(
                cmd, shell=True, executable='/bin/bash')
            p.wait()
        
        os.chdir(cwd)

        return OPIO({
            "task_path": task_path,
            # "log_scf" : task_path/outlog,
            # "err" : task_path/errlog,
        })



