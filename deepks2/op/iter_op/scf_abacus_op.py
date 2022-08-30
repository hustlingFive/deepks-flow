import os, shutil,subprocess
from pathlib import Path
from deepks2.utils.run_command import run_command
from deepks2.utils.chdir import set_directory
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



class PrepScfAbacus(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "scf_abacus_config" : Union[dict, List[dict]],
            "system" : Artifact(Path),
            # "stru_file" : Artifact(Path),
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
        # systems_train = ["./share/systems_train/group.00","./share/systems_train/group.01","./share/systems_train/group.02"]
        # systems_test = ["./share/systems_test/group.03"]
        scf_abacus_config = ip["scf_abacus_config"]
        system = ip["system"]
        # stru_file = ip["stru_file"]
        SYS_TRAIN = "systems_train"
        SYS_TEST = "systems_test"
        group_size = scf_abacus_config.pop("group_size", 150)


        system_train = []
        system_test = []
        train_parent = system/SYS_TRAIN
        test_parent = system/SYS_TEST
        for group in os.listdir(train_parent):
            system_train.append(train_parent/group)
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
                    # shutil.copytree(stru_file, train_parent/group_name ,dirs_exist_ok = True)
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
                    # shutil.copytree(stru_file, test_parent/group_name ,dirs_exist_ok = True)
                shutil.copytree((Path(dir)/this).resolve(),(test_parent/group_name/"ABACUS"/(str(group)+str(this).zfill(4))))
                counter+=1


        op = OPIO({
            "task_names" : task_names,
            "task_paths" : task_paths,
        })
        return op


    # def _script_rand_seed(
    #         self,
    #         input_dict,
    # ):
    #     jtmp = input_dict.copy()
    #     if jtmp['model']['descriptor']['type'] == 'hybrid':
    #         for desc in jtmp['model']['descriptor']['list']:
    #             desc['seed'] = random.randrange(sys.maxsize) % (2**32)
    #     else:
    #         jtmp['model']['descriptor']['seed'] = random.randrange(sys.maxsize) % (2**32)
    #     jtmp['model']['fitting_net']['seed'] = random.randrange(sys.maxsize) % (2**32)
    #     jtmp['training']['seed'] = random.randrange(sys.maxsize) % (2**32)
    #     return jtmp



class RunScfAbacus(OP):
   

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "scf_abacus_config" : Union[dict, List[dict]],
            # "task_name" : str,
            "task_path" : Artifact(Path),
            "stru_file" : Artifact(Path),
        })
    
    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_path" : Artifact(Path),
            "err" : Artifact(Path),
            # "log_data" : Artifact(Path),
            "log_scf" : Artifact(Path),
            # "task_name": str
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        print("start")
        cwd = os.getcwd()

        task_path = ip["task_path"]
        scf_abacus_config = ip["scf_abacus_config"]
        run_cmd = scf_abacus_config.pop("run_cmd")
        abacus_path = scf_abacus_config.pop("abacus_path")
        # resources = scf_abacus_config.pop("resources")
        cpus_per_task = scf_abacus_config.pop("cpus_per_task",3)

        outlog="out.log"
        errlog="err.log"

        stru_file = ip["stru_file"]
        shutil.copytree(stru_file, task_path,dirs_exist_ok = True)
        
        os.chdir(task_path)

        for f in os.listdir(task_path/"ABACUS"):
            print(f)
            cmd=str(f"ulimit -s unlimited && \
                . /opt/intel/oneapi/setvars.sh && \
                cd ABACUS/{f}/ &&  \
                {run_cmd} -n {cpus_per_task} {abacus_path} > {outlog} 2>{errlog}  &&  \
                echo {f}`grep convergence ./OUT.ABACUS/running_scf.log` > conv  &&  \
                echo {f}`grep convergence ./OUT.ABACUS/running_scf.log`")
            p = subprocess.Popen(
                cmd, shell=True, executable='/bin/bash')
            p.wait()
        
        os.chdir(cwd)

        return OPIO({
            "task_path": task_path,
            "log_scf" : task_path/outlog,
            "err" : task_path/errlog,
            # "task_name":ip["task_name"]
            # "labeled_data": work_dir / out_name,
        })


#     @staticmethod
#     def vasp_args():
#         doc_vasp_cmd = "The command of VASP"
#         doc_vasp_log = "The log file name of VASP"
#         doc_vasp_out = "The output dir name of labeled data. In `deepmd/npy` format provided by `dpdata`."
#         return [
#             Argument("command", str, optional=True, default='mpirun', doc=doc_vasp_cmd),
#             Argument("log", str, optional=True, default="log", doc=doc_vasp_log),
#             Argument("out", str, optional=True, default="out", doc=doc_vasp_out),
#         ]

#     @staticmethod
#     def normalize_config(data = {}):
#         ta = RunScfAbacus.vasp_args()
#         base = Argument("base", dict, ta)
#         data = base.normalize_value(data, trim_pattern="_*")
#         base.check_value(data, strict=True)
#         return data

    
# config_args = RunScfAbacus.vasp_args
