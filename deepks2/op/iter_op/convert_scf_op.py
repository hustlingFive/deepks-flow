from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
import os
import shutil
from typing import Tuple, List, Set, Union
from pathlib import Path
import numpy as np


class ConvertScfAbacus(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "scf_config": Union[dict, List[dict]],
            "n_iter": int,
            "stru_file": Artifact(Path),
            "model": Artifact(Path, optional=True),
            "system": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "system": Artifact(Path),
            "model": Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip: OPIO,
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
        for root, dirs, files in os.walk(sys_train):
            for dir in dirs:
                systems_train.append(os.path.join(root, dir))
        for root, dirs, files in os.walk(sys_test):
            for dir in dirs:
                systems_test.append(os.path.join(root, dir))

        no_model = True if n_iter == 0 else False

        orb_files = convert_scf_config["orb_files"]
        pp_files = convert_scf_config["pp_files"]
        proj_file = convert_scf_config["proj_file"]
        orb_files = ["../../"+str(os.path.basename(s)) for s in orb_files]
        pp_files = ["../../"+str(os.path.basename(s)) for s in pp_files]
        proj_file = ["../../"+str(os.path.basename(s)) for s in proj_file]
        

        # from deepks.iterate.template_abacus import convert_data

        convert_scf_config.update(
            systems_train=systems_train,
            systems_test=systems_test,
            model_file=model_file,
            no_model=no_model,
            orb_files=orb_files,
            pp_files=pp_files,
            proj_file=proj_file,
        )

        ConvertScfAbacus.convert_data(**convert_scf_config)

        os.chdir(cwd)

        return OPIO({
            "system": system,
            "model": system/CMODEL_FILE,
        })

    @staticmethod
    # need parameters: orb_files, pp_files, proj_file
    def convert_data(systems_train, systems_test=None, *,
                     no_model=True, model_file=None, pp_files=[],
                     lattice_vector=np.eye(3, dtype=int), dispatcher=None, **pre_args):

        CMODEL_FILE = "model.ptg"
        NAME_TYPE = {   'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
            'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13,
        'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
        'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
        'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31,
        'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
        'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
        'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49,
        'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
        'Ba': 56, #'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
            ## 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67,
            ## 'Er': 68, 'Tm': 69, 'Yb': 70, 
            ## 'Lu': 71, 
        'Hf': 72, 'Ta': 73,
        'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79,
        'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 
            ## 'Po': 84, #'At': 85,
            ## 'Rn': 86, #'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
            ## 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97,
            ## 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
            ## 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108,
            ## 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Uut': 113,
            ## 'Fl': 114, 'Uup': 115, 'Lv': 116, 'Uus': 117, 'Uuo': 118
        } #dict
        TYPE_NAME ={v:k for k, v in NAME_TYPE.items()}
        from deepks.utils import load_sys_paths
        from deepks.iterate.generator_abacus import make_abacus_scf_kpt, make_abacus_scf_input, make_abacus_scf_stru

        # trace a model (if necessary)
        if not no_model:
            if model_file is not None:
                from deepks.model import CorrNet
                model = CorrNet.load(model_file)
                model.compile_save(CMODEL_FILE)
                # set 'deepks_scf' to 1, and give abacus the path of traced model file
                pre_args.update(
                    deepks_scf=1, model_file="../../"+CMODEL_FILE)
            else:
                raise FileNotFoundError(
                    f"No required model file in {os.getcwd()}")
        # split systems into groups
        nsys_trn = len(systems_train)
        nsys_tst = len(systems_test)
        #ntask_trn = int(np.ceil(nsys_trn / sub_size))
        #ntask_tst = int(np.ceil(nsys_tst / sub_size))
        train_sets = [systems_train[i::nsys_trn] for i in range(nsys_trn)]
        test_sets = [systems_test[i::nsys_tst] for i in range(nsys_tst)]
        systems = systems_train+systems_test
        sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]
        # init sys_data (dpdata)
        for i, sset in enumerate(train_sets+test_sets):
            atom_data = np.load(f"{sys_paths[i]}/atom.npy")
            if os.path.isfile(f"{sys_paths[i]}/box.npy"):
                cell_data = np.load(f"{sys_paths[i]}/box.npy")
            nframes = atom_data.shape[0]
            natoms = atom_data.shape[1]
            atoms = atom_data[1, :, 0]
            # atoms.sort() # type order
            types = np.unique(atoms)  # index in type list
            ntype = types.size
            from collections import Counter
            nta = Counter(atoms)  # dict {itype: nta}, natom in each type
            if not os.path.exists(f"{sys_paths[i]}/ABACUS"):
                os.mkdir(f"{sys_paths[i]}/ABACUS")
            # pre_args.update({"lattice_vector":lattice_vector})
            # if "stru_abacus.yaml" exists, update STRU args in pre_args:
            pre_args_new = dict(zip(pre_args.keys(), pre_args.values()))
            if os.path.exists(f"{sys_paths[i]}/stru_abacus.yaml"):
                from deepks.utils import load_yaml
                stru_abacus = load_yaml(f"{sys_paths[i]}/stru_abacus.yaml")
                for k, v in stru_abacus.items():
                    pre_args_new[k] = v
            for f in range(nframes):
                if not os.path.exists(f"{sys_paths[i]}/ABACUS/{f}"):
                    os.mkdir(f"{sys_paths[i]}/ABACUS/{f}")
                # create STRU file
                if not os.path.isfile(f"{sys_paths[i]}/ABACUS/{f}/STRU"):
                    Path(f"{sys_paths[i]}/ABACUS/{f}/STRU").touch()
                # create sys_data for each frame
                frame_data = atom_data[f]
                # frame_sorted=frame_data[np.lexsort(frame_data[:,::-1].T)] #sort cord by type
                sys_data = {'atom_names': [TYPE_NAME[it] for it in nta.keys()], 'atom_numbs': list(nta.values()),
                            # 'cells': np.array([lattice_vector]), 'coords': [frame_sorted[:,1:]]}
                            'cells': np.array([lattice_vector]), 'coords': [frame_data[:, 1:]]}
                if os.path.isfile(f"{sys_paths[i]}/box.npy"):
                    sys_data = {'atom_names': [TYPE_NAME[it] for it in nta.keys()], 'atom_numbs': list(nta.values()),
                                'cells': [cell_data[f]], 'coords': [frame_data[:, 1:]]}
                # write STRU file
                with open(f"{sys_paths[i]}/ABACUS/{f}/STRU", "w") as stru_file:
                    stru_file.write(make_abacus_scf_stru(
                        sys_data, pp_files, pre_args_new))
                # write INPUT file
                with open(f"{sys_paths[i]}/ABACUS/{f}/INPUT", "w") as input_file:
                    input_file.write(make_abacus_scf_input(pre_args))
                # write KPT file (gamma_only)
                with open(f"{sys_paths[i]}/ABACUS/{f}/KPT", "w") as kpt_file:
                    kpt_file.write(make_abacus_scf_kpt(pre_args))
