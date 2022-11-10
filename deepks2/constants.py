import numpy as np
DEFAULT_SCF_MACHINE = {
    "template_config": None, 
    "executor": None,
}

# args not specified here may cause error
DEFAULT_TRN_MACHINE = {
    "template_config": None, 
    "executor": None,
}

BOHRIUM_SCF_ABACUS_CONVERT = {
    "extra": {
        "job_name":"deepks_scf_abacus_convert"
    }
}
BOHRIUM_SCF_ABACUS_PREP = {
    "extra": {
        "job_name":"deepks_scf_abacus_prep"
    }
}
BOHRIUM_SCF_ABACUS_RUN = {
    "extra": {
        "job_name":"deepks_scf_abacus_run"
    }
}
BOHRIUM_SCF_ABACUS_GATHER = {
    "extra": {
        "job_name":"deepks_scf_abacus_gather"
    }
}
BOHRIUM_DEEPKS_TRAIN = {
    "extra": {
        "job_name":"deepks_train"
    }
}

SCF_ARGS_NAME = "scf_input.yaml"
SCF_ARGS_NAME_ABACUS="scf_abacus.yaml"   #for abacus, caoyu add 2021-07-26
INIT_SCF_NAME_ABACUS="init_scf_abacus.yaml"   #for abacus init, caoyu add 2021-12-17
TRN_ARGS_NAME = "train_input.yaml"
INIT_SCF_NAME = "init_scf.yaml"
INIT_TRN_NAME = "init_train.yaml"

DATA_TRAIN = "data_train"
DATA_TEST  = "data_test"
PROJ_BASIS = "proj_basis.npz"

SCF_STEP_DIR = "00.scf"
TRN_STEP_DIR = "01.train"

RECORD = "RECORD"

SYS_TRAIN = "systems_train"
SYS_TEST = "systems_test"
DEFAULT_TRAIN = "systems_train.raw"
DEFAULT_TEST = "systems_test.raw"

MODEL_FILE = "model.pth"
CMODEL_FILE = "model.ptg"
RESTART_MODEL = "old_model.pth"

LOG_TRAIN = "log.train"
SCF_OUT_LOG = "out.log"
SCF_ERR_LOG = "err.log"

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

ABACUS_CMD="bash run_abacus.sh"

DEFAULT_SCF_ARGS_ABACUS={
    "orb_files": [],  # "orb" atomic number order
    "pp_files": [],  # "upf" atomic number order
    "proj_file": [],  # "orb"
    "ntype": 1,
    "nspin": 1,
    "symmetry": 0,
    "nbands": None,
    "ecutwfc": 50,
    "scf_thr": 1.0e-7,
    "scf_nmax": 50,
    "dft_functional": "pbe", 
    "basis_type": "lcao",
    "gamma_only": 1,
    "k_points": None,
    "kspacing": None,
    "smearing_method":"gaussian",
    "smearing_sigma":0.02,
    "mixing_type": "pulay",
    "mixing_beta": 0.4,
    "cal_force": 0,
    "cal_stress": 0,
    "deepks_bandgap": 0,
    "deepks_out_labels":1,
    "deepks_scf":0,
    "lattice_constant": 1,
    "lattice_vector": np.eye(3,dtype=int).tolist(),
    "coord_type": "Cartesian",
    "run_cmd": "mpirun",
    "sub_size": 1,
    "abacus_path": "/usr/local/bin/ABACUS.mpi",
    "group_size" : 100, # parallelism config
    "cpus_per_task" : 2, # parallelism config
}

DEFAULT_TRAIN_ARGS = {
   "model_args":None, 
   "data_args":None, 
   "preprocess_args":None, 
   "train_args":None, 
   "fit_elem":False,
   "proj_basis":None
}

default_image = 'base_dflow_deepks'
default_host = '127.0.0.1:2746'
