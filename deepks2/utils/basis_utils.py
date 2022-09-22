import numpy as np
# below are basis set settings

_zeta = 1.5**np.array([17,13,10,7,5,3,2,1,0,-1,-2,-3])
_coef = np.diag(np.ones(_zeta.size)) - np.diag(np.ones(_zeta.size-1), k=1)
_table = np.concatenate([_zeta.reshape(-1,1), _coef], axis=1)
DEFAULT_BASIS = [[0, *_table.tolist()], [1, *_table.tolist()], [2, *_table.tolist()]]
DEFAULT_SYMB = "Ne"

def load_basis(basis):
    if basis is None:
        return DEFAULT_BASIS
    elif isinstance(basis, np.ndarray) and basis.ndim == 2:
        return [[ll, *basis.tolist()] for ll in range(3)]
    elif not isinstance(basis, str):
        return basis
    elif basis.endswith(".npy"):
        table = np.load(basis)
        return [[ll, *table.tolist()] for ll in range(3)]
    elif basis.endswith(".npz"):
        all_tables = np.load(basis)
        return [[int(name.split("_L")[-1]) if "_L" in name else ii, *table.tolist()] 
                for ii, (name, table) in enumerate(all_tables.items())]
    else:
        raise RuntimeError('unknown input format of basis: ', type(basis))

        # from pyscf import gto
        # symb = DEFAULT_SYMB
        # if "@" in basis:
        #     basis, symb = basis.split("@")
        # return gto.basis.load(basis, symb=symb)


def save_basis(file, basis):
    """Save the basis to npz file from internal format of pyscf"""
    tables = {f"arr_{i}_L{l}":np.array(b) for i, (l,*b) in enumerate(basis)}
    np.savez(file, **tables)


def get_shell_sec(basis):
    if not isinstance(basis, (list, tuple)):
        basis = load_basis(basis)
    shell_sec = []
    for l, c0, *cr in basis:
        shell_sec.extend([2*l+1] * (len(c0)-1))
    return shell_sec
    
