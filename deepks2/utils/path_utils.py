from pathlib import Path
import os
import shutil
from glob import glob
from deepks2.utils.arg_utils import check_list
# below are path related utils

def get_abs_path(p):
    if p is None:
        return None
    else:
        return Path(p).absolute()


def get_sys_name(p):
    if p.endswith(os.path.sep):
        return p.rstrip(os.path.sep)
    if p.endswith(".xyz"):
        return p[:-4]
    return p


def get_with_prefix(p, base=None, prefer=None, nullable=False):
    """
    Get file path by searching its prefix.
    If `base` is a directory, equals to get "base/p*".
    Otherwise, equals to get "base.p*".
    Only one result will be return. 
    If more than one match, give the first one with suffix in `prefer`.
    """
    if not base:
        base = "./"
    if os.path.isdir(base):
        pattern = os.path.join(base, p)
    else:
        pattern = f"{base.rstrip('.')}.{p}"
    matches = glob(pattern + "*")
    if len(matches) == 1:
        return matches[0]
    prefer = check_list(prefer)
    for suffix in prefer:
        if pattern+suffix in matches:
            return pattern+suffix
    if nullable:
        return None
    raise FileNotFoundError(f"{pattern}* not exists or has more than one matches")

    
def link_file(src, dst, use_abs=False):
    src, dst = Path(src), Path(dst)
    assert src.exists(), f'{src} does not exist'
    src_path = os.path.abspath(src) if use_abs else os.path.relpath(src, dst.parent)
    if not dst.exists():
        if not dst.parent.exists():
            os.makedirs(dst.parent)
        os.symlink(src_path, dst)
    elif not os.path.samefile(src, dst):
        os.remove(dst)
        os.symlink(src_path, dst)


def copy_file(src, dst):
    src, dst = Path(src), Path(dst)
    assert src.exists(), f'{src} does not exist'
    if not dst.exists():
        if not dst.parent.exists():
            os.makedirs(dst.parent)
        shutil.copy2(src, dst)
    elif not os.path.samefile(src, dst):
        os.remove(dst)
        shutil.copy2(src, dst)


def create_dir(dirname, backup=False):
    dirname = Path(dirname)
    if not dirname.exists():
        os.makedirs(dirname)
    elif backup and dirname != Path('.'):
        os.makedirs(dirname.parent, exist_ok=True)
        counter = 0
        bckname = str(dirname) + f'.bck.{counter:03d}'
        while os.path.exists(bckname):
            counter += 1
            bckname = str(dirname) + f'.bck.{counter:03d}'
        dirname.rename(bckname)
        os.makedirs(dirname)
    else:
        assert dirname.is_dir(), f'{dirname} is not a dir'