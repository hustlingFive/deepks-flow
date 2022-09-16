import os
import sys
import argparse
try:
    import deepks2
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from deepks2.utils.deepks_utils import load_yaml, deep_update



def main_cli(args=None):
    parser = argparse.ArgumentParser(
                prog="deepks",
                description="A program to generate accurate energy functionals.")
    parser.add_argument("command", 
                        help="specify the sub-command to run, possible choices: "
                             "train, test, scf, stats, iterate")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                        help="arguments to be passed to the sub-command")

    args = parser.parse_args(args)

    # sepatate all sub_cli to make them useable independently 
    if args.command.upper().startswith("ITER"):
        sub_cli = iter_cli
    elif args.command.upper().startswith("HF"):
        sub_cli = hfscf_cli
    else:
        return ValueError(f"unsupported sub-command: {args.command}")
    
    sub_cli(args.args)


def iter_cli(args=None):
    parser = argparse.ArgumentParser(
                prog="deepks2 iterate",
                description="Run the iteration procedure to train a SCF model.",
                argument_default=argparse.SUPPRESS)
    parser.add_argument("argfile", nargs="*", default=[],
                        help='the input yaml file for args, '
                             'if more than one, the latter has higher priority')
    parser.add_argument("-s", "--systems-train", nargs="*",
                        help='systems for training, '
                             'can be xyz files or folders with npy data')
    parser.add_argument("-t", "--systems-test", nargs="*",
                        help='systems for training, '
                             'can be xyz files or folders with npy data')
    parser.add_argument("-n", "--n-iter", type=int,
                        help='the number of iterations to run')
    parser.add_argument("--workdir",
                        help='working directory, default is current directory')
    parser.add_argument("--share-folder", 
                        help='folder to store share files, default is "share"')
    parser.add_argument("--cleanup", action="store_true", dest="cleanup",
                        help='if set, clean up files used for job dispatching')
    parser.add_argument("--no-strict", action="store_false", dest="strict",
                        help='if set, allow other arguments to be passed to task')
    # allow cli specified argument files
    sub_names = ["scf-input", "scf-machine", "train-input", "train-machine",
                 "init-model", "init-scf", "init-train", "scf-abacus"]
    for name in sub_names:
        parser.add_argument(f"--{name}",
            help='if specified, subsitude the original arguments with given file')
    
    args = parser.parse_args(args)
    argdict = {}
    for fl in args.argfile:
        argdict = deep_update(argdict, load_yaml(fl))
    del args.argfile
    argdict.update(vars(args))
    
    from deepks2.flow.submit_iter import submit_iterate
    submit_iterate(**argdict)

def hfscf_cli(args=None):
    parser = argparse.ArgumentParser(
                prog="deepks2 hfscf",
                description="Calculate and save SCF results using given model.",
                argument_default=argparse.SUPPRESS)
    parser.add_argument("argfile", nargs="*", default=[],
                        help='the input yaml file for args, '
                             'if more than one, the latter has higher priority')
    
    args = parser.parse_args(args)
    argdict = {}
    for fl in args.argfile:
        argdict = deep_update(argdict, load_yaml(fl))
    del args.argfile
    argdict.update(vars(args))
    
    from deepks2.flow.submit_hfscf import submit_hfscf
    submit_hfscf(**argdict)


if __name__ == "__main__":
    main_cli()