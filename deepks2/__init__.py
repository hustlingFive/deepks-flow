__author__ = "Yifan Shan"

# try:
#     from ._version import version as __version__
# except ImportError:
#     __version__ = 'unkown'

__all__ = [
    "op",
    "flow",
    "step",
    "utils"
]

def __getattr__(name):
    from importlib import import_module
    if name in __all__:
        return import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
