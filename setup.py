import os
from sys import exit
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

try:
    julia_home = os.environ["JULIA_HOME"]
except KeyError:
    print("JULIA_HOME environment variable is missing")
    exit(1)


def join(*args):
    return os.path.normpath(os.path.join(*args))


def julia_include_dirs(julia_home):
    relatives = [
        ("src",),
        ("src", "support"),
        ("usr", "include"),
    ]

    return [join(julia_home, "..", "..", *r) for r in relatives] + ["."]

core = Extension("core",
                 [join("libjulia", "core.pyx")],
                 include_dirs=julia_include_dirs(julia_home),
                 libraries=["julia"],
                 runtime_library_dirs=[join(julia_home, "..", "lib")],
                 define_macros=[])

setup(
    name="libjulia",
    description="Python binding library for libjulia",
    author="Kenta Sato",
    packages=["libjulia"],
    include_dirs=[np.get_include()],
    ext_modules=cythonize(core)
)
