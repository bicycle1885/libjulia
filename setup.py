import os
from sys import exit
from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    have_cython = True
except ImportError:
    have_cython = False

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


def numpy_include_dirs():
    return [np.get_include()]


def generate_extension(name, sources):
    return Extension(
        name,
        sources,
        include_dirs=(julia_include_dirs(julia_home)
                      + numpy_include_dirs()),
        libraries=["julia"],
        runtime_library_dirs=[join(julia_home, "..", "lib")],
        define_macros=[]
    )


if have_cython:
    core = generate_extension("core", [join("libjulia", "core.pyx")])
    ext_modules = cythonize([core])
else:
    core = generate_extension("core", [join("libjulia", "core.c")])
    ext_modules = [core]

setup(
    name="libjulia",
    description="Python binding library for libjulia",
    author="Kenta Sato",
    packages=["libjulia"],
    ext_modules=ext_modules
)
