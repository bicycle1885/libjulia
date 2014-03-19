from libc.stdlib cimport malloc, free
from julia cimport *


### Wrappers

cdef class Function:
    """Julia function wrapper"""

    cdef jl_function_t* thisptr

    cdef set_function(self, jl_function_t* function):
        self.thisptr = function

    def __call__(self, *args):
        cdef:
            uint32_t i
            jl_value_t **jl_args = NULL
            uint32_t nargs = len(args)

        if nargs == 0:
            return jl2py(jl_call0(self.thisptr))

        jl_args = <jl_value_t**>malloc(sizeof(jl_value_t*) * nargs)

        if jl_args == NULL:
            raise MemoryError("failed to allocate memory")

        try:
            for i in range(nargs):
                jl_args[i] = py2jl(args[i])

            return jl2py(jl_call(self.thisptr, jl_args, nargs))
        finally:
            free(jl_args)


cdef class Module:
    """Julia module wrapper"""

    cdef jl_module_t* thisptr

    cdef set_module(self, jl_module_t* module):
        self.thisptr = module

    def __getattr__(self, attr):
        cdef:
            jl_sym_t* sym = jl_symbol(<char*>attr)
            jl_value_t* v = jl_get_global(self.thisptr, sym)

        return jl2py(v)


### Type Converters
# py2jl: Python => Julia
# jl2py: Julia  => Python

cdef jl_value_t* py2jl(object py_value):
    cdef object v = py_value

    if v is None:
        return jl_nothing
    elif isinstance(v, bool):
        if v:
            return jl_box_bool(1)
        else:
            return jl_box_bool(0)
    elif isinstance(v, int):
        return jl_box_int64(<int64_t>v)
    elif isinstance(v, float):
        return jl_box_float64(<double>v)

    raise TypeError("the type of the python value is not supported")


cdef object jl2py(jl_value_t* jl_value):
    cdef:
        jl_value_t* v = jl_value
        Function f

    if jl_is_nothing(v):
        return None
    elif jl_is_bool(v):
        return <bint>jl_unbox_bool(v)
    elif jl_is_int32(v):
        return <int32_t>jl_unbox_int32(v)
    elif jl_is_int64(v):
        return <int64_t>jl_unbox_int64(v)
    elif jl_is_uint32(v):
        return <uint32_t>jl_unbox_int32(v)
    elif jl_is_uint64(v):
        return <uint64_t>jl_unbox_int64(v)
    elif jl_is_float32(v):
        return <float>jl_unbox_float32(v)
    elif jl_is_float64(v):
        return <double>jl_unbox_float64(v)
    elif jl_is_function(v):
        f = Function()
        f.set_function(<jl_function_t*>v)
        return f
    elif jl_is_array(v):
        pass

    raise TypeError("the type of the julia value is not supported")


def init(julia_home_dir=None):
    cdef char* julia_home_dir_c

    if julia_home_dir:
        julia_home_dir_c = julia_home_dir
    else:
        julia_home_dir_c = NULL  # guess

    jl_init(julia_home_dir_c)


def eval_string(s):
    return jl2py(jl_eval_string(<char*>s))


cdef get_function(jl_module_t* module, char* name):
    cdef:
        jl_function_t* jl_function
        Function f

    jl_function = jl_get_function(module, name)

    if jl_function == NULL:
        raise NameError("function '{}' is not found in the module".format(name))

    f = Function()
    f.set_function(jl_function)
    return f


def get_base_function(name):

    if not isinstance(name, str):
        raise TypeError("'str' is required")

    return get_function(jl_base_module, <char*>name)


def load_base_module():
    cdef:
        Module m = Module()
    m.set_module(jl_base_module)
    return m
