from julia cimport *

cdef extern from "dtypes.h":
    jl_datatype_t* nptype2jltype(int type_num)
