#include "assert.h"
#include "julia.h"
#include "numpy/arrayobject.h"


jl_datatype_t* nptype2jltype(int type_num)
{
    switch (type_num) {
        case NPY_BOOL:
            return jl_bool_type;
        // short/long/...?
        case NPY_INT8:
            return jl_int8_type;
        case NPY_UINT8:
            return jl_uint8_type;
        case NPY_INT16:
            return jl_int16_type;
        case NPY_UINT16:
            return jl_uint16_type;
        case NPY_INT32:
            return jl_int32_type;
        case NPY_UINT32:
            return jl_uint32_type;
        case NPY_INT64:
            return jl_int64_type;
        case NPY_UINT64:
            return jl_uint64_type;
        case NPY_FLOAT32:
            return jl_float32_type;
        case NPY_FLOAT64:
            return jl_float64_type;
        default:
            assert(false);
    }
}
