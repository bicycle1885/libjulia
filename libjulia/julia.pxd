from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,

    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

cdef extern from "julia.h":

    # C-level types
    ctypedef struct jl_value_t
    ctypedef struct jl_sym_t
    ctypedef struct jl_tuple_t
    ctypedef struct jl_array_t
    ctypedef struct jl_lambda_info_t
    ctypedef struct jl_function_t
    ctypedef struct jl_typector_t
    ctypedef struct jl_typename_t
    ctypedef struct jl_uniontype_t
    ctypedef struct jl_fielddesc_t
    ctypedef struct jl_datatype_t
    ctypedef struct jl_tvar_t
    ctypedef struct jl_module_t

    # singleton values
    jl_value_t* jl_true
    jl_value_t* jl_false
    jl_value_t* jl_nothing
    jl_tuple_t* jl_null

    # init
    void jl_init(char*)

    # eval
    jl_value_t* jl_eval_string(char*)

    # calling functions
    jl_value_t* jl_call(jl_function_t*, jl_value_t**, uint32_t)
    jl_value_t* jl_call0(jl_function_t*)

    # gc
    void JL_GC_PUSHARGS(jl_value_t**, size_t)
    void JL_GC_POP()

    # data accessors
    jl_value_t* jl_typeof(jl_tuple_t*)
    size_t jl_tuple_len(jl_tuple_t*)
    jl_value_t* jl_tupleref(jl_tuple_t* t, size_t i)
    void jl_tupleset(jl_tuple_t* t, size_t i, jl_value_t* x)
    size_t jl_array_len(jl_array_t*)
    void* jl_array_data(jl_array_t*)

    # arrays
    jl_value_t *jl_apply_array_type(jl_datatype_t *type, size_t dim)
    jl_array_t *jl_alloc_array_1d(jl_value_t *atype, size_t nr)
    jl_array_t *jl_alloc_array_2d(jl_value_t *atype, size_t nr, size_t nc)
    jl_array_t *jl_alloc_array_3d(jl_value_t *atype, size_t nr, size_t nc, size_t z)
    jl_array_t *jl_ptr_to_array(jl_value_t *atype, void *data, jl_tuple_t *dims, int own_buffer)

    # modules
    jl_module_t* jl_main_module
    jl_module_t* jl_internal_main_module
    jl_module_t* jl_core_module
    jl_module_t* jl_base_module
    jl_module_t* jl_current_module
    jl_value_t* jl_get_global(jl_module_t*, jl_sym_t*)
    jl_function_t* jl_get_function(jl_module_t*, char*)

    # predicates
    bint jl_is_nothing(jl_value_t*)
    bint jl_is_tuple(jl_value_t*)
    bint jl_is_int32(jl_value_t*)
    bint jl_is_int64(jl_value_t*)
    bint jl_is_uint32(jl_value_t*)
    bint jl_is_uint64(jl_value_t*)
    bint jl_is_float32(jl_value_t*)
    bint jl_is_float64(jl_value_t*)
    bint jl_is_bool(jl_value_t*)
    bint jl_is_ascii_string(jl_value_t*)
    bint jl_is_utf8_string(jl_value_t*)
    bint jl_is_byte_string(jl_value_t*)
    bint jl_is_function(jl_value_t*)
    bint jl_is_array(void *v)

    # constructors
    jl_sym_t* jl_symbol(char*)
    jl_tuple_t *jl_tuple(size_t n, ...)
    jl_tuple_t *jl_tuple1(void *a)
    jl_tuple_t *jl_tuple2(void *a, void *b)
    jl_tuple_t *jl_alloc_tuple(size_t n)
    jl_tuple_t *jl_alloc_tuple_uninit(size_t n)
    jl_tuple_t *jl_tuple_append(jl_tuple_t *a, jl_tuple_t *b)
    jl_tuple_t *jl_tuple_fill(size_t n, jl_value_t *v)

    # boxing functions
    jl_value_t* jl_box_bool(int8_t)
    jl_value_t* jl_box_int64(int64_t)
    jl_value_t* jl_box_float64(double)

    # unboxing functions
    int8_t jl_unbox_bool(jl_value_t*)
    int32_t jl_unbox_int32(jl_value_t*)
    int64_t jl_unbox_int64(jl_value_t*)
    uint32_t jl_unbox_uint32(jl_value_t*)
    uint64_t jl_unbox_uint64(jl_value_t*)
    float jl_unbox_float32(jl_value_t*)
    double jl_unbox_float64(jl_value_t*)
