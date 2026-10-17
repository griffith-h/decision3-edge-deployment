
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const at::BFloat16* in_ptr0,
                       const at::BFloat16* in_ptr1,
                       const double* in_ptr2,
                       const at::BFloat16* in_ptr3,
                       float* out_ptr0,
                       at::BFloat16* out_ptr1,
                       const int64_t ks0,
                       const int64_t ks1,
                       const int64_t ks2)
{
    #pragma omp parallel num_threads(3)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(ks1); x1+=static_cast<int64_t>(1L))
                    {
                        {
                            {
                                auto tmp0 = in_ptr0[static_cast<int64_t>(ks2*x1 + ks1*ks2*x0)];
                                auto tmp1 = c10::convert<at::BFloat16>(tmp0);
                                auto tmp2 = c10::convert<float>(tmp1);
                                auto tmp3 = float(tmp2 * tmp2);
                                tmp_acc0 = tmp_acc0 + tmp3;
                            }
                        }
                    }
                    out_ptr0[static_cast<int64_t>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(8L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(ks1); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(8L)))) && x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(8L*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L))))))
                        {
                            for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                                auto tmp3 =
                                [&]
                                {
                                    __at_align__ std::array<at::BFloat16, 16> tmpbuf;
                                    #pragma GCC unroll 8
                                    for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                    {
                                        tmpbuf[x1_inner] = in_ptr0[static_cast<int64_t>(ks2*x1 + ks2*x1_inner + ks1*ks2*x0 + ks1*ks2*x0_inner)];
                                    }
                                    return at::vec::Vectorized<at::BFloat16>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp6 = out_ptr0[static_cast<int64_t>(x0 + x0_inner)];
                                auto tmp10 = in_ptr2[static_cast<int64_t>(0L)];
                                auto tmp19 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr3 + static_cast<int64_t>(x1 + ks1*x0 + ks1*x0_inner), static_cast<int64_t>(8));
                                auto tmp1 = (tmp0);
                                auto tmp2 = at::vec::convert<float,2,at::BFloat16,1>(tmp1);
                                auto tmp4 = (tmp3);
                                auto tmp5 = at::vec::convert<float,2,at::BFloat16,1>(tmp4);
                                auto tmp7 = ks1;
                                auto tmp8 = c10::convert<float>(tmp7);
                                auto tmp9 = tmp6 / tmp8;
                                auto tmp11 = c10::convert<float>(tmp10);
                                auto tmp12 = float(tmp9 + tmp11);
                                auto tmp13 = 1 / std::sqrt(tmp12);
                                auto tmp14 = at::vec::VectorizedN<float,2>(tmp13);
                                auto tmp15 = tmp5 * tmp14;
                                auto tmp16 = at::vec::convert<at::BFloat16,1,float,2>(tmp15);
                                auto tmp17 = tmp2 * tmp15;
                                auto tmp18 = at::vec::convert<at::BFloat16,1,float,2>(tmp17);
                                auto tmp20 = (tmp19);
                                auto tmp21 = at::vec::convert<float,2,at::BFloat16,1>(tmp20);
                                auto tmp22 = tmp21.neg();
                                auto tmp23 = tmp22.exp();
                                auto tmp24 = static_cast<float>(1.0);
                                auto tmp25 = at::vec::VectorizedN<float,2>(tmp24);
                                auto tmp26 = tmp23 + tmp25;
                                auto tmp27 = tmp21 / tmp26;
                                auto tmp28 = tmp17 * tmp27;
                                auto tmp29 = at::vec::convert<at::BFloat16,1,float,2>(tmp28);
                                tmp29.store(out_ptr1 + static_cast<int64_t>(x1 + ks1*x0 + ks1*x0_inner), static_cast<int64_t>(8));
                            }
                        }
                        if(C10_UNLIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(8L)))) && x1 >= static_cast<int64_t>(8L*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L)))) && x1 < static_cast<int64_t>(ks1)))
                        {
                            for (long x0_inner = 0; x0_inner < static_cast<int64_t>(8); x0_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(ks1 + (-8L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L)))));
                                auto tmp3 =
                                [&]
                                {
                                    __at_align__ std::array<at::BFloat16, 16> tmpbuf;
                                    #pragma GCC unroll 8
                                    for (long x1_inner = 0; x1_inner < static_cast<int64_t>(ks1 + (-8L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L)))); x1_inner++)
                                    {
                                        tmpbuf[x1_inner] = in_ptr0[static_cast<int64_t>(ks2*x1 + ks2*x1_inner + ks1*ks2*x0 + ks1*ks2*x0_inner)];
                                    }
                                    return at::vec::Vectorized<at::BFloat16>::loadu(tmpbuf.data(), static_cast<int64_t>(ks1 + (-8L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L)))));
                                }
                                ()
                                ;
                                auto tmp6 = out_ptr0[static_cast<int64_t>(x0 + x0_inner)];
                                auto tmp10 = in_ptr2[static_cast<int64_t>(0L)];
                                auto tmp19 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr3 + static_cast<int64_t>(x1 + ks1*x0 + ks1*x0_inner), static_cast<int64_t>(ks1 + (-8L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L)))));
                                auto tmp1 = (tmp0);
                                auto tmp2 = at::vec::convert<float,2,at::BFloat16,1>(tmp1);
                                auto tmp4 = (tmp3);
                                auto tmp5 = at::vec::convert<float,2,at::BFloat16,1>(tmp4);
                                auto tmp7 = ks1;
                                auto tmp8 = c10::convert<float>(tmp7);
                                auto tmp9 = tmp6 / tmp8;
                                auto tmp11 = c10::convert<float>(tmp10);
                                auto tmp12 = float(tmp9 + tmp11);
                                auto tmp13 = 1 / std::sqrt(tmp12);
                                auto tmp14 = at::vec::VectorizedN<float,2>(tmp13);
                                auto tmp15 = tmp5 * tmp14;
                                auto tmp16 = at::vec::convert<at::BFloat16,1,float,2>(tmp15);
                                auto tmp17 = tmp2 * tmp15;
                                auto tmp18 = at::vec::convert<at::BFloat16,1,float,2>(tmp17);
                                auto tmp20 = (tmp19);
                                auto tmp21 = at::vec::convert<float,2,at::BFloat16,1>(tmp20);
                                auto tmp22 = tmp21.neg();
                                auto tmp23 = tmp22.exp();
                                auto tmp24 = static_cast<float>(1.0);
                                auto tmp25 = at::vec::VectorizedN<float,2>(tmp24);
                                auto tmp26 = tmp23 + tmp25;
                                auto tmp27 = tmp21 / tmp26;
                                auto tmp28 = tmp17 * tmp27;
                                auto tmp29 = at::vec::convert<at::BFloat16,1,float,2>(tmp28);
                                tmp29.store(out_ptr1 + static_cast<int64_t>(x1 + ks1*x0 + ks1*x0_inner), static_cast<int64_t>(ks1 + (-8L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L)))));
                            }
                        }
                        if(C10_UNLIKELY(x0 >= static_cast<int64_t>(8L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(8L)))) && x0 < static_cast<int64_t>(ks0) && x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(8L*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L))))))
                        {
                            for (long x0_inner = 0; x0_inner < static_cast<int64_t>(ks0 + (-8L)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(8L)))); x0_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                                auto tmp3 =
                                [&]
                                {
                                    __at_align__ std::array<at::BFloat16, 16> tmpbuf;
                                    #pragma GCC unroll 8
                                    for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                    {
                                        tmpbuf[x1_inner] = in_ptr0[static_cast<int64_t>(ks2*x1 + ks2*x1_inner + ks1*ks2*x0 + ks1*ks2*x0_inner)];
                                    }
                                    return at::vec::Vectorized<at::BFloat16>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                }
                                ()
                                ;
                                auto tmp6 = out_ptr0[static_cast<int64_t>(x0 + x0_inner)];
                                auto tmp10 = in_ptr2[static_cast<int64_t>(0L)];
                                auto tmp19 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr3 + static_cast<int64_t>(x1 + ks1*x0 + ks1*x0_inner), static_cast<int64_t>(8));
                                auto tmp1 = (tmp0);
                                auto tmp2 = at::vec::convert<float,2,at::BFloat16,1>(tmp1);
                                auto tmp4 = (tmp3);
                                auto tmp5 = at::vec::convert<float,2,at::BFloat16,1>(tmp4);
                                auto tmp7 = ks1;
                                auto tmp8 = c10::convert<float>(tmp7);
                                auto tmp9 = tmp6 / tmp8;
                                auto tmp11 = c10::convert<float>(tmp10);
                                auto tmp12 = float(tmp9 + tmp11);
                                auto tmp13 = 1 / std::sqrt(tmp12);
                                auto tmp14 = at::vec::VectorizedN<float,2>(tmp13);
                                auto tmp15 = tmp5 * tmp14;
                                auto tmp16 = at::vec::convert<at::BFloat16,1,float,2>(tmp15);
                                auto tmp17 = tmp2 * tmp15;
                                auto tmp18 = at::vec::convert<at::BFloat16,1,float,2>(tmp17);
                                auto tmp20 = (tmp19);
                                auto tmp21 = at::vec::convert<float,2,at::BFloat16,1>(tmp20);
                                auto tmp22 = tmp21.neg();
                                auto tmp23 = tmp22.exp();
                                auto tmp24 = static_cast<float>(1.0);
                                auto tmp25 = at::vec::VectorizedN<float,2>(tmp24);
                                auto tmp26 = tmp23 + tmp25;
                                auto tmp27 = tmp21 / tmp26;
                                auto tmp28 = tmp17 * tmp27;
                                auto tmp29 = at::vec::convert<at::BFloat16,1,float,2>(tmp28);
                                tmp29.store(out_ptr1 + static_cast<int64_t>(x1 + ks1*x0 + ks1*x0_inner), static_cast<int64_t>(8));
                            }
                        }
                        if(C10_UNLIKELY(x0 >= static_cast<int64_t>(8L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(8L)))) && x0 < static_cast<int64_t>(ks0) && x1 >= static_cast<int64_t>(8L*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L)))) && x1 < static_cast<int64_t>(ks1)))
                        {
                            for (long x0_inner = 0; x0_inner < static_cast<int64_t>(ks0 + (-8L)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(8L)))); x0_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(ks1 + (-8L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L)))));
                                auto tmp3 =
                                [&]
                                {
                                    __at_align__ std::array<at::BFloat16, 16> tmpbuf;
                                    #pragma GCC unroll 8
                                    for (long x1_inner = 0; x1_inner < static_cast<int64_t>(ks1 + (-8L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L)))); x1_inner++)
                                    {
                                        tmpbuf[x1_inner] = in_ptr0[static_cast<int64_t>(ks2*x1 + ks2*x1_inner + ks1*ks2*x0 + ks1*ks2*x0_inner)];
                                    }
                                    return at::vec::Vectorized<at::BFloat16>::loadu(tmpbuf.data(), static_cast<int64_t>(ks1 + (-8L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L)))));
                                }
                                ()
                                ;
                                auto tmp6 = out_ptr0[static_cast<int64_t>(x0 + x0_inner)];
                                auto tmp10 = in_ptr2[static_cast<int64_t>(0L)];
                                auto tmp19 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr3 + static_cast<int64_t>(x1 + ks1*x0 + ks1*x0_inner), static_cast<int64_t>(ks1 + (-8L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L)))));
                                auto tmp1 = (tmp0);
                                auto tmp2 = at::vec::convert<float,2,at::BFloat16,1>(tmp1);
                                auto tmp4 = (tmp3);
                                auto tmp5 = at::vec::convert<float,2,at::BFloat16,1>(tmp4);
                                auto tmp7 = ks1;
                                auto tmp8 = c10::convert<float>(tmp7);
                                auto tmp9 = tmp6 / tmp8;
                                auto tmp11 = c10::convert<float>(tmp10);
                                auto tmp12 = float(tmp9 + tmp11);
                                auto tmp13 = 1 / std::sqrt(tmp12);
                                auto tmp14 = at::vec::VectorizedN<float,2>(tmp13);
                                auto tmp15 = tmp5 * tmp14;
                                auto tmp16 = at::vec::convert<at::BFloat16,1,float,2>(tmp15);
                                auto tmp17 = tmp2 * tmp15;
                                auto tmp18 = at::vec::convert<at::BFloat16,1,float,2>(tmp17);
                                auto tmp20 = (tmp19);
                                auto tmp21 = at::vec::convert<float,2,at::BFloat16,1>(tmp20);
                                auto tmp22 = tmp21.neg();
                                auto tmp23 = tmp22.exp();
                                auto tmp24 = static_cast<float>(1.0);
                                auto tmp25 = at::vec::VectorizedN<float,2>(tmp24);
                                auto tmp26 = tmp23 + tmp25;
                                auto tmp27 = tmp21 / tmp26;
                                auto tmp28 = tmp17 * tmp27;
                                auto tmp29 = at::vec::convert<at::BFloat16,1,float,2>(tmp28);
                                tmp29.store(out_ptr1 + static_cast<int64_t>(x1 + ks1*x0 + ks1*x0_inner), static_cast<int64_t>(ks1 + (-8L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(8L)))));
                            }
                        }
                    }
                }
            }
        }
    }
}

// Python bindings to call kernel():
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <sstream>
#include <cstdlib>
#include <cerrno>

// This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
// We manually link it below to workaround issues with fbcode build.
static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
    static_assert(std::is_pointer_v<T>, "arg type must be pointer or long");
    return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
}
template <> inline int64_t parse_arg<int64_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
    if(result == -1 && PyErr_Occurred()) [[unlikely]]
        throw std::runtime_error("expected int arg");
    return result;
}
template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
    if(result == reinterpret_cast<void*>(-1) && PyErr_Occurred()) [[unlikely]]
        throw std::runtime_error("expected int arg");
    return reinterpret_cast<uintptr_t>(result);
}
template <> inline float parse_arg<float>(PyObject* args, size_t n) {
    auto result = PyFloat_AsDouble(PyTuple_GET_ITEM(args, n));
    if(result == -1.0 && PyErr_Occurred()) [[unlikely]]
        throw std::runtime_error("expected float arg");
    return static_cast<float>(result);
}



static PyObject* kernel_py(PyObject* self, PyObject* args) {
    try {
        if(!PyTuple_CheckExact(args)) [[unlikely]]
            throw std::runtime_error("tuple args required");
        if(PyTuple_GET_SIZE(args) != 9) [[unlikely]]
            throw std::runtime_error("requires 9 args");
        kernel(parse_arg<at::BFloat16*>(args, 0), parse_arg<at::BFloat16*>(args, 1), parse_arg<double*>(args, 2), parse_arg<at::BFloat16*>(args, 3), parse_arg<float*>(args, 4), parse_arg<at::BFloat16*>(args, 5), parse_arg<int64_t>(args, 6), parse_arg<int64_t>(args, 7), parse_arg<int64_t>(args, 8)); Py_RETURN_NONE;
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}

static PyMethodDef py_methods[] = {
    {"kernel", kernel_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef py_module =
    {PyModuleDef_HEAD_INIT, "kernel", NULL, -1, py_methods};

PyMODINIT_FUNC PyInit_kernel(void) {
    const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
    if(!str_addr) {
        PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
        return nullptr;
    }

    char* endptr = nullptr;
    errno = 0;
    uintptr_t addr = std::strtoull(str_addr, &endptr, 10);
    if(errno != 0 || endptr == str_addr || addr == 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to parse _TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
        return nullptr;
    }
    _torchinductor_pyobject_tensor_data_ptr =
        reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
    PyObject* module = PyModule_Create(&py_module);
    if (module == NULL) {
        return NULL;
    }
    #ifdef Py_GIL_DISABLED
        PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
    #endif
    return module;
}
