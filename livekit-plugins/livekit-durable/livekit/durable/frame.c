#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#if PY_MAJOR_VERSION != 3 || (PY_MINOR_VERSION < 10 || PY_MINOR_VERSION > 14)
#error Python 3.10-3.14 is required
#endif


/* only for internal headers */
#define Py_BUILD_CORE
#include "internal/pycore_code.h"
#if PY_MINOR_VERSION == 14
#include "internal/pycore_interpframe_structs.h"
#include "internal/pycore_stackref.h"
#endif
#if PY_MINOR_VERSION >= 11
#include "internal/pycore_frame.h"
#endif
#undef Py_BUILD_CORE



// This is a redefinition of the private PyTryBlock from <= 3.10.
// https://github.com/python/cpython/blob/3.8/Include/frameobject.h#L10
// https://github.com/python/cpython/blob/3.9/Include/cpython/frameobject.h#L11
// https://github.com/python/cpython/blob/3.10/Include/cpython/frameobject.h#L22
typedef struct {
  int b_type;
  int b_handler;
  int b_level;
} PyTryBlock;

#if PY_MINOR_VERSION >= 11

typedef _PyInterpreterFrame Frame;

#elif PY_MINOR_VERSION == 10

typedef signed char PyFrameState;

// https://github.com/python/cpython/blob/3.10/Include/cpython/frameobject.h#L28
struct _frame {
  PyObject_VAR_HEAD struct _frame *f_back; /* previous frame, or NULL */
  PyCodeObject *f_code;                    /* code segment */
  PyObject *f_builtins;    /* builtin symbol table (PyDictObject) */
  PyObject *f_globals;     /* global symbol table (PyDictObject) */
  PyObject *f_locals;      /* local symbol table (any mapping) */
  PyObject **f_valuestack; /* points after the last local */
  PyObject *f_trace;       /* Trace function */
  int f_stackdepth;        /* Depth of value stack */
  char f_trace_lines;      /* Emit per-line trace events? */
  char f_trace_opcodes;    /* Emit per-opcode trace events? */

  /* Borrowed reference to a generator, or NULL */
  PyObject *f_gen;

  int f_lasti;          /* Last instruction if called */
  int f_lineno;         /* Current line number. Only valid if non-zero */
  int f_iblock;         /* index in f_blockstack */
  PyFrameState f_state; /* What state the frame is in */
  PyTryBlock f_blockstack[CO_MAXBLOCKS]; /* for try and loop blocks */
  PyObject *f_localsplus[1];             /* locals+stack, dynamically sized */
};

typedef struct _frame Frame;

enum _framestate {
  FRAME_CREATED = -2,
  FRAME_SUSPENDED = -1,
  FRAME_EXECUTING = 0,
  FRAME_RETURNED = 1,
  FRAME_UNWINDING = 2,
  FRAME_RAISED = 3,
  FRAME_CLEARED = 4
};

#endif

// This is a redefinition of the private PyCoroWrapper from 3.8-3.13.
// https://github.com/python/cpython/blob/3.8/Objects/genobject.c#L840
// https://github.com/python/cpython/blob/3.9/Objects/genobject.c#L830
// https://github.com/python/cpython/blob/3.10/Objects/genobject.c#L884
// https://github.com/python/cpython/blob/3.11/Objects/genobject.c#L1016
// https://github.com/python/cpython/blob/3.12/Objects/genobject.c#L1003
// https://github.com/python/cpython/blob/v3.13.0a5/Objects/genobject.c#L985
typedef struct {
  PyObject_HEAD PyCoroObject *cw_coroutine;
} PyCoroWrapper;

#if PY_MINOR_VERSION <= 10
static int get_frame_iblock_limit(Frame *frame) { return CO_MAXBLOCKS; }

static int get_frame_iblock(Frame *frame) { return frame->f_iblock; }

static void set_frame_iblock(Frame *frame, int iblock) {
  assert(iblock >= 0 && iblock < get_frame_iblock_limit(frame));
  frame->f_iblock = iblock;
}

static PyTryBlock *get_frame_blockstack(Frame *frame) {
  PyTryBlock *blockstack = frame->f_blockstack;
  assert(blockstack);
  return blockstack;
}
#else
// iblock not applicable for pythom >= 3.11
static int get_frame_iblock_limit(Frame *frame) { return 1; }

static int get_frame_iblock(Frame *frame) { return 0; }

static void set_frame_iblock(Frame *frame, int iblock) { assert(!iblock); }

static PyTryBlock *get_frame_blockstack(Frame *frame) { return NULL; }
#endif

static Frame *get_frame(PyGenObject *gen_like) {
#if PY_MINOR_VERSION >= 14
  Frame *frame = (Frame *)(&gen_like->gi_iframe);
#elif PY_MINOR_VERSION >= 11
  Frame *frame = (Frame *)(struct _PyInterpreterFrame *)(gen_like->gi_iframe);
#elif PY_MINOR_VERSION == 10
  Frame *frame = (Frame *)(gen_like->gi_frame);
#endif
  assert(frame);
  return frame;
}

static PyCodeObject *get_frame_code(Frame *frame) {
#if PY_MINOR_VERSION >= 14
  PyObject *executable = PyStackRef_AsPyObjectBorrow(frame->f_executable);
  PyCodeObject *code = (PyCodeObject *)executable;
#elif PY_MINOR_VERSION >= 13
  PyCodeObject *code = (PyCodeObject *)frame->f_executable;
#elif PY_MINOR_VERSION >= 10
  PyCodeObject *code = (PyCodeObject *)frame->f_code;
#endif
  assert(code);
  return code;
}

static int get_frame_lasti(Frame *frame) {
  PyCodeObject *code = get_frame_code(frame);
#if PY_MINOR_VERSION == 11 || PY_MINOR_VERSION == 12
  assert(frame->prev_instr);
  return (int)((intptr_t)frame->prev_instr - (intptr_t)_PyCode_CODE(code));
#elif PY_MINOR_VERSION == 10
  return frame->f_lasti;
#else
  assert(frame->instr_ptr);
  return (int)((intptr_t)frame->instr_ptr - (intptr_t)_PyCode_CODE(code));
#endif
}

static void set_frame_lasti(Frame *frame, int lasti) {
  PyCodeObject *code = get_frame_code(frame);
#if PY_MINOR_VERSION == 11 || PY_MINOR_VERSION == 12
  frame->prev_instr =
      (_Py_CODEUNIT *)((intptr_t)_PyCode_CODE(code) + (intptr_t)lasti);
#elif PY_MINOR_VERSION == 10
  frame->f_lasti = lasti;
#else
  frame->instr_ptr =
      (_Py_CODEUNIT *)((intptr_t)_PyCode_CODE(code) + (intptr_t)lasti);
#endif
}

static int get_frame_state(PyGenObject *gen_like) {
#if PY_MINOR_VERSION == 10
  Frame *frame = (Frame *)(gen_like->gi_frame);
  if (!frame)
    return FRAME_CLEARED;
  return frame->f_state;
#else
  return gen_like->gi_frame_state;
#endif
}

static void set_frame_state(PyGenObject *gen_like, int fs) {
#if PY_MINOR_VERSION == 10
  Frame *frame = get_frame(gen_like);
  frame->f_state = (PyFrameState)fs;

#else
  gen_like->gi_frame_state = (int8_t)fs;
#endif
}

static int valid_frame_state(int fs) {
#if PY_MINOR_VERSION >= 13
  return fs == FRAME_CREATED || fs == FRAME_SUSPENDED ||
         fs == FRAME_EXECUTING || fs == FRAME_COMPLETED ||
         fs == FRAME_CLEARED || fs == FRAME_SUSPENDED_YIELD_FROM;

#elif PY_MINOR_VERSION >= 11
  return fs == FRAME_CREATED || fs == FRAME_SUSPENDED ||
         fs == FRAME_EXECUTING || fs == FRAME_COMPLETED || fs == FRAME_CLEARED;
#elif PY_MINOR_VERSION == 10
  return fs == FRAME_CREATED || fs == FRAME_SUSPENDED ||
         fs == FRAME_EXECUTING || fs == FRAME_RETURNED ||
         fs == FRAME_UNWINDING || fs == FRAME_RAISED || fs == FRAME_CLEARED;
#endif
}

static int get_frame_stacktop_limit(Frame *frame) {
  PyCodeObject *code = get_frame_code(frame);
#if PY_MINOR_VERSION == 10
  return code->co_stacksize + code->co_nlocals;
#else
  return code->co_stacksize + code->co_nlocalsplus;
#endif
}

static int get_frame_stacktop(Frame *frame) {
#if PY_MINOR_VERSION >= 14
  ptrdiff_t stacktop = frame->stackpointer - frame->localsplus;
  assert(stacktop >= 0 && stacktop < get_frame_stacktop_limit(frame));
  return (int)stacktop;
#elif PY_MINOR_VERSION >= 11
  int stacktop = frame->stacktop;
  assert(stacktop >= 0 && stacktop < get_frame_stacktop_limit(frame));
  return stacktop;
#elif PY_MINOR_VERSION == 10
  assert(frame->f_localsplus);
  assert(frame->f_valuestack);
  int stacktop =
      (int)(frame->f_valuestack - frame->f_localsplus) + frame->f_stackdepth;
  assert(stacktop >= 0 && stacktop < get_frame_stacktop_limit(frame));
  return stacktop;

#endif
}

static void set_frame_stacktop(Frame *frame, int stacktop) {
#if PY_MINOR_VERSION >= 14
  assert(stacktop >= 0 && stacktop < get_frame_stacktop_limit(frame));
  frame->stackpointer = frame->localsplus + stacktop;
#elif PY_MINOR_VERSION >= 11
  assert(stacktop >= 0 && stacktop < get_frame_stacktop_limit(frame));
  frame->stacktop = stacktop;
#elif PY_MINOR_VERSION == 10
  assert(stacktop >= 0 && stacktop < get_frame_stacktop_limit(frame));
  assert(frame->f_localsplus);
  assert(frame->f_valuestack);
  int base = (int)(frame->f_valuestack - frame->f_localsplus);
  assert(stacktop >= base);
  frame->f_stackdepth = stacktop - base;

#endif
}

static const char *get_type_name(PyObject *obj) {
  PyObject *type = PyObject_Type(obj);
  if (!type) {
    return NULL;
  }
  PyObject *name = PyObject_GetAttrString(type, "__name__");
  Py_DECREF(type);
  if (!name) {
    return NULL;
  }
  const char *name_str = PyUnicode_AsUTF8(name);
  Py_DECREF(name);
  return name_str;
}

static PyGenObject *get_generator_like_object(PyObject *obj) {
  if (PyGen_Check(obj) || PyCoro_CheckExact(obj) ||
      PyAsyncGen_CheckExact(obj)) {
    // Note: In Python 3.9-3.13, the PyGenObject, PyCoroObject and
    // PyAsyncGenObject have the same layout, they just have different field
    // prefixes (gi_, cr_, ag_). We cast to PyGenObject here so that the
    // remainder of the code can use the gi_ prefix for all three cases.
    return (PyGenObject *)obj;
  }
  // If the object isn't a PyGenObject, PyCoroObject or PyAsyncGenObject, it may
  // still be a coroutine, for example a PyCoroWrapper. CPython unfortunately
  // does not export a function that checks whether a PyObject is a
  // PyCoroWrapper. We need to check the type name string.
  const char *type_name = get_type_name(obj);
  if (!type_name) {
    return NULL;
  }
  if (strcmp(type_name, "coroutine_wrapper") == 0) {
    // FIXME: improve safety here, e.g. by checking that the obj type matches a
    // known size
    PyCoroWrapper *wrapper = (PyCoroWrapper *)obj;
    // Cast the inner PyCoroObject to a PyGenObject. See the comment above.
    return (PyGenObject *)wrapper->cw_coroutine;
  }
  PyErr_SetString(PyExc_TypeError,
                  "Input object is not a generator or coroutine");
  return NULL;
}

static PyObject *ext_get_frame_state(PyObject *self, PyObject *args) {
  PyObject *arg;
  if (!PyArg_ParseTuple(args, "O", &arg)) {
    return NULL;
  }
  PyGenObject *gen_like = get_generator_like_object(arg);
  if (!gen_like) {
    return NULL;
  }
  int fs = get_frame_state(gen_like);
  return PyLong_FromLong((long)fs);
}

static PyObject *ext_get_frame_ip(PyObject *self, PyObject *args) {
  PyObject *obj;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }
  PyGenObject *gen_like = get_generator_like_object(obj);
  if (!gen_like) {
    return NULL;
  }
  if (get_frame_state(gen_like) >= FRAME_CLEARED) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot access cleared frame");
    return NULL;
  }
  Frame *frame = get_frame(gen_like);
  if (!frame) {
    return NULL;
  }
  int ip = get_frame_lasti(frame);
  return PyLong_FromLong((long)ip);
}

static PyObject *ext_get_frame_sp(PyObject *self, PyObject *args) {
  PyObject *obj;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }
  PyGenObject *gen_like = get_generator_like_object(obj);
  if (!gen_like) {
    return NULL;
  }
  if (get_frame_state(gen_like) >= FRAME_CLEARED) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot access cleared frame");
    return NULL;
  }
  Frame *frame = get_frame(gen_like);
  if (!frame) {
    return NULL;
  }
  int sp = get_frame_stacktop(frame);
  return PyLong_FromLong((long)sp);
}

static PyObject *ext_get_frame_bp(PyObject *self, PyObject *args) {
  PyObject *obj;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }
  PyGenObject *gen_like = get_generator_like_object(obj);
  if (!gen_like) {
    return NULL;
  }
  if (get_frame_state(gen_like) >= FRAME_CLEARED) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot access cleared frame");
    return NULL;
  }
  Frame *frame = get_frame(gen_like);
  if (!frame) {
    return NULL;
  }
  int bp = get_frame_iblock(frame);
  return PyLong_FromLong((long)bp);
}

static PyObject *ext_get_frame_stack_at(PyObject *self, PyObject *args) {
  PyObject *obj;
  int index;
  if (!PyArg_ParseTuple(args, "Oi", &obj, &index)) {
    return NULL;
  }
  PyGenObject *gen_like = get_generator_like_object(obj);
  if (!gen_like) {
    return NULL;
  }
  if (get_frame_state(gen_like) >= FRAME_CLEARED) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot access cleared frame");
    return NULL;
  }
  Frame *frame = get_frame(gen_like);
  if (!frame) {
    return NULL;
  }
  int sp = get_frame_stacktop(frame);
  if (index < 0 || index >= sp) {
    PyErr_SetString(PyExc_IndexError, "Index out of bounds");
    return NULL;
  }

  // NULL in C != None in Python. We need to preserve the fact that some items
  // on the stack are NULL (not yet available).
  PyObject *is_null = Py_False;

#if PY_MINOR_VERSION >= 14
  PyObject *stack_obj = PyStackRef_AsPyObjectBorrow(frame->localsplus[index]);
#elif PY_MINOR_VERSION >= 11
  PyObject *stack_obj = frame->localsplus[index];
#elif PY_MINOR_VERSION == 10
  PyObject *stack_obj = frame->f_localsplus[index];
#endif
  if (!stack_obj) {
    is_null = Py_True;
    stack_obj = Py_None;
  }
  return PyTuple_Pack(2, is_null, stack_obj);
}

static PyObject *ext_get_frame_block_at(PyObject *self, PyObject *args) {
  PyObject *obj;
  int index;
  if (!PyArg_ParseTuple(args, "Oi", &obj, &index)) {
    return NULL;
  }
  PyGenObject *gen_like = get_generator_like_object(obj);
  if (!gen_like) {
    return NULL;
  }
  if (get_frame_state(gen_like) >= FRAME_CLEARED) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot access cleared frame");
    return NULL;
  }
  Frame *frame = get_frame(gen_like);
  if (!frame) {
    return NULL;
  }
  int bp = get_frame_iblock(frame);
  if (index < 0 || index >= bp) {
    PyErr_SetString(PyExc_IndexError, "Index out of bounds");
    return NULL;
  }
  PyTryBlock *blockstack = get_frame_blockstack(frame);
  PyTryBlock *block = &blockstack[index];
  return Py_BuildValue("(iii)", block->b_type, block->b_handler,
                       block->b_level);
}

static PyObject *ext_set_frame_ip(PyObject *self, PyObject *args) {
  PyObject *obj;
  int ip;
  if (!PyArg_ParseTuple(args, "Oi", &obj, &ip)) {
    return NULL;
  }
  PyGenObject *gen_like = get_generator_like_object(obj);
  if (!gen_like) {
    return NULL;
  }
  if (get_frame_state(gen_like) >= FRAME_CLEARED) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot mutate cleared frame");
    return NULL;
  }
  Frame *frame = get_frame(gen_like);
  if (!frame) {
    return NULL;
  }
  set_frame_lasti(frame, ip);
  Py_RETURN_NONE;
}

static PyObject *ext_set_frame_sp(PyObject *self, PyObject *args) {
  PyObject *obj;
  int sp;
  if (!PyArg_ParseTuple(args, "Oi", &obj, &sp)) {
    return NULL;
  }
  PyGenObject *gen_like = get_generator_like_object(obj);
  if (!gen_like) {
    return NULL;
  }
  if (get_frame_state(gen_like) >= FRAME_CLEARED) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot mutate cleared frame");
    return NULL;
  }
  Frame *frame = get_frame(gen_like);
  if (!frame) {
    return NULL;
  }
  int limit = get_frame_stacktop_limit(frame);
  if (sp < 0 || sp >= limit) {
    PyErr_SetString(PyExc_IndexError, "Stack pointer out of bounds");
    return NULL;
  }
  int current_sp = get_frame_stacktop(frame);

#if PY_MINOR_VERSION >= 14
  _PyStackRef *localsplus = frame->localsplus;

  if (sp > current_sp) {
    for (int i = current_sp; i < sp; i++) {
      localsplus[i] = PyStackRef_NULL;
    }
  } else if (sp < current_sp) {
    for (int i = sp; i < current_sp; i++) {
      PyStackRef_XCLOSE(localsplus[i]);
      localsplus[i] = PyStackRef_NULL;
    }
  }
#else

#if PY_MINOR_VERSION >= 11
  PyObject **localsplus = frame->localsplus;

#elif PY_MINOR_VERSION == 10
  PyObject **localsplus = frame->f_localsplus;
#endif

  if (sp > current_sp) {
    for (int i = current_sp; i < sp; i++) {
      localsplus[i] = NULL;
    }
  } else if (sp < current_sp) {
    for (int i = sp; i < current_sp; i++) {
      Py_XDECREF(localsplus[i]);
      localsplus[i] = NULL;
    }
  }
#endif

  set_frame_stacktop(frame, sp);
  Py_RETURN_NONE;
}

static PyObject *ext_set_frame_bp(PyObject *self, PyObject *args) {
  PyObject *obj;
  int bp;
  if (!PyArg_ParseTuple(args, "Oi", &obj, &bp)) {
    return NULL;
  }
  PyGenObject *gen_like = get_generator_like_object(obj);
  if (!gen_like) {
    return NULL;
  }
  if (get_frame_state(gen_like) >= FRAME_CLEARED) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot mutate cleared frame");
    return NULL;
  }
  Frame *frame = get_frame(gen_like);
  if (!frame) {
    return NULL;
  }
  int limit = get_frame_iblock_limit(frame);
  if (bp < 0 || bp >= limit) {
    PyErr_SetString(PyExc_IndexError, "Block pointer out of bounds");
    return NULL;
  }
  set_frame_iblock(frame, bp);
  Py_RETURN_NONE;
}

static PyObject *ext_set_frame_state(PyObject *self, PyObject *args) {
  PyObject *obj;
  int fs;
  if (!PyArg_ParseTuple(args, "Oi", &obj, &fs)) {
    return NULL;
  }
  if (fs == FRAME_CLEARED) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Cannot set frame state to FRAME_CLEARED");
    return NULL;
  }
  PyGenObject *gen_like = get_generator_like_object(obj);
  if (!gen_like) {
    return NULL;
  }
  if (get_frame_state(gen_like) >= FRAME_CLEARED) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot mutate cleared frame");
    return NULL;
  }
  Frame *frame = get_frame(gen_like);
  if (!frame) {
    return NULL;
  }
  if (!valid_frame_state(fs)) {
    PyErr_SetString(PyExc_ValueError, "Invalid frame state");
    return NULL;
  }
  set_frame_state(gen_like, fs);
  Py_RETURN_NONE;
}

static PyObject *ext_set_frame_stack_at(PyObject *self, PyObject *args) {
  PyObject *obj;
  int index;
  PyObject *unset;
  PyObject *stack_obj;
  if (!PyArg_ParseTuple(args, "OiOO", &obj, &index, &unset, &stack_obj)) {
    return NULL;
  }
  if (!PyBool_Check(unset)) {
    PyErr_SetString(
        PyExc_TypeError,
        "Expected a boolean indicating whether to unset the stack object");
    return NULL;
  }
  PyGenObject *gen_like = get_generator_like_object(obj);
  if (!gen_like) {
    return NULL;
  }
  if (get_frame_state(gen_like) >= FRAME_CLEARED) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot mutate cleared frame");
    return NULL;
  }
  Frame *frame = get_frame(gen_like);
  if (!frame) {
    return NULL;
  }
  int sp = get_frame_stacktop(frame);
  if (index < 0 || index >= sp) {
    PyErr_SetString(PyExc_IndexError, "Index out of bounds");
    return NULL;
  }

#if PY_MINOR_VERSION >= 14
  _PyStackRef *localsplus = frame->localsplus;

  PyStackRef_XCLOSE(localsplus[index]);

  if (PyObject_IsTrue(unset)) {
    localsplus[index] = PyStackRef_NULL;
  } else {
    Py_INCREF(stack_obj);
    localsplus[index] = PyStackRef_FromPyObjectSteal(stack_obj);
  }

#else

#if PY_MINOR_VERSION >= 11
  PyObject **localsplus = frame->localsplus;
#elif PY_MINOR_VERSION == 10
  PyObject **localsplus = frame->f_localsplus;
#endif

  PyObject *prev = localsplus[index];
  if (PyObject_IsTrue(unset)) {
    localsplus[index] = NULL;
  } else {
    Py_INCREF(stack_obj);
    localsplus[index] = stack_obj;
  }
  Py_XDECREF(prev);
#endif
  Py_RETURN_NONE;
}

static PyObject *ext_set_frame_block_at(PyObject *self, PyObject *args) {
  PyObject *obj;
  int index;
  int b_type;
  int b_handler;
  int b_level;
  if (!PyArg_ParseTuple(args, "Oi(iii)", &obj, &index, &b_type, &b_handler,
                        &b_level)) {
    return NULL;
  }
  PyGenObject *gen_like = get_generator_like_object(obj);
  if (!gen_like) {
    return NULL;
  }
  if (get_frame_state(gen_like) >= FRAME_CLEARED) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot mutate cleared frame");
    return NULL;
  }
  Frame *frame = get_frame(gen_like);
  if (!frame) {
    return NULL;
  }
  int bp = get_frame_iblock(frame);
  if (index < 0 || index >= bp) {
    PyErr_SetString(PyExc_IndexError, "Index out of bounds");
    return NULL;
  }
  PyTryBlock *blockstack = get_frame_blockstack(frame);
  PyTryBlock *block = &blockstack[index];
  block->b_type = b_type;
  block->b_handler = b_handler;
  block->b_level = b_level;
  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"get_frame_ip", ext_get_frame_ip, METH_VARARGS,
     "Get instruction pointer of a generator or coroutine."},
    {"set_frame_ip", ext_set_frame_ip, METH_VARARGS,
     "Set instruction pointer of a generator or coroutine."},
    {"get_frame_sp", ext_get_frame_sp, METH_VARARGS,
     "Get stack pointer of a generator or coroutine."},
    {"set_frame_sp", ext_set_frame_sp, METH_VARARGS,
     "Set stack pointer of a generator or coroutine."},
    {"get_frame_bp", ext_get_frame_bp, METH_VARARGS,
     "Get block pointer of a generator or coroutine."},
    {"set_frame_bp", ext_set_frame_bp, METH_VARARGS,
     "Set block pointer of a generator or coroutine."},
    {"get_frame_stack_at", ext_get_frame_stack_at, METH_VARARGS,
     "Get an object from a generator or coroutine's stack, as an (is_null, "
     "obj) tuple."},
    {"set_frame_stack_at", ext_set_frame_stack_at, METH_VARARGS,
     "Set or unset an object on the stack of a generator or coroutine."},
    {"get_frame_block_at", ext_get_frame_block_at, METH_VARARGS,
     "Get a block from a generator or coroutine."},
    {"set_frame_block_at", ext_set_frame_block_at, METH_VARARGS,
     "Restore a block of a generator or coroutine."},
    {"get_frame_state", ext_get_frame_state, METH_VARARGS,
     "Get frame state of a generator or coroutine."},
    {"set_frame_state", ext_set_frame_state, METH_VARARGS,
     "Set frame state of a generator or coroutine."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "lk_durable", NULL,
                                    -1, methods};

PyMODINIT_FUNC PyInit_lk_durable(void) { return PyModule_Create(&module); }
