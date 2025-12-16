// This is a redefinition of the private/opaque frame object.
//
// https://github.com/python/cpython/blob/3.12/Include/internal/pycore_frame.h#L51
//
// In Python 3.10 and prior, `struct _frame` is both the PyFrameObject and
// PyInterpreterFrame. From Python 3.11 onwards, the two were split with the
// PyFrameObject (struct _frame) pointing to struct _PyInterpreterFrame.
struct Frame {
    PyCodeObject *f_code;
    struct Frame *previous; // struct _PyInterpreterFrame
    PyObject *f_funcobj;
    PyObject *f_globals;
    PyObject *f_builtins;
    PyObject *f_locals;
    PyFrameObject *frame_obj;
    _Py_CODEUNIT *prev_instr;
    int stacktop;
    uint16_t return_offset;
    char owner;
    PyObject *localsplus[1];
};

// This is a redefinition of private frame state constants.
// https://github.com/python/cpython/blob/3.12/Include/internal/pycore_frame.h#L34
typedef enum _framestate {
    FRAME_CREATED = -2,
    FRAME_SUSPENDED = -1,
    FRAME_EXECUTING = 0,
    FRAME_COMPLETED = 1,
    FRAME_CLEARED = 4
} FrameState;

/*
// This is the definition of PyFrameObject (aka. struct _frame) for reference
// to developers working on the extension.
//
// https://github.com/python/cpython/blob/3.12/Include/internal/pycore_frame.h#L16
typedef struct {
    PyObject_HEAD
    PyFrameObject *f_back;
    struct _PyInterpreterFrame *f_frame;
    PyObject *f_trace;
    int f_lineno;
    char f_trace_lines;
    char f_trace_opcodes;
    char f_fast_as_locals;
    PyObject *_f_frame_data[1];
} PyFrameObject;
*/

/*
// This is the definition of PyGenObject for reference to developers
// working on the extension.
//
// Note that PyCoroObject and PyAsyncGenObject have the same layout as
// PyGenObject, however the struct fields have a cr_ and ag_ prefix
// (respectively) rather than a gi_ prefix.
//
// https://github.com/python/cpython/blob/3.12/Include/cpython/genobject.h#L14
typedef struct {
    PyObject_HEAD
    PyObject *gi_weakreflist;
    PyObject *gi_name;
    PyObject *gi_qualname;
    _PyErr_StackItem gi_exc_state;
    PyObject *gi_origin_or_finalizer;
    char gi_hooks_inited;
    char gi_closed;
    char gi_running_async;
    int8_t gi_frame_state;
    PyObject *gi_iframe[1];
} PyGenObject;
*/

static Frame *get_frame(PyGenObject *gen_like) {
    Frame *frame = (Frame *)(struct _PyInterpreterFrame *)(gen_like->gi_iframe);
    assert(frame);
    return frame;
}

static PyCodeObject *get_frame_code(Frame *frame) {
    PyCodeObject *code = frame->f_code;
    assert(code);
    return code;
}

static int get_frame_lasti(Frame *frame) {
    // https://github.com/python/cpython/blob/3.12/Include/internal/pycore_frame.h#L77
    PyCodeObject *code = get_frame_code(frame);
    assert(frame->prev_instr);
    return (int)((intptr_t)frame->prev_instr - (intptr_t)_PyCode_CODE(code));
}

static void set_frame_lasti(Frame *frame, int lasti) {
    // https://github.com/python/cpython/blob/3.12/Include/internal/pycore_frame.h#L77
    PyCodeObject *code = get_frame_code(frame);
    frame->prev_instr = (_Py_CODEUNIT *)((intptr_t)_PyCode_CODE(code) + (intptr_t)lasti);
}

static int get_frame_state(PyGenObject *gen_like) {
    return gen_like->gi_frame_state;
}

static void set_frame_state(PyGenObject *gen_like, int fs) {
    gen_like->gi_frame_state = (int8_t)fs;
}

static int valid_frame_state(int fs) {
    return fs == FRAME_CREATED || fs == FRAME_SUSPENDED || fs == FRAME_EXECUTING || fs == FRAME_COMPLETED || fs == FRAME_CLEARED;
}

static int get_frame_stacktop_limit(Frame *frame) {
    PyCodeObject *code = get_frame_code(frame);
    return code->co_stacksize + code->co_nlocalsplus;
}

static int get_frame_stacktop(Frame *frame) {
    int stacktop = frame->stacktop;
    assert(stacktop >= 0 && stacktop < get_frame_stacktop_limit(frame));
    return stacktop;
}

static void set_frame_stacktop(Frame *frame, int stacktop) {
    assert(stacktop >= 0 && stacktop < get_frame_stacktop_limit(frame));
    frame->stacktop = stacktop;
}

static PyObject **get_frame_localsplus(Frame *frame) {
    PyObject **localsplus = frame->localsplus;
    assert(localsplus);
    return localsplus;
}

static int get_frame_iblock_limit(Frame *frame) {
    return 1; // not applicable >= 3.11
}

static int get_frame_iblock(Frame *frame) {
    return 0; // not applicable >= 3.11
}

static void set_frame_iblock(Frame *frame, int iblock) {
    assert(!iblock); // not applicable >= 3.11
}

static PyTryBlock *get_frame_blockstack(Frame *frame) {
    return NULL; // not applicable >= 3.11
}
