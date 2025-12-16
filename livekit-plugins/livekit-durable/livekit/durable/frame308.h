// This is a redefinition of the private/opaque frame object.
//
// https://github.com/python/cpython/blob/3.8/Include/frameobject.h#L16
//
// In Python <= 3.10, `struct _frame` is both the PyFrameObject and
// PyInterpreterFrame. From Python 3.11 onwards, the two were split with the
// PyFrameObject (struct _frame) pointing to struct _PyInterpreterFrame.
struct Frame {
    PyObject_VAR_HEAD
    struct Frame *f_back; // struct _frame
    PyCodeObject *f_code;
    PyObject *f_builtins;
    PyObject *f_globals;
    PyObject *f_locals;
    PyObject **f_valuestack;
    PyObject **f_stacktop;
    PyObject *f_trace;
    char f_trace_lines;
    char f_trace_opcodes;
    PyObject *f_gen;
    int f_lasti;
    int f_lineno;
    int f_iblock;
    char f_executing;
    PyTryBlock f_blockstack[CO_MAXBLOCKS];
    PyObject *f_localsplus[1];
};

// Python 3.9 and prior didn't have an explicit enum of frame states,
// but we can derive them based on the presence of a frame, and other
// information found on the frame, for compatibility with later versions.
typedef enum _framestate {
    FRAME_CREATED = -2,
    FRAME_EXECUTING = 0,
    FRAME_CLEARED = 4
} FrameState;

/*
// This is the definition of PyGenObject for reference to developers
// working on the extension.
//
// Note that PyCoroObject and PyAsyncGenObject have the same layout as
// PyGenObject, however the struct fields have a cr_ and ag_ prefix
// (respectively) rather than a gi_ prefix. In Python <= 3.10, PyCoroObject
// and PyAsyncGenObject have extra fields compared to PyGenObject. In Python
// 3.11 onwards, the three objects are identical (except for field name
// prefixes). The extra fields in Python <= 3.10 are not applicable to the
// extension at this time.
//
// https://github.com/python/cpython/blob/3.8/Include/genobject.h#L17
typedef struct {
    PyObject_HEAD
    PyFrameObject *gi_frame;
    char gi_running;
    PyObject *gi_code;
    PyObject *gi_weakreflist;
    PyObject *gi_name;
    PyObject *gi_qualname;
    _PyErr_StackItem gi_exc_state;
} PyGenObject;
*/

static Frame *get_frame(PyGenObject *gen_like) {
    Frame *frame = (Frame *)(gen_like->gi_frame);
    assert(frame);
    return frame;
}

static PyCodeObject *get_frame_code(Frame *frame) {
    PyCodeObject *code = frame->f_code;
    assert(code);
    return code;
}

static int get_frame_lasti(Frame *frame) {
    return frame->f_lasti;
}

static void set_frame_lasti(Frame *frame, int lasti) {
    frame->f_lasti = lasti;
}

static int get_frame_state(PyGenObject *gen_like) {
    // Python 3.8 doesn't have frame states, but we can derive
    // some for compatibility with later versions and to simplify
    // the extension.
    Frame *frame = (Frame *)(gen_like->gi_frame);
    if (!frame) {
        return FRAME_CLEARED;
    }
    return frame->f_executing ? FRAME_EXECUTING : FRAME_CREATED;
}

static void set_frame_state(PyGenObject *gen_like, int fs) {
    Frame *frame = get_frame(gen_like);
    frame->f_executing = (fs == FRAME_EXECUTING);
}

static int valid_frame_state(int fs) {
    return fs == FRAME_CREATED || fs == FRAME_EXECUTING || fs == FRAME_CLEARED;
}

static int get_frame_stacktop_limit(Frame *frame) {
    PyCodeObject *code = get_frame_code(frame);
    return code->co_stacksize + code->co_nlocals;
}

static int get_frame_stacktop(Frame *frame) {
    assert(frame->f_localsplus);
    int stacktop = (int)(frame->f_stacktop - frame->f_localsplus);
    assert(stacktop >= 0 && stacktop < get_frame_stacktop_limit(frame));
    return stacktop;
}

static void set_frame_stacktop(Frame *frame, int stacktop) {
    assert(stacktop >= 0 && stacktop < get_frame_stacktop_limit(frame));
    assert(frame->f_localsplus);
    frame->f_stacktop = frame->f_localsplus + stacktop;
}

static PyObject **get_frame_localsplus(Frame *frame) {
    PyObject **localsplus = frame->f_localsplus;
    assert(localsplus);
    return localsplus;
}

static int get_frame_iblock_limit(Frame *frame) {
    return CO_MAXBLOCKS;
}

static int get_frame_iblock(Frame *frame) {
    return frame->f_iblock;
}

static void set_frame_iblock(Frame *frame, int iblock) {
    assert(iblock >= 0 && iblock < get_frame_iblock_limit(frame));
    frame->f_iblock = iblock;
}

static PyTryBlock *get_frame_blockstack(Frame *frame) {
    PyTryBlock *blockstack = frame->f_blockstack;
    assert(blockstack);
    return blockstack;
}

