/*
 * blockguard.c
 *
 * Watchdog thread that detects asyncio event-loop blocking.
 * Polls the loop thread's current frame every poll_ms ms; fires if the
 * same (code object, line) is seen for >= threshold_ms ms.
 *
 * Output goes via write(2)/WriteFile -- deliberately NOT through logging
 * or fprintf, both of which can hold locks that the stuck loop thread
 * may already own, causing a deadlock.
 *
 * Compatible with CPython 3.10-3.14, POSIX and Windows.
 * PyFrameObject fields are opaque from 3.11; only public API is used.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#endif

typedef struct {
    volatile int    active;
    double          threshold_ms;
    double          poll_ms;
    PyThreadState  *loop_tstate;

#ifdef _WIN32
    HANDLE              watchdog_tid;
    CRITICAL_SECTION    mu;
    CONDITION_VARIABLE  cv;
#else
    pthread_t       watchdog_tid;
    pthread_mutex_t mu;
    pthread_cond_t  cv;
#endif
} GuardState;

static GuardState g;
static int g_initialized = 0;

static void
ensure_initialized(void)
{
    if (g_initialized) return;
    memset(&g, 0, sizeof(g));
#ifdef _WIN32
    InitializeCriticalSection(&g.mu);
    InitializeConditionVariable(&g.cv);
#else
    pthread_mutex_init(&g.mu, NULL);
#ifdef __linux__
    {
        pthread_condattr_t cattr;
        pthread_condattr_init(&cattr);
        pthread_condattr_setclock(&cattr, CLOCK_MONOTONIC);
        pthread_cond_init(&g.cv, &cattr);
        pthread_condattr_destroy(&cattr);
    }
#else
    pthread_cond_init(&g.cv, NULL);
#endif
#endif
    g_initialized = 1;
}

static void
write_stderr(const char *buf, size_t len)
{
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    if (h != INVALID_HANDLE_VALUE) {
        DWORD written;
        WriteFile(h, buf, (DWORD)len, &written, NULL);
    }
#else
    (void)write(STDERR_FILENO, buf, len);
#endif
}

typedef struct {
    PyCodeObject *code;
    int           lineno;
} FrameSnap;

static void
snap_clear(FrameSnap *s)
{
    Py_XDECREF(s->code);
    s->code   = NULL;
    s->lineno = -1;
}

/*
 * Returns a snapshot with a new strong reference to the code object.
 * Caller must call snap_clear() when done. Must be called with the GIL held.
 */
static FrameSnap
snap_frame(PyThreadState *tstate)
{
    FrameSnap s = {NULL, -1};
    if (!tstate) return s;

    PyFrameObject *f = PyThreadState_GetFrame(tstate);
    if (!f) return s;

    s.code   = PyFrame_GetCode(f);
    s.lineno = PyFrame_GetLineNumber(f);
    Py_DECREF(f);
    return s;
}

static int
snap_eq(const FrameSnap *a, const FrameSnap *b)
{
    return (a->code != NULL) && (a->code == b->code) && (a->lineno == b->lineno);
}

/*
 * The idle event loop sits in selectors.py:select() called from
 * base_events.py:_run_once(). We only skip this specific call chain,
 * not arbitrary selectors.py:select() calls (which would be a real block).
 */
static int
snap_is_idle_select(PyThreadState *tstate)
{
    if (!tstate) return 0;

    PyFrameObject *f = PyThreadState_GetFrame(tstate);
    if (!f) return 0;

    PyCodeObject *code = PyFrame_GetCode(f);
    const char *fn = PyUnicode_AsUTF8(code->co_filename);
    const char *name = PyUnicode_AsUTF8(code->co_name);
    Py_DECREF(code);

    if (!fn || !name || strcmp(name, "select") != 0) {
        Py_DECREF(f);
        return 0;
    }

    const char *p = strrchr(fn, '/');
    if (!p) p = strrchr(fn, '\\');
    const char *basename = p ? p + 1 : fn;

    if (strcmp(basename, "selectors.py") != 0) {
        Py_DECREF(f);
        return 0;
    }

    PyFrameObject *caller = PyFrame_GetBack(f);
    Py_DECREF(f);
    if (!caller) return 0;

    PyCodeObject *caller_code = PyFrame_GetCode(caller);
    const char *caller_fn = PyUnicode_AsUTF8(caller_code->co_filename);
    const char *caller_name = PyUnicode_AsUTF8(caller_code->co_name);
    Py_DECREF(caller_code);
    Py_DECREF(caller);

    if (!caller_fn || !caller_name) return 0;

    const char *cp = strrchr(caller_fn, '/');
    if (!cp) cp = strrchr(caller_fn, '\\');
    const char *caller_basename = cp ? cp + 1 : caller_fn;

    return strcmp(caller_basename, "base_events.py") == 0
        && strcmp(caller_name, "_run_once") == 0;
}

static int
append_str(PyObject *parts, PyObject *s)
{
    if (!s) { PyErr_Clear(); return -1; }
    int rc = PyList_Append(parts, s);
    Py_DECREF(s);
    if (rc < 0) { PyErr_Clear(); return -1; }
    return 0;
}

/*
 * Builds the traceback as a Python unicode string then emits it with
 * a single write. Individual frames that fail are skipped rather than
 * aborting the whole warning.
 */
static void
emit_warning(PyThreadState *tstate, double stuck_ms)
{
    PyObject *parts = PyList_New(0);
    if (!parts) { PyErr_Clear(); return; }

    #define MAX_DEPTH 48
    PyFrameObject *frames[MAX_DEPTH];
    int depth = 0;
    {
        PyFrameObject *f = PyThreadState_GetFrame(tstate);
        while (f && depth < MAX_DEPTH) {
            frames[depth++] = f;
            f = PyFrame_GetBack(f);
        }
        Py_XDECREF(f);
    }
    PyErr_Clear();

    append_str(parts, PyUnicode_FromFormat(
        "\n[blockguard] Event loop BLOCKED for %ld ms!\n"
        "  Stack (most recent call last):",
        (long)stuck_ms));

    if (depth == 0) {
        append_str(parts, PyUnicode_FromString(
            "\n    (no Python frames - blocked inside a C extension)"));
    } else {
        for (int i = depth - 1; i >= 0; i--) {
            PyFrameObject *fr = frames[i];
            PyCodeObject  *co = PyFrame_GetCode(fr);
            if (co) {
                append_str(parts, PyUnicode_FromFormat(
                    "\n    File \"%U\", line %d, in %U",
                    co->co_filename,
                    PyFrame_GetLineNumber(fr),
                    co->co_name));
                Py_DECREF(co);
            }
            Py_DECREF(fr);
        }
    }

    append_str(parts, PyUnicode_FromString(
        "\n  Use asyncio.to_thread() or loop.run_in_executor() "
        "for blocking work.\n"));

    {
        PyObject *sep = PyUnicode_FromString("");
        if (!sep) { PyErr_Clear(); goto done; }
        PyObject *msg = PyUnicode_Join(sep, parts);
        Py_DECREF(sep);
        if (!msg) { PyErr_Clear(); goto done; }

        Py_ssize_t nbytes;
        const char *buf = PyUnicode_AsUTF8AndSize(msg, &nbytes);
        if (buf && nbytes > 0)
            write_stderr(buf, (size_t)nbytes);

        Py_DECREF(msg);
    }

done:
    PyErr_Clear();
    Py_DECREF(parts);
}

static
#ifdef _WIN32
DWORD WINAPI
watchdog_body(LPVOID arg)
#else
void *
watchdog_body(void *arg)
#endif
{
    (void)arg;

    FrameSnap last          = {NULL, -1};
    double    stuck_ms      = 0.0;
    double    cooldown_left = 0.0;

    while (1) {
#ifdef _WIN32
        EnterCriticalSection(&g.mu);
        SleepConditionVariableCS(&g.cv, &g.mu, (DWORD)g.poll_ms);
        int still_active = g.active;
        LeaveCriticalSection(&g.mu);
#else
        struct timespec deadline;
#ifdef __linux__
        clock_gettime(CLOCK_MONOTONIC, &deadline);
#else
        clock_gettime(CLOCK_REALTIME, &deadline);
#endif
        {
            long add_ns = (long)(g.poll_ms * 1e6);
            deadline.tv_nsec += add_ns;
            while (deadline.tv_nsec >= 1000000000L) {
                deadline.tv_sec++;
                deadline.tv_nsec -= 1000000000L;
            }
        }
        pthread_mutex_lock(&g.mu);
        pthread_cond_timedwait(&g.cv, &g.mu, &deadline);
        int still_active = g.active;
        pthread_mutex_unlock(&g.mu);
#endif

        if (!still_active) break;

        PyGILState_STATE gil = PyGILState_Ensure();

        FrameSnap now = snap_frame(g.loop_tstate);

        if (now.code != NULL && snap_is_idle_select(g.loop_tstate)) {
            snap_clear(&now);
            snap_clear(&last);
            stuck_ms = 0.0;
            cooldown_left = 0.0;
        } else if (now.code != NULL && snap_eq(&now, &last)) {
            stuck_ms += g.poll_ms;
            if (cooldown_left > 0.0)
                cooldown_left -= g.poll_ms;
            snap_clear(&now);
        } else {
            snap_clear(&last);
            last          = now;
            stuck_ms      = 0.0;
            cooldown_left = 0.0;
        }

        if (stuck_ms >= g.threshold_ms && cooldown_left <= 0.0) {
            emit_warning(g.loop_tstate, stuck_ms);
            cooldown_left = g.threshold_ms * 10.0;
        }

        PyGILState_Release(gil);
    }

    if (last.code != NULL) {
        PyGILState_STATE gil = PyGILState_Ensure();
        snap_clear(&last);
        PyGILState_Release(gil);
    }
    return 0;
}

static PyObject *
py_install(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"threshold_ms", "poll_ms", NULL};
    double threshold_ms = 5000.0;
    double poll_ms      = 500.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|dd", kwlist,
                                     &threshold_ms, &poll_ms))
        return NULL;

    if (threshold_ms <= 0 || poll_ms <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "threshold_ms and poll_ms must be positive");
        return NULL;
    }
    if (poll_ms > threshold_ms) {
        PyErr_SetString(PyExc_ValueError,
                        "poll_ms must be <= threshold_ms");
        return NULL;
    }

    ensure_initialized();

    if (g.active) {
        PyErr_SetString(PyExc_RuntimeError, "blockguard is already installed");
        return NULL;
    }

    g.threshold_ms      = threshold_ms;
    g.poll_ms           = poll_ms;
    g.loop_tstate       = PyThreadState_Get();
    g.active            = 1;

    {
#ifdef _WIN32
        g.watchdog_tid = CreateThread(NULL, 0, watchdog_body, NULL, 0, NULL);
        if (g.watchdog_tid == NULL) {
            g.active = 0;
            errno = EAGAIN;
            PyErr_SetFromErrno(PyExc_OSError);
            return NULL;
        }
#else
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        int rc = pthread_create(&g.watchdog_tid, &attr, watchdog_body, NULL);
        pthread_attr_destroy(&attr);
        if (rc != 0) {
            g.active = 0;
            errno = rc;
            PyErr_SetFromErrno(PyExc_OSError);
            return NULL;
        }
#endif
    }

    {
        char buf[128];
        int n = snprintf(buf, sizeof(buf),
                         "[blockguard] watchdog started "
                         "(threshold=%.0f ms, poll=%.0f ms)\n",
                         threshold_ms, poll_ms);
        if (n > 0) write_stderr(buf, (size_t)(n < (int)sizeof(buf) ? n : (int)sizeof(buf) - 1));
    }

    Py_RETURN_NONE;
}

static PyObject *
py_uninstall(PyObject *self, PyObject *args)
{
    if (!g.active) {
        PyErr_SetString(PyExc_RuntimeError, "blockguard is not installed");
        return NULL;
    }

#ifdef _WIN32
    EnterCriticalSection(&g.mu);
    g.active = 0;
    WakeConditionVariable(&g.cv);
    LeaveCriticalSection(&g.mu);
#else
    pthread_mutex_lock(&g.mu);
    g.active = 0;
    pthread_cond_signal(&g.cv);
    pthread_mutex_unlock(&g.mu);
#endif

    Py_BEGIN_ALLOW_THREADS
#ifdef _WIN32
    WaitForSingleObject(g.watchdog_tid, INFINITE);
    CloseHandle(g.watchdog_tid);
#else
    pthread_join(g.watchdog_tid, NULL);
#endif
    Py_END_ALLOW_THREADS

    {
        const char msg[] = "[blockguard] watchdog stopped\n";
        write_stderr(msg, sizeof(msg) - 1);
    }

    Py_RETURN_NONE;
}

static PyMethodDef blockguard_methods[] = {
    {"install",   (PyCFunction)py_install, METH_VARARGS | METH_KEYWORDS,
     "install(threshold_ms=5000, poll_ms=500)\n"
     "Start the watchdog. Must be called from the event-loop thread."},
    {"uninstall", py_uninstall, METH_NOARGS,
     "uninstall()\nStop the watchdog thread."},
    {NULL}
};

static struct PyModuleDef blockguard_module = {
    PyModuleDef_HEAD_INIT, "blockguard",
    "Asyncio event-loop blocking detector.", -1,
    blockguard_methods
};

PyMODINIT_FUNC
PyInit_blockguard(void)
{
    return PyModule_Create(&blockguard_module);
}
