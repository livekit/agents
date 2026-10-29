// Deliberately limited Node-API host for the 0.2.7 inference ABI.
// No JavaScript evaluator, private engine ABI, or legacy Python is involved.
// A capsule belongs to its creating thread in an owned Python3152 child.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <node_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <poll.h>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <thread>
#include <utility>
#include <variant>
#include <vector>
#include <unistd.h>

#if PY_MAJOR_VERSION != 3 || PY_MINOR_VERSION != 15 || defined(Py_GIL_DISABLED)
#error "This proof requires ordinary GIL CPython3.15 headers."
#endif

struct Value;
using ValuePtr = std::shared_ptr<Value>;
struct Function { napi_callback callback; void* data; };
struct External { void* data; };
struct TypedArray { std::shared_ptr<std::vector<int16_t>> data; ValuePtr buffer; };
struct Promise { enum class State { Pending, Resolved, Rejected }; State state = State::Pending; ValuePtr result; };
struct Finalizer { napi_finalize callback; void* data; void* hint; };
struct Property { ValuePtr value; napi_callback getter = nullptr; napi_callback setter = nullptr; void* data = nullptr; napi_property_attributes attributes = napi_default_jsproperty; };
enum class Kind { Undefined, Number, String, Object, Function, External, TypedArray, ArrayBuffer, Promise, Error };
struct Value {
    napi_env env;
    Kind kind;
    std::variant<std::monostate, double, std::string, Function, External, TypedArray, Promise> data;
    std::map<std::string, Property> properties;
    std::vector<Finalizer> finalizers;
    Value(napi_env owner, Kind tag);
    ~Value();
};
struct napi_value__ { ValuePtr value; };
struct Scope { Scope* parent; bool escapable; bool escaped = false; std::vector<std::unique_ptr<napi_value__>> handles; };
struct napi_ref__ { std::weak_ptr<Value> weak; ValuePtr strong; uint32_t count; };
struct napi_deferred__ { ValuePtr promise; };
struct napi_callback_info__ { napi_env env; ValuePtr receiver; std::vector<ValuePtr> arguments; void* data; };
struct napi_async_work__ {
    enum class State { Created, Queued, Running, Complete };
    napi_async_execute_callback execute;
    napi_async_complete_callback complete;
    void* data;
    State state = State::Created;
};

// The dedicated parent pipe has no payload. Its last writer belongs to the
// SDK process; EOF must retire this child even while foreign code holds its GIL.
class Descriptor {
    int value_ = -1;
public:
    Descriptor() = default;
    explicit Descriptor(int fd) noexcept : value_(fd) {}
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    Descriptor(Descriptor&& other) noexcept : value_(std::exchange(other.value_, -1)) {}
    Descriptor& operator=(Descriptor&& other) noexcept {
        if (this != &other) { close(); value_ = std::exchange(other.value_, -1); }
        return *this;
    }
    int get() const noexcept { return value_; }
    void close() noexcept {
        if (value_ >= 0) {
            // Retrying close after EINTR can close a reused descriptor.
            (void)::close(std::exchange(value_, -1));
        }
    }
    ~Descriptor() { close(); }
};
static constexpr int PARENT_LOST_EXIT_CODE = 74;
class ParentWatch {
    Descriptor parent_, cancelled_, cancel_;
    std::thread thread_;
    const std::thread::id owner_ = std::this_thread::get_id();
    void observe() const noexcept {
        pollfd descriptors[2]{{parent_.get(), POLLIN, 0}, {cancelled_.get(), POLLIN, 0}};
        for (;;) {
            const auto ready = ::poll(descriptors, 2, -1);
            if (ready < 0) { if (errno == EINTR) continue; _exit(PARENT_LOST_EXIT_CODE); }
            if (descriptors[1].revents & POLLIN) return;
            if (descriptors[0].revents || descriptors[1].revents) _exit(PARENT_LOST_EXIT_CODE);
        }
    }
public:
    bool on_owner() const noexcept { return owner_ == std::this_thread::get_id(); }
    explicit ParentWatch(int descriptor) {
        struct stat metadata{};
        if (::fstat(descriptor, &metadata) < 0) throw std::system_error(errno, std::generic_category(), "inspect parent lifeline");
        const auto flags = ::fcntl(descriptor, F_GETFL);
        if (flags < 0) throw std::system_error(errno, std::generic_category(), "inspect parent lifeline flags");
        if (!S_ISFIFO(metadata.st_mode) || (flags & O_ACCMODE) != O_RDONLY) throw std::invalid_argument("parent lifeline must be a dedicated read-only pipe");
        const auto duplicate = ::fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
        if (duplicate < 0) throw std::system_error(errno, std::generic_category(), "duplicate parent lifeline");
        parent_ = Descriptor(duplicate);
        int ends[2]{};
        if (::pipe(ends) < 0) throw std::system_error(errno, std::generic_category(), "create parent-watch cancellation pipe");
        cancelled_ = Descriptor(ends[0]); cancel_ = Descriptor(ends[1]);
        if (::fcntl(cancelled_.get(), F_SETFD, FD_CLOEXEC) < 0 || ::fcntl(cancel_.get(), F_SETFD, FD_CLOEXEC) < 0 || ::fcntl(cancel_.get(), F_SETFL, O_NONBLOCK) < 0) throw std::system_error(errno, std::generic_category(), "isolate parent-watch cancellation pipe");
        thread_ = std::thread([this] { observe(); });
    }
    void stop() noexcept {
        if (!thread_.joinable()) return;
        const char cancel = 0;
        bool delivered = false;
        for (unsigned attempt = 0; attempt < 16; ++attempt) {
            const auto sent = ::write(cancel_.get(), &cancel, 1);
            if (sent == 1 || (sent < 0 && errno == EAGAIN)) { delivered = true; break; }
            if (errno != EINTR) break;
        }
        if (!delivered) _exit(PARENT_LOST_EXIT_CODE);
        try { thread_.join(); }
        catch (...) { _exit(PARENT_LOST_EXIT_CODE); }
    }
    ~ParentWatch() { stop(); }
};
static std::unique_ptr<ParentWatch> parent_watch;

struct Worker {
    std::mutex mutex;
    std::condition_variable ready, settled;
    napi_async_work work = nullptr;
    napi_env env = nullptr;
    bool stopping = false, done = false;
    std::exception_ptr failure;
    std::thread thread;
    Worker() : thread([this] {
        std::unique_lock lock(mutex);
        for (;;) {
            ready.wait(lock, [this] { return stopping || work != nullptr; });
            if (stopping) return;
            auto current = work;
            auto current_env = env;
            lock.unlock();
            try { current->execute(current_env, current->data); }
            catch (...) { failure = std::current_exception(); }
            lock.lock();
            work = nullptr;
            done = true;
            settled.notify_one();
        }
    }) {}
    void run(napi_env owner, napi_async_work item) {
        std::unique_lock lock(mutex);
        if (work != nullptr) throw std::runtime_error("inference worker is busy");
        done = false; failure = nullptr; env = owner; work = item; ready.notify_one();
        settled.wait(lock, [this] { return done; });
        auto error = std::exchange(failure, nullptr);
        if (error) {
            try { std::rethrow_exception(error); }
            catch (const std::exception& native_error) { throw std::runtime_error(native_error.what()); }
            catch (...) { throw std::runtime_error("native worker threw a non-standard C++ exception"); }
        }
    }
    void stop() {
        { std::lock_guard lock(mutex); stopping = true; ready.notify_one(); }
        if (thread.joinable()) thread.join();
        failure = nullptr;
    }
    ~Worker() { stop(); }
};

struct napi_env__ {
    std::thread::id owner = std::this_thread::get_id();
    Scope root{nullptr, false, false, {}};
    Scope* scope = &root;
    std::map<Scope*, std::unique_ptr<Scope>> scopes;
    std::set<napi_value> handles;
    std::map<napi_ref, std::unique_ptr<napi_ref__>> references;
    std::map<napi_deferred, std::unique_ptr<napi_deferred__>> deferred;
    std::map<napi_async_work, std::unique_ptr<napi_async_work__>> works;
    std::vector<napi_async_work> queued;
    std::vector<napi_callback_info> callbacks;
    std::set<std::string> calls;
    std::optional<napi_module> module;
    ValuePtr exports, pending;
    std::map<uint64_t, ValuePtr> vads;
    uint64_t next_vad = 1;
    std::array<char, 512> error_text{};
    napi_extended_error_info error{error_text.data(), nullptr, 0, napi_ok};
    size_t values = 0, edges = 0, finalizers = 0, predictions = 0;
    bool closing = false;
    void* library = nullptr;
    Worker worker;
    napi_env__() = default;
    ~napi_env__();
};

static thread_local napi_env active_env = nullptr;
static std::mutex library_mutex;
static bool library_attempted = false;
static void* pinned_library = nullptr; // The owned child, not a capsule, owns native singletons and image lifetime.
static constexpr size_t MAX_HANDLES = 8192, MAX_REFERENCES = 512, MAX_STRING = 4096, MAX_PROPERTIES = 256;
static constexpr size_t MAX_VAD_CONTEXTS = 128, VAD_WINDOW_SAMPLES = 512;
static constexpr const char* CAPSULE = "livekit_napi.Host3152";

Value::Value(napi_env owner, Kind tag) : env(owner), kind(tag) { ++env->values; }
Value::~Value() {
    for (auto finalizer : finalizers) {
        try { finalizer.callback(env, finalizer.data, finalizer.hint); ++env->finalizers; }
        catch (...) { std::abort(); } // A C finalizer may not unwind into its host.
    }
    env->edges -= properties.size();
    --env->values;
}

static bool owner(napi_env env) { return env != nullptr && env == active_env && env->owner == std::this_thread::get_id(); }
static napi_status status(napi_env env, napi_status code, const char* message = "") {
    env->error.error_code = code;
    std::snprintf(env->error_text.data(), env->error_text.size(), "%s", message);
    return code;
}
template<class F> static napi_status checked(napi_env env, const char* name, F&& action, bool retirement = false) noexcept {
    if (!owner(env)) return napi_invalid_arg;
    try {
        env->calls.insert(name);
        if (env->closing && !retirement) return status(env, napi_closing, "inference environment is closing");
        const auto result = action();
        if (result == napi_ok) return status(env, napi_ok);
        return status(env, result, "invalid or unsupported inference Node-API operation");
    } catch (const std::exception& error) { return status(env, napi_generic_failure, error.what()); }
    catch (...) { return status(env, napi_generic_failure, "native host allocation or callback failure"); }
}
#define CHECKED(env, ...) return checked(env, __func__, [&]() -> napi_status __VA_ARGS__)

static ValuePtr value(napi_env env, napi_value handle) {
    if (!handle || !env->handles.contains(handle)) throw std::runtime_error("invalid or expired Node-API handle");
    return handle->value;
}
static ValuePtr make(napi_env env, Kind kind) { if (env->values >= MAX_HANDLES) throw std::runtime_error("Node-API value capacity exceeded"); return std::make_shared<Value>(env, kind); }
static napi_value handle_in(napi_env env, ValuePtr item, Scope* scope) {
    if (env->handles.size() >= MAX_HANDLES) throw std::runtime_error("Node-API handle capacity exceeded");
    auto result = std::make_unique<napi_value__>(napi_value__{std::move(item)});
    auto raw = result.get(); scope->handles.push_back(std::move(result)); env->handles.insert(raw); return raw;
}
static napi_value handle(napi_env env, ValuePtr item) { return handle_in(env, std::move(item), env->scope); }
static ValuePtr undefined(napi_env env) { return make(env, Kind::Undefined); }
static std::string text(const char* source, size_t count) {
    if (!source) throw std::runtime_error("null string input");
    if (count == NAPI_AUTO_LENGTH) count = strnlen(source, MAX_STRING + 1);
    if (count > MAX_STRING) throw std::runtime_error("Node-API string capacity exceeded");
    return {source, count};
}
static ValuePtr string(napi_env env, std::string source) { auto item = make(env, Kind::String); item->data = std::move(source); return item; }
static std::string key(napi_env env, napi_value item) { auto v = value(env, item); if (v->kind != Kind::String) throw std::runtime_error("inference properties require string keys"); return std::get<std::string>(v->data); }
static bool object(const ValuePtr& v) { return v->kind == Kind::Object || v->kind == Kind::Function || v->kind == Kind::TypedArray || v->kind == Kind::ArrayBuffer || v->kind == Kind::Promise || v->kind == Kind::Error; }
static bool reaches(const ValuePtr& from, const ValuePtr& target, std::set<const Value*>& seen) {
    std::vector<std::pair<ValuePtr, size_t>> pending{{from, 0}};
    while (!pending.empty()) {
        auto [current, depth] = std::move(pending.back()); pending.pop_back();
        if (current == target) return true;
        if (depth > 128) throw std::runtime_error("inference object graph depth is unsupported");
        if (!seen.insert(current.get()).second) continue;
        for (const auto& entry : current->properties) if (entry.second.value) pending.emplace_back(entry.second.value, depth + 1);
        if (current->kind == Kind::Promise) { const auto& promise = std::get<Promise>(current->data); if (promise.result) pending.emplace_back(promise.result, depth + 1); }
        if (current->kind == Kind::TypedArray || current->kind == Kind::ArrayBuffer) { const auto& array = std::get<TypedArray>(current->data); if (array.buffer) pending.emplace_back(array.buffer, depth + 1); }
    }
    return false;
}
static void put(napi_env env, const ValuePtr& target, const std::string& name, Property property) {
    if (!object(target)) throw std::runtime_error("property owner is not an object");
    if (target->properties.size() >= MAX_PROPERTIES && !target->properties.contains(name)) throw std::runtime_error("Node-API property capacity exceeded");
    if (!target->properties.contains(name) && env->edges >= MAX_HANDLES) throw std::runtime_error("Node-API property graph capacity exceeded");
    std::set<const Value*> seen;
    if (property.value && reaches(property.value, target, seen)) throw std::runtime_error("cyclic objects are unsupported by the inference ABI host");
    auto [position, inserted] = target->properties.insert_or_assign(name, std::move(property));
    (void)position;
    if (inserted) ++env->edges;
}
static void release_scope(napi_env env, Scope* scope) {
    if (scope != env->scope || !scope->parent) throw std::runtime_error("Node-API scopes must close in stack order");
    env->scope = scope->parent;
    for (const auto& item : scope->handles) env->handles.erase(item.get());
    scope->handles.clear(); env->scopes.erase(scope);
}
struct ScopeOwner {
    napi_env env; Scope* scope;
    explicit ScopeOwner(napi_env owner_env) : env(owner_env) {
        if (env->scopes.size() >= 128) throw std::runtime_error("Node-API scope capacity exceeded");
        auto item = std::make_unique<Scope>(Scope{env->scope, false, false, {}});
        scope = item.get(); env->scopes.emplace(scope, std::move(item)); env->scope = scope;
    }
    ~ScopeOwner() { release_scope(env, scope); }
};
static ValuePtr invoke(napi_env env, Function function, const ValuePtr& receiver, std::vector<ValuePtr> arguments) {
    if (!function.callback) throw std::runtime_error("null native callback");
    napi_callback_info__ info{env, receiver, std::move(arguments), function.data};
    env->callbacks.push_back(&info);
    napi_value result = nullptr;
    try { result = function.callback(env, &info); }
    catch (const std::exception& error) { env->callbacks.pop_back(); throw std::runtime_error(error.what()); }
    catch (...) { env->callbacks.pop_back(); throw std::runtime_error("native callback threw a non-standard C++ exception"); }
    env->callbacks.pop_back();
    if (env->pending) return undefined(env);
    return result ? value(env, result) : undefined(env);
}
static ValuePtr property(napi_env env, const ValuePtr& target, const std::string& name) {
    if (!object(target)) throw std::runtime_error("property owner is not an object");
    auto found = target->properties.find(name);
    if (found == target->properties.end()) return undefined(env);
    if (found->second.getter) return invoke(env, {found->second.getter, found->second.data}, target, {});
    return found->second.value ? found->second.value : undefined(env);
}
static std::string error_message(napi_env env, const ValuePtr& error) {
    if (error->kind == Kind::String) return std::get<std::string>(error->data);
    auto message = property(env, error, "message");
    return message->kind == Kind::String ? std::get<std::string>(message->data) : "native inference rejected its promise";
}
static ValuePtr call_method(napi_env env, const ValuePtr& receiver, const char* name, std::vector<ValuePtr> arguments = {}) {
    auto fn = property(env, receiver, name);
    if (fn->kind != Kind::Function) throw std::runtime_error(std::string("missing native inference export: ") + name);
    auto result = invoke(env, std::get<Function>(fn->data), receiver, std::move(arguments));
    if (env->pending) { auto error = std::exchange(env->pending, nullptr); throw std::runtime_error(error_message(env, error)); }
    return result;
}
static ValuePtr call_export(napi_env env, const char* name, std::vector<ValuePtr> arguments = {}) {
    return call_method(env, env->exports, name, std::move(arguments));
}

extern "C" {
void napi_module_register(napi_module* module) {
    if (!owner(active_env) || !module) std::abort();
    try {
        active_env->calls.insert(__func__);
        if (module->nm_version != 1 || module->nm_flags != 0 || !module->nm_register_func || active_env->module) { status(active_env, napi_generic_failure, "unsupported or duplicate native module registration"); return; }
        active_env->module = *module;
    } catch (...) { status(active_env, napi_generic_failure, "native registration allocation failure"); }
}
void napi_fatal_error(const char* location, size_t location_len, const char* message, size_t message_len) {
    try { auto a = text(location, location_len); auto b = text(message, message_len); std::fprintf(stderr, "inference Node-API fatal: %s: %s\n", a.c_str(), b.c_str()); }
    catch (...) { std::fputs("inference Node-API fatal error\n", stderr); }
    std::abort();
}
napi_status napi_get_last_error_info(node_api_basic_env env, const napi_extended_error_info** result) {
    if (!owner(env) || !result) return napi_invalid_arg;
    try { env->calls.insert(__func__); *result = &env->error; return napi_ok; }
    catch (...) { return status(env, napi_generic_failure, "native error-info allocation failure"); }
}
napi_status napi_get_undefined(napi_env env, napi_value* result) { CHECKED(env, { if (!result) return napi_invalid_arg; *result = handle(env, undefined(env)); return napi_ok; }); }
napi_status napi_create_double(napi_env env, double number, napi_value* result) { CHECKED(env, { if (!result) return napi_invalid_arg; auto v = make(env, Kind::Number); v->data = number; *result = handle(env, v); return napi_ok; }); }
napi_status napi_create_object(napi_env env, napi_value* result) { CHECKED(env, { if (!result) return napi_invalid_arg; *result = handle(env, make(env, Kind::Object)); return napi_ok; }); }
napi_status napi_create_string_utf8(napi_env env, const char* source, size_t count, napi_value* result) { CHECKED(env, { if (!result) return napi_invalid_arg; *result = handle(env, string(env, text(source, count))); return napi_ok; }); }
napi_status napi_create_string_latin1(napi_env env, const char* source, size_t count, napi_value* result) { CHECKED(env, { if (!result) return napi_invalid_arg; auto latin = text(source, count); std::string utf8; for (char raw : latin) { const auto c = static_cast<unsigned char>(raw); if (c < 128) utf8 += static_cast<char>(c); else { utf8 += static_cast<char>(0xc0 | (c >> 6)); utf8 += static_cast<char>(0x80 | (c & 63)); } } *result = handle(env, string(env, utf8)); return napi_ok; }); }
napi_status napi_create_function(napi_env env, const char* name, size_t count, napi_callback callback, void* data, napi_value* result) { CHECKED(env, { if (!callback || !result) return napi_invalid_arg; if (name) (void)text(name, count); auto v = make(env, Kind::Function); v->data = Function{callback, data}; *result = handle(env, v); return napi_ok; }); }
napi_status napi_get_cb_info(napi_env env, napi_callback_info info, size_t* argc, napi_value* argv, napi_value* receiver, void** data) { CHECKED(env, { if (!info || std::find(env->callbacks.begin(), env->callbacks.end(), info) == env->callbacks.end() || (argv && (!argc || *argc > 16))) return napi_invalid_arg; if (argc) { if (argv) for (size_t i = 0; i < *argc; ++i) argv[i] = handle(env, i < info->arguments.size() ? info->arguments[i] : undefined(env)); *argc = info->arguments.size(); } if (receiver) *receiver = handle(env, info->receiver); if (data) *data = info->data; return napi_ok; }); }
napi_status napi_call_function(napi_env env, napi_value receiver, napi_value function, size_t argc, const napi_value* argv, napi_value* result) { CHECKED(env, { if (argc > 16 || (argc && !argv)) return napi_invalid_arg; auto fn = value(env, function); if (fn->kind != Kind::Function) return napi_function_expected; std::vector<ValuePtr> args; for (size_t i = 0; i < argc; ++i) args.push_back(value(env, argv[i])); auto returned = invoke(env, std::get<Function>(fn->data), value(env, receiver), std::move(args)); if (env->pending) return napi_pending_exception; if (result) *result = handle(env, returned); return napi_ok; }); }
napi_status napi_set_named_property(napi_env env, napi_value target, const char* name, napi_value entry) { CHECKED(env, { auto object_value = value(env, target); auto property_name = text(name, NAPI_AUTO_LENGTH); auto v = value(env, entry); auto found = object_value->properties.find(property_name); if (found != object_value->properties.end()) { auto& p = found->second; if (p.setter) { (void)invoke(env, {p.setter, p.data}, object_value, {v}); return env->pending ? napi_pending_exception : napi_ok; } if (!(p.attributes & napi_writable)) return napi_generic_failure; } put(env, object_value, property_name, Property{v}); return napi_ok; }); }
napi_status napi_get_named_property(napi_env env, napi_value target, const char* name, napi_value* result) { CHECKED(env, { if (!result) return napi_invalid_arg; *result = handle(env, property(env, value(env, target), text(name, NAPI_AUTO_LENGTH))); return env->pending ? napi_pending_exception : napi_ok; }); }
napi_status napi_get_property(napi_env env, napi_value target, napi_value name, napi_value* result) { CHECKED(env, { if (!result) return napi_invalid_arg; *result = handle(env, property(env, value(env, target), key(env, name))); return env->pending ? napi_pending_exception : napi_ok; }); }
napi_status napi_has_property(napi_env env, napi_value target, napi_value name, bool* result) { CHECKED(env, { if (!result) return napi_invalid_arg; auto v = value(env, target); if (!object(v)) return napi_object_expected; *result = v->properties.contains(key(env, name)); return napi_ok; }); }
napi_status napi_define_properties(napi_env env, napi_value target, size_t count, const napi_property_descriptor* properties) { CHECKED(env, { if (count > MAX_PROPERTIES || (count && !properties)) return napi_invalid_arg; auto v = value(env, target); for (size_t i = 0; i < count; ++i) { const auto& d = properties[i]; if (d.attributes & ~(napi_writable | napi_enumerable | napi_configurable | napi_static)) return napi_invalid_arg; auto name = d.utf8name ? text(d.utf8name, NAPI_AUTO_LENGTH) : key(env, d.name); Property p{nullptr, d.getter, d.setter, d.data, d.attributes}; if (d.method) { p.value = make(env, Kind::Function); p.value->data = Function{d.method, d.data}; } else if (d.value) p.value = value(env, d.value); put(env, v, name, std::move(p)); } return napi_ok; }); }
napi_status napi_typeof(napi_env env, napi_value entry, napi_valuetype* result) { CHECKED(env, { if (!result) return napi_invalid_arg; switch (value(env, entry)->kind) { case Kind::Undefined: *result = napi_undefined; break; case Kind::Number: *result = napi_number; break; case Kind::String: *result = napi_string; break; case Kind::Function: *result = napi_function; break; case Kind::External: *result = napi_external; break; default: *result = napi_object; } return napi_ok; }); }
napi_status napi_is_typedarray(napi_env env, napi_value entry, bool* result) { CHECKED(env, { if (!result) return napi_invalid_arg; *result = value(env, entry)->kind == Kind::TypedArray; return napi_ok; }); }
napi_status napi_get_typedarray_info(napi_env env, napi_value entry, napi_typedarray_type* type, size_t* length, void** data, napi_value* arraybuffer, size_t* offset) { CHECKED(env, { auto v = value(env, entry); if (v->kind != Kind::TypedArray) return napi_invalid_arg; auto& typed = std::get<TypedArray>(v->data); auto& bytes = *typed.data; if (type) *type = napi_int16_array; if (length) *length = bytes.size(); if (data) *data = bytes.data(); if (arraybuffer) { if (!typed.buffer) { typed.buffer = make(env, Kind::ArrayBuffer); typed.buffer->data = TypedArray{typed.data, nullptr}; } *arraybuffer = handle(env, typed.buffer); } if (offset) *offset = 0; return napi_ok; }); }
napi_status napi_create_external(napi_env env, void* data, node_api_basic_finalize finalize, void* hint, napi_value* result) { CHECKED(env, { if (!result) return napi_invalid_arg; auto v = make(env, Kind::External); v->data = External{data}; if (finalize) v->finalizers.push_back({finalize, data, hint}); *result = handle(env, v); return napi_ok; }); }
napi_status napi_get_value_external(napi_env env, napi_value entry, void** result) { CHECKED(env, { if (!result) return napi_invalid_arg; auto v = value(env, entry); if (v->kind != Kind::External) return napi_invalid_arg; *result = std::get<External>(v->data).data; return napi_ok; }); }
napi_status napi_create_reference(napi_env env, napi_value entry, uint32_t count, napi_ref* result) { CHECKED(env, { if (!result || env->references.size() >= MAX_REFERENCES) return napi_invalid_arg; auto v = value(env, entry); if (!object(v) && v->kind != Kind::External) return napi_object_expected; auto ref = std::make_unique<napi_ref__>(napi_ref__{v, count ? v : nullptr, count}); *result = ref.get(); env->references.emplace(*result, std::move(ref)); return napi_ok; }); }
napi_status napi_delete_reference(napi_env env, napi_ref ref) { return checked(env, __func__, [&]() -> napi_status { auto found = env->references.find(ref); if (found == env->references.end()) return napi_invalid_arg; auto owned = std::move(found->second); env->references.erase(found); return napi_ok; }, true); }
napi_status napi_get_reference_value(napi_env env, napi_ref ref, napi_value* result) { CHECKED(env, { if (!result || !env->references.contains(ref)) return napi_invalid_arg; auto v = ref->weak.lock(); *result = v ? handle(env, v) : nullptr; return napi_ok; }); }
napi_status napi_add_finalizer(napi_env env, napi_value entry, void* data, node_api_basic_finalize finalize, void* hint, napi_ref* result) { CHECKED(env, { if (!finalize) return napi_invalid_arg; auto v = value(env, entry); if (!object(v)) return napi_object_expected; if (result) { auto s = napi_create_reference(env, entry, 0, result); if (s != napi_ok) return s; } v->finalizers.push_back({finalize, data, hint}); return napi_ok; }); }
napi_status napi_open_handle_scope(napi_env env, napi_handle_scope* result) { CHECKED(env, { if (!result || env->scopes.size() >= 128) return napi_invalid_arg; auto s = std::make_unique<Scope>(Scope{env->scope, false, false, {}}); auto raw = s.get(); env->scopes.emplace(raw, std::move(s)); env->scope = raw; *result = reinterpret_cast<napi_handle_scope>(raw); return napi_ok; }); }
napi_status napi_close_handle_scope(napi_env env, napi_handle_scope scope) { CHECKED(env, { auto raw = reinterpret_cast<Scope*>(scope); if (!env->scopes.contains(raw) || raw != env->scope || raw->escapable) return napi_handle_scope_mismatch; release_scope(env, raw); return napi_ok; }); }
napi_status napi_open_escapable_handle_scope(napi_env env, napi_escapable_handle_scope* result) { CHECKED(env, { if (!result || env->scopes.size() >= 128) return napi_invalid_arg; auto s = std::make_unique<Scope>(Scope{env->scope, true, false, {}}); auto raw = s.get(); env->scopes.emplace(raw, std::move(s)); env->scope = raw; *result = reinterpret_cast<napi_escapable_handle_scope>(raw); return napi_ok; }); }
napi_status napi_close_escapable_handle_scope(napi_env env, napi_escapable_handle_scope scope) { CHECKED(env, { auto raw = reinterpret_cast<Scope*>(scope); if (!env->scopes.contains(raw) || raw != env->scope || !raw->escapable) return napi_handle_scope_mismatch; release_scope(env, raw); return napi_ok; }); }
napi_status napi_escape_handle(napi_env env, napi_escapable_handle_scope scope, napi_value entry, napi_value* result) { CHECKED(env, { auto raw = reinterpret_cast<Scope*>(scope); if (!result || !env->scopes.contains(raw) || !raw->escapable || raw != env->scope) return napi_invalid_arg; if (raw->escaped) return napi_escape_called_twice; *result = handle_in(env, value(env, entry), raw->parent); raw->escaped = true; return napi_ok; }); }
static napi_status create_error(napi_env env, napi_value code, napi_value message, napi_value* result, const char* name) { if (!result) return napi_invalid_arg; auto msg = value(env, message); if (msg->kind != Kind::String) return napi_string_expected; auto v = make(env, Kind::Error); put(env, v, "message", Property{msg}); put(env, v, "name", Property{string(env, name)}); if (code) { auto c = value(env, code); if (c->kind != Kind::String) return napi_string_expected; put(env, v, "code", Property{c}); } *result = handle(env, v); return napi_ok; }
napi_status napi_create_error(napi_env env, napi_value code, napi_value message, napi_value* result) { CHECKED(env, { return create_error(env, code, message, result, "Error"); }); }
napi_status napi_create_type_error(napi_env env, napi_value code, napi_value message, napi_value* result) { CHECKED(env, { return create_error(env, code, message, result, "TypeError"); }); }
napi_status napi_create_range_error(napi_env env, napi_value code, napi_value message, napi_value* result) { CHECKED(env, { return create_error(env, code, message, result, "RangeError"); }); }
napi_status napi_throw(napi_env env, napi_value error) { CHECKED(env, { env->pending = value(env, error); return napi_ok; }); }
napi_status napi_is_exception_pending(napi_env env, bool* result) { CHECKED(env, { if (!result) return napi_invalid_arg; *result = env->pending != nullptr; return napi_ok; }); }
napi_status napi_get_and_clear_last_exception(napi_env env, napi_value* result) { CHECKED(env, { if (!result) return napi_invalid_arg; *result = handle(env, env->pending ? std::exchange(env->pending, nullptr) : undefined(env)); return napi_ok; }); }
napi_status napi_create_promise(napi_env env, napi_deferred* deferred, napi_value* promise) { CHECKED(env, { if (!deferred || !promise || env->deferred.size() >= 2) return napi_invalid_arg; auto v = make(env, Kind::Promise); v->data = Promise{}; auto d = std::make_unique<napi_deferred__>(napi_deferred__{v}); *deferred = d.get(); env->deferred.emplace(*deferred, std::move(d)); *promise = handle(env, v); return napi_ok; }); }
static napi_status settle(napi_env env, napi_deferred deferred, napi_value entry, Promise::State state) { auto found = env->deferred.find(deferred); if (found == env->deferred.end()) return napi_invalid_arg; auto& promise = std::get<Promise>(found->second->promise->data); if (promise.state != Promise::State::Pending) return napi_invalid_arg; auto result = value(env, entry); std::set<const Value*> seen; if (result->kind == Kind::Promise || reaches(result, found->second->promise, seen)) return napi_generic_failure; promise.result = std::move(result); promise.state = state; env->deferred.erase(found); return napi_ok; }
napi_status napi_resolve_deferred(napi_env env, napi_deferred deferred, napi_value result) { CHECKED(env, { return settle(env, deferred, result, Promise::State::Resolved); }); }
napi_status napi_reject_deferred(napi_env env, napi_deferred deferred, napi_value result) { CHECKED(env, { return settle(env, deferred, result, Promise::State::Rejected); }); }
napi_status napi_create_async_work(napi_env env, napi_value resource, napi_value name, napi_async_execute_callback execute, napi_async_complete_callback complete, void* data, napi_async_work* result) { CHECKED(env, { if (!execute || !complete || !result || env->works.size() >= 2) return napi_invalid_arg; if (resource && !object(value(env, resource))) return napi_object_expected; if (value(env, name)->kind != Kind::String) return napi_string_expected; auto w = std::make_unique<napi_async_work__>(napi_async_work__{execute, complete, data}); *result = w.get(); env->works.emplace(*result, std::move(w)); return napi_ok; }); }
napi_status napi_queue_async_work(node_api_basic_env env, napi_async_work work) { CHECKED(env, { if (!env->works.contains(work) || work->state != napi_async_work__::State::Created) return napi_invalid_arg; if (!env->queued.empty()) return napi_queue_full; work->state = napi_async_work__::State::Queued; env->queued.push_back(work); return napi_ok; }); }
napi_status napi_delete_async_work(napi_env env, napi_async_work work) { CHECKED(env, { if (!env->works.contains(work) || (work->state != napi_async_work__::State::Complete && work->state != napi_async_work__::State::Created)) return napi_invalid_arg; env->works.erase(work); return napi_ok; }); }
}

static void drain(napi_env env, const ValuePtr& returned) {
    if (returned->kind != Kind::Promise) throw std::runtime_error("native predict did not return a promise");
    auto& promise = std::get<Promise>(returned->data);
    while (promise.state == Promise::State::Pending) {
        if (env->queued.size() != 1) throw std::runtime_error("native predict left an unsettled promise without one bounded work item");
        auto work = env->queued.back(); env->queued.clear(); work->state = napi_async_work__::State::Running;
        std::exception_ptr execution_failure;
        try { env->worker.run(env, work); }
        catch (...) { execution_failure = std::current_exception(); }
        work->state = napi_async_work__::State::Complete;
        auto callback = work->complete; auto data = work->data;
        ScopeOwner completion(env);
        try { callback(env, execution_failure ? napi_generic_failure : napi_ok, data); }
        catch (const std::exception& error) { throw std::runtime_error(error.what()); }
        catch (...) { throw std::runtime_error("native completion threw a non-standard C++ exception"); }
        if (execution_failure) std::rethrow_exception(execution_failure);
        if (env->pending) { auto error = std::exchange(env->pending, nullptr); throw std::runtime_error(error_message(env, error)); }
    }
    if (promise.state == Promise::State::Rejected) throw std::runtime_error(error_message(env, promise.result));
}

napi_env__::~napi_env__() {
    worker.stop(); // Drop transported exception objects while their image remains mapped.
    closing = true;
    pending.reset(); vads.clear(); exports.reset(); deferred.clear();
    for (const auto& entry : root.handles) handles.erase(entry.get());
    root.handles.clear(); scopes.clear();
    // Keep weak reference records valid while native finalizers delete them.
    // Move each strong root out before release: its finalizer may erase any ref.
    for (;;) {
        auto strong = std::find_if(references.begin(), references.end(), [](const auto& entry) { return entry.second->strong != nullptr; });
        if (strong == references.end()) break;
        auto retained = std::move(strong->second->strong);
        retained.reset();
    }
    references.clear(); works.clear();
    if (values != 0) std::abort(); // Remaining values would outlive their owning environment.
    // No public vendor threadpool shutdown exists. Keep its image mapped until
    // the owned CPython3152 child exits; the parent must reap that process.
    if (active_env == this) active_env = nullptr;
}

static napi_env capsule(PyObject* object_value) {
    auto env = static_cast<napi_env>(PyCapsule_GetPointer(object_value, CAPSULE));
    if (env && !owner(env)) { PyErr_SetString(PyExc_RuntimeError, "inference capsule belongs to its creating child thread"); return nullptr; }
    return env;
}
static void destroy_capsule(PyObject* object_value) {
    if (!PyCapsule_IsValid(object_value, CAPSULE)) return;
    auto env = static_cast<napi_env>(PyCapsule_GetPointer(object_value, CAPSULE));
    if (env && owner(env)) delete env;
    else { PyErr_SetString(PyExc_RuntimeError, "inference capsule finalized outside its owning child thread"); PyErr_WriteUnraisable(object_value); }
}
static PyObject* py_open(PyObject*, PyObject* path) {
    if (active_env) { PyErr_SetString(PyExc_RuntimeError, "one inference environment per child thread is supported"); return nullptr; }
    Py_ssize_t count = 0; const char* filename = PyUnicode_AsUTF8AndSize(path, &count);
    if (!filename) return nullptr;
    if (count <= 0 || count > 4096 || std::memchr(filename, 0, static_cast<size_t>(count))) { PyErr_SetString(PyExc_ValueError, "invalid native artifact path"); return nullptr; }
    try {
        auto env = std::make_unique<napi_env__>(); active_env = env.get();
        { std::lock_guard lock(library_mutex);
          if (library_attempted) throw std::runtime_error("native artifact is pinned to one host lifetime; start a new 3.15.2 child");
          library_attempted = true; pinned_library = dlopen(filename, RTLD_NOW | RTLD_LOCAL); env->library = pinned_library;
        }
        if (!env->library) throw std::runtime_error(dlerror());
        if (!env->module) throw std::runtime_error("artifact did not register the admitted inference N-API ABI");
        auto exports = handle(env.get(), make(env.get(), Kind::Object));
        napi_value returned = nullptr;
        try { returned = env->module->nm_register_func(env.get(), exports); }
        catch (const std::exception& error) { throw std::runtime_error(error.what()); }
        catch (...) { throw std::runtime_error("native initialization threw a non-standard C++ exception"); }
        if (env->pending) throw std::runtime_error(error_message(env.get(), env->pending));
        env->exports = value(env.get(), returned ? returned : exports);
        auto result = PyCapsule_New(env.get(), CAPSULE, destroy_capsule);
        if (!result) return nullptr;
        (void)env.release(); return result;
    } catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); return nullptr; }
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "non-standard native host exception"); return nullptr; }
}
static PyObject* initialize_model(PyObject* object_value, const char* name) {
    auto env = capsule(object_value); if (!env) return nullptr;
    try {
        if (!env->queued.empty()) throw std::runtime_error("inference work remains unsettled; retire the owning child");
        ScopeOwner scope(env);
        if (call_export(env, name)->kind != Kind::Undefined) throw std::runtime_error("native prewarm returned unexpected data");
        if (std::strcmp(name, "initVad") == 0) {
            auto window = property(env, env->exports, "VAD_WINDOW_SAMPLES");
            if (window->kind != Kind::Number || std::get<double>(window->data) != static_cast<double>(VAD_WINDOW_SAMPLES)) throw std::runtime_error("native VAD window contract changed");
        }
        Py_RETURN_NONE;
    } catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); return nullptr; }
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "non-standard native host exception"); return nullptr; }
}
static PyObject* py_init_eot(PyObject*, PyObject* object_value) { return initialize_model(object_value, "initEot"); }
static PyObject* py_init_vad(PyObject*, PyObject* object_value) { return initialize_model(object_value, "initVad"); }
static bool context_token(PyObject* object_value, uint64_t& result) {
    if (!PyLong_CheckExact(object_value)) { PyErr_SetString(PyExc_TypeError, "VAD context token must be an exact integer"); return false; }
    const auto token = PyLong_AsUnsignedLongLong(object_value);
    if (PyErr_Occurred()) return false;
    result = static_cast<uint64_t>(token); return true;
}
static ValuePtr vad_context(napi_env env, uint64_t token) {
    const auto found = env->vads.find(token);
    if (found == env->vads.end()) throw std::out_of_range("unknown or retired VAD context");
    return found->second;
}
static PyObject* py_create_vad(PyObject*, PyObject* object_value) {
    auto env = capsule(object_value); if (!env) return nullptr;
    try {
        if (!env->queued.empty()) throw std::runtime_error("inference work remains unsettled; retire the owning child");
        if (env->vads.size() >= MAX_VAD_CONTEXTS || env->next_vad == UINT64_MAX) throw std::runtime_error("native VAD context capacity exceeded");
        ScopeOwner scope(env); auto context = call_export(env, "createVad");
        if (context->kind != Kind::Object || property(env, context, "predict")->kind != Kind::Function || property(env, context, "reset")->kind != Kind::Function) throw std::runtime_error("native VAD context contract changed");
        const auto token = env->next_vad++; env->vads.emplace(token, std::move(context));
        auto result = PyLong_FromUnsignedLongLong(token);
        if (!result) env->vads.erase(token);
        return result;
    } catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); return nullptr; }
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "non-standard native host exception"); return nullptr; }
}
static PyObject* predict(PyObject* args, bool vad) {
    PyObject *object_value = nullptr, *token_value = nullptr, *pcm = nullptr;
    if (vad) { if (!PyArg_ParseTuple(args, "OOO:predict_vad", &object_value, &token_value, &pcm)) return nullptr; }
    else if (!PyArg_ParseTuple(args, "OO:predict_eot", &object_value, &pcm)) return nullptr;
    auto env = capsule(object_value); if (!env) return nullptr;
    uint64_t token = 0; if (vad && !context_token(token_value, token)) return nullptr;
    Py_buffer buffer{};
    if (PyObject_GetBuffer(pcm, &buffer, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) return nullptr;
    if (buffer.ndim != 1 || buffer.itemsize != 2 || !buffer.format || std::strcmp(buffer.format, "h") || !PyBuffer_IsContiguous(&buffer, 'C') || buffer.suboffsets) { PyBuffer_Release(&buffer); PyErr_SetString(PyExc_TypeError, "inference requires contiguous one-dimensional native int16 PCM"); return nullptr; }
    if (buffer.len < 0 || buffer.len % 2 || (vad ? buffer.len != static_cast<Py_ssize_t>(VAD_WINDOW_SAMPLES * 2) : buffer.len > 38400)) { PyBuffer_Release(&buffer); PyErr_SetString(PyExc_ValueError, vad ? "VAD requires exactly512int16 samples" : "EOT accepts at most19200int16 samples"); return nullptr; }
    double probability = 0; std::exception_ptr failure;
    try {
        if (!env->queued.empty()) throw std::runtime_error("inference work remains unsettled; retire the owning child");
        auto receiver = vad ? vad_context(env, token) : env->exports;
        auto typed = make(env, Kind::TypedArray); TypedArray input{std::make_shared<std::vector<int16_t>>(static_cast<size_t>(buffer.len) / 2), nullptr};
        if (buffer.len) std::memcpy(input.data->data(), buffer.buf, static_cast<size_t>(buffer.len));
        typed->data = std::move(input); PyBuffer_Release(&buffer);
        ValuePtr returned;
        { ScopeOwner call(env); returned = call_method(env, receiver, "predict", {typed}); }
        typed.reset(); // Only the native addon's own references may retain input now.
        Py_BEGIN_ALLOW_THREADS
        try {
            drain(env, returned); auto result = std::get<Promise>(returned->data).result;
            if (!result || result->kind != Kind::Number) throw std::runtime_error("native inference returned a non-number probability");
            probability = std::get<double>(result->data);
            if (!std::isfinite(probability) || probability < 0 || probability > 1) throw std::runtime_error("native inference returned an invalid probability");
            ++env->predictions;
        } catch (...) { failure = std::current_exception(); }
        Py_END_ALLOW_THREADS
        if (failure) std::rethrow_exception(failure);
        return PyFloat_FromDouble(probability);
    } catch (const std::out_of_range& error) { if (buffer.obj) PyBuffer_Release(&buffer); PyErr_SetString(PyExc_ValueError, error.what()); return nullptr; }
    catch (const std::exception& error) { if (buffer.obj) PyBuffer_Release(&buffer); PyErr_SetString(PyExc_RuntimeError, error.what()); return nullptr; }
    catch (...) { if (buffer.obj) PyBuffer_Release(&buffer); PyErr_SetString(PyExc_RuntimeError, "non-standard native host exception"); return nullptr; }
}
static PyObject* py_predict_eot(PyObject*, PyObject* args) { return predict(args, false); }
static PyObject* py_predict_vad(PyObject*, PyObject* args) { return predict(args, true); }
static PyObject* change_vad(PyObject* args, bool drop) {
    PyObject *object_value = nullptr, *token_value = nullptr;
    if (!PyArg_ParseTuple(args, "OO:change_vad", &object_value, &token_value)) return nullptr;
    auto env = capsule(object_value); if (!env) return nullptr;
    uint64_t token = 0; if (!context_token(token_value, token)) return nullptr;
    try {
        if (!env->queued.empty()) throw std::runtime_error("inference work remains unsettled; retire the owning child");
        auto context = vad_context(env, token);
        if (drop) env->vads.erase(token);
        else { ScopeOwner scope(env); if (call_method(env, context, "reset")->kind != Kind::Undefined) throw std::runtime_error("native VAD reset returned unexpected data"); }
        Py_RETURN_NONE;
    } catch (const std::out_of_range& error) { PyErr_SetString(PyExc_ValueError, error.what()); return nullptr; }
    catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); return nullptr; }
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "non-standard native host exception"); return nullptr; }
}
static PyObject* py_reset_vad(PyObject*, PyObject* args) { return change_vad(args, false); }
static PyObject* py_drop_vad(PyObject*, PyObject* args) { return change_vad(args, true); }
static PyObject* py_stats(PyObject*, PyObject* object_value) {
    auto env = capsule(object_value); if (!env) return nullptr;
    auto calls = PyTuple_New(static_cast<Py_ssize_t>(env->calls.size())); if (!calls) return nullptr;
    Py_ssize_t i = 0;
    for (const auto& call : env->calls) { auto item = PyUnicode_FromString(call.c_str()); if (!item) { Py_DECREF(calls); return nullptr; } PyTuple_SET_ITEM(calls, i++, item); }
    return Py_BuildValue("{s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:N}", "vad_contexts", static_cast<Py_ssize_t>(env->vads.size()), "values", static_cast<Py_ssize_t>(env->values), "handles", static_cast<Py_ssize_t>(env->handles.size()), "references", static_cast<Py_ssize_t>(env->references.size()), "async_work", static_cast<Py_ssize_t>(env->works.size()), "finalizers", static_cast<Py_ssize_t>(env->finalizers), "predictions", static_cast<Py_ssize_t>(env->predictions), "calls", calls);
}
static PyObject* py_close(PyObject*, PyObject* object_value) {
    auto env = capsule(object_value); if (!env) return nullptr;
    if (!env->queued.empty()) { PyErr_SetString(PyExc_RuntimeError, "inference work remains unsettled; retire the owning child"); return nullptr; }
    if (PyCapsule_SetName(object_value, "livekit_napi.Closed3152") < 0) return nullptr;
    delete env; Py_RETURN_NONE;
}
static PyObject* py_start_parent_watch(PyObject*, PyObject* descriptor) {
    if (!PyLong_CheckExact(descriptor)) { PyErr_SetString(PyExc_TypeError, "parent lifeline descriptor must be an exact integer"); return nullptr; }
    const auto fd = PyLong_AsLong(descriptor); if (PyErr_Occurred()) return nullptr;
    if (fd < 0 || fd > INT_MAX) { PyErr_SetString(PyExc_ValueError, "invalid parent lifeline descriptor"); return nullptr; }
    try {
        std::lock_guard lock(library_mutex);
        if (parent_watch) throw std::runtime_error("parent lifeline is already owned by this child");
        if (library_attempted) throw std::runtime_error("parent lifeline must start before loading the native artifact");
        parent_watch = std::make_unique<ParentWatch>(static_cast<int>(fd));
        Py_RETURN_NONE;
    } catch (const std::invalid_argument& error) { PyErr_SetString(PyExc_ValueError, error.what()); return nullptr; }
    catch (const std::system_error& error) { PyErr_SetString(PyExc_OSError, error.what()); return nullptr; }
    catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); return nullptr; }
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "non-standard parent-watch failure"); return nullptr; }
}
static PyObject* py_stop_parent_watch(PyObject*, PyObject*) {
    if (parent_watch && !parent_watch->on_owner()) { PyErr_SetString(PyExc_RuntimeError, "parent lifeline belongs to its creating child thread"); return nullptr; }
    if (active_env) { PyErr_SetString(PyExc_RuntimeError, "retire the native environment before stopping its parent lifeline"); return nullptr; }
    parent_watch.reset(); Py_RETURN_NONE;
}
static PyMethodDef methods[] = {
    {"start_parent_watch", py_start_parent_watch, METH_O, "Own a dedicated parent pipe before loading any native model."},
    {"stop_parent_watch", py_stop_parent_watch, METH_NOARGS, "Join the parent watcher after the native environment has retired."},
    {"open", py_open, METH_O, "Load an admitted N-API artifact inside this owned3152 child."},
    {"init_eot", py_init_eot, METH_O, "Initialize the genuine addon EOT singleton."},
    {"predict_eot", py_predict_eot, METH_VARARGS, "Predict from at most19200contiguous int16 samples."},
    {"init_vad", py_init_vad, METH_O, "Initialize the genuine addon VAD singleton."},
    {"create_vad", py_create_vad, METH_O, "Own one independent VAD stream context until drop or child retirement."},
    {"predict_vad", py_predict_vad, METH_VARARGS, "Advance one VAD context from exactly512contiguous int16 samples."},
    {"reset_vad", py_reset_vad, METH_VARARGS, "Reset the selected VAD stream's native recurrent state."},
    {"drop_vad", py_drop_vad, METH_VARARGS, "Retire the selected settled VAD stream context."},
    {"stats", py_stats, METH_O, "Inspect private lifetime proof counters."},
    {"close", py_close, METH_O, "Release a settled environment on its owning child thread."},
    {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_napi_host315", "Source-owned host for the inference Node-API contract.", -1, methods, nullptr, nullptr, nullptr, nullptr};
PyMODINIT_FUNC PyInit__napi_host315() {
    if (std::strncmp(Py_GetVersion(), "3.15.2 ", 7)) { PyErr_SetString(PyExc_ImportError, "this qualification requires CPython3.15.2"); return nullptr; }
    auto result = PyModule_Create(&module); if (!result) return nullptr;
    if (PyModule_AddIntConstant(result, "VAD_WINDOW_SAMPLES", static_cast<long>(VAD_WINDOW_SAMPLES)) < 0 || PyModule_AddIntConstant(result, "MAX_VAD_CONTEXTS", static_cast<long>(MAX_VAD_CONTEXTS)) < 0 || PyModule_AddIntConstant(result, "PARENT_LOST_EXIT_CODE", PARENT_LOST_EXIT_CODE) < 0) { Py_DECREF(result); return nullptr; }
    return result;
}
