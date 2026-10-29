// Source-owned working fake at the exact public N-API seam; no vendor engine.
#include <node_api.h>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>

static void require(bool condition, const char* location) {
    if (!condition) { std::fprintf(stderr, "fake N-API conformance failed: %s\n", location); std::abort(); }
}
#define OK(call) require((call) == napi_ok, #call)
static void hang_at(const char* location) {
    std::fprintf(stdout, "NATIVE_HANG %s\n", location); std::fflush(stdout);
    for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
}
static bool mode_is(const char* expected) {
    const auto actual = std::getenv("LIVEKIT_NAPI_FAKE_MODE");
    return actual && std::string(actual) == expected;
}
static std::atomic<size_t> finalized = 0;
static std::atomic<size_t> custom_destroyed = 0;
struct CustomFailure : std::exception {
    const char* what() const noexcept override { return "source-owned fake custom worker exception"; }
    ~CustomFailure() override { ++custom_destroyed; }
};
static bool initialized = false;
static napi_value nothing(napi_env env) { napi_value value = nullptr; OK(napi_get_undefined(env, &value)); return value; }
static napi_value echo(napi_env env, napi_callback_info info) {
    size_t count = 1; napi_value value = nullptr;
    OK(napi_get_cb_info(env, info, &count, &value, nullptr, nullptr)); return value;
}
static void external_finalizer(napi_env, void* data, void*) { delete static_cast<int*>(data); ++finalized; }
struct FinalizerReference { napi_ref reference = nullptr; };
static void reference_finalizer(napi_env env, void* data, void*) {
    auto owned = static_cast<FinalizerReference*>(data);
    OK(napi_delete_reference(env, owned->reference));
    delete owned; ++finalized;
}

static void conformance(napi_env env) {
    napi_handle_scope outer = nullptr; OK(napi_open_handle_scope(env, &outer));
    napi_value message = nullptr, latin = nullptr, error = nullptr, object = nullptr, fn = nullptr, returned = nullptr;
    OK(napi_create_string_utf8(env, "source-owned fake failure", NAPI_AUTO_LENGTH, &message));
    const char e_acute[] = {static_cast<char>(0xe9), 0};
    OK(napi_create_string_latin1(env, e_acute, 1, &latin));
    OK(napi_create_error(env, nullptr, message, &error));
    OK(napi_create_type_error(env, nullptr, latin, &error));
    OK(napi_create_range_error(env, nullptr, message, &error));
    OK(napi_throw(env, error)); bool pending = false; OK(napi_is_exception_pending(env, &pending)); require(pending, "pending exception");
    OK(napi_get_and_clear_last_exception(env, &error)); OK(napi_is_exception_pending(env, &pending)); require(!pending, "cleared exception");
    require(napi_typeof(env, message, nullptr) == napi_invalid_arg, "invalid argument status");
    const napi_extended_error_info* info = nullptr; OK(napi_get_last_error_info(env, &info)); require(info && info->error_code == napi_invalid_arg, "retained error info");
    OK(napi_create_object(env, &object)); OK(napi_create_function(env, "echo", NAPI_AUTO_LENGTH, echo, nullptr, &fn));
    OK(napi_set_named_property(env, object, "echo", fn)); OK(napi_get_named_property(env, object, "echo", &returned));
    OK(napi_call_function(env, object, returned, 1, &message, &returned));
    napi_valuetype type = napi_undefined; OK(napi_typeof(env, returned, &type)); require(type == napi_string, "function invocation");
    napi_value name = nullptr; OK(napi_create_string_utf8(env, "echo", NAPI_AUTO_LENGTH, &name));
    bool found = false; OK(napi_has_property(env, object, name, &found)); require(found, "property presence"); OK(napi_get_property(env, object, name, &returned));
    napi_property_descriptor descriptor{"method", nullptr, echo, nullptr, nullptr, nullptr, napi_default_method, nullptr};
    OK(napi_define_properties(env, object, 1, &descriptor));
    napi_value external = nullptr; auto external_data = new int(7); OK(napi_create_external(env, external_data, external_finalizer, nullptr, &external));
    void* observed = nullptr; OK(napi_get_value_external(env, external, &observed)); require(observed == external_data, "external pointer preservation");
    napi_ref reference = nullptr; OK(napi_create_reference(env, external, 1, &reference)); OK(napi_get_reference_value(env, reference, &returned)); OK(napi_delete_reference(env, reference));
    napi_escapable_handle_scope inner = nullptr; OK(napi_open_escapable_handle_scope(env, &inner));
    napi_value number = nullptr, escaped = nullptr; OK(napi_create_double(env, 0.125, &number)); OK(napi_escape_handle(env, inner, number, &escaped));
    require(napi_escape_handle(env, inner, number, &escaped) == napi_escape_called_twice, "single escape");
    OK(napi_close_escapable_handle_scope(env, inner)); OK(napi_typeof(env, escaped, &type)); require(type == napi_number, "escaped handle remains live");
    napi_deferred deferred = nullptr; napi_value promise = nullptr; OK(napi_create_promise(env, &deferred, &promise)); OK(napi_reject_deferred(env, deferred, error));
    require(napi_reject_deferred(env, deferred, error) == napi_invalid_arg, "double settlement refused");
    OK(napi_create_promise(env, &deferred, &promise));
    napi_value cycle = nullptr; OK(napi_create_object(env, &cycle)); OK(napi_set_named_property(env, cycle, "promise", promise));
    require(napi_resolve_deferred(env, deferred, cycle) == napi_generic_failure, "promise-result cycle refused");
    OK(napi_reject_deferred(env, deferred, error));
    require(napi_set_named_property(env, cycle, "self", cycle) == napi_generic_failure, "direct property cycle refused");
    OK(napi_close_handle_scope(env, outer));
}

static napi_value init_eot(napi_env env, napi_callback_info) {
    const auto mode = std::getenv("LIVEKIT_NAPI_FAKE_MODE");
    if (mode && std::string(mode) == "warm-int") throw 7;
    if (mode_is("warm-hang")) hang_at("prewarm");
    if (!initialized) { conformance(env); initialized = true; }
    return nothing(env);
}
struct InputLife { bool executed = false, completed = false; };
static void input_finalizer(napi_env, void* data, void*) {
    auto life = static_cast<InputLife*>(data); require(life->executed && life->completed, "input finalizer follows worker settlement"); delete life; ++finalized;
}
struct VadState { double state = 0; };
static void vad_finalizer(napi_env, void* data, void*) { if (mode_is("finalizer-hang")) hang_at("finalizer"); delete static_cast<VadState*>(data); ++finalized; }
struct Work {
    napi_env env; napi_async_work work = nullptr; napi_ref input = nullptr; napi_deferred deferred = nullptr;
    const int16_t* pcm; size_t count; double probability = 0; std::thread::id owner; InputLife* life;
    std::string mode;
    VadState* vad = nullptr; napi_ref receiver = nullptr;
};
static void execute(napi_env env, void* data) {
    auto work = static_cast<Work*>(data); require(std::this_thread::get_id() != work->owner, "execute runs on worker thread");
    bool pending = false; require(napi_is_exception_pending(env, &pending) == napi_invalid_arg, "owner-thread Node-API confinement");
    if (work->mode == "hang") hang_at("predict");
    if (work->mode == "worker-int" || work->mode == "worker-custom") {
        work->life->executed = true;
        if (work->mode == "worker-custom") throw CustomFailure{};
        throw 7;
    }
    double total = 0; for (size_t i = 0; i < work->count; ++i) total += std::abs(static_cast<double>(work->pcm[i]));
    work->probability = work->count ? total / (32768.0 * static_cast<double>(work->count)) : 0;
    if (work->vad) { work->vad->state = (work->vad->state + work->probability) / 2; work->probability = work->vad->state; }
    work->life->executed = true;
}
static void complete(napi_env env, napi_status status, void* data) {
    auto work = static_cast<Work*>(data);
    const auto failed_worker = work->mode == "worker-int" || work->mode == "worker-custom";
    require(status == (failed_worker ? napi_generic_failure : napi_ok) && std::this_thread::get_id() == work->owner, "complete runs on owner thread");
    if (work->mode == "worker-custom") require(custom_destroyed == 1, "foreign exception released before completion and image retirement");
    napi_value input = nullptr; OK(napi_get_reference_value(env, work->input, &input)); require(input != nullptr, "persistent input remains live after callback scope");
    size_t count = 0; void* pcm = nullptr; OK(napi_get_typedarray_info(env, input, nullptr, &count, &pcm, nullptr, nullptr));
    require(pcm == work->pcm && count == work->count, "input pointer and extent remain stable through work");
    work->life->completed = true;
    napi_value result = nullptr;
    if (work->mode == "reject" || failed_worker) { napi_value message = nullptr; OK(napi_create_string_utf8(env, "source-owned fake rejection", NAPI_AUTO_LENGTH, &message)); OK(napi_create_error(env, nullptr, message, &result)); OK(napi_reject_deferred(env, work->deferred, result)); }
    else { OK(napi_create_double(env, work->mode == "nan" ? NAN : work->probability, &result)); OK(napi_resolve_deferred(env, work->deferred, result)); }
    if (work->receiver) { napi_value receiver = nullptr; OK(napi_get_reference_value(env, work->receiver, &receiver)); require(receiver != nullptr, "VAD receiver survives async settlement"); OK(napi_delete_reference(env, work->receiver)); }
    OK(napi_delete_reference(env, work->input)); OK(napi_delete_async_work(env, work->work)); delete work;
}
static void unused_execute(napi_env, void*) {}
static void unused_complete(napi_env, napi_status, void*) {}
static napi_value prediction(napi_env env, napi_callback_info info, bool vad) {
    if (!initialized) { conformance(env); initialized = true; }
    size_t argc = 2; napi_value arguments[2]{}, receiver = nullptr; OK(napi_get_cb_info(env, info, &argc, arguments, &receiver, nullptr)); require(argc == 1, "one PCM argument");
    bool typed = false; OK(napi_is_typedarray(env, arguments[0], &typed)); require(typed, "typed PCM input");
    napi_typedarray_type type = napi_int8_array; size_t count = 0, offset = 1; void* pcm = nullptr; napi_value buffer = nullptr;
    OK(napi_get_typedarray_info(env, arguments[0], &type, &count, &pcm, &buffer, &offset)); require(type == napi_int16_array && offset == 0 && count <= 19200, "typed PCM extent");
    napi_valuetype buffer_type = napi_undefined; OK(napi_typeof(env, buffer, &buffer_type)); require(buffer_type == napi_object, "owned backing buffer");
    require(napi_set_named_property(env, buffer, "view", arguments[0]) == napi_generic_failure, "typed-array backing cycle refused");
    const auto mode_env = std::getenv("LIVEKIT_NAPI_FAKE_MODE"); std::string mode = mode_env ? mode_env : "normal";
    if (mode == "throw-int") throw 7;
    if (mode == "throw") { napi_value message = nullptr, error = nullptr; OK(napi_create_string_utf8(env, "source-owned fake synchronous error", NAPI_AUTO_LENGTH, &message)); OK(napi_create_type_error(env, nullptr, message, &error)); OK(napi_throw(env, error)); return nullptr; }
    auto life = new InputLife; auto work = new Work{env, nullptr, nullptr, nullptr, static_cast<const int16_t*>(pcm), count, 0, std::this_thread::get_id(), life, mode};
    if (vad) {
        require(count == 512, "VAD consumes exactly one window");
        napi_value state = nullptr; void* raw = nullptr;
        OK(napi_get_named_property(env, receiver, "_state", &state)); OK(napi_get_value_external(env, state, &raw));
        work->vad = static_cast<VadState*>(raw); require(work->vad != nullptr, "VAD method receives its own context object");
        OK(napi_create_reference(env, receiver, 1, &work->receiver));
    }
    napi_value promise = nullptr, name = nullptr;
    OK(napi_create_promise(env, &work->deferred, &promise)); OK(napi_create_reference(env, arguments[0], 1, &work->input));
    OK(napi_add_finalizer(env, arguments[0], life, input_finalizer, nullptr, nullptr));
    OK(napi_create_string_utf8(env, "fake-predict", NAPI_AUTO_LENGTH, &name));
    OK(napi_create_async_work(env, nullptr, name, execute, complete, work, &work->work)); OK(napi_queue_async_work(env, work->work));
    napi_async_work saturation = nullptr; OK(napi_create_async_work(env, nullptr, name, unused_execute, unused_complete, nullptr, &saturation));
    require(napi_queue_async_work(env, saturation) == napi_queue_full, "bounded async queue"); OK(napi_delete_async_work(env, saturation));
    return promise;
}
static napi_value predict(napi_env env, napi_callback_info info) { return prediction(env, info, false); }
static napi_value predict_vad(napi_env env, napi_callback_info info) { return prediction(env, info, true); }
static napi_value reset_vad(napi_env env, napi_callback_info info) {
    if (mode_is("reset-hang")) hang_at("reset");
    size_t argc = 0; napi_value receiver = nullptr, state = nullptr; void* raw = nullptr;
    OK(napi_get_cb_info(env, info, &argc, nullptr, &receiver, nullptr)); require(argc == 0, "VAD reset receives no samples");
    OK(napi_get_named_property(env, receiver, "_state", &state)); OK(napi_get_value_external(env, state, &raw));
    require(raw != nullptr, "VAD reset receives its own context object"); static_cast<VadState*>(raw)->state = 0;
    return nothing(env);
}
static napi_value create_vad(napi_env env, napi_callback_info) {
    napi_value receiver = nullptr, state = nullptr, run = nullptr, reset = nullptr;
    OK(napi_create_object(env, &receiver));
    auto context = new VadState; OK(napi_create_external(env, context, vad_finalizer, nullptr, &state));
    OK(napi_set_named_property(env, receiver, "_state", state));
    OK(napi_create_function(env, "predict", NAPI_AUTO_LENGTH, predict_vad, nullptr, &run));
    OK(napi_create_function(env, "reset", NAPI_AUTO_LENGTH, reset_vad, nullptr, &reset));
    OK(napi_set_named_property(env, receiver, "predict", run)); OK(napi_set_named_property(env, receiver, "reset", reset));
    return receiver;
}
static napi_value initialize(napi_env env, napi_value exports) {
    const auto mode = std::getenv("LIVEKIT_NAPI_FAKE_MODE");
    if (mode && std::string(mode) == "init-int") throw 7;
    if (mode_is("open-hang")) hang_at("module-init");
    napi_value init = nullptr, prediction = nullptr, create = nullptr, window = nullptr;
    OK(napi_create_function(env, "initEot", NAPI_AUTO_LENGTH, init_eot, nullptr, &init));
    OK(napi_create_function(env, "predict", NAPI_AUTO_LENGTH, predict, nullptr, &prediction));
    OK(napi_set_named_property(env, exports, "initEot", init)); OK(napi_set_named_property(env, exports, "predict", prediction));
    OK(napi_set_named_property(env, exports, "initVad", init));
    OK(napi_create_function(env, "createVad", NAPI_AUTO_LENGTH, create_vad, nullptr, &create));
    OK(napi_set_named_property(env, exports, "createVad", create));
    OK(napi_create_double(env, 512, &window)); OK(napi_set_named_property(env, exports, "VAD_WINDOW_SAMPLES", window));
    for (bool strong : {false, true}) {
        napi_value holder = nullptr; OK(napi_create_object(env, &holder)); auto data = new FinalizerReference;
        if (strong) { OK(napi_create_reference(env, holder, 1, &data->reference)); OK(napi_add_finalizer(env, holder, data, reference_finalizer, nullptr, nullptr)); }
        else OK(napi_add_finalizer(env, holder, data, reference_finalizer, nullptr, &data->reference));
        OK(napi_set_named_property(env, exports, strong ? "strong-finalizer-proof" : "weak-finalizer-proof", holder));
    }
    return exports;
}
__attribute__((constructor)) static void register_module() {
    if (mode_is("dlopen-hang")) hang_at("dlopen-constructor");
    static napi_module module{1, 0, __FILE__, initialize, "source_owned_fake", nullptr, {nullptr, nullptr, nullptr, nullptr}};
    napi_module_register(&module);
}
