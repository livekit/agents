# Bindings and the Python host

ABI declarations, generated APIs, host invocation and the Python-specific boundary.

[Contributor index](README.md) · [Execution paths](architecture.md#execution-paths)

## Shared binding semantics

**Declare semantics once in Rust, generate both bindings, and run the same
conformance cases against each.** Features reuse these contracts rather than
adding their own cancellation wrapper, iterator pump or conversion rules.

Generated Python remains part of the Rust-declared binding system. Keep language-specific
SDK adaptation, object identity and hydration in the host; generate mechanical field
conversion rather than maintaining a second schema inventory.

ABI declarations describe API meaning independently of the Rust implementation
types. Runtime and binding implementations may change without changing that
meaning. A Rust future does not itself define cancellation propagation; an
`Option<T>` does not define omitted-argument behavior; a Rust iterator does not
choose between a lazy host iterator and an eagerly collected list.

### Declared contracts

| Concern | Declared semantics |
| --- | --- |
| Presence | Required, omitted, explicit null and value; defaults apply to the declared cases, not general truthiness |
| Scalars | Boolean and Unicode semantics, integer ranges, floating-point precision and permitted non-finite values; no implicit truthiness conversion |
| Collections | Typed lists, tuples, arrays and map keys/values; ordering, snapshot versus live access and validation of every element |
| Records and variants | Required fields, defaults and default factories, enum/union discriminants, unknown-field policy and validation paths |
| Buffers and media | Encoding, element type, layout, format, copying/borrowing, retained lifetime and permitted concurrent access |
| Time | Duration versus timestamp, units, precision and monotonic versus wall-clock domain |
| Calls | Typed constructors, functions, methods and properties; synchronous versus asynchronous execution, argument/default rules and result conversion |
| Callback | Arguments, allowed return forms, execution context, reentrancy, failure conversion and cancellation relationship |
| Stream | Item/control types, lazy consumption, buffering, backpressure, EOF, failure and cleanup |
| Notifications | Typed event payload, subscription lifetime, delivery order, buffering and unsubscribe behavior |
| Resource | Native handles versus host references, identity, ownership/borrowing, permitted sharing, lifetime and close responsibility |
| Errors | Typed failures, host exceptions, causes, validation errors and cancellation; conversion in both directions |
| Cancellation | Target operation, propagation to owned work, reason/outcome and observable settlement |

These are internal contract families, not a public wrapper class for each row.
Each declaration composes supported families; adding a type extends the shared
model and its conformance cases before either binding accepts it.

For example, a nullable update has three meanings: omit to preserve the current
value, explicit null to clear it, or supply a replacement. Rust can represent
this with `Maybe<Option<T>>`; Python uses `NOT_GIVEN`/`None`/a value and Node uses
`undefined`/`null`/a value. An invalid null is rejected when the declaration does
not permit it. False, zero, an empty string and an empty collection remain
values, not omission.

Time and numeric declarations state units, precision and accepted ranges.
Conversions cannot silently truncate an integer, reinterpret a timestamp as a
duration or lose precision crossing JavaScript's number range. Media/buffer
contracts state whether data is copied, borrowed or retained and how long it
remains valid.

Borrowed buffers preserve Rust aliasing and concurrency guarantees for their
full retained lifetime, in both binding directions. Retaining an exporter or
creating a read-only view does not establish immutable backing storage. Copy
mutable host buffers at the boundary unless an enforced ownership or
synchronization contract makes borrowing safe throughout native execution.
Obtain copies through the host's supported access rules. Releasing the GIL,
returning to the host or scheduling native work cannot leave Rust references
exposed to incompatible mutation through another alias.
Conformance tests include writable aliases and host mutation after handoff,
verifying copied values or the declared access protection during native work.

Duration conversion rejects unsupported non-finite or out-of-range values
before constructing native durations.

Records and collections cross as validated values unless explicitly declared as
live resources. Mutating a converted list or record does not silently mutate
Rust state. Mutable defaults are created per instance. Typed maps declare their
key conversion; arbitrary JSON is a separate, explicit value contract, not a
fallback for unknown types. Aliases, newtypes and generic parameters retain
their declared validation and conversion rules.

### Calls, references and failures

Constructors, getters, setters, methods and free functions use the same typed
conversion rules. Callback signatures declare whether a synchronous result,
awaitable or stream is accepted. A synchronous call cannot block an event loop
waiting for a callback on that same loop. Reentrant callbacks enter through the
runtime's normal event dispatch, never recursively into a core transition.

Native Rust objects cross through typed handles. Python/Node objects cross
through typed proxies or explicitly opaque host references. Opaque references
preserve identity within their originating host and are not inspected by core
code. Retaining a reference keeps its object alive; releasing or closing it
follows the declared ownership contract. Stale or foreign-runtime handles fail
explicitly rather than becoming null or another object. Host references and
native handles are not serializable application state. These bindings do not
transfer live Python objects into a JavaScript runtime or vice versa.

Notifications reuse typed callbacks or sources. Subscription ownership defines
when delivery starts and ends, whether unsubscribe waits for in-flight delivery,
and how late events are rejected. Delivery follows the declared buffering and
ordering policy; adding an event API does not introduce another queue protocol.

Declared domain errors map to the corresponding host error types with structured
details. Unclassified host exceptions retain diagnostic type, message and cause;
an exception returned to its originating host retains the original exception
where its lifetime permits. Validation errors identify the argument or field
path. Cancellation, end-of-stream and failure remain distinct. Binding entry
points contain unwinding Rust panics before the host boundary. Process aborts
are supervision failures, not recoverable API exceptions. Do not report success
or reuse potentially corrupted operation state after a panic.

Cancellation follows the [shared operation lifecycle](runtime.md#cancellation-and-task-ownership).

### Conformance

The shared declaration model feeds Python and Node generation. Emitters express
that model in each language without independently choosing defaults, ownership
or error behavior. Common defaults live in the shared contract vocabulary;
features do not repeat metadata for rules that are already defined there.

Resolve declared types and aliases through that model. Do not infer semantics
from a last path segment such as `Value`, a callback's name, or whichever methods
a host object happens to expose. Unsupported or ambiguous declarations fail
generation explicitly. Add a missing contract capability to the generator,
rather than falling back to `Any`, string-based reflection or handwritten glue.

Run equivalent conformance cases against Python-to-Rust and Rust-to-Python,
and Node-to-Rust and Rust-to-Node. Cases cover absent/null/empty values, defaults,
invalid fields/tags, range/precision boundaries, collection and buffer ownership,
per-instance defaults, scalar/awaitable/stream callback results, synchronous
calls and reentrancy, subscription/unsubscribe races, error/cause conversion,
object identity, stale and foreign handles, partial output followed by failure,
backpressure, borrowed resources, cancellation before start and during a pull,
late completion and cleanup.
The specification and inputs are shared; adapters use each language's normal
APIs. Tests assert the same outcomes, not identical scheduler traces.

Feature tests add domain behavior on top of these primitives. They do not
reimplement the primitive or redefine its contract. Native Rust plugins and
host-language plugins pass the same capability contracts through their bindings.
## Binding rules

### Declarations and placement

All handwritten API export declarations live in `lk-agents-abi`. Backend-local interpreter
interface metadata lives beside the Python adapter that uses it; it cannot introduce a public
native export or provider call. Both resolve into the binding model. Other `#[host(...)]` declarations and export metadata remain ABI-owned. There is no `Host` derive, and
standalone class/binding/export forms are rejected. Neither subject
cores nor runtime implementations declare a second export surface.

| ABI declaration | Defines | Implementation or generated output |
| --- | --- | --- |
| `#[host(record)]` | Boundary data: options, events, chat items, reports and error payloads | Generated models and conversions at the language boundary |
| `#[host]` for a synchronous operation | Constructor, transition or synchronous API contract | Subject core for decisions; runtime or an async I/O crate for host views and resources |
| `#[host(proxy)]` on a trait | Provider, user I/O, tool runner, hook or host SDK contract | Generated proxy in `lk-agents-py`, entered through HostInvoker |
| `#[host(proxy(base = "tts.py"))]` on a trait | Provider contract whose host class is a plugin base class, hand-written in that file beside the trait | Generated proxy; the base written into its generated module after the records, checked to define every declared call |
| `#[host(facade(...))]` on a trait | Public host class over a native class ([Facades](#facades)) | Generated Python class; a handwritten subclass only for Python-specific leftovers |
| `#[host]` for an async operation, stream or awaitable | Operation awaited by the host | Runtime or async I/O implementation and generated async bridge |

ABI contains contracts and data, not I/O or host-language implementations. It
has no pyo3, Tokio, futures or async-trait dependency: dynamic ports return std
boxed futures and class traits use native `async fn`. Generated pyo3 conversions
are free functions in `lk-agents-py`, beside the generated adapters and proxies.
Generation must not introduce pyo3, executors or runtime dependencies into ABI.
ABI never imports subject-core or runtime implementations.

Core bindings take data and IDs and return data. Bind an item only when used
by the public API, a plugin base or host glue. Glue-only names use `_core_*`.
Avoid duplicate wire records, per-language schemas and bindings that only
rename another binding.

A proxy addresses the host object's own surface; no handwritten view renames it. A method
declares what it reaches: `name = "a.b"` calls the method along that attribute path,
`getter = "a.b"` (with `call` or `read_once` in place, with `await` on the host's loop) reads the
attribute instead, and
`function = "module.name"` calls a module function with the host object first, for glue that
does more than rename. A `#[host(kw)]` parameter, and every one after it, is passed by keyword.
A proxy can target an SDK class directly (`proxy(host = "livekit.rtc.ByteStreamWriter")`), and
`proxy(.., sdk)` reads each `wire_enum` the host returns as its SDK number through the enum's
`TryFrom<i32>`: an unknown number reads as the enum's `Default` and is dropped from a list. An
adapter that carries state beside its SDK object (a room's event bus, a source's playout ledger)
holds that object as `sdk`, and its declarations read through it:

```rust
#[crate::host(proxy(host = "livekit.agents.voice.room_io.Room", sdk))]
pub trait RtcRoom: Send + Sync {
    #[host(call, getter = "sdk.connection_state")]
    fn connection_state(&self) -> Result<ConnectionState, RtcError>;
    #[host(read_once, getter = "local")]
    fn local(&self) -> &Arc<dyn LocalParticipant>;
    #[host(call, name = "sdk.unregister_byte_stream_handler")]
    fn unregister_byte_stream_handler(&self, topic: &str) -> Result<(), RtcError>;
}
```

`view = "module.Class"` remains for host objects whose calls need state the host class does not
carry, such as a turn detector's opened stream.

A proxy method also declares how it reaches the host, so no `_core_*` method exists only to
adapt a call: `await` and `notify` accept a host method that returns a value or an awaitable,
`emit = "event"` emits the arguments on the host's event emitter, and `await, iter` pulls the
next item of a host `AsyncIterator` (the host itself, or the attribute along `getter`), opened by
its first pull, with its end read as `None` and a `#[host(cancel)]` reaching the pending pull. A proxy trait can also be emitted as a
`typing.Protocol` for host code that types its implementations. Rust never calls the host to
reach another native object: a capability one native owner needs from another is passed
natively.

SDK event subscriptions are declared on the SDK proxy (event name, argument conversions, the
native entry they feed); codegen emits the `on(...)` wiring (`_generated/rtc_events.py`):

```rust
#[crate::host(proxy(host = "livekit.rtc.Room", view = "livekit.agents.voice.room_io.Room", sdk,
    events(module = "rtc_events", on = "sdk", entries = RtcRuntime, pass(bus = "self.bus"),
        participant_disconnected(args(participant), reason = "participant.disconnect_reason"),
        ..)))]
```

A `wire_enum` crosses as its SDK number wherever an SDK holds it so: an `sdk` proxy's returns, a
`#[host(sdk)]` function argument and the fields of a `record(.., sdk)`. A proxy's
`#[host(kwargs)]` argument is a `typed_dict` record whose keys are the call's keyword arguments,
so an SDK call taking keyword options needs no adapter method.

Keep export attributes in one documented vocabulary shared by parsing and
generation. New keys require multiple concrete uses, not feature-specific glue.

### Boundary values

- No tuple crosses the boundary. A value with more than one field is a `#[host(record)]`;
  codegen rejects tuple types in ABI signatures.
- ABI declares every default, constant and environment variable the host sees, once:
  `#[host(default)]` on parameters and fields, `#[host(constant)]` for constants, and
  environment resolution in the native configuration (as `ServerConfig` resolves
  `LIVEKIT_URL`). Python never retypes one.
- Every string vocabulary is a `wire_strs` enum or a record tag declared in ABI. Its Python
  `Literal` is generated; Rust matches the enum, never the string.
- An options record can emit a keyword-arguments `TypedDict` (`kwargs`), so a public
  constructor takes `**options: Unpack[XOptionsKwargs]` and passes them on unfiltered.
- `Duration` crosses as `datetime.timedelta`.
- A Python object that carries a record's data converts through `record(host_type = ...)`:
  SDK objects, OpenTelemetry API types, public events and outcome classes. No handwritten
  field-by-field copy or `isinstance` ladder converts between a host object and a record.
  A boundary field reads a nested attribute (`path = "trace_flags.sampled"`, emitted through
  `emit_class`), `fixed(..)` names constructor constants, a transparent union tells host classes
  apart (`host_type`, a member's `path`, a last `otherwise` member), `native_enum(host_type = ..)`
  reads a foreign enum by member name, and `record(.., typed_dict, tag = "type")` is a dict
  union keyed by its tag (ASGI messages).

### Facades

A public Python class over a native class is declared as its own ABI trait, and that trait reads
as the Python class: one method per Python method, in order. The native class is named once on
the trait. A method with no attribute calls the native method of the same name. A delegate is a
typed associated item that the constructor receives, and a method forwarding to it says so with
`via`:

```rust
/// `llm.DuplexRealtimeAdapter`: runs a `DuplexModel` inside a `VoiceSession`.
#[host(facade(path = "livekit.agents.llm.duplex", extended, name = "DuplexRealtimeAdapter",
              base = "llm.realtime.RealtimeModel", native = DuplexRealtimeModel,
              legs(session), base_args(capabilities)))]
pub trait DuplexRealtimeAdapterFacade {
    /// The wrapped plugin model.
    #[host(delegate)]
    type DuplexModel: DuplexHost;

    #[host(new)]
    fn new(duplex_model: Self::DuplexModel,
           #[host(kw, default = None)] gate: Option<Arc<dyn AudioGateFactory>>,
           #[host(default = 0.8)] audio_timeout: f64) -> Self;

    #[host(getter, via = DuplexModel)] fn model(&self) -> String;
    #[host(getter, via = DuplexModel)] fn provider(&self) -> String;
    #[host(via = DuplexModel)] async fn aclose(&self);
}
```

The generated class (`_generated/facades_<x>.py`, re-exported from `path`) subclasses `base`;
its public `__init__` takes the `#[host(new)]` parameters with their ABI defaults and builds the
native object from them, by name; `base_args(..)` then builds its base from the native getters
of the same names, and `keep(..)` keeps parameters as `self._<param>` for an extending class;
`host_args(param = "module.object")` fills a native constructor parameter the Python signature
does not take with a host object (`inference.VAD`'s Silero model factory). A facade named as its
base binds the base as `Base<Name>`, and a `base` with no module names a top-level
`livekit.agents` class (`VAD`). A facade with no `#[host(new)]` declares `_native`, which its
handwritten subclass sets. A chain
facade declares `availability(observer, members, kind, event, record)`: it is its chain's
observer, keeps `members` (as its `#[host(new)]` takes them) as `_members`, emits each
availability change as `record`, names the serving member's `model`, `provider` and
`metrics_metadata`, and closes its native object, or its members when the native declares no
`aclose`. Method adaptations
are a fixed vocabulary: `none_ends_iteration` (a native `None` raises `StopAsyncIteration`),
`wrap` (the result becomes a record), `native = m` (another native method), `future`,
`discard`, `not_given`, `origin`, `returns`, `annotation`, `resolve`, and the context-manager
form `exit = m`, which turns a native begin/end pair into `async with`:

```rust
/// `RunContext`'s hold of the call's floor.
#[host(facade(path = "livekit.agents.voice.events", name = "RunContext", extended))]
pub trait RunContextFacade {
    /// The call's session, read at each use.
    #[host(delegate = "_call()._core")]
    type Call: VoiceSession;

    /// Wait for idle, then hold the floor while inside.
    #[host(via = Call, native = wait_for_idle_and_hold, exit = release_idle_hold,
           bind(origin = "_origin"), yields = "_call()")]
    async fn foreground(&self) -> Result<(), IdleRefused>;
}
```

The target (`self._native`, or the `via` delegate) is read once as the block enters, so the exit
reaches the object that entered; the native exit is idempotent, a no-op once that object closed.
`exit = Class::m` is the entered native value's own method, `abort = m` runs instead when the
block raises, `yields = "path"` hands the block `self.path` and `returns = "module.Name"` the
entered value wrapped. A delegate is held as `self._x`, or read at each call from a declared path
(`delegate = "_call()._core"`). `bind(arg = "path")` passes `self.path` as a native argument the
Python method does not take; `returns` with `each = Record::field` maps a native list first
(`ChatContext([x.payload for x in rows])`); `chain` answers `self`; `setter = prop` assigns a
property. `from_native` builds the class from the native object it is handed
(`Partition(native, agent)`), keeping the `#[host(new)]` parameters, and `foreign` lets `base` name a
class outside `livekit.agents` (`opentelemetry.trace.Span`). `generic = "T"` makes the class
`Generic[T]` over its own TypeVar, which its annotations name (`RunContext.foreground` yields
`VoiceSession[Context_T]`).

A stream facade (`base` a provider stream: `llm.LLMStream`, `RecognizeStream`,
`SynthesizeStream`, `ChunkedStream`, `VADStream`) names the native operation its `_run` opens
on its owner, the facade of the same native class: `stream(op = synthesize, owner = "_tts",
native_retries)`. The operation's arguments are the stream's `self._<param>` attributes; the base
class pumps input and output (`_core_run_native`), and `native_retries` marks a native operation
that retries on its own, so the base's attempts do not. Per-call arguments are declared once, as
the operation's parameters: with `method = "stream"` the owner facade gets that provider method,
taking every parameter but those read from the base stream's `state(..)`, and the stream's
constructor passes `base_args(..)` to its base and keeps the rest. `rename(op_param = "name")`
gives an operation parameter the provider method's own name (`synthesize(text)` passing
`input_text`), `provider_default(p)` makes a parameter not given take the owner's native getter
`p` (a provider's own `conn_options`), and the stream opens the operation on the owner's `leg`
when it declares one (the job's view of an inference client):

```rust
#[host(facade(path = "livekit.agents.stt", base = "stt.RecognizeStream", native = FallbackStt,
              stream(op = stream, owner = "_stt", native_retries, method = "stream",
                     base_args(conn_options), state(start_time_offset),
                     default(conn_options = "..types.DEFAULT_API_CONNECT_OPTIONS"),
                     annotation(language = "str"), not_given)))]
pub trait FallbackRecognizeStream {}
```

Logic a facade cannot declare goes in a handwritten subclass of the generated class when it is
specific to Python (SDK objects, user callbacks, reflection): the facade is then `extended`, not
re-exported, and the handwritten class of its name in `path` subclasses it, setting `_native` when
no `#[host(new)]` builds it. Logic that Rust can own cleanly moves to Rust instead, and the facade
stays fully generated.

### Generation

`lk-agents-codegen` discovers declarations through Rust modules, resolves them into one model, and
generates, as checked-in files:

- A Rust trait for every declaration, classes and providers alike, which a core,
  the runtime or an async I/O crate implements; the compiler checks completeness.
- The pyo3 adapters, proxies and record conversions in `lk-agents-py`, which call
  those traits. The `#[host(..)]` attribute (`lk-agents-host-derive`) strips its markers and
  writes only the Rust a declaration implies: a record's `new` and `Default`, a wire enum's
  `ALL`, `as_str` and `from_wire`, interface helpers.
- The complete `_core.pyi`, `core_names.json` and Python models under `_generated/`.
- Forwarding modules, and the plugin bases copied verbatim from their hand-written `.py` files.
- Language emitters consume the same resolved classes, records, enums, constants and errors;
  backend-local interpreter interfaces do not enter ABI. A Node emitter uses these contracts
  with its own environment, references and scheduling.

Provider traits define their Python hooks and `_core_<method>` roles. Missing
required roles fail generation. Generated Python passes ruff and mypy.
The plugin base classes are hand-written `.py` files beside their ABI proxy traits; codegen merges their imports and writes them verbatim, and refuses a proxy call its base class does not define.

Generated files carry a first-line marker and are never edited manually.
`--check` rejects drift. Handwritten Python contains public API behavior,
user callback invocation, host SDK integration or process launch/bootstrap.
Modules that only forward or re-export are generated, with public names
re-exported by their owning package's `__init__`. A declaration's `path = "livekit.agents.x"`
is its public module: `_generated/public_x.py` re-exports every name placed there, and `x`
star-imports it, holding no import list of those names; its literal `__all__` lists them, which
mypy requires. Records, exceptions, flags, `constant(path = ..)`, `function, path = ..` and str
classes are placed by `path`; a native class by `class(module = ..)`; a plugin base file's public
classes at their host class's module. A handwritten module lists only its own names and extends
them with the generated list, the one spelling pyright follows:
`__all__ = ["AudioOutput"]` then `__all__ += _public.__all__`.

### Module registration

Maturin builds `livekit.agents._core` from `rust/crates/lk-agents-py`, configured
in `python/livekit-agents/pyproject.toml`. It ships inside the
`livekit-agents` wheel as `cp311-abi3` for Python 3.11 and later.

`lk-agents-py` is the only crate containing pyo3: the registration shell, the
generated conversions, adapters and proxies, and HostInvoker's Python entry
(GIL, context copy and coroutine scheduling) behind its declared port. Its
handwritten part stays minimal; whatever can be generated is generated. Runtime
initialization, fork guards, async bridging and resource handles belong in
`lk-agents-runtime` as language-neutral Rust. Generated registration supplies the module's names. Tests compare `dir(_core)`
with `core_names.json`. CI accepts each built wheel with
`.github/check_wheel.py`: its `cp311-abi3` tag, its
contents, `dir(_core)` against the shipped `_core.pyi`, and a size record.

### Python binding layout

The handwritten files under `lk-agents-py/src/` use a `py_` prefix for Python-specific
representations and execution. `py_calls.rs` handles interpreter entry, captured contexts,
notifications and future bridging; `py_streams.rs` adapts Python iterators and native sources.
`py_values.rs` handles scalars, buffers, JSON, union normalization, metadata codecs and missing
values. `py_objects.rs` retains object references and recovers native capabilities;
`py_errors.rs` preserves exceptions; `py_logging.rs` forwards Rust log records;
`py_stall.rs` is the event-loop stall detector, reading stacks with py-spy over the process's
own memory, never under the GIL.
`lib.rs` selects implementations and registers exports.
Domain classes are declared in ABI and generated into `generated/`; resource ownership and
feature behavior stay with their native implementations. Colocated Python tests in
`src/py_bindings_tests.py` exercise the Cargo-built bindings and process/fork journeys.
The `py_bindings` Cargo integration target discovers sibling `*_tests.py` files and runs them
with pytest. Public-package targets share one test bootstrap that registers the freshly linked
module through the same `register` function used by Python imports, before loading the package. Assertions
are ordinary Python, without embedded source strings. This is the standard for this crate;
these files are not also collected by the standalone Python test gate. Run both languages with:

```bash
uv run --no-sync cargo test --manifest-path rust/Cargo.toml -p lk-agents-py --features ffmpeg
```

The uv environment supplies the matching Python interpreter and installed test dependencies.
No extension rebuild or separate pytest command is needed. Rust tests retain coverage of
native ownership, reference registries and cancellation internals.
Their host objects and callback bodies live in `src/py_test_fixtures.py`; the test-only
`py_test_support.rs` loader compiles that file with its real path for tracebacks. Rust passes
values to named fixture functions that return explicit objects or accept explicit arguments.
There is no `locals()` export or dictionary-update protocol. Native assertions and isolated
Cargo targets remain, without inline Python programs or dynamically assembled expressions.

Opaque `HostRef` and `MediaRef` declarations select one generic Python reference codec.
They are distinct aliases of one `ObjectRef<Kind>` implementation, with host and media marker
types; no constructor trait or macro selector is required. The binding keeps one typed
`retain`/`resolve` implementation. The table accepts a reference and validates
its registry origin and live entry together; callers never resolve bare numeric IDs or do a
separate membership check. Captured exceptions keep their separate type and table, whose `take` operation
validates the reference and consumes it once across all clones.

ABI `exceptions(...)` declarations generate ordinary Python exception classes: bases,
constructor fields and defaults, public module identity and representations. The public error
modules re-export them; semantic cause-chain display and HTTP retry normalization remain host
hooks. Native-to-host error mappings and declared `capture` wrappers are generated too.
`record(..., host_type = "...")` generates field conversions for an existing host class without
replacing its authored methods (for example, SDK audio frames).

Record declarations generate complete converters: field reads and writes, renames, omission,
flattening, constructor selection, validation and nested SDK metadata paths. Buffer ownership,
clock translation and other value codecs receive one declared value; they never infer host
field names. Native normalization consumes typed records. Handwritten primitives declare their
binding paths with `codec(..., py = "module::converter")` in ABI, so generated callers
reference their actual owner directly. There is no handwritten converter re-export inventory.

The Python module's implementation selection is generated from concrete Rust `impl Trait for Type`
items in the backend's Cargo dependency closure, matching ABI class traits by resolved identity.
Unrelated backends and test crates cannot contribute selections. Missing, generic or ambiguous implementations
fail generation; names are not matched by suffix or guessed from class names. Types owned by other
crates must be reachable through the runtime's existing public re-exports. Python's logging bridge
remains implemented in the Python backend. There is no second handwritten class catalogue.

Framework marker classes, the missing-value sentinel, connection option dataclasses, interruption
events and transcript chunks are generated from ABI declarations. `TranscriptChunk` carries text
and optional alignment metadata as ordinary fields across every binding; it has no scalar string
semantics. Node adapters normalize plain text explicitly, and metadata lists convert every entry
instead of filtering by a handwritten Python class. `APIConnectOptions` retains explicit Python
methods for asyncio deadlines; generated dataclasses declare those hooks alongside their fields.
Framework data schemas include the public session and playback events, room options, tool
metadata and outcomes, provider option dictionaries, worker enums and evaluation results. An
explicit `dataclass`, `typed_dict`, enum or transparent alias can declare only a model; native
converters appear when `extract` or `emit` is requested. The declaration owns constructor field
order, keyword-only fields, frozen/slot behavior, excluded fields and discriminator keys.
A struct record is a keyword-only, slotted dataclass, frozen (and so hashable) only when its Rust
type derives `Hash`: an identity used as a key, such as `FrameId`. Freezing is not the default
because `@dataclass` generates `__setattr__`, `__delattr__`, `__hash__` and pickling hooks for
each frozen class when the module is imported, which made up a third of the records' import
time; `tests/smoke/test_import_time.py` bounds the modules the import loads.
Python validation and serialization operations remain named hooks with behavioral contracts.

Callback-bearing dataclasses use `deferred` to keep ordinary postponed Python annotations while
their public facade is initializing. Their types are imported for static checking, without
importing a facade that is still defining those same types. Other public dataclasses evaluate
their annotation expressions before assigning their public module identity. Neither path
recursively evaluates a callback's forward references during package import.

Public result unions are generated as well. A dataclass record may subclass another of its
module (`base = "_Settled"`), inheriting its fields and members; it declares a model only,
since its Rust struct holds none of the base's fields. `generic = "Output_T"` makes a dataclass
`Generic[Output_T]` and a transparent union a `TypeAliasType` over that parameter, which the
module declares as a type variable defaulting to `Any`; a union member names its parameter
with `annotation = "RunCompleted[Output_T]"`. A `cached` property keeps a stateful view (the
`expect` cursor) on an unslotted dataclass. Member annotations are imported for the type
checker only, except a toggle's class, tested at run time. `RunResult[Output]` is
`RunCompleted[Output_T] | RunInputRequired | RunNoOutput` over `_Settled`, whose
`new_messages()`, `all_messages()` and `expect` are `_bindings` hooks; `settled_result` builds a
branch from a `RunSettled`.

A `codec(..., flag(module = "..."))` generates a Python `Flag` from the native newtype's public
bit constants; the generator discovers those constants by resolved type identity. Exception
schemas preserve read-only message properties, no-argument constructors and qualified bases.
Python controllers, interpreter state, SDK resource adapters and provider plugin
implementations remain authored Python; their existence does not make them
shared boundary records.

Native newtypes can declare `codec(via = u8)` to generate checked `TryFrom`/`Into` conversions.
A nominal host enum declares `attribute = "value"` alongside its host class; generated readers
check that class and generated writers reconstruct it. Connection defaults use
`defaults_from = DEFAULT_API_CONNECT_OPTIONS`, resolving the native constant as data without
executing Rust code or repeating field defaults. Marker and missing-sentinel declarations preserve
identity behavior; dataclass and Pydantic declarations choose their model semantics explicitly.
Transparent records declare scalar alternatives and nominal host variants without handwritten
language-specific discriminators.

Backend-local `#[host(interface(...))]` unit structs name the low-level interpreter interface
surface (asyncio, logging, reflection,
buffer methods and class imports). Discovery follows the backend's Rust modules and skips
generated files. Each object interface names its Python type, such as
`asyncio.AbstractEventLoop`, `asyncio.Task` or `contextvars.Context`; module functions belong to
their modules. The generator adds inherent helpers to those named Rust namespaces; there is no
anonymous constant manifest. These helpers accept Python objects, not checked nominal wrappers:
`MemoryView::cast(obj)` reads the declared `cast` member, while `MemoryView::import(py)` resolves
the real Python type. Structural probes check only their declared required members.
Imports use that same identity, without a separate import catalogue. Fields are
read-only unless explicitly declared writable. Nominal interfaces cannot add structural probes.
Duck-typed contracts are explicit structural protocols: `aclose()` is separate
from `AsyncIterator`, which does not promise closing. Generated method signatures own keyword
parameters, including `EventLoop.create_task(coro, context=...)`. Record fields use whole generated converters instead of
per-field accessor wrappers. Codegen rejects handwritten member lookups and generic field-helper
indirection, including computed keys and constants. Runtime keys are confined to the generic
bridges called by generated code, application map serialization and reflected Python field
objects. Handwritten public re-exports of generated bindings are rejected too.

Binding tests are grouped by the boundary they exercise. Explicit Cargo targets isolate
synthetic package fixtures, public Python models, generated proxies, native resource handles
and reference-table accounting in separate interpreters. Codec behavior belongs to media tests;
provider and orchestration behavior belongs to inference, core, voice or runtime tests.

### The GIL boundary

No lock holder waits for the GIL: Rust never calls the host while holding a
lock, so no lock and the GIL wait on each other:

- Every generated synchronous entry (constructor, method, getter, setter,
  function, an eager await's call, a duplex stream's `push`/`flush`/`end_input`)
  converts its arguments with the GIL, runs the Rust call inside `py.detach`,
  then converts the result with the GIL again. A mutable class's methods keep
  the GIL: PyO3's borrow of the instance, which the GIL serializes, is the
  class's only guard, and released it would refuse another thread's call. A
  getter that provably takes no lock may opt out with `#[host(getter, attached)]`.
- Live state shared with the host has one owner. Rust takes a lock only to read
  or update state, then calls the host (a proxy, a sink, HostInvoker) after the
  guard drops; host objects taken out of a locked queue are dropped after it.
  Log lines are queued and never wait for the GIL, so logging under a lock is
  allowed.
- A pyclass's `Drop` runs with the GIL, so it may take only locks whose
  holders never call the host.
- `lk-agents-py` holds no lock while Python code runs; a lock that must is
  PyO3's `MutexExt::lock_py_attached` or a `PyOnceLock`.
- Locks are plain `parking_lot` locks; std's poisoning `Mutex`/`RwLock` stay
  banned in the I/O crates.
## Plugins

A plugin supplies an implementation of a Rust-defined capability, such as STT,
TTS, LLM, realtime, VAD, a transform or a media endpoint. Provider contracts and
export metadata live in ABI. Rust implementations satisfy the runtime interface;
Python/Node implementations satisfy generated host interfaces. Session and agent
code consume the same capability regardless of implementation language.

Plugin authors implement provider/domain behavior. Generated/runtime support
owns host invocation, stream adaptation, correlation, cancellation and error
conversion. Plugin code does not reproduce those mechanisms. Configuration,
capabilities, identity and supported update behavior are typed and validated
before use; unsupported operations fail explicitly.

Python loads `lk-agents-py` as the `_core` extension. Node loads the Node
binding. The host owns its interpreter, event loop and application objects;
Rust calls host implementations through the generated proxy for that binding.

```text
Python application -> Python binding -> Rust runtime -> native Rust capability
                                                   -> Python plugin/I/O proxy

Node application   -> Node binding   -> Rust runtime -> native Rust capability
                                                   -> Node plugin/I/O proxy
```

A Rust implementation is reusable through both bindings. Python authors
implement the Python-facing capability; Node authors implement the Node-facing
capability. Both adapt to the same Rust interface and receive the same lifecycle,
stream and error semantics. Binding generation owns language adaptation, not
the plugin's business logic.

Each plugin instance has an explicit owner and close lifecycle. Sharing a
stateless definition does not share live streams or session-specific state.
Capabilities determine permitted concurrency and updates. Sessions supply
runtime resources and cancellation scope; secrets and connections are never
persisted as application state.

Test native Rust implementations through both bindings. Test Python and Node
implementations through their respective generated proxies, using the same
capability/stream conformance cases. Provider-specific protocol tests remain
separate from binding adaptation tests.
