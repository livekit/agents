//! The values a proxy may read once, when it is taken from the host.
//!
//! A `read_once` value is answered from that read for the proxy's whole life, so a session that
//! took it keeps it. Only a value its host object cannot change after construction may be read
//! that way: anything `update_options`, a setter or an attribute can change is read with `call`
//! each time it is used, or is owned by the engine and changed through it (a port object below).
//! A new entry names why its value cannot change.

/// One value read once: its host target (`name`, `getter` or `function`), the host class it is
/// read from when the target alone is ambiguous, and why it cannot change.
struct ReadOnce {
    target: &'static str,
    host: Option<&'static str>,
    #[allow(dead_code, reason = "the reviewed justification of each entry, read in this file")]
    why: &'static str,
}

const fn any(target: &'static str, why: &'static str) -> ReadOnce {
    ReadOnce { target, host: None, why }
}

const fn on(host: &'static str, target: &'static str, why: &'static str) -> ReadOnce {
    ReadOnce { target, host: Some(host), why }
}

const READ_ONCE: &[ReadOnce] = &[
    // Identity and fixed shape of a provider.
    any("_core_instance_id", "`id(self)`: the object's identity"),
    any("_core_label", "its class path"),
    any(
        "_core_capabilities",
        "fixed at construction: an option that would change a capability is a constructor argument, \
         and `update_options` refuses to change it",
    ),
    any("_core_num_channels", "the channel count is a constructor argument of the base class"),
    any("_core_retries_itself", "whether it retries itself is its class's (an adapter, a chain)"),
    any(
        "_core_conn_options",
        "the connection policy given at construction; a bound view's is its binding",
    ),
    any("_core_framework_info", "the running framework's version and platform"),
    // Port objects: built with their owner and never replaced; what changes goes through them.
    any("_core_options", "the engine-owned TTS options cell `update_options` writes"),
    any("_lifetime", "the toolset's lifetime port, built in its constructor"),
    any("plugin._input", "the duplex plugin's input queue, built in its constructor"),
    any("_tracker", "the output's playback tracker, built in its constructor"),
    any("_toolsets", "the session's toolset scopes port, built in its constructor"),
    any("_run", "the job's execution port, built with the job"),
    any("bus", "the room adapter's event sink, built in its constructor"),
    any("rpc_router", "the room adapter's RPC routes, built in its constructor"),
    any("local", "the room's local participant, fixed for the connection"),
    // Audio outputs: an output's chain and rate are constructor arguments.
    any("label", "an I/O object's name, given at construction"),
    on(
        "livekit.agents.voice.io.AudioOutput",
        "livekit.agents._bindings.audio_output_capabilities",
        "`can_pause` over a chain whose links and capabilities are set in `__init__`",
    ),
    on(
        "livekit.agents.voice.io.AudioOutput",
        "sample_rate",
        "the rate the sink requires, set in `__init__`",
    ),
    // One RPC invocation's request, which never changes.
    any("request_id", "the RPC invocation's id"),
    any("caller_identity", "who made the RPC invocation"),
    any("payload", "the RPC invocation's payload"),
    any("response_timeout", "the RPC invocation's deadline"),
    any("method", "the RPC method invoked"),
    // Room identity, which the SDK never changes for a participant, publication or stream.
    any("info.stream_id", "a data stream's id"),
    any("sdk.sid", "a participant's or publication's server id"),
    any("sdk.identity", "a participant's identity"),
    any("sdk.kind", "a participant's or track's kind"),
    any("sdk.name", "a publication's track name"),
    any("sdk.source", "a publication's track source"),
];

/// Refuses `read_once` on a target outside [`READ_ONCE`].
pub(super) fn check(at: &str, host: &str, target: &str) -> Result<(), String> {
    let allowed =
        READ_ONCE.iter().any(|r| r.target == target && r.host.is_none_or(|class| class == host));
    if allowed {
        return Ok(());
    }
    Err(format!(
        "{at}: `{target}` of `{host}` is not read once: only a value its host object cannot change \
         after construction is (the list in decl/resolve/read_once.rs, each with its reason); read \
         it with `call` each time it is used, or keep it in an engine-owned port"
    ))
}
