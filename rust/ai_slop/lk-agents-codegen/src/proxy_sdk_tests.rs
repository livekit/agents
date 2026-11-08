use crate::decl::tests::{proxy, read};

const STATE: &str = "#[crate::host(wire_enum)] pub enum State { Off, On }";

// Catches: an SDK number read or written as the enum's string, an unknown number refused instead
// of read as the default, or `sdk` accepted on a value that has no SDK number.
#[test]
fn sdk_parameters_and_record_fields_cross_as_sdk_numbers() {
    let decls = read(&format!(
        r#"{STATE}
        #[crate::host(record(module = "rtc", emit, sdk))]
        pub struct Options {{ pub name: String, pub state: State, pub last: Option<State> }}
        #[crate::host(functions)]
        pub trait Entries {{
            #[host(function)] fn changed(#[host(sdk)] state: State, #[host(default = None, sdk)] last: Option<State>);
        }}"#
    ))
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(
        llm.contains("state.extract::<i32>().map(|v| <::lk_agents_abi::llm::class::State as ::std::convert::TryFrom<i32>>::try_from(v).unwrap_or_default())?")
            && llm.contains("last.map(|v| v.extract::<i32>()"),
        "{llm}"
    );
    let shell = crate::records::conversions(&decls).expect("emits").text;
    assert!(
        shell.contains("kwargs.set_item(\"state\", ::std::convert::Into::<i32>::into(state))?;")
            && shell.contains("last.map(::std::convert::Into::<i32>::into)")
            && shell.contains("kwargs.set_item(\"name\""),
        "{shell}"
    );
    let models = crate::records::python(&decls).expect("models");
    let model = &models.iter().find(|m| m.path == "rtc.py").expect("rtc.py").text;
    assert!(
        model.contains("    state: int\n") && model.contains("    last: int | None\n"),
        "{model}"
    );
    let stubs = crate::class_stub::emit(&decls).expect("stubs");
    let stub = stubs.iter().map(|s| s.out.text.as_str()).collect::<String>();
    assert!(stub.contains("def changed(state: int, last: int | None = None) -> None"), "{stub}");
    let err = read(
        "#[crate::host(functions)] pub trait E { #[host(function)] fn f(#[host(sdk)] s: String); }",
    )
    .expect_err("a string has no SDK number");
    assert!(err.contains("sdk applies to a wire enum"), "{err}");
}

// Catches: a typed_dict argument passed positionally instead of as the call's keyword arguments,
// or `kwargs` accepted on what is not an owned record.
#[test]
fn a_kwargs_parameter_spreads_its_typed_dict() {
    let source = format!(
        "#[crate::host(record(module = \"opts\", typed_dict, emit))] pub struct Opts {{ pub topic: String }}\n{}",
        proxy(
            "#[host(await, name = \"sdk.stream\")] async fn stream(&self, cx: &Cx, #[host(kwargs)] opts: Opts) -> Result<(), HostError>;"
        )
    );
    let decls = read(&source).expect("resolves");
    let p = &decls.proxies[0];
    let emitted = crate::proxy::proxy(p).expect("emits");
    assert!(
        emitted.contains("PyAny>; 0] = []")
            && emitted.contains("::pyo3::types::PyDictMethods::update(&kwargs, ("),
        "{emitted}"
    );
    let err = read(&proxy(
        "#[host(await)] async fn stream(&self, cx: &Cx, #[host(kwargs)] opts: &str) -> Result<(), HostError>;",
    ))
    .expect_err("a string is not a typed_dict");
    assert!(err.contains("`kwargs` passes an owned typed_dict record"), "{err}");
}

const ROOM: &str = r#"
#[crate::host(native(refused = "expected a bus"))] pub trait Bus: Send + Sync {}
#[crate::host(class(name = "EventBus", native = Bus))] pub trait EventBus: Send + Sync + Sized + 'static { #[host(new)] fn new() -> Self; }
#[crate::host(proxy(host = "livekit.agents.room_io.Participant"))]
pub trait Participant: Send + Sync { #[host(call, getter = "sdk.identity")] fn identity(&self) -> Result<String, HostError>; }
#[crate::host(proxy(host = "livekit.agents.room_io.Room", events(
    module = "room_events", on = "sdk", entries = RoomEntries, pass(bus = "self.bus"),
    joined(args(participant)),
    changed(args(changed, participant), when(participant = "livekit.rtc.RemoteParticipant")),
    left(args(participant), reason = "participant.reason"),
    dtmf(args(event), digit = "event.digit", who = "event.participant.identity"),
)))]
pub trait Room: Send + Sync { #[host(call, getter = "sdk.name")] fn name(&self) -> Result<String, HostError>; }
#[crate::host(functions)]
pub trait RoomEntries {
    #[host(function, name = "_core_joined")] fn joined(bus: std::sync::Arc<dyn Bus>, participant: std::sync::Arc<dyn Participant>);
    #[host(function, name = "_core_changed")] fn changed(bus: std::sync::Arc<dyn Bus>, participant: std::sync::Arc<dyn Participant>);
    #[host(function, name = "_core_left")] fn left(bus: std::sync::Arc<dyn Bus>, participant: std::sync::Arc<dyn Participant>, #[host(sdk)] reason: State);
    #[host(function, name = "_core_dtmf")] fn dtmf(bus: std::sync::Arc<dyn Bus>, digit: String, who: Option<String>);
}
"#;

// Catches: an SDK event delivered to the wrong entry, an argument passed by position instead of
// by name, an adapter not built over the room, an optional path read through `None`, a filtered
// event delivered anyway, or an entry parameter left without a source.
#[test]
fn sdk_events_subscribe_each_event_to_its_entry() {
    let source = format!("use crate::errors::HostError;\n{STATE}\n{ROOM}");
    let decls = read(&source).expect("resolves");
    let out = super::python(&decls).expect("emits");
    let [module] = out.as_slice() else { panic!("one subscription module") };
    assert_eq!(module.path, "room_events.py");
    let text = &module.text;
    for want in [
        "from livekit import rtc as _rtc\n",
        "def subscribe(room: Room) -> None:",
        "    from livekit.agents import room_io as _room_io\n",
        "    on = room.sdk.on\n    bus = room.bus\n",
        "    def joined(participant: Any) -> None:\n        _core._core_joined(bus, _room_io.Participant(room, participant))\n\n    on(\"joined\", joined)\n",
        "    def changed(_changed: Any, participant: Any) -> None:\n        if not isinstance(participant, _rtc.RemoteParticipant):\n            return\n",
        "_core._core_left(bus, _room_io.Participant(room, participant), participant.reason)",
        "        who_owner = event.participant\n        _core._core_dtmf(bus, event.digit, None if who_owner is None else who_owner.identity)\n",
    ] {
        assert!(text.contains(want), "missing `{want}` in:\n{text}");
    }
    for (from, to, want) in [
        ("#[host(sdk)] reason", "reason", "an SDK number is a #[host(sdk)] parameter"),
        ("digit = \"event.digit\", ", "", "entry parameter `digit` has no source"),
        (
            "    joined(args(participant)),\n",
            "    joined(args(participant)),\n    gone(args(participant)),\n",
            "has no entry `gone`",
        ),
    ] {
        let decls = read(&source.replace(from, to)).expect("resolves");
        let Err(err) = super::python(&decls) else { panic!("expected `{want}`") };
        assert!(err.contains(want), "{err}");
    }
    let err = read(&source.replace("reason = \"participant.reason\"", "reason = \"other.reason\""))
        .expect_err("a path starts at an argument");
    assert!(err.contains("starts at an argument"), "{err}");
}
