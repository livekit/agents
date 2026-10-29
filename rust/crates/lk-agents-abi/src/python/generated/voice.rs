//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `voice` proxy bridges and in-place records.

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::avatar::ClearBufferListener {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::avatar::AvatarMedia {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/background_audio.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::background_audio::FrameStream {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::background_audio::FrameStream>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::io::PlaybackListener {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::io::PlaybackListener>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::io::AudioInput {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::io::AudioInput>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::io::VideoInput {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::io::VideoInput>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::io::AudioOutput {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::io::AudioOutput>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::io::TextOutput {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::io::TextOutput>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_effects.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::session_effects::SessionHooks {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>(obj)
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_effects.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::session_effects::SessionEvents {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::session_effects::SessionEvents>(obj)
    }
}
