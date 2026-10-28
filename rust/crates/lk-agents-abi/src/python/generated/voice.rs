//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `voice` records.

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::avatar::ClearBufferListener {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::avatar::ClearBufferListener>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`ClearBufferListener` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::avatar::AvatarAudioReceiver>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`AvatarAudioReceiver` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::avatar::AvatarVideoGenerator>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`AvatarVideoGenerator` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/avatar.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::avatar::AvatarMedia {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::avatar::AvatarMedia>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`AvatarMedia` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/background_audio.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::background_audio::FrameStream {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::background_audio::FrameStream>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`FrameStream` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::io::PlaybackListener {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::io::PlaybackListener>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`PlaybackListener` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::io::AudioInput {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::io::AudioInput>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`AudioInput` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::io::VideoInput {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::io::VideoInput>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`VideoInput` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::io::AudioOutput {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::io::AudioOutput>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`AudioOutput` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/io.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::io::TextOutput {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::io::TextOutput>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`TextOutput` is a declared proxy: it is taken from the host object, not a view"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/voice/session_effects.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::voice::session_effects::SessionHooks {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::voice::session_effects::SessionHooks>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`SessionHooks` is a declared proxy: it is taken from the host object, not a view"))
    }
}
