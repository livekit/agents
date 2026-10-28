//! Generated speech exports backed by the voice core's text algorithms.

pub mod host;
pub mod markup;
pub mod search;
pub mod stream;
pub mod stream_adapter;
pub mod tokenize;

#[cfg(test)]
mod fallback_tts_tests;
#[cfg(test)]
mod native_tts_tests;

#[cfg(test)]
mod judge_tests;
