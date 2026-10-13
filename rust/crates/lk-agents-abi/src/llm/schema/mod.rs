//! `llm/utils.py`'s schema half: pure JSON transforms. The pydantic steps stay in the host; Rust
//! starts from the emitted schema.

mod defaults;
mod openai;

pub use defaults::{SchemaError, inject_schema_defaults, inject_schema_defaults_of};
pub use openai::{
    build_legacy_openai_schema, build_strict_openai_schema, to_openai_response_format,
};
