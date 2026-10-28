//! Image, tool and formatting error boundary contracts.
mod error;
pub mod host;
pub mod image;
mod strict;
pub mod tools;
pub use error::FormatError;
pub use image::{BYTES_TAG, ImageResolver, SerializedImage};
pub use strict::{MAX_DEPTH, StrictError};
