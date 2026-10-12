//! `utils/images/` plus `serialize_image`: the driver resolves a `MediaRef` to host-converted
//! [`RgbaFrame`], this encodes it, and `lk-agents-format` gets a plain value, never an encoder.
//! `ImageContent._cache` is not ported: its key ignores the inference size (a live bug upstream).

mod encode;
mod serialize;

pub use encode::{
    EncodeOptions, ImageFormat, MediaError, ResizeOptions, ResizeStrategy, RgbaFrame, encode,
    encode_image,
};
pub use serialize::{
    FRAME_MIME_TYPE, MimeMismatch, SUPPORTED_DATA_URL_MIME_TYPES, serialize_image,
    serialize_image_frame, serialize_image_url,
};
