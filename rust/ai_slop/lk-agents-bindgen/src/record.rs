//! What a record's fields inherit from the record, read the same way by every declaration reader.

use crate::Metas;

/// What a field without its own word inherits from its record, read the same way by the binding
/// and by `lk-agents-codegen`:
///
/// * `#[binding(default)]` on the record: every field reads as `default`;
/// * a `#[python(typed_dict, total = "False")]` record with a read target: the same (any key may be
///   absent);
/// * `#[binding(skip_if_none)]` on the record: every `Option` field is `skip_if_none`;
/// * `#[python(zero_defaults)]`: every number and `Option` field defaults to its `Default` on the
///   host (`0`, `0.0`, `None`), unless it is `#[python(required)]` or says its own.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecordDefaults {
    pub default: bool,
    pub skip_if_none: bool,
    pub zero: bool,
}

impl RecordDefaults {
    pub fn of(attrs: &[syn::Attribute]) -> syn::Result<RecordDefaults> {
        let b = Metas::named(attrs, "binding")?;
        let py = Metas::named(attrs, "python")?;
        let open_dict = py.flag("typed_dict") && py.text("total")? == Some("False");
        Ok(RecordDefaults {
            default: b.flag("default") || (open_dict && b.flag("extract")),
            skip_if_none: b.flag("skip_if_none"),
            zero: py.flag("zero_defaults"),
        })
    }
}
