//! `#[table]` on a `#[derive(Host)]` record: the record is a row of a SQLite table whose DDL is
//! livekit/protocol `livekit/agent/thread_schema_v1.sql`.
//!
//! The record gets two inherent consts: `COLUMNS`, the column names in declaration order, and
//! `SELECT`, the column list a read uses, `json(col)` for a `serde_json::Value` (JSONB) column.
//! `#[host(rename = "type")]` names the column when the Rust field cannot.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Expr, Token};

/// Nothing when the record carries no `#[table]`.
pub fn table(input: &syn::DeriveInput) -> syn::Result<TokenStream> {
    if !input.attrs.iter().any(|a| a.path().is_ident("table")) {
        return Ok(TokenStream::new());
    }
    let syn::Data::Struct(syn::DataStruct { fields: syn::Fields::Named(fields), .. }) = &input.data
    else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "#[table] is for a struct with named fields",
        ));
    };
    let columns = fields.named.iter().map(column).collect::<syn::Result<Vec<_>>>()?;
    let column_names: Vec<&str> = columns.iter().map(|(name, _)| name.as_str()).collect();
    let select = columns
        .iter()
        .map(|(name, json)| if *json { format!("json({name})") } else { name.clone() })
        .collect::<Vec<_>>()
        .join(", ");
    let ident = &input.ident;
    Ok(quote! {
        impl #ident {
            /// The columns, in declaration order.
            pub const COLUMNS: &'static [&'static str] = &[#(#column_names),*];
            /// The column list a read selects: `json(col)` for a JSONB column.
            pub const SELECT: &'static str = #select;
        }
    })
}

/// The column name, and whether it holds JSON.
fn column(f: &syn::Field) -> syn::Result<(String, bool)> {
    let ident = f.ident.as_ref().expect("named field");
    let mut name = ident.to_string().trim_start_matches("r#").to_owned();
    for a in f.attrs.iter().filter(|a| lk_agents_bindgen::is_host(a)) {
        a.parse_nested_meta(|m| {
            if m.path.is_ident("rename") {
                name = m.value()?.parse::<syn::LitStr>()?.value();
            } else if m.input.peek(Token![=]) {
                let _: Expr = m.value()?.parse()?;
            }
            Ok(())
        })?;
    }
    let ty = option_inner(&f.ty).unwrap_or(&f.ty);
    let json = matches!(ty, syn::Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "Value"));
    Ok((name, json))
}

fn option_inner(t: &syn::Type) -> Option<&syn::Type> {
    let syn::Type::Path(p) = t else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return None };
    match a.args.first()? {
        syn::GenericArgument::Type(t) => Some(t),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(src: &str) -> String {
        let input: syn::DeriveInput = syn::parse_str(src).unwrap();
        table(&input).unwrap().to_string()
    }

    #[test]
    fn select_follows_the_fields() {
        let out = render(
            r#"
            #[table]
            struct LaneRow {
                lane: String,
                state: Option<Value>,
                #[host(rename = "type")]
                r#type: String,
            }
            "#,
        );
        assert!(out.contains("\"lane\" , \"state\" , \"type\""), "{out}");
        assert!(out.contains("\"lane, json(state), type\""), "{out}");
    }

    #[test]
    fn a_record_without_table_renders_nothing() {
        assert!(render("struct Plain { a: String }").is_empty());
    }
}
