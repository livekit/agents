//! `#[table("name")]` on a `#[derive(Host)]` record: the record is a row of that table of
//! livekit/protocol `livekit/agent/thread_schema_v1.sql`, one field per column, named as the column.
//!
//! The record implements the calling crate's `sql::Table`: `COLUMNS`, `SELECT` (the column list a
//! read uses, `json(col)` for a JSONB column) and `from_row`. Each field is checked against the DDL
//! when the crate compiles: a column of the table, nullable exactly when the field is an `Option`,
//! JSONB exactly when the field is read as JSON (a `serde_json::Value`, or a typed field marked
//! `#[table(json)]`).
//!
//! `#[table("tasks", remote = path::TaskRow)]` declares the columns of a record defined in another
//! crate (`#[derive(Table)]` on a column declaration whose fields mirror it): the impl is for
//! `path::TaskRow`, built field by field, so a field the two disagree on fails to compile.

use proc_macro2::TokenStream;
use quote::quote;

/// Nothing when the record carries no `#[table]`.
pub fn table(input: &syn::DeriveInput) -> syn::Result<TokenStream> {
    let Some(attr) = input.attrs.iter().find(|a| a.path().is_ident("table")) else {
        return Ok(TokenStream::new());
    };
    let (table, remote) = attr.parse_args_with(|input: syn::parse::ParseStream| {
        let table = input.parse::<syn::LitStr>()?.value();
        let mut remote = None;
        if input.parse::<Option<syn::Token![,]>>()?.is_some() {
            let key = input.parse::<syn::Ident>()?;
            if key != "remote" {
                return Err(syn::Error::new_spanned(key, "expected `remote = path`"));
            }
            input.parse::<syn::Token![=]>()?;
            remote = Some(input.parse::<syn::Path>()?);
        }
        Ok((table, remote))
    })?;
    let syn::Data::Struct(syn::DataStruct { fields: syn::Fields::Named(fields), .. }) = &input.data
    else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "#[table] is for a struct with named fields",
        ));
    };
    let columns = fields.named.iter().map(column).collect::<syn::Result<Vec<_>>>()?;
    let column_names: Vec<&str> = columns.iter().map(|c| c.name.as_str()).collect();
    let select = columns
        .iter()
        .map(|c| if c.json { format!("json({})", c.name) } else { c.name.clone() })
        .collect::<Vec<_>>()
        .join(", ");
    let reads = columns.iter().enumerate().map(|(i, c)| {
        let field = &c.field;
        quote! { #field: cells.get(#i)? }
    });
    let ident = &input.ident;
    let target = match &remote {
        Some(path) => quote! { #path },
        None => quote! { #ident },
    };
    let checks = columns.iter().map(|c| {
        let (name, json, nullable) = (&c.name, c.json, c.nullable);
        let message = format!(
            "{ident}.{name}: the `{table}` table of the thread schema has no {}{} column `{name}`",
            if nullable { "nullable" } else { "NOT NULL" },
            if json { " JSONB" } else { "" },
        );
        quote! {
            const _: () = assert!(
                crate::sql::is_column(crate::thread::schema::THREAD_SCHEMA, #table, #name, #json, #nullable),
                #message,
            );
        }
    });
    Ok(quote! {
        #(#checks)*

        impl crate::sql::Table for #target {
            const COLUMNS: &'static [&'static str] = &[#(#column_names),*];
            const SELECT: &'static str = #select;

            fn from_row(row: &crate::sql::Row) -> Result<Self, crate::StorageError> {
                let cells = crate::sql::Cells::new(row, Self::COLUMNS)?;
                Ok(#target { #(#reads),* })
            }
        }
    })
}

struct Column {
    field: syn::Ident,
    name: String,
    json: bool,
    nullable: bool,
}

fn column(f: &syn::Field) -> syn::Result<Column> {
    let ident = f.ident.clone().expect("named field");
    let name = ident.to_string().trim_start_matches("r#").to_owned();
    let mut json_flag = false;
    for a in f.attrs.iter().filter(|a| a.path().is_ident("table")) {
        a.parse_nested_meta(|m| {
            if m.path.is_ident("json") {
                json_flag = true;
                Ok(())
            } else {
                Err(m.error("expected `json`"))
            }
        })?;
    }
    let nullable = option_inner(&f.ty).is_some();
    let ty = option_inner(&f.ty).unwrap_or(&f.ty);
    let json = json_flag
        || matches!(ty, syn::Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "Value"));
    Ok(Column { field: ident, name, json, nullable })
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
