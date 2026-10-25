//! The SQL machinery: typed row reads ([`FromCell`], [`Cells`]), the [`Statement`] trait and its
//! `statement!` macro, and the compile-time checks of statements and records.

use serde_json::Value;

use crate::StorageError;
use crate::conn::{Row, SqlValue};

/// Typed reads of one row's cells, positional, each failure naming the column.
pub struct Cells<'a> {
    row: &'a Row,
    columns: &'static [&'static str],
}

impl<'a> Cells<'a> {
    pub fn new(row: &'a Row, columns: &'static [&'static str]) -> Result<Self, StorageError> {
        if row.len() != columns.len() {
            return Err(StorageError::Decode(format!(
                "expected {} columns, got {}",
                columns.len(),
                row.len()
            )));
        }
        Ok(Self { row, columns })
    }

    /// Cell `i`, typed; a mismatch names the column.
    pub fn get<T: FromCell>(&self, i: usize) -> Result<T, StorageError> {
        T::from_cell(&self.row[i]).map_err(|expected| {
            StorageError::Decode(format!("{}: expected {expected}", self.columns[i]))
        })
    }
}

/// A column's Rust type, read from its cell; the error is what was expected.
pub trait FromCell: Sized {
    fn from_cell(cell: &SqlValue) -> Result<Self, String>;
}

impl FromCell for i64 {
    fn from_cell(cell: &SqlValue) -> Result<Self, String> {
        cell.as_i64().ok_or_else(|| "an integer".into())
    }
}

impl FromCell for String {
    fn from_cell(cell: &SqlValue) -> Result<Self, String> {
        cell.as_str().map(str::to_owned).ok_or_else(|| "text".into())
    }
}

/// A JSONB column, read as `json(col)` text.
impl FromCell for Value {
    fn from_cell(cell: &SqlValue) -> Result<Self, String> {
        let text = cell.as_str().ok_or("JSON text")?;
        serde_json::from_str(text).map_err(|e| format!("JSON: {e}"))
    }
}

/// A typed JSONB column.
macro_rules! json_cell {
    ($($t:ty),*) => {$(
        impl $crate::sql::FromCell for $t {
            fn from_cell(cell: &$crate::conn::SqlValue) -> Result<Self, String> {
                serde_json::from_value(<serde_json::Value as $crate::sql::FromCell>::from_cell(cell)?).map_err(|e| format!("JSON: {e}"))
            }
        }
    )*};
}

impl<T: FromCell> FromCell for Option<T> {
    fn from_cell(cell: &SqlValue) -> Result<Self, String> {
        if cell.is_null() { Ok(None) } else { T::from_cell(cell).map(Some) }
    }
}

/// Whether `column` is a column of `table` in `ddl`, JSONB exactly when `json`, nullable exactly
/// when `nullable`. Each `#[table]` record asserts it of each field at compile time.
pub const fn is_column(ddl: &str, table: &str, column: &str, json: bool, nullable: bool) -> bool {
    let (d, t, c) = (ddl.as_bytes(), table.as_bytes(), column.as_bytes());
    let mut i = 0;
    while !(at(d, i, b"CREATE TABLE ") && at(d, i + 13, t) && at(d, i + 13 + t.len(), b" (")) {
        i += 1;
        if i >= d.len() {
            return false;
        }
    }
    // one column per line until the table's `);`
    loop {
        while i < d.len() && d[i] != b'\n' {
            i += 1;
        }
        while i < d.len() && (d[i] == b'\n' || d[i] == b' ') {
            i += 1;
        }
        if i >= d.len() || at(d, i, b");") {
            return false;
        }
        if at(d, i, c) && i + c.len() < d.len() && d[i + c.len()] == b' ' {
            let mut j = i + c.len();
            while d[j] == b' ' {
                j += 1;
            }
            let is_json = at(d, j, b"JSONB");
            let mut not_null = false;
            while j < d.len() && d[j] != b'\n' && !at(d, j, b"--") {
                not_null = not_null || at(d, j, b"NOT NULL") || at(d, j, b"PRIMARY KEY");
                j += 1;
            }
            return is_json == json && not_null != nullable;
        }
    }
}

const fn at(d: &[u8], i: usize, pat: &[u8]) -> bool {
    if i + pat.len() > d.len() {
        return false;
    }
    let mut k = 0;
    while k < pat.len() {
        if d[i + k] != pat[k] {
            return false;
        }
        k += 1;
    }
    true
}

/// One statement: its SQL, with `:field` placeholders, and the values it binds, as fields.
pub(crate) trait Statement {
    /// As written, with `:field` placeholders.
    const SQL: &'static str;
    /// The fields, in the order their values are sent.
    const FIELDS: &'static [&'static str];
    /// As sent: each `:field` as `?N`.
    fn sql() -> &'static str;
    fn params(self) -> Vec<SqlValue>;
}

pub(crate) fn wire<S: Statement>(s: S) -> crate::conn::wire::Statement {
    crate::conn::statement(S::sql(), s.params())
}

/// [`Statement::SQL`] with each `:field` as `?N`, `N` the field's position in `fields`.
pub(crate) fn numbered(sql: &str, fields: &[&str]) -> String {
    let mut out = String::with_capacity(sql.len());
    let mut rest = sql;
    while let Some(at) = rest.find(':') {
        out.push_str(&rest[..at]);
        let tail = &rest[at + 1..];
        let len =
            tail.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(tail.len());
        let name = &tail[..len];
        let i = fields
            .iter()
            .position(|f| *f == name)
            .unwrap_or_else(|| panic!("`:{name}` is not a field of the statement {sql}"));
        out.push_str(&format!("?{}", i + 1));
        rest = &tail[len..];
    }
    out.push_str(rest);
    out.trim().to_owned()
}

/// Whether `sql`'s `:placeholders` are exactly `fields`: each placeholder a field, each field used.
/// Checked when a statement is declared, so a mismatch does not compile.
pub(crate) const fn placeholders_are(sql: &str, fields: &[&str]) -> bool {
    let sql = sql.as_bytes();
    let mut used = [false; 16];
    let mut i = 0;
    while i < sql.len() {
        if sql[i] == b':' {
            let start = i + 1;
            let mut end = start;
            while end < sql.len() && (sql[end].is_ascii_alphanumeric() || sql[end] == b'_') {
                end += 1;
            }
            let mut f = 0;
            while f < fields.len()
                && !(end - start == fields[f].len() && at(sql, start, fields[f].as_bytes()))
            {
                f += 1;
            }
            if f == fields.len() {
                return false;
            }
            used[f] = true;
            i = end;
        } else {
            i += 1;
        }
    }
    let mut f = 0;
    while f < fields.len() {
        if !used[f] {
            return false;
        }
        f += 1;
    }
    true
}

/// Declares a statement: `Name<'a> { field: Type, .. } => "SQL with :field placeholders"`. A JSON
/// column's field is a [`SqlValue`] made with [`SqlValue::json`].
macro_rules! statement {
    (
        $(#[$doc:meta])*
        $name:ident $(<$lt:lifetime>)? { $($field:ident: $ty:ty),* $(,)? } => $sql:literal
    ) => {
        $(#[$doc])*
        pub(crate) struct $name$(<$lt>)? {
            $(pub $field: $ty,)*
        }

        const _: () = assert!(
            $crate::sql::placeholders_are($sql, &[$(stringify!($field)),*]),
            concat!(stringify!($name), ": every :placeholder must be a field and every field used"),
        );

        impl$(<$lt>)? $crate::sql::Statement for $name$(<$lt>)? {
            const SQL: &'static str = $sql;
            const FIELDS: &'static [&'static str] = &[$(stringify!($field)),*];
            fn sql() -> &'static str {
                static SENT: std::sync::LazyLock<String> =
                    std::sync::LazyLock::new(|| $crate::sql::numbered(<$name>::SQL, <$name>::FIELDS));
                &SENT
            }
            fn params(self) -> Vec<$crate::conn::SqlValue> {
                vec![$(self.$field.into()),*]
            }
        }
    };
}

pub(crate) use {json_cell, statement};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::THREAD_SCHEMA;

    #[test]
    fn a_column_matches_by_name_nullability_and_json() {
        let ddl = THREAD_SCHEMA;
        assert!(is_column(ddl, "tasks", "ask", true, true));
        assert!(!is_column(ddl, "tasks", "ask", false, true), "JSONB read as text");
        assert!(!is_column(ddl, "tasks", "status", false, true), "NOT NULL read as an Option");
        assert!(!is_column(ddl, "tasks", "nope", false, false));
        assert!(!is_column(ddl, "items", "status", false, false), "another table's column");
        assert!(is_column(ddl, "events", "id", false, false), "a primary key is not null");
    }

    #[test]
    fn placeholders_are_fields_and_sent_numbered() {
        assert!(placeholders_are("x = :a AND y = :b AND z = :a", &["a", "b"]));
        assert!(!placeholders_are("x = :a", &["a", "b"]), "an unused field");
        assert!(!placeholders_are("x = :ab", &["a"]), "a placeholder that is no field");
        assert_eq!(
            numbered("x = :b AND y = :a AND z = :b", &["a", "b"]),
            "x = ?2 AND y = ?1 AND z = ?2"
        );
    }
}
