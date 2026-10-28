//! The backend-neutral SQL vocabulary: bound values ([`SqlValue`]), executable [`Statement`]s and
//! their results, typed row reads ([`FromCell`], [`Cells`]), the [`Bind`] trait with its
//! `statement!` macro, and the compile-time checks of statements and records. Both backends
//! execute these as they are; neither's wire or driver types appear here.

use std::borrow::Cow;

use serde::Serialize;
use serde_json::Value;

use crate::StorageError;

mod script;
pub use script::schema_statements;

/// One SQLite storage class, as a bind parameter or a cell.
#[derive(Debug, Clone, PartialEq)]
pub enum SqlValue {
    Null,
    Int(i64),
    Double(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl SqlValue {
    /// A JSON column's bind value: its text, for `jsonb(..)`.
    pub fn json<T: Serialize + ?Sized>(v: &T) -> SqlValue {
        SqlValue::Text(serde_json::to_string(v).expect("a JSON-shaped value"))
    }

    /// `NULL` for `None`, else [`SqlValue::json`].
    pub fn opt_json<T: Serialize>(v: Option<&T>) -> SqlValue {
        v.map_or(SqlValue::Null, SqlValue::json)
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            SqlValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            SqlValue::Text(s) => Some(s),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, SqlValue::Null)
    }
}

impl From<&str> for SqlValue {
    fn from(s: &str) -> Self {
        SqlValue::Text(s.to_owned())
    }
}

impl From<String> for SqlValue {
    fn from(s: String) -> Self {
        SqlValue::Text(s)
    }
}

impl From<i64> for SqlValue {
    fn from(i: i64) -> Self {
        SqlValue::Int(i)
    }
}

impl From<bool> for SqlValue {
    fn from(b: bool) -> Self {
        SqlValue::Int(i64::from(b))
    }
}

impl From<f64> for SqlValue {
    fn from(d: f64) -> Self {
        SqlValue::Double(d)
    }
}

impl From<Vec<u8>> for SqlValue {
    fn from(b: Vec<u8>) -> Self {
        SqlValue::Blob(b)
    }
}

impl<T: Into<SqlValue>> From<Option<T>> for SqlValue {
    fn from(v: Option<T>) -> Self {
        v.map_or(SqlValue::Null, Into::into)
    }
}

pub type Row = Vec<SqlValue>;

/// The first cell of the first row: a `RETURNING version`, a `COUNT(*)`, a single column read.
pub fn first_cell(rows: &[Row]) -> Option<&SqlValue> {
    rows.first().and_then(|row| row.first())
}

/// One executable statement: SQL with positional `?N` parameters and the values they bind.
#[derive(Debug, Clone, PartialEq)]
pub struct Statement {
    pub sql: Cow<'static, str>,
    pub params: Vec<SqlValue>,
}

impl Statement {
    /// A statement that binds nothing: DDL.
    pub fn raw(sql: &'static str) -> Self {
        Self { sql: Cow::Borrowed(sql), params: Vec::new() }
    }
}

/// One statement's answer in a batch: the rows it returned, or the rows it changed.
#[derive(Debug, Clone, PartialEq)]
pub enum StatementResult {
    Rows(Vec<Row>),
    Exec { rows_affected: i64, last_insert_id: i64 },
}

impl StatementResult {
    /// The rows of a statement that returned some; empty for one that did not.
    pub fn rows(&self) -> &[Row] {
        match self {
            StatementResult::Rows(r) => r,
            StatementResult::Exec { .. } => &[],
        }
    }

    pub fn rows_affected(&self) -> i64 {
        match self {
            StatementResult::Rows(_) => 0,
            StatementResult::Exec { rows_affected, .. } => *rows_affected,
        }
    }

    /// The first cell as an integer: a `RETURNING version`, a `COUNT(*)`.
    pub fn first_int(&self) -> Option<i64> {
        first_cell(self.rows()).and_then(SqlValue::as_i64)
    }
}

/// An atomic batch's answer: one result per statement, in order. Its existence means the batch
/// committed.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchResult {
    pub results: Vec<StatementResult>,
}

impl BatchResult {
    /// The result of the statement at `i`; a missing one is a malformed answer, never a
    /// zero-row success.
    pub fn at(&self, i: usize) -> Result<&StatementResult, StorageError> {
        self.results.get(i).ok_or_else(|| {
            StorageError::Protocol(format!(
                "a batch answered {} results, statement {i} has none",
                self.results.len()
            ))
        })
    }
}

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
/// A record read from one table's rows: `#[table]` (lk-agents-host-derive) implements it.
pub trait Table: Sized {
    /// The columns, in declaration order.
    const COLUMNS: &'static [&'static str];
    /// The column list a read selects: `json(col)` for a JSONB column.
    const SELECT: &'static str;
    /// The record of a row read with [`Self::SELECT`].
    fn from_row(row: &Row) -> Result<Self, StorageError>;
}

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
            fn from_cell(cell: &$crate::sql::SqlValue) -> Result<Self, String> {
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

/// A declared statement: its SQL, with `:field` placeholders, and the values it binds, as fields.
pub(crate) trait Bind {
    /// As written, with `:field` placeholders.
    const SQL: &'static str;
    /// The fields, in the order their values are sent.
    const FIELDS: &'static [&'static str];
    /// As sent: each `:field` as `?N`.
    fn sql() -> &'static str;
    /// The executable statement with this declaration's values.
    fn bind(self) -> Statement;
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

        impl$(<$lt>)? $crate::sql::Bind for $name$(<$lt>)? {
            const SQL: &'static str = $sql;
            const FIELDS: &'static [&'static str] = &[$(stringify!($field)),*];
            fn sql() -> &'static str {
                static SENT: std::sync::LazyLock<String> =
                    std::sync::LazyLock::new(|| $crate::sql::numbered(<$name>::SQL, <$name>::FIELDS));
                &SENT
            }
            fn bind(self) -> $crate::sql::Statement {
                $crate::sql::Statement {
                    sql: std::borrow::Cow::Borrowed(<Self as $crate::sql::Bind>::sql()),
                    params: vec![$(self.$field.into()),*],
                }
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
        assert!(is_column(ddl, "tasks", "internal", true, true));
        assert!(!is_column(ddl, "tasks", "internal", false, true), "JSONB read as text");
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
