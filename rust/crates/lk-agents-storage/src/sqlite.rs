//! Executes the backend-neutral statements on a SQLite connection: what the local backend runs
//! on its files and the storage simulation on an in-memory database. Synchronous; the caller
//! owns the connection and where it runs.

use rusqlite::types::ValueRef;

use crate::StorageError;
use crate::sql::{BatchResult, Row, SqlValue, Statement, StatementResult};

/// A lock another process held past the busy timeout applied nothing: retryable. Anything else
/// keeps SQLite's text under the `sqlite` code.
pub fn sqlite_error(e: rusqlite::Error) -> StorageError {
    match &e {
        rusqlite::Error::SqliteFailure(f, _)
            if matches!(
                f.code,
                rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
            ) =>
        {
            StorageError::Unavailable(e.to_string())
        }
        _ => StorageError::Server { code: "sqlite".into(), message: e.to_string() },
    }
}

fn bind(stmt: &mut rusqlite::Statement<'_>, params: &[SqlValue]) -> Result<(), StorageError> {
    for (i, v) in params.iter().enumerate() {
        let i = i + 1;
        let bound = match v {
            SqlValue::Null => stmt.raw_bind_parameter(i, rusqlite::types::Null),
            SqlValue::Int(n) => stmt.raw_bind_parameter(i, n),
            SqlValue::Double(d) => stmt.raw_bind_parameter(i, d),
            SqlValue::Text(s) => stmt.raw_bind_parameter(i, s),
            SqlValue::Blob(b) => stmt.raw_bind_parameter(i, b),
        };
        bound.map_err(sqlite_error)?;
    }
    Ok(())
}

fn cell(v: ValueRef<'_>) -> Result<SqlValue, StorageError> {
    Ok(match v {
        ValueRef::Null => SqlValue::Null,
        ValueRef::Integer(n) => SqlValue::Int(n),
        ValueRef::Real(d) => SqlValue::Double(d),
        ValueRef::Text(t) => SqlValue::Text(
            std::str::from_utf8(t)
                .map_err(|e| StorageError::Decode(format!("a text cell: {e}")))?
                .to_owned(),
        ),
        ValueRef::Blob(b) => SqlValue::Blob(b.to_vec()),
    })
}

/// One statement: its rows when it has columns (`SELECT`, `RETURNING`), else what it changed.
fn run(conn: &rusqlite::Connection, s: &Statement) -> Result<StatementResult, StorageError> {
    let mut stmt = conn.prepare_cached(&s.sql).map_err(sqlite_error)?;
    bind(&mut stmt, &s.params)?;
    let width = stmt.column_count();
    if width == 0 {
        let changed = stmt.raw_execute().map_err(sqlite_error)?;
        return Ok(StatementResult::Exec {
            rows_affected: changed as i64,
            last_insert_id: conn.last_insert_rowid(),
        });
    }
    let mut rows = stmt.raw_query();
    let mut out = Vec::new();
    while let Some(row) = rows.next().map_err(sqlite_error)? {
        let row = (0..width)
            .map(|i| cell(row.get_ref(i).map_err(sqlite_error)?))
            .collect::<Result<Row, _>>()?;
        out.push(row);
    }
    Ok(StatementResult::Rows(out))
}

pub fn query(conn: &rusqlite::Connection, s: &Statement) -> Result<Vec<Row>, StorageError> {
    match run(conn, s)? {
        StatementResult::Rows(rows) => Ok(rows),
        StatementResult::Exec { .. } => Ok(Vec::new()),
    }
}

/// The statements in one immediate transaction: all committed, or none.
pub fn batch(
    conn: &mut rusqlite::Connection,
    statements: &[Statement],
) -> Result<BatchResult, StorageError> {
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(sqlite_error)?;
    let results = statements.iter().map(|s| run(&tx, s)).collect::<Result<Vec<_>, _>>()?;
    tx.commit().map_err(sqlite_error)?;
    Ok(BatchResult { results })
}
