//! Executes the backend-neutral statements on a SQLite connection: what the local backend runs
//! on its files and the storage simulation on an in-memory database. Synchronous; the caller
//! owns the connection and where it runs.

use rusqlite::types::ValueRef;

use crate::StorageError;
use crate::sql::{BatchResult, Row, SqlValue, Statement, StatementResult};

/// The database's own failure (a lock held past the busy timeout, I/O, a full disk, corruption,
/// memory) applied nothing and is retryable: `Unavailable`. What the caller asked wrongly (a
/// refused constraint, which is how a fence aborts a batch, a type mismatch, SQL it rejects)
/// stays `Invalid`.
pub fn sqlite_error(e: rusqlite::Error) -> StorageError {
    use rusqlite::ErrorCode as Code;
    match &e {
        rusqlite::Error::SqliteFailure(f, _)
            if matches!(
                f.code,
                Code::DatabaseBusy
                    | Code::DatabaseLocked
                    | Code::SystemIoFailure
                    | Code::DiskFull
                    | Code::DatabaseCorrupt
                    | Code::NotADatabase
                    | Code::CannotOpen
                    | Code::ReadOnly
                    | Code::PermissionDenied
                    | Code::FileLockingProtocolFailed
                    | Code::NoLargeFileSupport
                    | Code::OutOfMemory
                    | Code::InternalMalfunction
                    | Code::SchemaChanged
            ) =>
        {
            StorageError::Unavailable(format!("sqlite: {e}"))
        }
        _ => StorageError::Invalid(format!("sqlite: {e}")),
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
                .map_err(|e| StorageError::Unavailable(format!("a text cell: {e}")))?
                .to_owned(),
        ),
        ValueRef::Blob(b) => SqlValue::Blob(b.to_vec()),
    })
}

/// One statement: its rows when it has columns (`SELECT`, `RETURNING`), else what it changed.
fn run(
    conn: &rusqlite::Connection,
    s: &Statement,
    read_only: bool,
) -> Result<StatementResult, StorageError> {
    // SQLite parses trigger bodies and quoted/comment semicolons itself. Preparing the
    // complete tail before stepping prevents silently accepting only the first statement.
    let mut statements = rusqlite::Batch::new(conn, &s.sql);
    let mut stmt = statements
        .next()
        .map_err(sqlite_error)?
        .ok_or_else(|| StorageError::Invalid("expected one SQL statement".into()))?;
    if statements.next().map_err(sqlite_error)?.is_some() {
        return Err(StorageError::Invalid("expected one SQL statement".into()));
    }
    if read_only && (!stmt.readonly() || stmt.column_count() == 0) {
        return Err(StorageError::Invalid("query requires a read-only SQL statement".into()));
    }
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
    match run(conn, s, true)? {
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
    let results = statements.iter().map(|s| run(&tx, s, false)).collect::<Result<Vec<_>, _>>()?;
    tx.commit().map_err(sqlite_error)?;
    Ok(BatchResult { results })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_cannot_change_data_or_schema() {
        let mut conn = rusqlite::Connection::open_in_memory().unwrap();
        batch(
            &mut conn,
            &[
                Statement::raw("CREATE TABLE notes (body TEXT)"),
                Statement::raw("INSERT INTO notes VALUES ('saved')"),
            ],
        )
        .unwrap();
        for sql in [
            "UPDATE notes SET body = 'changed' RETURNING body",
            "DELETE FROM notes RETURNING body",
            "INSERT INTO notes VALUES ('new')",
            "CREATE TABLE unwanted (body TEXT)",
            "DROP TABLE notes",
        ] {
            assert!(query(&conn, &Statement::raw(sql)).is_err(), "accepted {sql}");
            assert_eq!(
                query(&conn, &Statement::raw("SELECT body FROM notes")).unwrap(),
                vec![vec![SqlValue::Text("saved".into())]]
            );
        }
        assert_eq!(
            query(
                &conn,
                &Statement::raw("SELECT name FROM sqlite_master WHERE name = 'unwanted'"),
            )
            .unwrap(),
            Vec::<Row>::new(),
        );
    }

    #[test]
    fn a_statement_tail_is_refused_before_either_write() {
        let mut conn = rusqlite::Connection::open_in_memory().unwrap();
        batch(&mut conn, &[Statement::raw("CREATE TABLE notes (body TEXT)")]).unwrap();
        let sql = "INSERT INTO notes VALUES ('first'); INSERT INTO notes VALUES ('second')";
        assert!(batch(&mut conn, &[Statement::raw(sql)]).is_err());
        assert!(query(&conn, &Statement::raw(sql)).is_err());
        assert!(query(&conn, &Statement::raw("SELECT body FROM notes")).unwrap().is_empty());

        // SQLite owns parsing: semicolons in identifiers, strings, comments and trigger
        // bodies remain part of the single statement.
        batch(
            &mut conn,
            &[
                Statement::raw("CREATE TABLE audit (body TEXT)"),
                Statement::raw(
                    "CREATE TRIGGER \"changed;\" AFTER INSERT ON notes BEGIN \
                     INSERT INTO audit VALUES ('first;'); \
                     INSERT INTO audit VALUES ('second;'); END; -- tail;",
                ),
                Statement::raw("INSERT INTO notes VALUES ('one;two'); /* comment; */ ;"),
            ],
        )
        .unwrap();
        assert_eq!(
            query(&conn, &Statement::raw("SELECT body FROM audit ORDER BY rowid")).unwrap(),
            vec![vec![SqlValue::Text("first;".into())], vec![SqlValue::Text("second;".into())]]
        );
    }
}
