//! Dedicated scope records, loaded with the lane and saved in its fenced transaction.

use lk_agents_abi::storage::context::{ContextScope, ContextScopeWrite};
use serde_json::Value;

use crate::StorageError;
use crate::db::Fence;
use crate::sql::{Cells, Row, SqlValue, Statement};

// Scope records are framework-private schema, independent of the platform's v1 thread tables.
pub const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS context_scopes (
    scope_id TEXT PRIMARY KEY,
    lane TEXT NOT NULL,
    revision INTEGER NOT NULL,
    fields JSONB NOT NULL,
    contracts JSONB NOT NULL,
    manifests JSONB NOT NULL
)";

pub fn select(lane: &str) -> Statement {
    Statement {
        sql: "SELECT scope_id, revision, json(fields), json(contracts), json(manifests) \
              FROM context_scopes WHERE lane = ?1 ORDER BY scope_id"
            .into(),
        params: vec![lane.into()],
    }
}

pub fn from_rows(rows: &[Row]) -> Result<Vec<ContextScope>, StorageError> {
    rows.iter()
        .map(|row| {
            let columns =
                Cells::new(row, &["scope_id", "revision", "fields", "contracts", "manifests"])?;
            let fields: Value = columns.get(2)?;
            let contracts: Value = columns.get(3)?;
            let manifests: Value = columns.get(4)?;
            Ok(ContextScope {
                scope_id: columns.get(0)?,
                revision: columns.get(1)?,
                fields: serde_json::from_value(fields).map_err(decode)?,
                contracts: serde_json::from_value(contracts).map_err(decode)?,
                manifests: serde_json::from_value(manifests).map_err(decode)?,
            })
        })
        .collect()
}

fn decode(error: serde_json::Error) -> StorageError {
    StorageError::Decode(format!("context scope: {error}"))
}

/// A revision mismatch fails a NOT NULL constraint, rolling back the enclosing batch. A
/// mismatched lane fence writes nothing, so the normal commit decoder reports VersionConflict.
pub fn write(
    lane: &str,
    write: &ContextScopeWrite,
    fence: Option<&Fence>,
) -> Result<Statement, StorageError> {
    if write.scope.scope_id.is_empty()
        || write.expected_revision < 0
        || write.scope.revision <= write.expected_revision
    {
        return Err(StorageError::Invalid("invalid context scope revision write".into()));
    }
    Ok(Statement {
        sql: "INSERT INTO context_scopes (scope_id, lane, revision, fields, contracts, manifests) \
              SELECT ?1, ?2, \
                CASE WHEN COALESCE((SELECT revision FROM context_scopes WHERE scope_id = ?1), 0) = ?3 \
                     THEN ?4 ELSE NULL END, jsonb(?5), jsonb(?6), jsonb(?7) \
              WHERE ?8 IS NULL OR EXISTS \
                (SELECT 1 FROM lanes WHERE lane = ?2 AND version = ?8 AND lease_token = ?9) \
              ON CONFLICT(scope_id) DO UPDATE SET \
                revision = CASE WHEN context_scopes.lane = excluded.lane \
                                THEN excluded.revision ELSE NULL END, \
                fields = excluded.fields, contracts = excluded.contracts, manifests = excluded.manifests"
            .into(),
        params: vec![
            write.scope.scope_id.as_str().into(),
            lane.into(),
            write.expected_revision.into(),
            write.scope.revision.into(),
            SqlValue::json(&write.scope.fields),
            SqlValue::json(&write.scope.contracts),
            SqlValue::json(&write.scope.manifests),
            fence.map(|fence| fence.version).into(),
            fence.map(|fence| fence.token.as_str()).into(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ContextActivation;
    use crate::sqlite::{batch, query};
    use lk_agents_abi::storage::context::ContextField;
    use serde_json::json;

    #[test]
    fn scope_saves_are_durable_and_stale_revisions_rollback_the_entire_batch() {
        let mut conn = rusqlite::Connection::open_in_memory().unwrap();
        batch(&mut conn, &crate::db::bootstrap()).unwrap();
        let field = ContextField {
            name: "draft".into(),
            schema: json!({"type": "string"}),
            value: json!("saved"),
        };
        let mut activation = ContextActivation::new(Vec::new(), None, false).unwrap();
        activation.load("root", "Booking", &[field.clone()]).unwrap();
        activation.save("root", "Booking", 0, &[field]).unwrap();
        let first = activation.writes().remove(0);
        batch(&mut conn, &[write("desk", &first, None).unwrap()]).unwrap();
        let saved = from_rows(&query(&conn, &select("desk")).unwrap()).unwrap();
        assert_eq!(saved, vec![first.scope.clone()]);
        assert!(from_rows(&query(&conn, &select("other")).unwrap()).unwrap().is_empty());

        let stale = write("desk", &first, None).unwrap();
        let result = batch(&mut conn, &[Statement::raw("CREATE TABLE partial (body TEXT)"), stale]);
        assert!(result.is_err());
        assert!(
            query(&conn, &Statement::raw("SELECT name FROM sqlite_master WHERE name = 'partial'"))
                .unwrap()
                .is_empty()
        );
        assert_eq!(from_rows(&query(&conn, &select("desk")).unwrap()).unwrap(), saved);
    }
}
