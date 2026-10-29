//! Application SQL (docs/context.md#application-records-and-transactions). Both data planes
//! execute reads and atomic batches, never an interactive transaction: reads observe committed
//! records, writes are staged and applied together when the transaction ends, so a read after a
//! write in the same transaction is refused rather than answered without that write.

use std::sync::Mutex;

use lk_agents_abi::storage::context::{AppStatement, AppTransaction};
use serde_json::{Number, Value};

use super::store::{ContextStore, Database};
use crate::StorageError;
use crate::sql::{SqlValue, Statement};

/// One application transaction: independent, or joining the managed turn it began in.
pub struct Transaction {
    database: Database,
    /// The managed turn the writes join, and that turn's generation.
    joins: Option<(ContextStore, u64)>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    staged: Vec<AppStatement>,
    ended: bool,
}

impl Transaction {
    pub(crate) fn new(database: Database, joins: Option<(ContextStore, u64)>) -> Self {
        Self { database, joins, state: Mutex::default() }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("application transaction")
    }

    fn ensure_open(state: &State) -> Result<(), StorageError> {
        if state.ended {
            return Err(StorageError::Invalid("the transaction has ended".into()));
        }
        Ok(())
    }
}

impl AppTransaction for Transaction {
    async fn execute(&self, sql: String, params: Vec<Value>) -> Result<(), StorageError> {
        let write = AppStatement { sql, params };
        statement(&write)?;
        let mut state = self.state();
        Self::ensure_open(&state)?;
        state.staged.push(write);
        Ok(())
    }

    async fn fetch(
        &self,
        sql: String,
        params: Vec<Value>,
    ) -> Result<Vec<Vec<Value>>, StorageError> {
        let read = statement(&AppStatement { sql, params })?;
        {
            let state = self.state();
            Self::ensure_open(&state)?;
            if !state.staged.is_empty() {
                return Err(StorageError::Invalid(
                    "a transaction reads committed records only: read before its first write, \
                     or in a transaction after this one commits"
                        .into(),
                ));
            }
        }
        let rows = self.database.query(read).await?;
        rows.iter().map(|row| row.iter().map(cell).collect()).collect()
    }

    async fn commit(&self) -> Result<(), StorageError> {
        let staged = {
            let mut state = self.state();
            Self::ensure_open(&state)?;
            state.ended = true;
            std::mem::take(&mut state.staged)
        };
        if staged.is_empty() {
            return Ok(());
        }
        match &self.joins {
            Some((store, turn)) => store.join_turn(*turn, staged).await,
            None => {
                let statements = staged.iter().map(statement).collect::<Result<_, _>>()?;
                self.database.batch(statements).await
            }
        }
    }

    fn rollback(&self) {
        let mut state = self.state();
        state.ended = true;
        state.staged.clear();
    }
}

/// An application statement as the data plane executes it.
pub fn statement(s: &AppStatement) -> Result<Statement, StorageError> {
    let params = s.params.iter().map(param).collect::<Result<_, _>>()?;
    Ok(Statement { sql: s.sql.clone().into(), params })
}

fn param(value: &Value) -> Result<SqlValue, StorageError> {
    let invalid = || {
        StorageError::Invalid(format!(
            "SQL parameter {value}: expected null, a boolean, an integer, a float or a string"
        ))
    };
    Ok(match value {
        Value::Null => SqlValue::Null,
        Value::Bool(b) => (*b).into(),
        Value::Number(n) if n.is_f64() => SqlValue::Double(n.as_f64().ok_or_else(invalid)?),
        Value::Number(n) => SqlValue::Int(n.as_i64().ok_or_else(invalid)?),
        Value::String(s) => SqlValue::Text(s.clone()),
        Value::Array(_) | Value::Object(_) => return Err(invalid()),
    })
}

fn cell(value: &SqlValue) -> Result<Value, StorageError> {
    Ok(match value {
        SqlValue::Null => Value::Null,
        SqlValue::Int(i) => (*i).into(),
        SqlValue::Double(d) => Number::from_f64(*d).map_or(Value::Null, Value::Number),
        SqlValue::Text(s) => s.clone().into(),
        SqlValue::Blob(_) => {
            return Err(StorageError::Invalid(
                "a BLOB column cannot be read as an application row value: select hex(...)".into(),
            ));
        }
    })
}
