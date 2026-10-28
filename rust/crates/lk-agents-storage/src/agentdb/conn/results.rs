//! Validate one query or atomic batch's ordered response before accepting its commit tip.

use super::{WsdbError, decode_batch, pb};
use crate::sql::{BatchResult, Row, StatementResult};

enum ResultSet {
    Rows { width: usize, rows: Vec<Row> },
    Exec { rows_affected: i64, last_insert_id: i64 },
}

pub(super) struct Results {
    sets: Vec<ResultSet>,
    expected: usize,
    rows: u64,
}

impl Results {
    pub(super) fn new(expected: usize) -> Self {
        Self { sets: Vec::new(), expected, rows: 0 }
    }

    fn start(&self, statement: u32) -> Result<(), WsdbError> {
        if statement as usize != self.sets.len() || self.sets.len() >= self.expected {
            return Err(WsdbError::Protocol(format!(
                "result for statement {statement}, expected {} of {}",
                self.sets.len(),
                self.expected,
            )));
        }
        Ok(())
    }

    pub(super) fn columns(&mut self, columns: pb::Columns) -> Result<(), WsdbError> {
        self.start(columns.statement)?;
        self.sets.push(ResultSet::Rows { width: columns.names.len(), rows: Vec::new() });
        Ok(())
    }

    pub(super) fn batch(&mut self, batch: pb::ColumnBatch) -> Result<(), WsdbError> {
        if batch.statement as usize + 1 != self.sets.len() {
            return Err(WsdbError::Protocol("rows arrived outside their statement".into()));
        }
        let Some(ResultSet::Rows { width, rows }) = self.sets.last_mut() else {
            return Err(WsdbError::Protocol("rows arrived without Columns".into()));
        };
        if batch.columns.len() != *width {
            return Err(WsdbError::Protocol("row width differs from Columns".into()));
        }
        let decoded = decode_batch(&batch)?;
        self.rows += u64::from(batch.rows);
        rows.extend(decoded);
        Ok(())
    }

    pub(super) fn exec(&mut self, result: pb::ExecResult) -> Result<(), WsdbError> {
        self.start(result.statement)?;
        self.sets.push(ResultSet::Exec {
            rows_affected: result.rows_affected,
            last_insert_id: result.last_insert_id,
        });
        Ok(())
    }

    pub(super) fn finish(self, done: &pb::Done) -> Result<BatchResult, WsdbError> {
        if self.sets.len() != self.expected || self.rows != done.total_rows {
            return Err(WsdbError::Protocol(format!(
                "Done reports {} rows after {} rows and {} of {} statements",
                done.total_rows,
                self.rows,
                self.sets.len(),
                self.expected,
            )));
        }
        Ok(BatchResult {
            results: self
                .sets
                .into_iter()
                .map(|set| match set {
                    ResultSet::Rows { rows, .. } => StatementResult::Rows(rows),
                    ResultSet::Exec { rows_affected, last_insert_id } => {
                        StatementResult::Exec { rows_affected, last_insert_id }
                    }
                })
                .collect(),
        })
    }
}
