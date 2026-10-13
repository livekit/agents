//! The data-plane wire: the prost types of `livekit.AgentDB.Wire` from `livekit-protocol`, a
//! bind value, and the column-oriented row batch decoded back into rows.

use serde::Serialize;

pub use livekit_protocol::agent_db::wire;

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
    /// A JSON column's bind value: its text, for `jsonb(?)`. JSON values and the column records
    /// (string keys only) always serialize.
    pub fn json<T: Serialize + ?Sized>(v: &T) -> SqlValue {
        SqlValue::Text(serde_json::to_string(v).expect("a JSON-shaped value"))
    }

    /// `NULL` for `None`, else [`SqlValue::json`].
    pub fn opt_json<T: Serialize>(v: Option<&T>) -> SqlValue {
        v.map_or(SqlValue::Null, SqlValue::json)
    }

    pub fn to_wire(&self) -> wire::Value {
        use wire::value::Value as V;
        let value = match self {
            SqlValue::Null => V::NullValue(true),
            SqlValue::Int(i) => V::IntValue(*i),
            SqlValue::Double(d) => V::DoubleValue(*d),
            SqlValue::Text(s) => V::TextValue(s.clone()),
            SqlValue::Blob(b) => V::BlobValue(b.clone()),
        };
        wire::Value { value: Some(value) }
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

/// One result row, one cell per column.
pub type Row = Vec<SqlValue>;

/// A statement with positional parameters.
pub fn statement(sql: &str, params: Vec<SqlValue>) -> wire::Statement {
    wire::Statement {
        sql: sql.to_owned(),
        params: params.iter().map(SqlValue::to_wire).collect(),
        lang: 0,
    }
}

/// A batch's rows. Values are typed per row: walk `types` and take the next entry from the
/// matching array; text and blobs are concatenated with one exclusive end offset each. A
/// truncated column reads as `NULL`, as the Go client does, rather than failing the batch.
pub fn decode_batch(b: &wire::ColumnBatch) -> Vec<Row> {
    use wire::ValueType;
    let rows = b.rows as usize;
    let mut out: Vec<Row> = (0..rows).map(|_| Vec::with_capacity(b.columns.len())).collect();
    for col in &b.columns {
        let (mut ints, mut doubles, mut texts, mut blobs) = (0usize, 0usize, 0usize, 0usize);
        let (mut text_at, mut blob_at) = (0usize, 0usize);
        for (r, row) in out.iter_mut().enumerate() {
            let tag =
                col.types.get(r).copied().map(i32::from).and_then(|t| ValueType::try_from(t).ok());
            let cell = match tag {
                Some(ValueType::Int) if ints < col.ints.len() => {
                    ints += 1;
                    SqlValue::Int(col.ints[ints - 1])
                }
                Some(ValueType::Double) if doubles < col.doubles.len() => {
                    doubles += 1;
                    SqlValue::Double(col.doubles[doubles - 1])
                }
                Some(ValueType::Text) if texts < col.text_ends.len() => {
                    let end = col.text_ends[texts] as usize;
                    texts += 1;
                    let cell = if end >= text_at && end <= col.text_data.len() {
                        SqlValue::Text(
                            String::from_utf8_lossy(&col.text_data[text_at..end]).into_owned(),
                        )
                    } else {
                        SqlValue::Null
                    };
                    text_at = end;
                    cell
                }
                Some(ValueType::Blob) if blobs < col.blob_ends.len() => {
                    let end = col.blob_ends[blobs] as usize;
                    blobs += 1;
                    let cell = if end >= blob_at && end <= col.blob_data.len() {
                        SqlValue::Blob(col.blob_data[blob_at..end].to_vec())
                    } else {
                        SqlValue::Null
                    };
                    blob_at = end;
                    cell
                }
                _ => SqlValue::Null,
            };
            row.push(cell);
        }
    }
    out
}

/// The inverse of [`decode_batch`], for tests and fakes: rows into one column-oriented batch.
pub fn encode_batch(rows: &[Row], width: usize) -> wire::ColumnBatch {
    use wire::ValueType;
    let mut columns: Vec<wire::Column> = (0..width).map(|_| wire::Column::default()).collect();
    for row in rows {
        for (i, col) in columns.iter_mut().enumerate() {
            match row.get(i).unwrap_or(&SqlValue::Null) {
                SqlValue::Null => col.types.push(ValueType::Null as u8),
                SqlValue::Int(v) => {
                    col.types.push(ValueType::Int as u8);
                    col.ints.push(*v);
                }
                SqlValue::Double(v) => {
                    col.types.push(ValueType::Double as u8);
                    col.doubles.push(*v);
                }
                SqlValue::Text(s) => {
                    col.types.push(ValueType::Text as u8);
                    col.text_data.extend_from_slice(s.as_bytes());
                    col.text_ends.push(col.text_data.len() as u32);
                }
                SqlValue::Blob(b) => {
                    col.types.push(ValueType::Blob as u8);
                    col.blob_data.extend_from_slice(b);
                    col.blob_ends.push(col.blob_data.len() as u32);
                }
            }
        }
    }
    wire::ColumnBatch { columns, rows: rows.len() as u32, statement: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mixed_column_round_trips() {
        let rows: Vec<Row> = vec![
            vec![SqlValue::Int(1), SqlValue::Text("a".into()), SqlValue::Null],
            vec![
                SqlValue::Text("mixed".into()),
                SqlValue::Text("".into()),
                SqlValue::Blob(vec![1, 2]),
            ],
            vec![SqlValue::Double(2.5), SqlValue::Text("bb".into()), SqlValue::Int(-7)],
        ];
        let batch = encode_batch(&rows, 3);
        assert_eq!(decode_batch(&batch), rows);
    }

    #[test]
    fn a_truncated_column_reads_as_null() {
        let mut batch = encode_batch(&[vec![SqlValue::Int(1)], vec![SqlValue::Int(2)]], 1);
        batch.columns[0].ints.pop();
        assert_eq!(decode_batch(&batch), vec![vec![SqlValue::Int(1)], vec![SqlValue::Null]]);
    }

    #[test]
    fn bind_values_take_the_five_classes() {
        use wire::value::Value as V;
        assert_eq!(SqlValue::from(None::<String>).to_wire().value, Some(V::NullValue(true)));
        assert_eq!(SqlValue::from(true).to_wire().value, Some(V::IntValue(1)));
        assert_eq!(
            SqlValue::json(&serde_json::json!({"a": 1})).to_wire().value,
            Some(V::TextValue("{\"a\":1}".into()))
        );
    }
}
