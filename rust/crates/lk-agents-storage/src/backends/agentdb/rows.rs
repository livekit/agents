//! Rows as the `wsdb` wire carries them: bound values, statements and column-oriented batches.

use super::connection::{MAX_FRAME, WsdbError, wire};
use crate::sql::{Row, SqlValue, Statement};

/// A bound value as the wire carries it.
fn to_wire(v: &SqlValue) -> wire::Value {
    use wire::value::Value as V;
    let value = match v {
        SqlValue::Null => V::NullValue(true),
        SqlValue::Int(i) => V::IntValue(*i),
        SqlValue::Double(d) => V::DoubleValue(*d),
        SqlValue::Text(s) => V::TextValue(s.clone()),
        SqlValue::Blob(b) => V::BlobValue(b.clone()),
    };
    wire::Value { value: Some(value) }
}

/// A statement as the wire carries it.
pub fn statement(s: &Statement) -> wire::Statement {
    wire::Statement {
        sql: s.sql().to_owned(),
        params: s.params().iter().map(to_wire).collect(),
        lang: 0,
    }
}

/// A batch's rows. Values are typed per row: walk `types` and take the next entry from the
/// matching array; text and blobs are concatenated with one exclusive end offset each. A column
/// that does not hold what its types announce, or text that is not UTF-8, is a malformed answer:
/// refused, never read as `NULL`.
pub fn decode_batch(b: &wire::ColumnBatch) -> Result<Vec<Row>, WsdbError> {
    use wire::ValueType;
    let malformed = |what: &str| WsdbError::Protocol(format!("a malformed column batch: {what}"));
    let rows = b.rows as usize;
    let row_bytes = b
        .columns
        .len()
        .checked_mul(std::mem::size_of::<SqlValue>())
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Row>()))
        .and_then(|bytes| bytes.checked_mul(rows));
    if row_bytes.is_none_or(|bytes| bytes > MAX_FRAME)
        || (rows > 0 && b.columns.is_empty())
        || b.columns.iter().any(|column| column.types.len() != rows)
    {
        return Err(malformed("row dimensions exceed the decode budget or column types"));
    }
    let mut out: Vec<Row> = (0..rows).map(|_| Vec::with_capacity(b.columns.len())).collect();
    for col in &b.columns {
        let (mut ints, mut doubles) = (col.ints.iter(), col.doubles.iter());
        let (mut texts, mut blobs) = (col.text_ends.iter(), col.blob_ends.iter());
        let (mut text_at, mut blob_at) = (0usize, 0usize);
        for (row, tag) in out.iter_mut().zip(&col.types) {
            let span = |ends: &mut std::slice::Iter<'_, u32>, at: &mut usize, len: usize| {
                let end = *ends.next()? as usize;
                let start = std::mem::replace(at, end);
                (start <= end && end <= len).then_some(start..end)
            };
            let cell = match ValueType::try_from(i32::from(*tag)) {
                Ok(ValueType::Null) => SqlValue::Null,
                Ok(ValueType::Int) => SqlValue::Int(*ints.next().ok_or_else(|| malformed("ints"))?),
                Ok(ValueType::Double) => {
                    SqlValue::Double(*doubles.next().ok_or_else(|| malformed("doubles"))?)
                }
                Ok(ValueType::Text) => {
                    let range = span(&mut texts, &mut text_at, col.text_data.len())
                        .ok_or_else(|| malformed("text offsets"))?;
                    let text = std::str::from_utf8(&col.text_data[range])
                        .map_err(|_| malformed("text that is not UTF-8"))?;
                    SqlValue::Text(text.to_owned())
                }
                Ok(ValueType::Blob) => {
                    let range = span(&mut blobs, &mut blob_at, col.blob_data.len())
                        .ok_or_else(|| malformed("blob offsets"))?;
                    SqlValue::Blob(col.blob_data[range].to_vec())
                }
                Err(_) => return Err(malformed("an unknown value type")),
            };
            row.push(cell);
        }
        if ints.next().is_some()
            || doubles.next().is_some()
            || texts.next().is_some()
            || blobs.next().is_some()
            || text_at != col.text_data.len()
            || blob_at != col.blob_data.len()
        {
            return Err(malformed("unannounced column values"));
        }
    }
    Ok(out)
}

/// The inverse of [`decode_batch`], for tests: rows into one column-oriented batch.
#[cfg(test)]
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
