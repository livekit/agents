//! Split the protocol's table/index DDL into statements for an atomic connection batch.

pub fn schema_statements(schema: &str) -> impl Iterator<Item = &str> {
    let bytes = schema.as_bytes();
    let (mut start, mut i) = (0, 0);
    let mut quote = None;
    let (mut line_comment, mut block_comment, mut has_sql) = (false, false, false);
    std::iter::from_fn(move || {
        while i < bytes.len() {
            let byte = bytes[i];
            let next = bytes.get(i + 1).copied();
            i += 1;
            if line_comment {
                line_comment = byte != b'\n';
            } else if block_comment {
                if byte == b'*' && next == Some(b'/') {
                    block_comment = false;
                    i += 1;
                }
            } else if let Some(end) = quote {
                if byte == end {
                    if end != b']' && next == Some(end) {
                        i += 1;
                    } else {
                        quote = None;
                    }
                }
            } else {
                match (byte, next) {
                    (b'-', Some(b'-')) => {
                        line_comment = true;
                        i += 1;
                    }
                    (b'/', Some(b'*')) => {
                        block_comment = true;
                        i += 1;
                    }
                    (b'\'' | b'"' | b'`' | b'[', _) => {
                        quote = Some(if byte == b'[' { b']' } else { byte });
                        has_sql = true;
                    }
                    (b';', _) => {
                        let statement = &schema[start..i];
                        start = i;
                        if has_sql {
                            has_sql = false;
                            return Some(statement.trim());
                        }
                    }
                    _ => has_sql |= !byte.is_ascii_whitespace(),
                }
            }
        }
        if has_sql {
            has_sql = false;
            Some(schema[start..].trim())
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::THREAD_SCHEMA;

    #[test]
    fn a_fresh_thread_schema_accepts_a_task() {
        // Keep the regression even when an upstream schema revision rewords its comments.
        let schema = THREAD_SCHEMA.replace(
            "CREATE TABLE tasks (",
            "CREATE TABLE tasks (\n-- Task states; submitted, working, completed.\n",
        );
        let mut db = rusqlite::Connection::open_in_memory().unwrap();
        let tx = db.transaction().unwrap();
        for statement in schema_statements(&schema) {
            tx.execute(statement, []).unwrap_or_else(|e| panic!("{e}: {statement}"));
        }
        tx.commit().unwrap();
        db.execute(
            "INSERT INTO tasks (task_id, lane, seq, message_id, status, updated_at, data) \
             VALUES ('AST_first', 'front-desk', 1, 'first', 'submitted', 1, '{}')",
            [],
        )
        .unwrap();
        let task: (String, String) = db
            .query_row(
                "SELECT message_id, status FROM tasks WHERE task_id = 'AST_first'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(task, ("first".into(), "submitted".into()));
    }

    #[test]
    fn schema_comments_and_quoted_values_preserve_statement_boundaries() {
        let schema = r#"
-- A semicolon here; must not terminate a statement.
/* Nor here; */ CREATE TABLE "semi;colon" (
    [key;name] TEXT DEFAULT 'it''s; -- /* text */',
    `value;name` TEXT
);
INSERT INTO "semi;colon" (`value;name`) VALUES ('hello;world');
; -- A comment-only tail;
/* with another semicolon; */
"#;
        let db = rusqlite::Connection::open_in_memory().unwrap();
        for statement in schema_statements(schema) {
            db.execute(statement, []).unwrap_or_else(|e| panic!("{e}: {statement}"));
        }
        let row: (String, String) = db
            .query_row(r#"SELECT * FROM "semi;colon""#, [], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap();
        assert_eq!(row, ("it's; -- /* text */".into(), "hello;world".into()));
    }
}
