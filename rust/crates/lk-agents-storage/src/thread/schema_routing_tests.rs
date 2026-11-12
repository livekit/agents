use rusqlite::{Connection, StatementStatus, named_params, params};

use super::schema::{THREAD_SCHEMA, THREAD_SCHEMA_VERSION};
use crate::sql::Bind;

fn database() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    for statement in crate::sql::schema_statements(THREAD_SCHEMA) {
        conn.execute(statement, []).unwrap();
    }
    conn.execute("INSERT INTO schema_version VALUES (?1)", [THREAD_SCHEMA_VERSION]).unwrap();
    conn
}

#[test]
fn every_partition_write_requires_an_explicit_nonempty_key() {
    let conn = database();
    for (table, columns, values) in [
        ("items", "id, seq, payload", "'item', 1, jsonb('{}')"),
        ("tasks", "task_id, status, updated_at_ms, data", "'task', 'submitted', 1, jsonb('{}')"),
        ("partitions", "version", "0"),
        ("events", "occurred_at_ms, run_id, type, payload", "1, 'run', 'run_started', jsonb('{}')"),
        (
            "inbox",
            "task_id, message_id, message, received_at_ms",
            "'task', 'message', jsonb('{}'), 1",
        ),
    ] {
        let omitted = format!("INSERT INTO {table} ({columns}) VALUES ({values})");
        assert!(conn.execute(&omitted, []).is_err(), "{table} accepted an omitted key");
        let explicit =
            format!("INSERT INTO {table} (partition_key, {columns}) VALUES (?1, {values})");
        for invalid in [None, Some("")] {
            assert!(conn.execute(&explicit, [invalid]).is_err(), "{table} accepted {invalid:?}");
        }
        conn.execute(&explicit, [lk_agents_abi::storage::DEFAULT_PARTITION_KEY]).unwrap();
    }
}

#[test]
fn task_inbox_lease_and_retry_receipt_share_millisecond_units() {
    use super::schema::{
        FRAMEWORK_SCHEMA, InsertInbox, InsertPartition, InsertSubmittedTask, PruneAppliedWrites,
        RecordAppliedWrite, TakeLease,
    };
    let conn = database();
    conn.execute_batch(FRAMEWORK_SCHEMA).unwrap();
    conn.create_scalar_function(
        "unixepoch",
        1,
        rusqlite::functions::FunctionFlags::SQLITE_UTF8,
        |_| Ok(2000.125_f64),
    )
    .unwrap();
    conn.execute(InsertPartition::SQL, named_params! {":partition_key": "default"}).unwrap();
    conn.execute(InsertInbox::SQL, named_params! {
        ":partition_key": "default", ":task_id": "task", ":message_id": "message", ":message": "{}",
    }).unwrap();
    conn.execute(
        InsertSubmittedTask::SQL,
        named_params! {
            ":partition_key": "default", ":task_id": "task", ":name": None::<String>,
            ":message_id": "message", ":data": "{}", ":internal": None::<String>,
        },
    )
    .unwrap();
    let lease: i64 = conn
        .query_row(
            TakeLease::SQL,
            named_params! {
                ":partition_key": "default", ":holder": "worker", ":token": "token",
                ":lease_ms": 1500, ":task_id": "task", ":wait_ms": 60000,
            },
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(lease, 0);
    conn.execute(RecordAppliedWrite::SQL, named_params! {":id": "receipt"}).unwrap();
    for (table, column, expected) in [
        ("tasks", "updated_at_ms", 2_000_125_i64),
        ("inbox", "received_at_ms", 2_000_125),
        ("partitions", "lease_expires_at_ms", 2_001_625),
        ("applied_writes", "applied_at_ms", 2_000_125),
    ] {
        let value: i64 =
            conn.query_row(&format!("SELECT {column} FROM {table}"), [], |row| row.get(0)).unwrap();
        assert_eq!(value, expected, "{table}.{column}");
    }
    conn.execute("INSERT INTO applied_writes VALUES ('old', ?1)", [2_000_125_i64 - 86_400_001])
        .unwrap();
    assert_eq!(conn.execute(PruneAppliedWrites::SQL, []).unwrap(), 1);
}

#[test]
fn inbox_inputs_retain_their_own_destination_in_cross_endpoint_acceptance_order() {
    let mut conn = database();
    let tx = conn.transaction().unwrap();
    for (id, endpoint) in [("first", "/chat"), ("second", "/confirm"), ("third", "/chat")] {
        tx.execute(
            "INSERT INTO tasks (task_id, partition_key, endpoint, message_id, status, updated_at_ms, data)
             VALUES (?1, 'default', ?2, ?1, 'submitted', 1000, jsonb('{}'))",
            params![id, endpoint],
        )
        .unwrap();
        tx.execute(
            "INSERT INTO inbox (partition_key, task_id, message_id, message, received_at_ms)
             VALUES ('default', ?1, ?1, jsonb('{}'), 1)",
            [id],
        )
        .unwrap();
    }
    tx.commit().unwrap();

    let routes = conn
        .prepare(
            "SELECT t.task_id, t.endpoint FROM inbox i
         JOIN tasks t ON t.partition_key = i.partition_key AND t.task_id = i.task_id
         WHERE i.partition_key = 'default' ORDER BY t.pos",
        )
        .unwrap()
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        routes,
        [
            ("first".into(), "/chat".into()),
            ("second".into(), "/confirm".into()),
            ("third".into(), "/chat".into()),
        ]
    );

    // A retry cannot insert the same incoming message through another endpoint.
    let duplicate = conn
        .execute(
            "INSERT INTO tasks (task_id, partition_key, endpoint, message_id, status, updated_at_ms, data)
         VALUES ('retry', 'default', '/confirm', 'first', 'submitted', 1001, jsonb('{}'))
         ON CONFLICT (partition_key, message_id) WHERE message_id IS NOT NULL DO NOTHING",
            [],
        )
        .unwrap();
    assert_eq!(duplicate, 0);
    assert_eq!(
        conn.query_row("SELECT endpoint FROM tasks WHERE task_id = 'first'", [], |r| r
            .get::<_, String>(0))
            .unwrap(),
        "/chat"
    );
}

#[test]
fn incremental_event_receipts_deduplicate_per_partition_without_restricting_managed_events() {
    let conn = database();
    let append =
        "INSERT INTO events (occurred_at_ms, partition_key, run_id, event_key, type, payload)
                  VALUES (1000, ?1, 'run', ?2, 'message_delta', jsonb('{\"text\":\"Hello\"}'))
                  ON CONFLICT (partition_key, event_key) WHERE event_key IS NOT NULL DO NOTHING";
    assert_eq!(conn.execute(append, params!["default", "run:chunk:1"]).unwrap(), 1);
    assert_eq!(conn.execute(append, params!["default", "run:chunk:1"]).unwrap(), 0);
    assert_eq!(conn.execute(append, params!["booking", "run:chunk:1"]).unwrap(), 1);
    // Normal managed events keep their run-id commit guards; they need no new key.
    assert_eq!(conn.execute(append, params!["default", None::<String>]).unwrap(), 1);
    assert_eq!(conn.execute(append, params!["default", None::<String>]).unwrap(), 1);
    assert!(conn.execute(append, params!["default", ""]).is_err());
    assert_eq!(
        conn.query_row("SELECT count(*) FROM events WHERE event_key = 'run:chunk:1'", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
}

#[test]
fn destinations_cannot_be_empty_but_non_request_tasks_need_no_endpoint() {
    let conn = database();
    let insert = "INSERT INTO tasks (task_id, partition_key, endpoint, status, updated_at_ms, data)
                  VALUES ('task', 'default', ?1, 'submitted', 1000, jsonb('{}'))";
    assert!(conn.execute(insert, [""]).is_err());
    assert_eq!(conn.execute(insert, [None::<String>]).unwrap(), 1);
}

#[test]
fn item_pages_seek_past_other_partitions_in_both_directions() {
    use super::schema::{SelectItemsAfter, SelectItemsBefore};

    let conn = database();
    conn.execute_batch(
        "WITH RECURSIVE numbers(n) AS (
           SELECT 1 UNION ALL SELECT n + 1 FROM numbers WHERE n < 20003
         ) INSERT INTO items (id, partition_key, seq, payload)
         SELECT 'item-' || n,
           CASE WHEN n IN (1, 10002, 20003) THEN 'wanted' ELSE 'other' END,
           n, jsonb('{}') FROM numbers",
    )
    .unwrap();

    for (sql, cursor, expected_all) in
        [(SelectItemsAfter::SQL, 1_i64, 2_i64), (SelectItemsBefore::SQL, 20003_i64, 20002_i64)]
    {
        let cursor_name = if cursor == 1 { ":after" } else { ":before" };
        for partition_key in [Some("wanted"), None] {
            let mut statement = conn.prepare(sql).unwrap();
            let positions = statement
                .query_map(
                    &[
                        (":partition_key", &partition_key as &dyn rusqlite::ToSql),
                        (cursor_name, &cursor),
                        (":limit", &1_i64),
                    ],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(positions, [if partition_key.is_some() { 10002 } else { expected_all }]);
            assert_eq!(statement.get_status(StatementStatus::Sort), 0);
            // The work stays bounded by the page, even across 10,000 unrelated rows.
            let steps = statement.get_status(StatementStatus::VmStep);
            assert!(steps < 500, "{partition_key:?}: {steps} VM steps");
        }
    }
}

#[test]
fn task_update_pages_use_their_ordering_index() {
    use super::schema::SelectThreadTasksPage;

    let conn = database();
    conn.execute_batch(
        "WITH RECURSIVE numbers(n) AS (
           SELECT 1 UNION ALL SELECT n + 1 FROM numbers WHERE n < 1000
         ) INSERT INTO tasks (task_id, partition_key, status, updated_at_ms, data)
         SELECT 'task-' || n, 'default', 'completed', n, jsonb('{}') FROM numbers",
    )
    .unwrap();
    let mut statement = conn.prepare(SelectThreadTasksPage::SQL).unwrap();
    let ids = statement
        .query_map(
            named_params! {
                ":partition_key": None::<String>, ":status": None::<String>,
                ":updated_after": 0, ":before_at": i64::MAX, ":before_id": "", ":limit": 1,
            },
            |row| row.get::<_, String>(0),
        )
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(ids, ["task-1000"]);
    assert_eq!(statement.get_status(StatementStatus::Sort), 0);
    assert!(statement.get_status(StatementStatus::VmStep) < 500);
}
