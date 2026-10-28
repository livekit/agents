//! The thread rows as the host reads them (`lk_agents_abi::storage`): chat items and A2A
//! messages become their protobuf JSON, which cannot fail for these types.

use lk_agents_abi::storage as host;
use serde::Serialize;
use serde_json::Value;

use crate::StorageError;
use crate::db::TaskPage;
use crate::schema::{EventRow, InboxRow, ItemRow, LaneRow, TaskData, TaskRow};

fn json<T: Serialize>(v: &T) -> Value {
    serde_json::to_value(v).unwrap_or_default()
}

impl From<ItemRow> for host::ItemRow {
    fn from(row: ItemRow) -> Self {
        host::ItemRow { payload: json(&row.payload), id: row.id, message_id: row.message_id }
    }
}

impl TryFrom<host::ItemRow> for ItemRow {
    type Error = StorageError;

    /// A host item whose payload is not a `ChatItem` is an invalid argument.
    fn try_from(row: host::ItemRow) -> Result<Self, StorageError> {
        let payload = serde_json::from_value(row.payload)
            .map_err(|e| StorageError::Invalid(format!("item {}: {e}", row.id)))?;
        Ok(ItemRow { id: row.id, message_id: row.message_id, payload })
    }
}

impl From<TaskData> for host::TaskData {
    fn from(data: TaskData) -> Self {
        host::TaskData {
            history: data.history.iter().map(json).collect(),
            artifacts: data.artifacts.map(|a| a.iter().map(json).collect()),
            metadata: data.metadata,
            status_message: data.status_message.as_ref().map(json),
        }
    }
}

impl From<TaskRow> for host::TaskRecord {
    fn from(row: TaskRow) -> Self {
        host::TaskRecord {
            task_id: row.task_id,
            lane: row.lane,
            seq: row.seq,
            name: row.name,
            message_id: row.message_id,
            status: row.status,
            updated_at: row.updated_at,
            data: row.data.into(),
            internal: row.internal,
        }
    }
}

impl From<TaskPage> for host::TaskPage {
    fn from(page: TaskPage) -> Self {
        host::TaskPage {
            tasks: page.tasks.into_iter().map(Into::into).collect(),
            next_page_token: page.next_page_token,
            page_size: page.page_size,
            total_size: page.total_size,
        }
    }
}

impl From<LaneRow> for host::LaneRow {
    fn from(row: LaneRow) -> Self {
        host::LaneRow {
            lane: row.lane,
            version: row.version,
            state: row.state,
            lease_holder: row.lease_holder,
            lease_until: row.lease_until,
            waiter: row.waiter,
            drained_id: row.drained_id,
        }
    }
}

impl From<EventRow> for host::EventRow {
    fn from(row: EventRow) -> Self {
        host::EventRow {
            id: row.id,
            ts_ms: row.ts_ms,
            lane: row.lane,
            run_id: row.run_id,
            r#type: row.r#type,
            payload: row.payload,
        }
    }
}

impl From<InboxRow> for host::InboxRow {
    fn from(row: InboxRow) -> Self {
        host::InboxRow {
            message: json(&row.message),
            id: row.id,
            lane: row.lane,
            task_id: row.task_id,
            message_id: row.message_id,
            received_at: row.received_at,
        }
    }
}
