//! The storage-owned thread rows as the host reads them (`lk_agents_abi::storage`).

use lk_agents_abi::storage as host;
use crate::db::TaskPage;
use crate::schema::{EventRow, LaneRow};

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
