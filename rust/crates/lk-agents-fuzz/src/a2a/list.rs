//! A client paging through `ListTasks` while the turns change the tasks: the storage core's
//! `list_tasks` plan and decoding, one page per step, with a page size and filters the seed
//! draws. Sometimes it presents a page token under other filters first, which must be refused.

use fastrand::Rng;
use lk_agents_storage::sql::Table;
use lk_agents_storage::db::{self as plans, Decode, TaskQuery};
use lk_agents_storage::run::{TaskRow, TaskState};

use super::{LANE, Observed};
use crate::storage::db::SimDb;

/// One listing: its filters, the tasks before and after it, every page's rows in order, and
/// whether a token presented under other filters was refused.
#[derive(Debug, Clone)]
pub struct Listing {
    pub status: Option<TaskState>,
    pub updated_after: i64,
    /// `(task_id, updated_at, status)` of every task when the listing started and ended.
    pub before: Vec<(String, i64, TaskState)>,
    pub after: Vec<(String, i64, TaskState)>,
    pub listed: Vec<String>,
    /// `Some(refused)` when a page token was tried under other filters.
    pub foreign_token: Option<bool>,
}

impl Listing {
    /// Whether a task with this status and update time is in the listing's filters.
    pub fn matches(&self, updated_at: i64, status: TaskState) -> bool {
        self.status.is_none_or(|s| s == status) && updated_at > self.updated_after
    }
}

pub struct Lister {
    at: i64,
    listings: u8,
    current: Option<(Listing, i64, Option<String>)>,
}

/// Every task's identity, update time and status, as the checker reads them.
fn snapshot(db: &SimDb) -> Vec<(String, i64, TaskState)> {
    db.inspect(&plans::tasks(LANE, None, 10_000).statement)
        .iter()
        .map(|r| TaskRow::from_row(r).expect("a task row"))
        .map(|t| (t.task_id, t.updated_at, t.status))
        .collect()
}

impl Lister {
    pub fn new(at: i64) -> Self {
        Lister { at, listings: 3, current: None }
    }

    pub fn done(&self) -> bool {
        self.listings == 0
    }

    pub fn wake(&self) -> i64 {
        self.at
    }

    pub fn step(&mut self, now: i64, db: &mut SimDb, rng: &mut Rng, seen: &mut Observed) {
        self.at = now + rng.i64(1..400);
        let (mut listing, page_size, token) = match self.current.take() {
            Some(current) => current,
            None => {
                let status = match rng.u8(0..4) {
                    0 => Some(TaskState::Completed),
                    1 => Some(TaskState::Working),
                    _ => None,
                };
                let before = snapshot(db);
                let updated_after = match rng.bool() {
                    true => before.iter().map(|t| t.1).min().unwrap_or(0),
                    false => 0,
                };
                let listing = Listing {
                    status,
                    updated_after,
                    before,
                    after: Vec::new(),
                    listed: Vec::new(),
                    foreign_token: None,
                };
                (listing, rng.i64(1..5), None)
            }
        };
        if let Some(token) = &token
            && listing.foreign_token.is_none()
            && rng.u8(0..3) == 0
        {
            // the same token under other filters
            let other = TaskQuery {
                status: match listing.status {
                    Some(_) => None,
                    None => Some(TaskState::Completed),
                },
                updated_after: listing.updated_after,
                page_size,
                page_token: Some(token.clone()),
            };
            listing.foreign_token = Some(plans::list_tasks(&other).is_err());
        }
        let query = TaskQuery {
            status: listing.status,
            updated_after: listing.updated_after,
            page_size,
            page_token: token.clone(),
        };
        let plan = plans::list_tasks(&query).expect("a token this listing made");
        let page = match db.batch(&plan.statements, None) {
            Ok(out) => plan.decoder.decode(&out).expect("a task page"),
            // the database is unreachable: the same page again later
            Err(_) => {
                self.current = Some((listing, page_size, token));
                return;
            }
        };
        listing.listed.extend(page.tasks.into_iter().map(|t| t.task_id));
        if page.next_page_token.is_empty() {
            listing.after = snapshot(db);
            seen.listings.push(listing);
            self.listings -= 1;
        } else {
            self.current = Some((listing, page_size, Some(page.next_page_token)));
        }
    }
}
