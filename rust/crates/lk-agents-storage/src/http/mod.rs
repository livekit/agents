//! Local ThreadService: generated wire contract over the shared storage engine.

use std::collections::BTreeMap;
use std::sync::Arc;

use lk_agents_abi::a2a::auth::{
    AuthOperation, AuthPermissions as _, AuthPrincipal, Credentials as _,
};
use lk_agents_net::a2a_auth::Credentials;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::backends::with_thread;
use crate::cursor::Cursor;
use crate::sql::{Bind, Cells};
use crate::thread::schema::{SelectItemsAfter, SelectItemsBefore, SelectThreadTasksInserted};
use crate::{Storage, TaskState, Thread};
use generated::*;
use livekit_protocol::agent::agent_thread;
pub use lk_agents_abi::errors::ApiError;

pub mod generated;
mod server;
pub use server::ThreadServer;

pub struct Context {
    principal: AuthPrincipal,
}

pub struct ThreadService {
    storage: Arc<dyn Storage>,
    project_id: String,
    credentials: Credentials,
}

impl ThreadService {
    pub fn new(
        storage: Arc<dyn Storage>,
        project_id: String,
        api_key: String,
        api_secret: String,
    ) -> Self {
        Self { storage, project_id, credentials: Credentials::new(api_key, api_secret) }
    }

    fn authorize(
        &self,
        context: &Context,
        project: &str,
        operation: AuthOperation,
        id: &str,
        scope: Option<&str>,
    ) -> Result<(), ApiError> {
        if project != self.project_id && project != "p_" {
            return Err(ApiError::forbidden("project does not match this worker"));
        }
        let grant = context
            .principal
            .grant
            .as_ref()
            .ok_or_else(|| ApiError::forbidden("a thread grant is required"))?;
        // Project-wide create grants also allow threads with no scope. Endpoint-call grants
        // remain separate and never authorize access to the threads service.
        let allowed = (operation == AuthOperation::Create && grant.all && grant.create)
            || Credentials::_core_auth_grant_allows(
                grant.clone(),
                operation,
                id.to_owned(),
                scope.map(str::to_owned),
            );
        if !allowed {
            return Err(ApiError::forbidden("thread permission denied"));
        }
        Ok(())
    }

    pub async fn request(
        &self,
        method: &str,
        path: &str,
        query: &str,
        authorization: Option<&str>,
        body: &[u8],
    ) -> Result<Value, ApiError> {
        let token = authorization
            .and_then(lk_agents_abi::a2a::auth::bearer_token)
            .ok_or_else(|| ApiError::unauthorized("a bearer token is required"))?;
        let principal = self
            .credentials
            .verify(token.to_owned(), String::new(), String::new())
            .map_err(|_| ApiError::unauthorized("invalid bearer token"))?;
        let path = decode_path(path)?;
        let segments: Vec<_> = path.iter().map(String::as_str).collect();
        let query = decode_query(query)?;
        let context = Context { principal };

        generated::dispatch(self, &context, method, &segments, &query, body).await
    }
}

trait HttpError {
    fn invalid(message: impl std::fmt::Display) -> Self;
    fn unauthorized(message: impl Into<String>) -> Self;
    fn forbidden(message: impl Into<String>) -> Self;
    fn not_found(message: impl Into<String>) -> Self;
    fn internal(error: impl std::fmt::Display) -> Self;
}

impl HttpError for ApiError {
    fn invalid(message: impl std::fmt::Display) -> Self {
        Self::status(message.to_string(), 400, None, Some(false))
    }

    fn unauthorized(message: impl Into<String>) -> Self {
        Self::status(message, 401, None, Some(false))
    }

    fn forbidden(message: impl Into<String>) -> Self {
        Self::status(message, 403, None, Some(false))
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self::status(message, 404, None, Some(false))
    }

    fn internal(_error: impl std::fmt::Display) -> Self {
        Self::status("storage request failed", 500, None, Some(false))
    }
}

fn decode_path(path: &str) -> Result<Vec<String>, ApiError> {
    path.trim_start_matches('/')
        .split('/')
        .map(|segment| {
            percent_encoding::percent_decode_str(segment)
                .decode_utf8()
                .map(|decoded| decoded.into_owned())
                .map_err(|_| ApiError::invalid("path must contain valid UTF-8"))
        })
        .collect()
}

fn decode_query(query: &str) -> Result<BTreeMap<String, String>, ApiError> {
    // form_urlencoded is intentionally lossy; reject invalid UTF-8 before it can replace it.
    percent_encoding::percent_decode_str(query)
        .decode_utf8()
        .map_err(|_| ApiError::invalid("query must contain valid UTF-8"))?;

    let mut parameters = BTreeMap::new();
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        if parameters.insert(key.into_owned(), value.into_owned()).is_some() {
            return Err(ApiError::invalid("duplicate query parameter"));
        }
    }
    Ok(parameters)
}

fn check_query(query: &BTreeMap<String, String>, allowed: &[&str]) -> Result<(), ApiError> {
    if let Some(key) = query.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(ApiError::invalid(format!("unknown query parameter {key}")));
    }
    Ok(())
}

fn parameter<T: for<'de> Deserialize<'de> + Default>(
    query: &BTreeMap<String, String>,
    key: &str,
    numeric: bool,
) -> Result<T, ApiError> {
    let Some(value) = query.get(key) else { return Ok(T::default()) };
    let value = if numeric {
        json!(value.parse::<i32>().map_err(ApiError::invalid)?)
    } else {
        json!(value)
    };
    serde_json::from_value(value).map_err(ApiError::invalid)
}

fn page_info(next_cursor: String) -> Option<PageInfo> {
    Some(PageInfo { has_more: !next_cursor.is_empty(), next_cursor })
}

fn page_size(value: i32) -> Result<i64, ApiError> {
    if value < 0 {
        return Err(ApiError::invalid("page.pageSize cannot be negative"));
    }
    Ok(if value == 0 { 100 } else { value.min(200) }.into())
}

fn validate_fields(scope: &str, attributes: &BTreeMap<String, String>) -> Result<(), ApiError> {
    if scope.len() > 256
        || attributes.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>() > 16 * 1024
    {
        return Err(ApiError::invalid("scope or attributes exceed their size limit"));
    }
    Ok(())
}

fn ttl(value: Option<Duration>, create: bool) -> Result<Option<Duration>, ApiError> {
    match value {
        None if create => Ok(Some(Duration { seconds: 86400, nanos: 0 })),
        Some(d) if create && d.seconds == 0 && d.nanos == 0 => ttl(None, true),
        Some(d)
            if d.seconds < 0
                || d.nanos < 0
                || (d.seconds == 0 && d.nanos == 0)
                || d.seconds > 30 * 86400
                || (d.seconds == 30 * 86400 && d.nanos > 0) =>
        {
            Err(ApiError::invalid("idleTtl must be positive and at most 30 days"))
        }
        other => Ok(other),
    }
}

impl ThreadService {
    async fn thread(
        &self,
        context: &Context,
        project: &str,
        id: &str,
        operation: AuthOperation,
    ) -> Result<agent_thread::Thread, ApiError> {
        // Check the project before looking up a thread in its local registry.
        if project != self.project_id && project != "p_" {
            return Err(ApiError::forbidden("project does not match this worker"));
        }
        let thread = self.storage.get_thread(id).await?;
        self.authorize(context, project, operation, id, Some(&thread.scope))?;
        Ok(thread)
    }
}

#[async_trait::async_trait]
impl generated::ThreadService for ThreadService {
    async fn list_threads(
        &self,
        context: &Context,
        request: ListThreadsRequest,
    ) -> Result<ListThreadsResponse, ApiError> {
        self.authorize(
            context,
            &request.project_id,
            AuthOperation::List,
            "",
            Some(&request.scope),
        )?;
        if request.scope.is_empty() {
            return Err(ApiError::invalid("scope is required"));
        }
        let page = self
            .storage
            .list_threads(agent_thread::ListRequest {
                scope: request.scope,
                page_token: request.page_cursor,
                limit: page_size(request.page_page_size)? as i32,
            })
            .await?;
        Ok(ListThreadsResponse { items: page.threads, page_info: page_info(page.next_page_token) })
    }

    async fn create_thread(
        &self,
        context: &Context,
        request: CreateThreadRequest,
    ) -> Result<CreateThreadResponse, ApiError> {
        self.authorize(
            context,
            &request.project_id,
            AuthOperation::Create,
            "",
            Some(&request.body.scope),
        )?;
        validate_fields(&request.body.scope, &request.body.attributes)?;
        let thread = self
            .storage
            .create_thread(agent_thread::CreateRequest {
                scope: request.body.scope,
                attributes: request.body.attributes.into_iter().collect(),
                idle_ttl: ttl(request.body.idle_ttl, true)?,
            })
            .await?;
        Ok(CreateThreadResponse { thread: Some(thread) })
    }

    async fn get_thread(
        &self,
        context: &Context,
        request: GetThreadRequest,
    ) -> Result<GetThreadResponse, ApiError> {
        Ok(GetThreadResponse {
            thread: Some(
                self.thread(context, &request.project_id, &request.thread_id, AuthOperation::Read)
                    .await?,
            ),
        })
    }

    async fn delete_thread(
        &self,
        context: &Context,
        request: DeleteThreadRequest,
    ) -> Result<DeleteThreadResponse, ApiError> {
        self.thread(context, &request.project_id, &request.thread_id, AuthOperation::Delete)
            .await?;
        self.storage.delete_thread(&request.thread_id).await?;
        Ok(DeleteThreadResponse {})
    }

    async fn update_thread(
        &self,
        context: &Context,
        request: UpdateThreadRequest,
    ) -> Result<UpdateThreadResponse, ApiError> {
        self.thread(context, &request.project_id, &request.thread_id, AuthOperation::Write).await?;
        validate_fields("", &request.body.attributes)?;
        let thread = self
            .storage
            .update_thread(agent_thread::UpdateRequest {
                thread_id: request.thread_id,
                attributes: request.body.attributes.into_iter().collect(),
                idle_ttl: ttl(request.body.idle_ttl, false)?,
            })
            .await?;
        Ok(UpdateThreadResponse { thread: Some(thread) })
    }

    async fn get_thread_items(
        &self,
        context: &Context,
        request: GetThreadItemsRequest,
    ) -> Result<GetThreadItemsResponse, ApiError> {
        self.thread(context, &request.project_id, &request.thread_id, AuthOperation::Read).await?;
        let id = request.thread_id.clone();
        with_thread(&*self.storage, &id, "thread-api", async |thread| items(thread, request).await)
            .await
    }

    async fn get_thread_tasks(
        &self,
        context: &Context,
        request: GetThreadTasksRequest,
    ) -> Result<GetThreadTasksResponse, ApiError> {
        self.thread(context, &request.project_id, &request.thread_id, AuthOperation::Read).await?;
        let id = request.thread_id.clone();
        with_thread(&*self.storage, &id, "thread-api", async |thread| tasks(thread, request).await)
            .await
    }
}

async fn items(
    thread: &Thread,
    request: GetThreadItemsRequest,
) -> Result<GetThreadItemsResponse, ApiError> {
    let desc = request.sort_order == SortOrder::SortOrderDesc;
    let listing = ("http-items", &request.thread_id, &request.agent, desc);
    let after =
        Cursor::position(Some(&request.page_cursor), listing, if desc { i64::MAX } else { 0 })?;
    let limit = page_size(request.page_page_size)?;
    thread.ensure_schema().await?;
    let lane = Some(request.agent.as_str()).filter(|lane| !lane.is_empty());
    let page = if desc {
        SelectItemsBefore { lane, before: after, limit: limit + 1 }.bind()
    } else {
        SelectItemsAfter { lane, after, limit: limit + 1 }.bind()
    };
    let rows = thread.conn().query(page).await?;
    let more = rows.len() > limit as usize;
    let mut items = Vec::new();
    let mut last = after;
    for row in rows.iter().take(limit as usize) {
        let (cells, item) = Cells::followed_by::<crate::ItemRow>(row, &["rowid", "lane", "seq"])?;
        last = cells.get(0)?;
        items.push(agent_thread::Item {
            agent: cells.get(1)?,
            seq: cells.get(2)?,
            message_id: item.message_id.unwrap_or_default(),
            item: Some(item.payload),
        });
    }
    let next = if more { Cursor::encode(listing, last) } else { String::new() };
    Ok(GetThreadItemsResponse { items, page_info: page_info(next) })
}

pub(crate) fn task_status(status: TaskState) -> TaskStatus {
    match status {
        TaskState::Submitted => TaskStatus::Submitted,
        TaskState::Working => TaskStatus::Working,
        TaskState::InputRequired => TaskStatus::InputRequired,
        TaskState::Completed => TaskStatus::Completed,
        TaskState::Failed => TaskStatus::Failed,
        TaskState::Canceled => TaskStatus::Canceled,
        TaskState::Rejected => TaskStatus::Rejected,
    }
}

pub(crate) fn task_filter(status: TaskStatus) -> Option<TaskState> {
    match status {
        TaskStatus::Unspecified => None,
        TaskStatus::Submitted => Some(TaskState::Submitted),
        TaskStatus::Working => Some(TaskState::Working),
        TaskStatus::InputRequired => Some(TaskState::InputRequired),
        TaskStatus::Completed => Some(TaskState::Completed),
        TaskStatus::Failed => Some(TaskState::Failed),
        TaskStatus::Canceled => Some(TaskState::Canceled),
        TaskStatus::Rejected => Some(TaskState::Rejected),
    }
}

async fn tasks(
    thread: &Thread,
    request: GetThreadTasksRequest,
) -> Result<GetThreadTasksResponse, ApiError> {
    let status = task_filter(request.status).map(TaskState::as_str);
    let listing = ("http-tasks", &request.thread_id, status);
    let after = Cursor::position(Some(&request.page_cursor), listing, 0_i64)?;
    let limit = page_size(request.page_page_size)?;
    thread.ensure_schema().await?;
    let page = SelectThreadTasksInserted {
        lane: None,
        status,
        updated_after: i64::MIN,
        after,
        limit: limit + 1,
    };
    let rows = thread.conn().query(page.bind()).await?;
    let more = rows.len() > limit as usize;
    let mut items = Vec::new();
    let mut last = after;
    for row in rows.iter().take(limit as usize) {
        let (cells, task) = Cells::followed_by::<crate::thread::schema::TaskRow>(row, &["rowid"])?;
        last = cells.get(0)?;
        items.push(TaskRecord {
            task_id: task.task_id,
            thread_id: request.thread_id.clone(),
            name: task.name,
            status: Some(task_status(task.status)),
            updated_at: Some(Timestamp {
                seconds: task.updated_at / 1000,
                nanos: ((task.updated_at % 1000) * 1_000_000) as i32,
            }),
        });
    }
    let next = if more { Cursor::encode(listing, last) } else { String::new() };
    Ok(GetThreadTasksResponse { items, page_info: page_info(next) })
}
