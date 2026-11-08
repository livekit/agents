//! Tool discovery export contracts, implemented by `lk_agents_core::tool_search`.

/// A native search pattern failed while matching.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct PatternError(pub String);

/// Model-facing tool discovery validation failed.
#[crate::host(error(kind = "livekit.agents.llm.tool_context.ToolError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ToolSearchError(pub String);

/// What the search reads of a tool's `SearchItem`.
#[crate::host(record(
    module = "tool_search",
    host_type = "livekit.agents._generated.toolset_events.SearchItem",
    boundary,
    extract
))]
#[derive(Debug, Clone, Default)]
pub struct SearchFields {
    pub name: String,
    pub description: String,
    pub parameters: std::collections::BTreeMap<String, String>,
}

/// One tool as the search sees it; identity stays with the host.
#[crate::host(class(name = "SearchDocument", pid = false))]
pub trait SearchDocument: Clone + Send + Sized + 'static {
    #[host(new)]
    fn new(item: SearchFields) -> Self;
    #[host(call)]
    fn keyword_score(&self, terms: Vec<String>) -> Result<f64, PatternError>;
}

#[crate::host(class(name = "BM25Index", pid = false))]
pub trait SearchIndex: Send + Sized + 'static {
    #[host(class(name = "SearchDocument"))]
    type Document: SearchDocument;
    #[host(new)]
    fn new(k1: f64, b: f64) -> Self;
    #[host(call)]
    fn build(&mut self, documents: Vec<Self::Document>);
    #[host(call)]
    fn clear(&mut self);
    #[host(call)]
    fn score(&self, doc: Self::Document, terms: Vec<String>) -> f64;
    #[host(call)]
    fn search(&self, query: String, docs: Vec<Self::Document>, limit: i64) -> Vec<usize>;
}

#[crate::host(functions(pid = false))]
pub trait ToolSearch {
    #[host(class(name = "SearchDocument"))]
    type Document: SearchDocument;
    #[host(function)]
    fn keyword_search(
        query: String,
        docs: Vec<Self::Document>,
        limit: i64,
    ) -> Result<Vec<usize>, PatternError>;
    #[host(function, pid = true)]
    fn tool_search_validate(query: String) -> Result<(), ToolSearchError>;
    #[host(function, pid = true)]
    fn tool_search_loaded(query: String, count: usize) -> Result<String, ToolSearchError>;
    /// `ToolSearchToolset`'s `tool_search` raw schema.
    #[host(function)]
    fn tool_search_schema(
        search_description: String,
        query_description: String,
    ) -> crate::llm::provider_format::tools::RawFunctionDescription;
    /// `ToolProxyToolset`'s `call_tool` raw schema.
    #[host(function)]
    fn call_tool_schema(
        description: String,
    ) -> crate::llm::provider_format::tools::RawFunctionDescription;
    #[host(function, pid = true)]
    fn tool_proxy_arguments(
        arguments: serde_json::Value,
        known: Vec<String>,
    ) -> Result<(String, serde_json::Value), ToolSearchError>;
}
