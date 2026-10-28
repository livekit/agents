//! Host tool discovery handles backed by deterministic text ranking.

use lk_agents_abi::tokenize::search::{
    self as abi, PatternError, SearchIndex, ToolSearch, ToolSearchError,
};
use lk_agents_voice::tokenize::search as voice;

#[derive(Clone)]
pub struct SearchDocument(voice::Document);

impl abi::SearchDocument for SearchDocument {
    fn new(name: String, description: String, parameters: Vec<(String, String)>) -> Self {
        Self(voice::Document::new(&name, &description, &parameters))
    }
    fn keyword_score(&self, terms: Vec<String>) -> Result<f64, PatternError> {
        self.0.keyword_score(&terms).map_err(|e| PatternError(e.to_string()))
    }
}

pub struct Bm25(voice::Bm25);
impl SearchIndex for Bm25 {
    type Document = SearchDocument;
    fn new(k1: f64, b: f64) -> Self {
        Self(voice::Bm25::new(k1, b))
    }
    fn build(&mut self, documents: Vec<SearchDocument>) {
        self.0.build(&documents.into_iter().map(|d| d.0).collect::<Vec<_>>());
    }
    fn clear(&mut self) {
        self.0.clear();
    }
    fn score(&self, doc: SearchDocument, terms: Vec<String>) -> f64 {
        self.0.score(&doc.0, &terms)
    }
    fn search(&self, query: String, docs: Vec<SearchDocument>, limit: i64) -> Vec<usize> {
        self.0.search(&query, &docs.into_iter().map(|d| d.0).collect::<Vec<_>>(), limit as isize)
    }
}

pub struct ToolDiscovery;
impl ToolSearch for ToolDiscovery {
    type Document = SearchDocument;
    fn keyword_search(
        query: String,
        docs: Vec<SearchDocument>,
        limit: i64,
    ) -> Result<Vec<usize>, PatternError> {
        voice::keyword_search(
            &query,
            &docs.into_iter().map(|d| d.0).collect::<Vec<_>>(),
            limit as isize,
        )
        .map_err(|e| PatternError(e.to_string()))
    }
    fn tool_search_validate(query: String) -> Result<(), ToolSearchError> {
        voice::validate_query(&query).map_err(|message| ToolSearchError(message.into()))
    }
    fn tool_search_loaded(query: String, count: usize) -> Result<String, ToolSearchError> {
        voice::loaded_result(&query, count)
            .map(str::to_owned)
            .map_err(|message| ToolSearchError(message.into()))
    }
    fn tool_proxy_arguments(
        arguments: serde_json::Value,
        known: Vec<String>,
    ) -> Result<(String, serde_json::Value), ToolSearchError> {
        voice::proxy_arguments(arguments, known).map_err(|message| ToolSearchError(message.into()))
    }
}
