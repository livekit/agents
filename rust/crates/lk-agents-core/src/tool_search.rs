//! Built-in tool discovery (`ToolSearchToolset`, `ToolProxyToolset`): keyword and BM25 ranking of
//! the tools a model may load, with native regex matching shared by all hosts.

use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::tokenize::search::{self as abi, PatternError, ToolSearchError};
use serde_json::Value;

use crate::text::is_py_space;

/// One tool as the search sees it; identity stays with the host.
#[derive(Clone, Debug)]
pub struct Document {
    name: String,
    description: String,
    parameters: String,
    frequencies: BTreeMap<String, usize>,
    length: usize,
}

fn words(text: &str, identifiers: bool) -> Vec<String> {
    let text = text.to_lowercase();
    let text = if identifiers { text.replace('_', " ") } else { text };
    text.split(is_py_space).filter(|s| !s.is_empty()).map(str::to_owned).collect()
}

impl abi::SearchDocument for Document {
    fn new(name: String, description: String, parameters: Vec<(String, String)>) -> Self {
        let mut tokens = Vec::new();
        for _ in 0..3 {
            tokens.extend(words(&name, true));
        }
        for _ in 0..2 {
            tokens.extend(words(&description, false));
        }
        for (name, description) in &parameters {
            tokens.extend(words(name, true));
            tokens.extend(words(description, false));
        }
        let length = tokens.len();
        let mut frequencies = BTreeMap::new();
        for token in tokens {
            *frequencies.entry(token).or_default() += 1;
        }
        Self {
            name: name.to_lowercase(),
            description: description.to_lowercase(),
            parameters: parameters
                .iter()
                .map(|(k, v)| format!("{k} {v}"))
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase(),
            frequencies,
            length,
        }
    }

    fn keyword_score(&self, terms: Vec<String>) -> Result<f64, PatternError> {
        KeywordQuery::new(&terms).score(self)
    }
}

enum Pattern {
    Regex(fancy_regex::Regex),
    Literal(String),
}

struct KeywordQuery {
    patterns: Vec<Pattern>,
}

impl KeywordQuery {
    /// An invalid pattern is a literal search.
    fn new(terms: &[String]) -> Self {
        Self {
            patterns: terms
                .iter()
                .map(|term| match fancy_regex::Regex::new(term) {
                    Ok(regex) => Pattern::Regex(regex),
                    Err(_) => Pattern::Literal(term.clone()),
                })
                .collect(),
        }
    }

    /// A matching error (backtracking limit) reaches the host.
    fn score(&self, doc: &Document) -> Result<f64, PatternError> {
        let mut score = 0.0;
        for pattern in &self.patterns {
            for (text, weight) in
                [(&doc.name, 3.0), (&doc.description, 2.0), (&doc.parameters, 1.0)]
            {
                let matches = match pattern {
                    Pattern::Regex(regex) => {
                        regex.is_match(text).map_err(|e| PatternError(e.to_string()))?
                    }
                    Pattern::Literal(term) => text.contains(term),
                };
                if matches {
                    score += weight;
                }
            }
        }
        Ok(score)
    }
}

/// Stable ranking, including Python's negative slice-stop semantics.
fn rank(scores: &[f64], limit: i64) -> Vec<usize> {
    let mut ranked: Vec<_> = scores.iter().enumerate().filter(|(_, s)| **s > 0.0).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(a.1));
    let take = if limit < 0 {
        ranked.len().saturating_sub(limit.unsigned_abs() as usize)
    } else {
        limit as usize
    };
    ranked.into_iter().take(take).map(|(i, _)| i).collect()
}

pub struct Bm25 {
    k1: f64,
    b: f64,
    average_length: f64,
    idf: BTreeMap<String, f64>,
}

impl abi::SearchIndex for Bm25 {
    type Document = Document;

    fn new(k1: f64, b: f64) -> Self {
        Self { k1, b, average_length: 0.0, idf: BTreeMap::new() }
    }

    fn build(&mut self, documents: Vec<Document>) {
        self.clear();
        if documents.is_empty() {
            return;
        }
        let n = documents.len() as f64;
        self.average_length = documents.iter().map(|d| d.length).sum::<usize>() as f64 / n;
        let mut counts = BTreeMap::<String, usize>::new();
        for doc in &documents {
            for term in doc.frequencies.keys() {
                *counts.entry(term.clone()).or_default() += 1;
            }
        }
        self.idf = counts
            .into_iter()
            .map(|(term, count)| {
                (term, ((n - count as f64 + 0.5) / (count as f64 + 0.5) + 1.0).ln())
            })
            .collect();
    }

    fn clear(&mut self) {
        self.idf.clear();
        self.average_length = 0.0;
    }

    fn score(&self, doc: Document, terms: Vec<String>) -> f64 {
        self.score_of(&doc, &terms)
    }

    fn search(&self, query: String, docs: Vec<Document>, limit: i64) -> Vec<usize> {
        let terms = words(&query, true);
        rank(&docs.iter().map(|doc| self.score_of(doc, &terms)).collect::<Vec<_>>(), limit)
    }
}

impl Bm25 {
    fn score_of(&self, doc: &Document, terms: &[String]) -> f64 {
        terms
            .iter()
            .filter_map(|term| {
                self.idf.get(term).map(|idf| {
                    let frequency = *doc.frequencies.get(term).unwrap_or(&0) as f64;
                    let normalization = if self.average_length > 0.0 {
                        1.0 - self.b + self.b * doc.length as f64 / self.average_length
                    } else {
                        1.0
                    };
                    idf * frequency * (self.k1 + 1.0) / (frequency + self.k1 * normalization)
                })
            })
            .sum()
    }
}

pub struct ToolDiscovery;

impl abi::ToolSearch for ToolDiscovery {
    type Document = Document;

    /// The query compiles once and every document gets the same matching policy.
    fn keyword_search(
        query: String,
        docs: Vec<Document>,
        limit: i64,
    ) -> Result<Vec<usize>, PatternError> {
        let mut seen = BTreeSet::new();
        let terms: Vec<_> =
            words(&query, false).into_iter().filter(|s| seen.insert(s.clone())).collect();
        let query = KeywordQuery::new(&terms);
        let scores = docs.iter().map(|doc| query.score(doc)).collect::<Result<Vec<_>, _>>()?;
        Ok(rank(&scores, limit))
    }

    fn tool_search_validate(query: String) -> Result<(), ToolSearchError> {
        if query.is_empty() { Err(ToolSearchError("query cannot be empty".into())) } else { Ok(()) }
    }

    fn tool_search_loaded(query: String, count: usize) -> Result<String, ToolSearchError> {
        if count == 0 {
            Err(ToolSearchError(format!("No tools found matching '{query}'.")))
        } else {
            Ok("Tools loaded successfully.".into())
        }
    }

    /// `ToolProxyToolset`'s `call_tool` arguments: the tool's name and its parameters (an object,
    /// or the JSON string of one), refused as the model's error when missing, mistyped, or naming
    /// no tool in `known`.
    fn tool_proxy_arguments(
        arguments: Value,
        known: Vec<String>,
    ) -> Result<(String, Value), ToolSearchError> {
        let refuse = |message: String| Err(ToolSearchError(message));
        let name = match arguments.get("name") {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
        };
        if name.is_empty() {
            return refuse("tool name cannot be empty".into());
        }
        let parameters = match arguments.get("parameters") {
            None | Some(Value::Null) => return refuse("parameters is required".into()),
            Some(p @ (Value::Object(_) | Value::String(_))) => p.clone(),
            Some(_) => return refuse("parameters must be a dictionary or a string".into()),
        };
        if !known.contains(&name) {
            return refuse(format!(
                "unknown tool '{name}', use tool_search to discover available tools"
            ));
        }
        Ok((name, parameters))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use abi::{SearchDocument as _, SearchIndex as _, ToolSearch as _};

    fn doc(name: &str, description: &str) -> Document {
        Document::new(name.into(), description.into(), Vec::new())
    }

    #[test]
    fn native_regex_and_search_policy() {
        let doc = doc("book_book", "[ literal");
        for pattern in
            [r"(book)_\1", r"book(?=_book)", r"(?<=book_)book", r"(?P<word>book)_(?P=word)"]
        {
            assert_eq!(doc.keyword_score(vec![pattern.into()]).unwrap(), 3.0);
        }
        assert_eq!(doc.keyword_score(vec!["[".into()]).unwrap(), 2.0);
        let search =
            |limit| ToolDiscovery::keyword_search("book".into(), vec![doc.clone(); 2], limit);
        assert_eq!(search(1).unwrap(), vec![0]);
        assert_eq!(search(-1).unwrap(), vec![0]);
        assert!(ToolDiscovery::tool_search_validate(String::new()).is_err());
        assert!(ToolDiscovery::tool_search_validate(" ".into()).is_ok());
        assert_eq!(
            ToolDiscovery::tool_search_loaded("x".into(), 0).unwrap_err().0,
            "No tools found matching 'x'."
        );
        assert_eq!(
            ToolDiscovery::tool_search_loaded("x".into(), 1).unwrap(),
            "Tools loaded successfully."
        );
    }

    #[test]
    fn rare_terms_and_stable_ties() {
        let docs = vec![doc("weather", ""), doc("flight", ""), doc("flight", "")];
        let mut index = Bm25::new(1.5, 0.75);
        index.build(docs.clone());
        assert!(
            index.score(docs[0].clone(), vec!["weather".into()])
                > index.score(docs[1].clone(), vec!["flight".into()])
        );
        assert_eq!(index.search("flight".into(), docs.clone(), 5), vec![1, 2]);
        assert_eq!(index.search("flight".into(), docs.clone(), -1), vec![1]);
        index.clear();
        assert!(index.search("flight".into(), docs, 5).is_empty());
    }
}
