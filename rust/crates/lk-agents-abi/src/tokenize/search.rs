//! Built-in tool discovery, including native regex matching shared by all hosts.
use std::collections::{HashMap, HashSet};

/// One tool as the search sees it; identity stays with the host.
#[derive(Clone, Debug)]
#[crate::class(name = "SearchDocument", from_py_object)]
pub struct Document {
    pub name: String,
    pub description: String,
    pub parameters: String,
    pub frequencies: HashMap<String, usize>,
    pub length: usize,
}

fn words(text: &str, identifiers: bool) -> Vec<String> {
    let text = text.to_lowercase();
    let text = if identifiers { text.replace('_', " ") } else { text };
    text.split(super::is_py_space).filter(|s| !s.is_empty()).map(str::to_owned).collect()
}

/// One tool as the search sees it; identity stays with the host.
#[crate::host(object(name = "SearchDocument", pid = false))]
impl Document {
    #[host(new)]
    pub fn new(name: &str, description: &str, parameters: &[(String, String)]) -> Self {
        let mut tokens = Vec::new();
        for _ in 0..3 {
            tokens.extend(words(name, true));
        }
        for _ in 0..2 {
            tokens.extend(words(description, false));
        }
        for (name, description) in parameters {
            tokens.extend(words(name, true));
            tokens.extend(words(description, false));
        }
        let length = tokens.len();
        let mut frequencies = HashMap::new();
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

    #[host(method)]
    pub fn keyword_score(&self, terms: &[String]) -> Result<f64, fancy_regex::Error> {
        KeywordQuery::new(terms).score(self)
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
    fn score(&self, doc: &Document) -> Result<f64, fancy_regex::Error> {
        let mut score = 0.0;
        for pattern in &self.patterns {
            for (text, weight) in
                [(&doc.name, 3.0), (&doc.description, 2.0), (&doc.parameters, 1.0)]
            {
                let matches = match pattern {
                    Pattern::Regex(regex) => regex.is_match(text)?,
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

pub fn keyword_terms(query: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    words(query, false).into_iter().filter(|s| seen.insert(s.clone())).collect()
}

/// Compile the query once and apply the same native matching policy to every document.
/// Invalid patterns are literal searches; runtime matching errors are surfaced to the host.
#[crate::host(export(pid = false))]
pub fn keyword_search(
    query: &str,
    docs: &[Document],
    limit: isize,
) -> Result<Vec<usize>, fancy_regex::Error> {
    let query = KeywordQuery::new(&keyword_terms(query));
    let scores = docs.iter().map(|doc| query.score(doc)).collect::<Result<Vec<_>, _>>()?;
    Ok(rank(&scores, limit))
}

/// Stable ranking, including Python's negative slice-stop semantics.
pub fn rank(scores: &[f64], limit: isize) -> Vec<usize> {
    let mut ranked: Vec<_> = scores.iter().enumerate().filter(|(_, s)| **s > 0.0).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(a.1));
    let take =
        if limit < 0 { ranked.len().saturating_sub(limit.unsigned_abs()) } else { limit as usize };
    ranked.into_iter().take(take).map(|(i, _)| i).collect()
}

#[crate::class(name = "BM25Index", mutable)]
pub struct Bm25 {
    k1: f64,
    b: f64,
    average_length: f64,
    idf: HashMap<String, f64>,
}

#[crate::host(object(name = "BM25Index", pid = false))]
impl Bm25 {
    #[host(new)]
    pub fn new(k1: f64, b: f64) -> Self {
        Self { k1, b, average_length: 0.0, idf: HashMap::new() }
    }
    #[host(method)]
    pub fn build(&mut self, documents: &[Document]) {
        self.clear();
        if documents.is_empty() {
            return;
        }
        let n = documents.len() as f64;
        self.average_length = documents.iter().map(|d| d.length).sum::<usize>() as f64 / n;
        let mut counts = HashMap::<String, usize>::new();
        for doc in documents {
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
    #[host(method)]
    pub fn clear(&mut self) {
        self.idf.clear();
        self.average_length = 0.0;
    }
    #[host(method)]
    pub fn score(&self, doc: &Document, terms: &[String]) -> f64 {
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
    #[host(method)]
    pub fn search(&self, query: &str, docs: &[Document], limit: isize) -> Vec<usize> {
        let terms = words(query, true);
        rank(&docs.iter().map(|doc| self.score(doc, &terms)).collect::<Vec<_>>(), limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_regex_and_search_policy() {
        let doc = Document::new("book_book", "[ literal", &[]);
        for pattern in
            [r"(book)_\1", r"book(?=_book)", r"(?<=book_)book", r"(?P<word>book)_(?P=word)"]
        {
            assert_eq!(doc.keyword_score(&[pattern.into()]).unwrap(), 3.0);
        }
        assert_eq!(doc.keyword_score(&["[".into()]).unwrap(), 2.0);
        assert_eq!(keyword_search("book", &[doc.clone(), doc.clone()], 1).unwrap(), vec![0]);
        assert_eq!(keyword_search("book", &[doc.clone(), doc], -1).unwrap(), vec![0]);
        assert!(validate_query("").is_err());
        assert!(validate_query(" ").is_ok());
        assert_eq!(loaded_result("x", 0).unwrap_err(), "No tools found matching 'x'.");
        assert_eq!(loaded_result("x", 1).unwrap(), "Tools loaded successfully.");
    }
    #[test]
    fn rare_terms_and_stable_ties() {
        let docs = [
            Document::new("weather", "", &[]),
            Document::new("flight", "", &[]),
            Document::new("flight", "", &[]),
        ];
        let mut index = Bm25::new(1.5, 0.75);
        index.build(&docs);
        assert!(
            index.score(&docs[0], &["weather".into()]) > index.score(&docs[1], &["flight".into()])
        );
        assert_eq!(index.search("flight", &docs, 5), vec![1, 2]);
        assert_eq!(index.search("flight", &docs, -1), vec![1]);
        index.clear();
        assert!(index.search("flight", &docs, 5).is_empty());
    }
}

/// Host-independent search validation and model-facing result policy.
#[crate::host(export(name = "tool_search_validate", err = "livekit.agents.llm.tool_context.ToolError"))]
pub fn validate_query(query: &str) -> Result<(), &'static str> {
    if query.is_empty() { Err("query cannot be empty") } else { Ok(()) }
}
#[crate::host(export(name = "tool_search_loaded", err = "livekit.agents.llm.tool_context.ToolError"))]
pub fn loaded_result(query: &str, count: usize) -> Result<&'static str, String> {
    if count == 0 {
        Err(format!("No tools found matching '{query}'."))
    } else {
        Ok("Tools loaded successfully.")
    }
}
