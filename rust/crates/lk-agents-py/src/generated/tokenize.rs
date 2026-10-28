//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `tokenize` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/tokenize/bindings.rs.
type TokenizeFunctionsImpl = <crate::Selected as super::Implementations>::TokenizeFunctions;

fn _extract_tokenize_functions_basic_split_sentences_min_sentence_len(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "min_sentence_len")
}

#[pyfunction]
#[pyo3(signature = (text, *, ignore_punctuation = true, split_character = false, retain_format = false))]
pub fn split_words(text: ::std::string::String, ignore_punctuation: bool, split_character: bool, retain_format: bool) -> PyResult<::std::vec::Vec<(::std::string::String, usize, usize)>> {
    let out = <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::split_words(text, ignore_punctuation, split_character, retain_format);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text, min_sentence_len = 20, *, retain_format = false))]
pub fn basic_split_sentences(text: ::std::string::String, #[pyo3(from_py_with = _extract_tokenize_functions_basic_split_sentences_min_sentence_len)] min_sentence_len: usize, retain_format: bool) -> PyResult<::std::vec::Vec<(::std::string::String, usize, usize)>> {
    let out = <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::basic_split_sentences(text, min_sentence_len, retain_format);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text, replacements, hold_last_word = false))]
pub fn replace_words(text: ::std::string::String, replacements: Bound<'_, PyAny>, hold_last_word: bool) -> PyResult<::std::option::Option<(::std::string::String, usize)>> {
    let replacements = replacements.cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(|dict| dict.iter().map(|(key, item)| Ok((key.extract::<::std::string::String>()?, item.extract::<::std::string::String>()?))).collect::<PyResult<::indexmap::IndexMap<_, _>>>())?;
    let out = <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::replace_words(text, replacements, hold_last_word);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text))]
pub fn split_paragraphs(text: ::std::string::String) -> PyResult<::std::vec::Vec<(::std::string::String, usize, usize)>> {
    let out = <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::split_paragraphs(text);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text))]
pub fn tokenize_paragraphs(text: ::std::string::String) -> PyResult<::std::vec::Vec<::std::string::String>> {
    let out = <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::tokenize_paragraphs(text);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text))]
pub fn xml_has_unclosed_tags(text: ::std::string::String) -> PyResult<bool> {
    let out = <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::xml_has_unclosed_tags(text);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (content, start_tag, end_tag, buffer, end_marker, is_final))]
pub fn strip_thinking(content: ::std::option::Option<::std::string::String>, start_tag: ::std::string::String, end_tag: ::std::string::String, buffer: ::std::string::String, end_marker: ::std::option::Option<::std::string::String>, is_final: bool) -> PyResult<(::std::option::Option<::std::string::String>, ::std::string::String, ::std::option::Option<::std::string::String>)> {
    let out = <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::strip_thinking(content, start_tag, end_tag, buffer, end_marker, is_final);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (parts))]
pub fn flatten_delta_text(parts: ::std::vec::Vec<::std::string::String>) -> PyResult<::std::option::Option<::std::string::String>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::flatten_delta_text(parts);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (tokenize_fnc, text))]
pub fn xml_tokenize(tokenize_fnc: Bound<'_, PyAny>, text: ::std::string::String) -> PyResult<::std::vec::Vec<(::std::string::String, usize, usize)>> {
    let tokenize_fnc = crate::generated::tokenize::TokenizerCallbackProxy::from_host(&tokenize_fnc)?;
    let out = <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::xml_tokenize(tokenize_fnc, text).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/bindings.rs.
type BasicWordTokenizerImpl = <crate::Selected as super::Implementations>::WordTokenizer;

/// Really naive implementation of a word tokenizer: words split on whitespace, punctuation
/// stripped unless asked to keep it.
#[pyclass(frozen, name = "BasicWordTokenizer", module = "livekit.agents._core")]
pub struct BasicWordTokenizer(pub(crate) ::std::sync::Arc<BasicWordTokenizerImpl>);

#[pymethods]
impl BasicWordTokenizer {
    #[new]
    #[pyo3(signature = (*, ignore_punctuation = true, split_character = false, retain_format = false))]
    fn new(ignore_punctuation: bool, split_character: bool, retain_format: bool) -> PyResult<BasicWordTokenizer> {
        let out = <BasicWordTokenizerImpl as ::lk_agents_abi::tokenize::bindings::WordTokenizer>::new(ignore_punctuation, split_character, retain_format);
        Ok(BasicWordTokenizer(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (text, *, language = None))]
    fn tokenize(&self, text: ::std::string::String, language: ::std::option::Option<::std::string::String>) -> PyResult<::std::vec::Vec<::std::string::String>> {
        let out = <BasicWordTokenizerImpl as ::lk_agents_abi::tokenize::bindings::WordTokenizer>::tokenize(&self.0, text, language);
        Ok(out)
    }

    #[pyo3(name = "_core_split")]
    #[pyo3(signature = (text))]
    fn split(&self, text: ::std::string::String) -> PyResult<::std::vec::Vec<(::std::string::String, usize, usize)>> {
        let out = <BasicWordTokenizerImpl as ::lk_agents_abi::tokenize::bindings::WordTokenizer>::split(&self.0, text);
        Ok(out)
    }

    #[getter]
    #[pyo3(name = "_core_min_token_len")]
    fn min_token_len(&self) -> PyResult<usize> {
        let out = <BasicWordTokenizerImpl as ::lk_agents_abi::tokenize::bindings::WordTokenizer>::min_token_len(&self.0);
        Ok(out)
    }

    #[getter]
    #[pyo3(name = "_core_min_ctx_len")]
    fn min_ctx_len(&self) -> PyResult<usize> {
        let out = <BasicWordTokenizerImpl as ::lk_agents_abi::tokenize::bindings::WordTokenizer>::min_ctx_len(&self.0);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/bindings.rs.
type BasicSentenceTokenizerImpl = <crate::Selected as super::Implementations>::SentenceTokenizer;

fn _extract_sentence_tokenizer_new_min_sentence_len(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "min_sentence_len")
}

fn _extract_sentence_tokenizer_new_stream_context_len(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "stream_context_len")
}

/// Really naive implementation of a sentence tokenizer: rule-based, and only English is really
/// tested.
#[pyclass(frozen, name = "BasicSentenceTokenizer", module = "livekit.agents._core")]
pub struct BasicSentenceTokenizer(pub(crate) ::std::sync::Arc<BasicSentenceTokenizerImpl>);

#[pymethods]
impl BasicSentenceTokenizer {
    #[new]
    #[pyo3(signature = (*, language = "english".to_owned(), min_sentence_len = 20, stream_context_len = 10, retain_format = false))]
    fn new(language: ::std::string::String, #[pyo3(from_py_with = _extract_sentence_tokenizer_new_min_sentence_len)] min_sentence_len: usize, #[pyo3(from_py_with = _extract_sentence_tokenizer_new_stream_context_len)] stream_context_len: usize, retain_format: bool) -> PyResult<BasicSentenceTokenizer> {
        let out = <BasicSentenceTokenizerImpl as ::lk_agents_abi::tokenize::bindings::SentenceTokenizer>::new(language, min_sentence_len, stream_context_len, retain_format);
        Ok(BasicSentenceTokenizer(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (text, *, language = None))]
    fn tokenize(&self, text: ::std::string::String, language: ::std::option::Option<::std::string::String>) -> PyResult<::std::vec::Vec<::std::string::String>> {
        let out = <BasicSentenceTokenizerImpl as ::lk_agents_abi::tokenize::bindings::SentenceTokenizer>::tokenize(&self.0, text, language);
        Ok(out)
    }

    #[pyo3(name = "_core_split")]
    #[pyo3(signature = (text))]
    fn split(&self, text: ::std::string::String) -> PyResult<::std::vec::Vec<(::std::string::String, usize, usize)>> {
        let out = <BasicSentenceTokenizerImpl as ::lk_agents_abi::tokenize::bindings::SentenceTokenizer>::split(&self.0, text);
        Ok(out)
    }

    #[getter]
    #[pyo3(name = "_core_min_token_len")]
    fn min_token_len(&self) -> PyResult<usize> {
        let out = <BasicSentenceTokenizerImpl as ::lk_agents_abi::tokenize::bindings::SentenceTokenizer>::min_token_len(&self.0);
        Ok(out)
    }

    #[getter]
    #[pyo3(name = "_core_min_ctx_len")]
    fn min_ctx_len(&self) -> PyResult<usize> {
        let out = <BasicSentenceTokenizerImpl as ::lk_agents_abi::tokenize::bindings::SentenceTokenizer>::min_ctx_len(&self.0);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/bindings.rs.
type TokenBufferImpl = <crate::Selected as super::Implementations>::BufferedTokens;

fn _extract_buffered_tokens_new_min_token_len(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "min_token_len")
}

fn _extract_buffered_tokens_new_min_ctx_len(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "min_ctx_len")
}

fn _extract_buffered_tokens_new_max_token_len(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<usize>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<usize>(&obj, "max_token_len")).map(Some) }
}

#[pyclass(name = "TokenBuffer", module = "livekit.agents._core")]
pub struct TokenBuffer(pub(crate) TokenBufferImpl);

#[pymethods]
impl TokenBuffer {
    #[new]
    #[pyo3(signature = (tokenize_fnc, *, min_token_len, min_ctx_len, max_token_len = None, xml_aware = false))]
    fn new(tokenize_fnc: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_buffered_tokens_new_min_token_len)] min_token_len: usize, #[pyo3(from_py_with = _extract_buffered_tokens_new_min_ctx_len)] min_ctx_len: usize, #[pyo3(from_py_with = _extract_buffered_tokens_new_max_token_len)] max_token_len: ::std::option::Option<usize>, xml_aware: bool) -> PyResult<TokenBuffer> {
        let tokenize_fnc = crate::generated::tokenize::TokenizerCallbackProxy::from_host(&tokenize_fnc)?;
        let out = <TokenBufferImpl as ::lk_agents_abi::tokenize::bindings::BufferedTokens>::new(tokenize_fnc, min_token_len, min_ctx_len, max_token_len, xml_aware);
        Ok(TokenBuffer(out))
    }

    /// The tokens ``text`` proves complete, batched.
    #[pyo3(signature = (text))]
    fn push(&mut self, text: ::std::string::String) -> PyResult<::std::vec::Vec<::std::string::String>> {
        let out = <TokenBufferImpl as ::lk_agents_abi::tokenize::bindings::BufferedTokens>::push(&mut self.0, text).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    fn flush(&mut self) -> PyResult<::std::vec::Vec<::std::string::String>> {
        let out = <TokenBufferImpl as ::lk_agents_abi::tokenize::bindings::BufferedTokens>::flush(&mut self.0).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/search.rs.
type SearchDocumentImpl = <crate::Selected as super::Implementations>::SearchDocument;

/// One tool as the search sees it; identity stays with the host.
#[pyclass(frozen, name = "SearchDocument", module = "livekit.agents._core")]
pub struct SearchDocument(pub(crate) ::std::sync::Arc<SearchDocumentImpl>);

#[pymethods]
impl SearchDocument {
    #[new]
    #[pyo3(signature = (name, description, parameters))]
    fn new(name: ::std::string::String, description: ::std::string::String, parameters: ::std::vec::Vec<(::std::string::String, ::std::string::String)>) -> PyResult<SearchDocument> {
        let out = <SearchDocumentImpl as ::lk_agents_abi::tokenize::search::SearchDocument>::new(name, description, parameters);
        Ok(SearchDocument(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (terms))]
    fn keyword_score(&self, terms: ::std::vec::Vec<::std::string::String>) -> PyResult<f64> {
        let out = <SearchDocumentImpl as ::lk_agents_abi::tokenize::search::SearchDocument>::keyword_score(&self.0, terms).map_err(|e: ::lk_agents_abi::tokenize::search::PatternError| <::lk_agents_abi::tokenize::search::PatternError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/search.rs.
type BM25IndexImpl = <crate::Selected as super::Implementations>::SearchIndex;

fn _extract_search_index_new_k1(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "k1")
}

fn _extract_search_index_new_b(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "b")
}

fn _extract_search_index_search_limit(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "limit")
}

#[pyclass(name = "BM25Index", module = "livekit.agents._core")]
pub struct BM25Index(pub(crate) BM25IndexImpl);

#[pymethods]
impl BM25Index {
    #[new]
    #[pyo3(signature = (k1, b))]
    fn new(#[pyo3(from_py_with = _extract_search_index_new_k1)] k1: f64, #[pyo3(from_py_with = _extract_search_index_new_b)] b: f64) -> PyResult<BM25Index> {
        let out = <BM25IndexImpl as ::lk_agents_abi::tokenize::search::SearchIndex>::new(k1, b);
        Ok(BM25Index(out))
    }

    #[pyo3(signature = (documents))]
    fn build(&mut self, documents: Bound<'_, PyAny>) -> PyResult<()> {
        let documents = documents.try_iter().and_then(|items| items.map(|item| { let item = item?; item.cast::<super::tokenize::SearchDocument>().map(|value| value.get().0.as_ref().clone()).map_err(::pyo3::PyErr::from) }).collect::<PyResult<Vec<_>>>())?;
        <BM25IndexImpl as ::lk_agents_abi::tokenize::search::SearchIndex>::build(&mut self.0, documents);
        Ok(())
    }

    fn clear(&mut self) -> PyResult<()> {
        <BM25IndexImpl as ::lk_agents_abi::tokenize::search::SearchIndex>::clear(&mut self.0);
        Ok(())
    }

    #[pyo3(signature = (doc, terms))]
    fn score(&self, doc: Bound<'_, PyAny>, terms: ::std::vec::Vec<::std::string::String>) -> PyResult<f64> {
        let doc = doc.cast::<super::tokenize::SearchDocument>().map(|value| value.get().0.as_ref().clone()).map_err(::pyo3::PyErr::from)?;
        let out = <BM25IndexImpl as ::lk_agents_abi::tokenize::search::SearchIndex>::score(&self.0, doc, terms);
        Ok(out)
    }

    #[pyo3(signature = (query, docs, limit))]
    fn search(&self, query: ::std::string::String, docs: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_search_index_search_limit)] limit: i64) -> PyResult<::std::vec::Vec<usize>> {
        let docs = docs.try_iter().and_then(|items| items.map(|item| { let item = item?; item.cast::<super::tokenize::SearchDocument>().map(|value| value.get().0.as_ref().clone()).map_err(::pyo3::PyErr::from) }).collect::<PyResult<Vec<_>>>())?;
        let out = <BM25IndexImpl as ::lk_agents_abi::tokenize::search::SearchIndex>::search(&self.0, query, docs, limit);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/search.rs.
type ToolSearchImpl = <crate::Selected as super::Implementations>::ToolSearch;

fn _extract_tool_search_keyword_search_limit(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "limit")
}

fn _extract_tool_search_tool_search_loaded_count(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "count")
}

#[pyfunction]
#[pyo3(signature = (query, docs, limit))]
pub fn keyword_search(query: ::std::string::String, docs: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_tool_search_keyword_search_limit)] limit: i64) -> PyResult<::std::vec::Vec<usize>> {
    let docs = docs.try_iter().and_then(|items| items.map(|item| { let item = item?; item.cast::<super::tokenize::SearchDocument>().map(|value| value.get().0.as_ref().clone()).map_err(::pyo3::PyErr::from) }).collect::<PyResult<Vec<_>>>())?;
    let out = <ToolSearchImpl as ::lk_agents_abi::tokenize::search::ToolSearch>::keyword_search(query, docs, limit).map_err(|e: ::lk_agents_abi::tokenize::search::PatternError| <::lk_agents_abi::tokenize::search::PatternError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (query))]
pub fn tool_search_validate(query: ::std::string::String) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    <ToolSearchImpl as ::lk_agents_abi::tokenize::search::ToolSearch>::tool_search_validate(query).map_err(|e: ::lk_agents_abi::tokenize::search::ToolSearchError| <::lk_agents_abi::tokenize::search::ToolSearchError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (query, count))]
pub fn tool_search_loaded(query: ::std::string::String, #[pyo3(from_py_with = _extract_tool_search_tool_search_loaded_count)] count: usize) -> PyResult<::std::string::String> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <ToolSearchImpl as ::lk_agents_abi::tokenize::search::ToolSearch>::tool_search_loaded(query, count).map_err(|e: ::lk_agents_abi::tokenize::search::ToolSearchError| <::lk_agents_abi::tokenize::search::ToolSearchError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (arguments, known))]
pub fn tool_proxy_arguments<'py>(py: Python<'py>, arguments: Bound<'_, PyAny>, known: ::std::vec::Vec<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    let arguments = ::lk_agents_abi::python::value_from_py(&arguments, "arguments")?;
    let out = <ToolSearchImpl as ::lk_agents_abi::tokenize::search::ToolSearch>::tool_proxy_arguments(arguments, known).map_err(|e: ::lk_agents_abi::tokenize::search::ToolSearchError| <::lk_agents_abi::tokenize::search::ToolSearchError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok({ let (x0, x1,) = out; ::pyo3::types::PyTuple::new(py, [::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?, ::lk_agents_abi::python::value_to_py(py, &x1)?])?.into_any() })
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/bindings.rs.

/// [`::lk_agents_abi::tokenize::bindings::TokenizerCallback`] over a host `collections.abc.Callable`.
#[derive(Clone)]
pub struct TokenizerCallbackProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl TokenizerCallbackProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::tokenize::bindings::TokenizerCallback>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::tokenize::bindings::TokenizerCallback>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::tokenize::bindings::TokenizerCallback>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(TokenizerCallbackProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::tokenize::bindings::TokenizerCallback>(|obj| TokenizerCallbackProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::tokenize::bindings::TokenizerCallback for TokenizerCallbackProxy {
    fn tokenize(&self, text: &str) -> ::std::result::Result<::std::vec::Vec<::lk_agents_abi::tokenize::bindings::TokenizerToken>, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("__call__")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| items.map(|item| { let item = item?; crate::generated::records::tokenizer::tokenizer_token_from_py(&item) }).collect::<PyResult<Vec<_>>>())).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}
