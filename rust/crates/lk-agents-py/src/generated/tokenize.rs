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
type TokenizeFunctionsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::TokenizeFunctions;
type TokenizeFunctionsSplitWordsOutput = ::std::vec::Vec<(::std::string::String, usize, usize)>;
type TokenizeFunctionsReplaceWordsOutput = ::std::option::Option<(::std::string::String, usize)>;
type TokenizeFunctionsTokenizeParagraphsOutput = ::std::vec::Vec<::std::string::String>;
type TokenizeFunctionsHyphenateWordOutput = ::std::vec::Vec<::std::string::String>;

#[pyfunction]
#[pyo3(signature = (text, *, ignore_punctuation = true, split_character = false, retain_format = false))]
pub fn split_words(
    py: Python<'_>,
    text: ::std::string::String,
    ignore_punctuation: bool,
    split_character: bool,
    retain_format: bool,
) -> PyResult<TokenizeFunctionsSplitWordsOutput> {
    let out = py.detach(|| <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::split_words(text, ignore_punctuation, split_character, retain_format));
    ::pyo3::PyResult::Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text, replacements, hold_last_word = false))]
pub fn replace_words(
    py: Python<'_>,
    text: ::std::string::String,
    replacements: Bound<'_, PyAny>,
    hold_last_word: bool,
) -> PyResult<TokenizeFunctionsReplaceWordsOutput> {
    let replacements = replacements
        .cast::<::pyo3::types::PyDict>()
        .map_err(::pyo3::PyErr::from)
        .and_then(|dict| {
            dict.iter()
                .map(|(key, item)| {
                    Ok((
                        key.extract::<::std::string::String>()?,
                        item.extract::<::std::string::String>()?,
                    ))
                })
                .collect::<PyResult<::indexmap::IndexMap<_, _>>>()
        })?;
    let out = py.detach(|| <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::replace_words(text, replacements, hold_last_word));
    ::pyo3::PyResult::Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text))]
pub fn tokenize_paragraphs(
    py: Python<'_>,
    text: ::std::string::String,
) -> PyResult<TokenizeFunctionsTokenizeParagraphsOutput> {
    let out = py.detach(|| <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::tokenize_paragraphs(text));
    ::pyo3::PyResult::Ok(out)
}

/// `hyphenate_word(word)`: the pieces, broken at the possible hyphenation points.
#[pyfunction]
#[pyo3(signature = (word))]
pub fn hyphenate_word(
    py: Python<'_>,
    word: ::std::string::String,
) -> PyResult<TokenizeFunctionsHyphenateWordOutput> {
    let out = py.detach(|| <TokenizeFunctionsImpl as ::lk_agents_abi::tokenize::bindings::TokenizeFunctions>::hyphenate_word(word));
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/bindings.rs.
type HostTokenizationImpl =
    <crate::generated::implementations::Selected as super::Implementations>::HostTokenization;
type HostTokenizationXmlTokenizeOutput = ::std::vec::Vec<(::std::string::String, usize, usize)>;

#[pyfunction]
#[pyo3(signature = (tokenize_fnc, text))]
pub fn xml_tokenize(
    py: Python<'_>,
    tokenize_fnc: Bound<'_, PyAny>,
    text: ::std::string::String,
) -> PyResult<HostTokenizationXmlTokenizeOutput> {
    let tokenize_fnc =
        crate::generated::tokenize::TokenizerCallbackProxy::from_host(tokenize_fnc.as_any())?;
    let out = py.detach(|| <HostTokenizationImpl as ::lk_agents_abi::tokenize::bindings::HostTokenization>::xml_tokenize(tokenize_fnc, text)).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/bindings.rs.
type BasicWordTokenizerImpl =
    <crate::generated::implementations::Selected as super::Implementations>::WordTokenizer;
type WordTokenizerTokenizeOutput = ::std::vec::Vec<::std::string::String>;
type WordTokenizerSplitOutput = ::std::vec::Vec<(::std::string::String, usize, usize)>;

/// Really naive implementation of a word tokenizer: words split on whitespace, punctuation
/// stripped unless asked to keep it.
#[pyclass(frozen, name = "BasicWordTokenizer", module = "livekit.agents._core")]
pub struct BasicWordTokenizer(pub(crate) ::std::sync::Arc<BasicWordTokenizerImpl>);

#[pymethods]
impl BasicWordTokenizer {
    #[new]
    #[pyo3(signature = (*, ignore_punctuation = true, split_character = false, retain_format = false))]
    fn new(
        py: Python<'_>,
        ignore_punctuation: bool,
        split_character: bool,
        retain_format: bool,
    ) -> PyResult<BasicWordTokenizer> {
        let out = py.detach(|| {
            <BasicWordTokenizerImpl as ::lk_agents_abi::tokenize::bindings::WordTokenizer>::new(
                ignore_punctuation,
                split_character,
                retain_format,
            )
        });
        ::pyo3::PyResult::Ok(BasicWordTokenizer(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (text, *, language = None))]
    fn tokenize(
        &self,
        py: Python<'_>,
        text: ::std::string::String,
        language: ::std::option::Option<::std::string::String>,
    ) -> PyResult<WordTokenizerTokenizeOutput> {
        let out = py.detach(|| <BasicWordTokenizerImpl as ::lk_agents_abi::tokenize::bindings::WordTokenizer>::tokenize(&self.0, text, language));
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(name = "_core_split")]
    #[pyo3(signature = (text))]
    fn split(
        &self,
        py: Python<'_>,
        text: ::std::string::String,
    ) -> PyResult<WordTokenizerSplitOutput> {
        let out = py.detach(|| {
            <BasicWordTokenizerImpl as ::lk_agents_abi::tokenize::bindings::WordTokenizer>::split(
                &self.0, text,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    #[pyo3(name = "_core_min_token_len")]
    fn min_token_len(&self, py: Python<'_>) -> PyResult<usize> {
        let out = py.detach(|| <BasicWordTokenizerImpl as ::lk_agents_abi::tokenize::bindings::WordTokenizer>::min_token_len(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    #[pyo3(name = "_core_min_ctx_len")]
    fn min_ctx_len(&self, py: Python<'_>) -> PyResult<usize> {
        let out = py.detach(|| <BasicWordTokenizerImpl as ::lk_agents_abi::tokenize::bindings::WordTokenizer>::min_ctx_len(&self.0));
        ::pyo3::PyResult::Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/bindings.rs.
type BasicSentenceTokenizerImpl =
    <crate::generated::implementations::Selected as super::Implementations>::SentenceTokenizer;

fn _extract_sentence_tokenizer_new_min_sentence_len(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SentenceTokenizerNewMinSentenceLenInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "min_sentence_len")
}
type SentenceTokenizerNewMinSentenceLenInput = usize;

fn _extract_sentence_tokenizer_new_stream_context_len(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SentenceTokenizerNewStreamContextLenInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "stream_context_len")
}
type SentenceTokenizerNewStreamContextLenInput = usize;
type SentenceTokenizerTokenizeOutput = ::std::vec::Vec<::std::string::String>;
type SentenceTokenizerSplitOutput = ::std::vec::Vec<(::std::string::String, usize, usize)>;

/// Really naive implementation of a sentence tokenizer: rule-based, and only English is really
/// tested.
#[pyclass(frozen, name = "BasicSentenceTokenizer", module = "livekit.agents._core")]
pub struct BasicSentenceTokenizer(pub(crate) ::std::sync::Arc<BasicSentenceTokenizerImpl>);

#[pymethods]
impl BasicSentenceTokenizer {
    #[new]
    #[pyo3(signature = (*, language = "english".to_owned(), min_sentence_len = 20, stream_context_len = 10, retain_format = false))]
    fn new(
        py: Python<'_>,
        language: ::std::string::String,
        #[pyo3(from_py_with = _extract_sentence_tokenizer_new_min_sentence_len)]
        min_sentence_len: SentenceTokenizerNewMinSentenceLenInput,
        #[pyo3(from_py_with = _extract_sentence_tokenizer_new_stream_context_len)]
        stream_context_len: SentenceTokenizerNewStreamContextLenInput,
        retain_format: bool,
    ) -> PyResult<BasicSentenceTokenizer> {
        let out = py.detach(|| <BasicSentenceTokenizerImpl as ::lk_agents_abi::tokenize::bindings::SentenceTokenizer>::new(language, min_sentence_len, stream_context_len, retain_format));
        ::pyo3::PyResult::Ok(BasicSentenceTokenizer(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (text, *, language = None))]
    fn tokenize(
        &self,
        py: Python<'_>,
        text: ::std::string::String,
        language: ::std::option::Option<::std::string::String>,
    ) -> PyResult<SentenceTokenizerTokenizeOutput> {
        let out = py.detach(|| <BasicSentenceTokenizerImpl as ::lk_agents_abi::tokenize::bindings::SentenceTokenizer>::tokenize(&self.0, text, language));
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(name = "_core_split")]
    #[pyo3(signature = (text))]
    fn split(
        &self,
        py: Python<'_>,
        text: ::std::string::String,
    ) -> PyResult<SentenceTokenizerSplitOutput> {
        let out = py.detach(|| <BasicSentenceTokenizerImpl as ::lk_agents_abi::tokenize::bindings::SentenceTokenizer>::split(&self.0, text));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    #[pyo3(name = "_core_min_token_len")]
    fn min_token_len(&self, py: Python<'_>) -> PyResult<usize> {
        let out = py.detach(|| <BasicSentenceTokenizerImpl as ::lk_agents_abi::tokenize::bindings::SentenceTokenizer>::min_token_len(&self.0));
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    #[pyo3(name = "_core_min_ctx_len")]
    fn min_ctx_len(&self, py: Python<'_>) -> PyResult<usize> {
        let out = py.detach(|| <BasicSentenceTokenizerImpl as ::lk_agents_abi::tokenize::bindings::SentenceTokenizer>::min_ctx_len(&self.0));
        ::pyo3::PyResult::Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/bindings.rs.
type TokenBufferImpl =
    <crate::generated::implementations::Selected as super::Implementations>::BufferedTokens;

fn _extract_buffered_tokens_new_min_token_len(
    obj: &Bound<'_, PyAny>,
) -> PyResult<BufferedTokensNewMinTokenLenInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "min_token_len")
}
type BufferedTokensNewMinTokenLenInput = usize;

fn _extract_buffered_tokens_new_min_ctx_len(
    obj: &Bound<'_, PyAny>,
) -> PyResult<BufferedTokensNewMinCtxLenInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "min_ctx_len")
}
type BufferedTokensNewMinCtxLenInput = usize;

fn _extract_buffered_tokens_new_max_token_len(
    obj: &Bound<'_, PyAny>,
) -> PyResult<BufferedTokensNewMaxTokenLenInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<usize>(obj.as_any(), "max_token_len")).map(Some)
    }
}
type BufferedTokensNewMaxTokenLenInput = ::std::option::Option<usize>;
type BufferedTokensPushOutput = ::std::vec::Vec<::std::string::String>;
type BufferedTokensFlushOutput = ::std::vec::Vec<::std::string::String>;

#[pyclass(name = "TokenBuffer", module = "livekit.agents._core")]
pub struct TokenBuffer(pub(crate) TokenBufferImpl);

#[pymethods]
impl TokenBuffer {
    #[new]
    #[pyo3(signature = (tokenize_fnc, *, min_token_len, min_ctx_len, max_token_len = None, xml_aware = false))]
    fn new(
        py: Python<'_>,
        tokenize_fnc: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_buffered_tokens_new_min_token_len)]
        min_token_len: BufferedTokensNewMinTokenLenInput,
        #[pyo3(from_py_with = _extract_buffered_tokens_new_min_ctx_len)]
        min_ctx_len: BufferedTokensNewMinCtxLenInput,
        #[pyo3(from_py_with = _extract_buffered_tokens_new_max_token_len)]
        max_token_len: BufferedTokensNewMaxTokenLenInput,
        xml_aware: bool,
    ) -> PyResult<TokenBuffer> {
        let tokenize_fnc =
            crate::generated::tokenize::TokenizerCallbackProxy::from_host(tokenize_fnc.as_any())?;
        let out = py.detach(|| {
            <TokenBufferImpl as ::lk_agents_abi::tokenize::bindings::BufferedTokens>::new(
                tokenize_fnc,
                min_token_len,
                min_ctx_len,
                max_token_len,
                xml_aware,
            )
        });
        ::pyo3::PyResult::Ok(TokenBuffer(out))
    }

    /// The tokens ``text`` proves complete, batched.
    #[pyo3(signature = (text))]
    fn push(&mut self, text: ::std::string::String) -> PyResult<BufferedTokensPushOutput> {
        let out = <TokenBufferImpl as ::lk_agents_abi::tokenize::bindings::BufferedTokens>::push(
            &mut self.0,
            text,
        )
        .map_err(|e: ::lk_agents_abi::errors::HostError| {
            <::lk_agents_abi::errors::HostError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        ::pyo3::PyResult::Ok(out)
    }

    fn flush(&mut self) -> PyResult<BufferedTokensFlushOutput> {
        let out = <TokenBufferImpl as ::lk_agents_abi::tokenize::bindings::BufferedTokens>::flush(
            &mut self.0,
        )
        .map_err(|e: ::lk_agents_abi::errors::HostError| {
            <::lk_agents_abi::errors::HostError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        ::pyo3::PyResult::Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/language/code.rs.
type LanguageFunctionsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::LanguageFunctions;
type LanguageFunctionsRegionOutput = ::std::option::Option<::std::string::String>;
type LanguageFunctionsLanguageNameOutput = ::std::option::Option<::std::string::String>;

/// Normalise a language code or name to BCP-47.
#[pyfunction]
#[pyo3(signature = (code))]
pub fn normalize_language(
    py: Python<'_>,
    code: ::std::string::String,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| <LanguageFunctionsImpl as ::lk_agents_abi::tokenize::language::LanguageFunctions>::normalize(&code));
    ::pyo3::PyResult::Ok(out)
}

/// Base language code (ISO 639-1 when possible).
///
/// E.g., ``'en'`` from ``'en-US'``, ``'zh'`` from ``'cmn-Hans-CN'``.
#[pyfunction]
#[pyo3(signature = (code))]
pub fn base_language(
    py: Python<'_>,
    code: ::std::string::String,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| <LanguageFunctionsImpl as ::lk_agents_abi::tokenize::language::LanguageFunctions>::base_language(&code));
    ::pyo3::PyResult::Ok(out)
}

/// Region code, e.g., ``'US'`` from ``'en-US'``, or ``None``.
#[pyfunction]
#[pyo3(signature = (code))]
pub fn language_region(
    py: Python<'_>,
    code: ::std::string::String,
) -> PyResult<LanguageFunctionsRegionOutput> {
    let out = py.detach(|| {
        <LanguageFunctionsImpl as ::lk_agents_abi::tokenize::language::LanguageFunctions>::region(
            &code,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

/// ISO 639-1 tag with region, e.g., ``'zh-CN'`` from ``'cmn-Hans-CN'``.
#[pyfunction]
#[pyo3(signature = (code))]
pub fn language_iso(
    py: Python<'_>,
    code: ::std::string::String,
) -> PyResult<::std::string::String> {
    let out = py.detach(|| {
        <LanguageFunctionsImpl as ::lk_agents_abi::tokenize::language::LanguageFunctions>::iso(
            &code,
        )
    });
    ::pyo3::PyResult::Ok(out)
}

/// Return the English language name (e.g. ``'english'``), or ``None`` if unknown.
#[pyfunction]
#[pyo3(signature = (code))]
pub fn language_name(
    py: Python<'_>,
    code: ::std::string::String,
) -> PyResult<LanguageFunctionsLanguageNameOutput> {
    let out = py.detach(|| <LanguageFunctionsImpl as ::lk_agents_abi::tokenize::language::LanguageFunctions>::language_name(&code));
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/search.rs.
type SearchDocumentImpl =
    <crate::generated::implementations::Selected as super::Implementations>::SearchDocument;

/// One tool as the search sees it; identity stays with the host.
#[pyclass(frozen, name = "SearchDocument", module = "livekit.agents._core")]
pub struct SearchDocument(pub(crate) ::std::sync::Arc<SearchDocumentImpl>);

#[pymethods]
impl SearchDocument {
    #[new]
    #[pyo3(signature = (name, description, parameters))]
    fn new(
        py: Python<'_>,
        name: ::std::string::String,
        description: ::std::string::String,
        parameters: ::std::vec::Vec<(::std::string::String, ::std::string::String)>,
    ) -> PyResult<SearchDocument> {
        let out = py.detach(|| {
            <SearchDocumentImpl as ::lk_agents_abi::tokenize::search::SearchDocument>::new(
                name,
                description,
                parameters,
            )
        });
        ::pyo3::PyResult::Ok(SearchDocument(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (terms))]
    fn keyword_score(
        &self,
        py: Python<'_>,
        terms: ::std::vec::Vec<::std::string::String>,
    ) -> PyResult<f64> {
        let out = py.detach(|| <SearchDocumentImpl as ::lk_agents_abi::tokenize::search::SearchDocument>::keyword_score(&self.0, terms)).map_err(|e: ::lk_agents_abi::tokenize::search::PatternError| <::lk_agents_abi::tokenize::search::PatternError as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/search.rs.
type BM25IndexImpl =
    <crate::generated::implementations::Selected as super::Implementations>::SearchIndex;

fn _extract_search_index_new_k1(obj: &Bound<'_, PyAny>) -> PyResult<SearchIndexNewK1Input> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "k1")
}
type SearchIndexNewK1Input = f64;

fn _extract_search_index_new_b(obj: &Bound<'_, PyAny>) -> PyResult<SearchIndexNewBInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "b")
}
type SearchIndexNewBInput = f64;
type SearchIndexSearchOutput = ::std::vec::Vec<usize>;

fn _extract_search_index_search_limit(
    obj: &Bound<'_, PyAny>,
) -> PyResult<SearchIndexSearchLimitInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "limit")
}
type SearchIndexSearchLimitInput = i64;

#[pyclass(name = "BM25Index", module = "livekit.agents._core")]
pub struct BM25Index(pub(crate) BM25IndexImpl);

#[pymethods]
impl BM25Index {
    #[new]
    #[pyo3(signature = (k1, b))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_search_index_new_k1)] k1: SearchIndexNewK1Input,
        #[pyo3(from_py_with = _extract_search_index_new_b)] b: SearchIndexNewBInput,
    ) -> PyResult<BM25Index> {
        let out = py.detach(|| {
            <BM25IndexImpl as ::lk_agents_abi::tokenize::search::SearchIndex>::new(k1, b)
        });
        ::pyo3::PyResult::Ok(BM25Index(out))
    }

    #[pyo3(signature = (documents))]
    fn build(&mut self, documents: Bound<'_, PyAny>) -> PyResult<()> {
        let documents = documents.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    item.cast::<super::tokenize::SearchDocument>()
                        .map(|value| value.get().0.as_ref().clone())
                        .map_err(::pyo3::PyErr::from)
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        <BM25IndexImpl as ::lk_agents_abi::tokenize::search::SearchIndex>::build(
            &mut self.0,
            documents,
        );
        Ok(())
    }

    fn clear(&mut self) -> PyResult<()> {
        <BM25IndexImpl as ::lk_agents_abi::tokenize::search::SearchIndex>::clear(&mut self.0);
        Ok(())
    }

    #[pyo3(signature = (doc, terms))]
    fn score(
        &self,
        doc: Bound<'_, PyAny>,
        terms: ::std::vec::Vec<::std::string::String>,
    ) -> PyResult<f64> {
        let doc = doc
            .cast::<super::tokenize::SearchDocument>()
            .map(|value| value.get().0.as_ref().clone())
            .map_err(::pyo3::PyErr::from)?;
        let out = <BM25IndexImpl as ::lk_agents_abi::tokenize::search::SearchIndex>::score(
            &self.0, doc, terms,
        );
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (query, docs, limit))]
    fn search(
        &self,
        query: ::std::string::String,
        docs: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_search_index_search_limit)]
        limit: SearchIndexSearchLimitInput,
    ) -> PyResult<SearchIndexSearchOutput> {
        let docs = docs.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    item.cast::<super::tokenize::SearchDocument>()
                        .map(|value| value.get().0.as_ref().clone())
                        .map_err(::pyo3::PyErr::from)
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        let out = <BM25IndexImpl as ::lk_agents_abi::tokenize::search::SearchIndex>::search(
            &self.0, query, docs, limit,
        );
        ::pyo3::PyResult::Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/search.rs.
type ToolSearchImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ToolSearch;
type ToolSearchKeywordSearchOutput = ::std::vec::Vec<usize>;

fn _extract_tool_search_keyword_search_limit(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ToolSearchKeywordSearchLimitInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "limit")
}
type ToolSearchKeywordSearchLimitInput = i64;

fn _extract_tool_search_tool_search_loaded_count(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ToolSearchToolSearchLoadedCountInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "count")
}
type ToolSearchToolSearchLoadedCountInput = usize;

#[pyfunction]
#[pyo3(signature = (query, docs, limit))]
pub fn keyword_search(
    py: Python<'_>,
    query: ::std::string::String,
    docs: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_tool_search_keyword_search_limit)]
    limit: ToolSearchKeywordSearchLimitInput,
) -> PyResult<ToolSearchKeywordSearchOutput> {
    let docs = docs.try_iter().and_then(|items| {
        items
            .map(|item| {
                let item = item?;
                item.cast::<super::tokenize::SearchDocument>()
                    .map(|value| value.get().0.as_ref().clone())
                    .map_err(::pyo3::PyErr::from)
            })
            .collect::<PyResult<Vec<_>>>()
    })?;
    let out = py.detach(|| <ToolSearchImpl as ::lk_agents_abi::tokenize::search::ToolSearch>::keyword_search(query, docs, limit)).map_err(|e: ::lk_agents_abi::tokenize::search::PatternError| <::lk_agents_abi::tokenize::search::PatternError as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok(out)
}

#[pyfunction]
#[pyo3(signature = (query))]
pub fn tool_search_validate(py: Python<'_>, query: ::std::string::String) -> PyResult<()> {
    crate::py_calls::check_pid()?;
    py.detach(|| <ToolSearchImpl as ::lk_agents_abi::tokenize::search::ToolSearch>::tool_search_validate(query)).map_err(|e: ::lk_agents_abi::tokenize::search::ToolSearchError| <::lk_agents_abi::tokenize::search::ToolSearchError as crate::py_errors::HostException>::to_py_err(&e))?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (query, count))]
pub fn tool_search_loaded(
    py: Python<'_>,
    query: ::std::string::String,
    #[pyo3(from_py_with = _extract_tool_search_tool_search_loaded_count)]
    count: ToolSearchToolSearchLoadedCountInput,
) -> PyResult<::std::string::String> {
    crate::py_calls::check_pid()?;
    let out = py.detach(|| <ToolSearchImpl as ::lk_agents_abi::tokenize::search::ToolSearch>::tool_search_loaded(query, count)).map_err(|e: ::lk_agents_abi::tokenize::search::ToolSearchError| <::lk_agents_abi::tokenize::search::ToolSearchError as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok(out)
}

#[pyfunction]
#[pyo3(signature = (arguments, known))]
pub fn tool_proxy_arguments<'py>(
    py: Python<'py>,
    arguments: Bound<'_, PyAny>,
    known: ::std::vec::Vec<::std::string::String>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let arguments = crate::py_values::value_from_py(arguments.as_any(), "arguments")?;
    let out = py.detach(|| <ToolSearchImpl as ::lk_agents_abi::tokenize::search::ToolSearch>::tool_proxy_arguments(arguments, known)).map_err(|e: ::lk_agents_abi::tokenize::search::ToolSearchError| <::lk_agents_abi::tokenize::search::ToolSearchError as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok({
        let (x0, x1) = out;
        ::pyo3::types::PyTuple::new(
            py,
            [
                ::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?,
                crate::py_values::value_to_py(py, &x1)?,
            ],
        )?
        .into_any()
    })
}

// Declared in rust/crates/lk-agents-abi/src/tokenize/bindings.rs.

/// [`::lk_agents_abi::tokenize::bindings::TokenizerCallback`] over a host `collections.abc.Callable`.
#[derive(Clone)]
pub struct TokenizerCallbackProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl TokenizerCallbackProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::tokenize::bindings::TokenizerCallback>>
    {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::tokenize::bindings::TokenizerCallback,
        >(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::tokenize::bindings::TokenizerCallback>>
    {
        if let Some(native) = crate::py_objects::native::<
            dyn ::lk_agents_abi::tokenize::bindings::TokenizerCallback,
        >(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::tokenize::bindings::TokenizerCallback>>
    {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(TokenizerCallbackProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::tokenize::bindings::TokenizerCallback for TokenizerCallbackProxy {
    fn tokenize(
        &self,
        text: &str,
    ) -> ::std::result::Result<
        ::std::vec::Vec<::lk_agents_abi::tokenize::bindings::TokenizerToken>,
        ::lk_agents_abi::errors::HostError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let text_arg = text.to_owned();
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [::pyo3::IntoPyObjectExt::into_bound_py_any(text_arg, py)?];
                Ok((host.bind(py).getattr("__call__")?, ::pyo3::types::PyTuple::new(py, args)?))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        crate::generated::records::tokenizer::tokenizer_token_from_py(item.as_any())
                    })
                    .collect::<PyResult<Vec<_>>>()
            }))
            .map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::tokenize::search::PatternError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::tokenize::search::ToolSearchError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        crate::py_errors::host_exception(
            "livekit.agents.llm.tool_context",
            "ToolError",
            ::std::string::ToString::to_string(self),
        )
    }
}
