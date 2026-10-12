use crate::transcription::filters::{MarkdownFilter, filter_emoji};

/// `_BUILTIN_TRANSFORMS`, in the reference's order.
pub const BUILTIN_TRANSFORMS: [&str; 2] = ["filter_markdown", "filter_emoji"];

pub trait TextTransform: Send {
    /// Feed a chunk; return the pieces that are now final. May be empty.
    fn push(&mut self, chunk: &str) -> Vec<String>;
    fn flush(&mut self) -> Vec<String>;
}

/// `ValueError: Invalid transform: ...`, raised eagerly at configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownTransform(pub String);

impl std::fmt::Display for UnknownTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid transform: {}, available transforms: {:?}", self.0, BUILTIN_TRANSFORMS)
    }
}

impl std::error::Error for UnknownTransform {}

/// `TextTransforms`: a builtin's name (a callable is refused at the boundary).
#[derive(Clone, Copy)]
pub enum TransformSpec {
    FilterMarkdown,
    FilterEmoji,
}

impl std::fmt::Debug for TransformSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformSpec::FilterMarkdown => f.write_str("filter_markdown"),
            TransformSpec::FilterEmoji => f.write_str("filter_emoji"),
        }
    }
}

/// Names only: a callable transform cannot be driven synchronously from the core, so it is refused
/// rather than silently dropped. An unknown name is refused eagerly, as the reference asks.
#[derive(Clone, Default)]
pub struct TransformSpecs(pub Vec<TransformSpec>);

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for TransformSpecs {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::exceptions::PyValueError;
        use pyo3::types::PyAnyMethods as _;
        let mut out = Vec::new();
        for item in obj.try_iter()? {
            let item = item?;
            let Ok(name) = item.extract::<String>() else {
                return Err(PyValueError::new_err(
                    "a callable tts_text_transform cannot cross into the core: the core's \
                     transform is `push(chunk) -> pieces` and a generator cannot be driven from \
                     it. Use \"filter_markdown\" / \"filter_emoji\", or apply your own transform \
                     in an `Agent.tts_node` override.",
                ));
            };
            out.push(TransformSpec::from_name(&name).map_err(|e| PyValueError::new_err(e.to_string()))?);
        }
        Ok(TransformSpecs(out))
    }
}

impl TransformSpec {
    pub fn from_name(name: &str) -> Result<TransformSpec, UnknownTransform> {
        match name {
            "filter_markdown" => Ok(TransformSpec::FilterMarkdown),
            "filter_emoji" => Ok(TransformSpec::FilterEmoji),
            other => Err(UnknownTransform(other.to_owned())),
        }
    }

    fn build(&self) -> Box<dyn TextTransform> {
        match self {
            TransformSpec::FilterMarkdown => Box::new(MarkdownTransform::default()),
            TransformSpec::FilterEmoji => Box::new(EmojiTransform),
        }
    }
}

/// [`filter_markdown`](crate::transcription::filters::MarkdownFilter) as a pipeline stage.
#[derive(Debug, Default)]
pub struct MarkdownTransform(MarkdownFilter);

impl TextTransform for MarkdownTransform {
    fn push(&mut self, chunk: &str) -> Vec<String> {
        self.0.push(chunk)
    }
    fn flush(&mut self) -> Vec<String> {
        self.0.flush()
    }
}

/// [`filter_emoji`] as a pipeline stage. Stateless, so it never holds a chunk back.
#[derive(Debug, Default)]
pub struct EmojiTransform;

impl TextTransform for EmojiTransform {
    fn push(&mut self, chunk: &str) -> Vec<String> {
        // Yields the filtered chunk even when empty, as the reference does.
        vec![filter_emoji(chunk)]
    }
    fn flush(&mut self) -> Vec<String> {
        Vec::new()
    }
}

/// `_apply_text_transforms`: the stages, in order.
pub struct Pipeline {
    stages: Vec<Box<dyn TextTransform>>,
}

impl Pipeline {
    /// Resolve and build. Fails on the first unknown name, before any text has been seen.
    pub fn new(specs: &[TransformSpec]) -> Pipeline {
        Pipeline { stages: specs.iter().map(TransformSpec::build).collect() }
    }
}

/// A pipeline is itself a transform, so one can be nested in another.
impl TextTransform for Pipeline {
    fn push(&mut self, chunk: &str) -> Vec<String> {
        let mut current = vec![chunk.to_owned()];
        for stage in &mut self.stages {
            let mut next = Vec::new();
            for piece in &current {
                next.extend(stage.push(piece));
            }
            current = next;
        }
        current
    }

    /// End of stream: each stage flushes after the pieces carried down from above.
    fn flush(&mut self) -> Vec<String> {
        let mut carried: Vec<String> = Vec::new();
        for stage in &mut self.stages {
            let mut next = Vec::new();
            for piece in &carried {
                next.extend(stage.push(piece));
            }
            next.extend(stage.flush());
            carried = next;
        }
        carried
    }
}

/// `text_transforms.replace`: substitute words as they stream past, holding back only a trailing
/// key prefix. `re.IGNORECASE`-style per-character folding via [`char::to_lowercase`].
#[lk_agents_abi::class(name = "ReplaceTransform", mutable)]
pub struct Replace {
    /// Keys longest-first (stable), so overlaps prefer the longer and ties keep insertion order.
    keys: Vec<Vec<char>>,
    values: Vec<String>,
    case_sensitive: bool,
    /// Every proper non-empty prefix of every key.
    prefixes: Vec<Vec<char>>,
    /// `max(len(k) - 1)`: the longest suffix that could still be a partial key.
    max_prefix: usize,
    buffer: String,
}

impl Replace {
    pub fn new<K: AsRef<str>, V: AsRef<str>>(
        replacements: &[(K, V)],
        case_sensitive: bool,
    ) -> Replace {
        let mut indexed: Vec<(Vec<char>, String)> = replacements
            .iter()
            .map(|(k, v)| (k.as_ref().chars().collect::<Vec<char>>(), v.as_ref().to_owned()))
            .collect();
        // `sorted(keys, key=len, reverse=True)` -- Python's sort is stable, and so is this one.
        indexed.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));

        let mut prefixes: Vec<Vec<char>> = Vec::new();
        let mut max_prefix = 0;
        for (k, _) in &indexed {
            max_prefix = max_prefix.max(k.len().saturating_sub(1));
            for n in 1..k.len() {
                let p = k[..n].to_vec();
                if !prefixes.contains(&p) {
                    prefixes.push(p);
                }
            }
        }
        // Longest first: `search` finds the leftmost start, which is the longest end-anchored run.
        prefixes.sort_by_key(|p| std::cmp::Reverse(p.len()));

        let (keys, values) = indexed.into_iter().unzip();
        Replace { keys, values, case_sensitive, prefixes, max_prefix, buffer: String::new() }
    }

    fn eq(&self, a: char, b: char) -> bool {
        if self.case_sensitive { a == b } else { a == b || a.to_lowercase().eq(b.to_lowercase()) }
    }

    fn matches_at(&self, text: &[char], at: usize, key: &[char]) -> bool {
        at + key.len() <= text.len()
            && key.iter().enumerate().all(|(i, &k)| self.eq(text[at + i], k))
    }

    /// `pattern.sub`: one left-to-right pass, longest key first at each position.
    fn apply(&self, text: &str) -> String {
        if self.keys.is_empty() {
            return text.to_owned();
        }
        let chars: Vec<char> = text.chars().collect();
        let mut out = String::with_capacity(text.len());
        let mut i = 0;
        while i < chars.len() {
            let hit = self.keys.iter().position(|k| !k.is_empty() && self.matches_at(&chars, i, k));
            match hit {
                Some(idx) => {
                    // Inserted literally, as `re.sub` with a function replacement.
                    out.push_str(&self.values[idx]);
                    i += self.keys[idx].len();
                }
                None => {
                    out.push(chars[i]);
                    i += 1;
                }
            }
        }
        out
    }

    /// How many characters at the end of `buffer` might still become a key.
    fn holdback(&self, buffer: &[char]) -> usize {
        if self.prefixes.is_empty() {
            return 0;
        }
        let window = buffer.len().min(self.max_prefix);
        for p in &self.prefixes {
            if p.len() <= window {
                let start = buffer.len() - p.len();
                if p.iter().enumerate().all(|(i, &c)| self.eq(buffer[start + i], c)) {
                    return p.len();
                }
            }
        }
        0
    }
}

/// `_core.ReplaceTransform`: the one pure transform that crosses as a value. `push(chunk)` /
/// `flush()` answer a list, not a joined string, because the synchroniser paces per piece; empty
/// means "held back". Not frozen: a transform is a buffer, and two threads pushing into one should
/// get `Already borrowed`, not an interleaved transcript.
#[lk_agents_abi::host(object(name = "ReplaceTransform"))]
impl Replace {
    /// `replacements` is a list of pairs because order breaks ties among equal-length keys.
    /// Case-insensitive matching is per-character `to_lowercase` (differs from `re.IGNORECASE` on
    /// e.g. `İ`). `case_sensitive` defaults to false, like `text_transforms.replace()`.
    #[host(new(defaults(case_sensitive = false)))]
    pub fn for_host(replacements: Vec<(String, String)>, case_sensitive: bool) -> Self {
        Replace::new(&replacements, case_sensitive)
    }

    #[host(method(name = "push"))]
    pub fn push_chunk(&mut self, chunk: &str) -> Vec<String> {
        TextTransform::push(self, chunk)
    }

    #[host(method(name = "flush"))]
    pub fn flush_held(&mut self) -> Vec<String> {
        TextTransform::flush(self)
    }
}

impl TextTransform for Replace {
    fn push(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.push_str(chunk);
        let applied = self.apply(&self.buffer);
        let chars: Vec<char> = applied.chars().collect();
        let flush_to = chars.len() - self.holdback(&chars);
        if flush_to == 0 {
            self.buffer = applied;
            return Vec::new();
        }
        self.buffer = chars[flush_to..].iter().collect();
        vec![chars[..flush_to].iter().collect()]
    }

    fn flush(&mut self) -> Vec<String> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        vec![std::mem::take(&mut self.buffer)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a key spanning a chunk boundary not being matched.
    #[test]
    fn a_replacement_survives_a_chunk_boundary() {
        let mut r = Replace::new(&[("livekit", "LiveKit")], true);
        let first = r.push("say live");
        assert!(!first.concat().contains("live"), "a key prefix must be held back: {first:?}");
        let mut out = first;
        out.extend(r.push("kit now"));
        out.extend(r.flush());
        assert_eq!(out.concat(), "say LiveKit now");
    }
}
