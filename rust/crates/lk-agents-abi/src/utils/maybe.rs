//! Omission is distinct from a value. `Maybe<Option<T>>` represents omitted, null, and non-null
//! values across the binding: `Agent(llm=None)` means "no LLM", not "unchanged".

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum Maybe<T> {
    #[default]
    NotGiven,
    /// For `Maybe<Option<T>>` an inner `None` is a *given* `None`.
    Given(T),
}

impl<T> Maybe<T> {
    pub fn is_given(&self) -> bool {
        matches!(self, Maybe::Given(_))
    }

    /// Not `into_option`, which would reintroduce the ambiguity.
    pub fn given(self) -> Option<T> {
        match self {
            Maybe::NotGiven => None,
            Maybe::Given(v) => Some(v),
        }
    }

    pub fn unwrap_or(self, default: T) -> T {
        self.given().unwrap_or(default)
    }
}

impl<T> std::fmt::Debug for Maybe<Option<T>> {
    /// `Arc<dyn Trait>` is not `Debug`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Maybe::NotGiven => "not-given",
            Maybe::Given(Some(_)) => "given",
            Maybe::Given(None) => "given-none",
        })
    }
}
