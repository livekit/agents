//! A `Vocabulary` enum's `assert_<enum>_is_covered::<G, S>()` requires a `Covers<K>` impl per
//! variant, so a new variant nobody has decided about stops compiling.

#[derive(Debug)]
pub enum Coverage<T> {
    Covered(T),
    /// Left out deliberately, for this stated reason.
    Skipped(&'static str),
}

impl<T> Coverage<T> {
    pub fn covered(self) -> Option<T> {
        match self {
            Coverage::Covered(with) => Some(with),
            Coverage::Skipped(_) => None,
        }
    }
}

/// "`Self` has decided what to do about variant `K`." One implementation per variant.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not cover `{K}`, so nothing here will ever produce that variant",
    label = "implement `Covers<{K}>` for `{Self}`: return `Coverage::Covered(..)` to handle it, or `Coverage::Skipped(\"why\")` to leave it out on purpose",
    note = "every variant of a driver boundary type must be covered or skipped -- lk-agents-voice/TEST_STRATEGY.md §5 \"Search-space inventory\""
)]
pub trait Covers<K> {
    /// Every variant of one vocabulary shares the type.
    type With;
    fn coverage() -> Coverage<Self::With>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_host_derive::Vocabulary;

    #[allow(dead_code)]
    #[derive(Vocabulary)]
    enum ProbeCommand {
        Say,
        UserTurn { text: String },
        Interrupt(bool),
    }

    struct Table;
    impl Covers<probe_command_vocabulary::Say> for Table {
        type With = &'static str;
        fn coverage() -> Coverage<&'static str> {
            Coverage::Covered("Just(Say)")
        }
    }
    impl Covers<probe_command_vocabulary::UserTurn> for Table {
        type With = &'static str;
        fn coverage() -> Coverage<&'static str> {
            Coverage::Covered("(frames, words)")
        }
    }
    impl Covers<probe_command_vocabulary::Interrupt> for Table {
        type With = &'static str;
        fn coverage() -> Coverage<&'static str> {
            Coverage::Skipped("shown skipped on purpose")
        }
    }

    // Catches: wrong or dropped keys, coverage out of declaration order, a skip counted as covered.
    #[test]
    fn the_derive_mints_one_key_per_variant_and_coverage_follows_declaration_order() {
        assert_probe_command_is_covered::<Table, &'static str>();
        assert_eq!(probe_command_vocabulary::NAMES, &["Say", "UserTurn", "Interrupt"]);
        let coverage = probe_command_coverage::<Table, &'static str>();
        let names: Vec<&str> = coverage.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["Say", "UserTurn", "Interrupt"]);
        let covered = coverage.iter().filter(|(_, c)| matches!(c, Coverage::Covered(_))).count();
        let skipped: Vec<&str> = coverage
            .iter()
            .filter_map(|(n, c)| matches!(c, Coverage::Skipped(_)).then_some(*n))
            .collect();
        assert_eq!((covered, skipped), (2, vec!["Interrupt"]));
    }
}
