use super::LegacyProjection;

/// One ordered projection change. The performer settles the old projection and
/// establishes the new one without pretending this changes either session's audio floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyPrimaryEffect {
    Cutover { previous: Option<LegacyProjection>, current: Option<LegacyProjection> },
}
