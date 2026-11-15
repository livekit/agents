use super::{Observed, Recording};

/// What a recogniser reported, in its own words: what the production speech rules
/// (`stt::normalizer::SpeechNormalizer`) take in. A provider that does not detect speech reports no
/// `Started` or `Ended`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SttReport {
    Started,
    Ended,
    Interim(String),
    Preflight(String),
    Final(String),
    /// Its connection dropped and it reconnected: the stream carries on.
    Dropped,
    /// It gave up: the stream fails.
    Failed,
    /// Its input ended and it had nothing more: the stream ends.
    Finished,
}

impl SttReport {
    /// A connection boundary: the open speech ends here, whatever it still owed.
    pub fn is_boundary(&self) -> bool {
        matches!(self, SttReport::Dropped | SttReport::Failed | SttReport::Finished)
    }
}

/// What a recognition stream handed its reader for one report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Recognised {
    Start,
    Interim(String),
    Preflight(String),
    Final(String),
    End,
}

/// One report to the n-th recognition stream (`Kind::Stt`), and what the stream made of it.
#[derive(Clone, Debug)]
pub struct Recognition {
    pub at: Observed,
    pub stt: u32,
    pub report: SttReport,
    pub emitted: Vec<Recognised>,
}

impl Recording {
    pub fn record_recognition(
        &mut self,
        at: Observed,
        stt: u32,
        report: SttReport,
        emitted: Vec<Recognised>,
    ) {
        self.recognition.push(Recognition { at, stt, report, emitted });
    }

    /// Every report a recognition stream took, in order, with what it handed its reader.
    pub fn recognition(&self) -> &[Recognition] {
        &self.recognition
    }

    /// Each start and end of speech a recognition stream handed its reader: `true` a start.
    pub fn recognised_edges(&self) -> impl Iterator<Item = (Observed, u32, bool)> + '_ {
        self.recognition.iter().flat_map(|r| {
            r.emitted.iter().filter_map(move |e| match e {
                Recognised::Start => Some((r.at, r.stt, true)),
                Recognised::End => Some((r.at, r.stt, false)),
                _ => None,
            })
        })
    }

    /// Whether a recognition stream started a speech after `since` and ended it.
    pub fn whole_speech_since(&self, since: Observed) -> bool {
        let mut started = std::collections::BTreeSet::new();
        self.recognised_edges().filter(|(at, ..)| *at > since).any(|(_, stt, start)| {
            if start {
                started.insert(stt);
                false
            } else {
                started.contains(&stt)
            }
        })
    }
}
