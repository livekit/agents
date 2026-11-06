//! Job profiles (docs/internals/telemetry.md "Job profiles"): a job's wall-clock samples of its
//! Python threads, kept in time order and encoded as a gzipped pprof `profile.proto` that Cloud
//! stores beside the session's audio recording. Each sample carries its thread and its time
//! (`end_timestamp_ns`), so a viewer can line it up with the recording's start.

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::{Arc, LazyLock};

use lk_agents_abi::telemetry::record::JobKey;
use parking_lot::Mutex;
use prost::Message as _;

/// The sampler's period, and the longest wall time one sample stands for.
pub const PERIOD_NANOS: i64 = 100_000_000;
const LONGEST_SAMPLE_NANOS: i64 = 10 * PERIOD_NANOS;
/// Samples kept with their own time (24 bytes each); later ones are merged by stack and thread.
pub const TIMED_SAMPLES: usize = 36_000;
/// Distinct stacks kept; a sample of a stack not seen before past them is counted, not kept.
pub const STACKS: usize = 20_000;
/// A stack's innermost frames kept; a deeper stack gets a `<truncated>` root frame.
pub const DEPTH: usize = 64;
/// Distinct thread names; the rest are counted as `other`.
pub const THREADS: usize = 64;
/// The largest encoded profile the recording upload carries: one Cloud refuses would lose the
/// whole recording.
pub const UPLOAD_BYTES: usize = 4 * 1024 * 1024;

/// One frame of a sampled stack, innermost first.
pub struct Frame<'a> {
    pub function: &'a str,
    pub file: &'a str,
    pub line: i64,
}

const TRUNCATED: Frame<'static> = Frame { function: "<truncated>", file: "", line: 0 };

/// A job's samples, interned as pprof stores them. Every table is bounded: [`STACKS`] stacks of
/// at most [`DEPTH`] + 1 frames (and so their functions, locations and strings), [`THREADS`]
/// thread names, [`TIMED_SAMPLES`] timed samples, one merged count per (stack, thread).
#[derive(Default)]
pub struct Profile {
    start_nanos: u64,
    last_nanos: u64,
    strings: Vec<String>,
    string_ids: HashMap<String, i64>,
    functions: HashMap<(i64, i64), u64>,
    locations: HashMap<(u64, i64), u64>,
    function_list: Vec<pprof::Function>,
    location_list: Vec<pprof::Location>,
    stacks: HashMap<Arc<[u64]>, u32>,
    stack_list: Vec<Arc<[u64]>>,
    threads: HashMap<String, i64>,
    /// (stack, thread name, Unix ns, wall ns, running), in time order.
    timed: Vec<(u32, i64, u64, i64, bool)>,
    merged: HashMap<(u32, i64, bool), i64>,
    /// Samples of a new stack past [`STACKS`].
    dropped: u64,
}

impl Profile {
    /// Adds the stack `frames` (innermost first) of `thread`, sampled at `at_nanos` (Unix ns),
    /// standing for `wall_nanos` of wall time; `running`: the thread was running, not waiting.
    pub fn add<'a>(
        &mut self,
        at_nanos: u64,
        wall_nanos: i64,
        thread: &str,
        running: bool,
        frames: impl Iterator<Item = Frame<'a>>,
    ) {
        let mut frames: Vec<Frame<'a>> = frames.take(DEPTH + 1).collect();
        if frames.is_empty() {
            return;
        }
        if frames.len() > DEPTH {
            frames[DEPTH] = TRUNCATED;
        }
        let Some(stack) = self.stack(&frames) else {
            self.dropped += 1;
            return;
        };
        if self.start_nanos == 0 {
            self.start_nanos = at_nanos;
        }
        self.last_nanos = self.last_nanos.max(at_nanos);
        let thread = self.thread(thread);
        let wall = wall_nanos.clamp(0, LONGEST_SAMPLE_NANOS);
        if self.timed.len() < TIMED_SAMPLES {
            self.timed.push((stack, thread, at_nanos, wall, running));
        } else {
            *self.merged.entry((stack, thread, running)).or_default() += wall;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.timed.is_empty()
    }

    /// Samples kept, timed or merged.
    pub fn samples(&self) -> usize {
        self.timed.len() + self.merged.len()
    }

    /// Samples not kept: their stack was new past [`STACKS`].
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The gzipped pprof profile.
    pub fn encode(&self) -> Vec<u8> {
        let mut strings = self.strings.clone();
        if strings.is_empty() {
            strings.push(String::new());
        }
        let mut string = |s: &str| match strings.iter().position(|x| x == s) {
            Some(at) => at as i64,
            None => {
                strings.push(s.to_owned());
                strings.len() as i64 - 1
            }
        };
        let (wall, nanoseconds) = (string("wall"), string("nanoseconds"));
        let (thread_name, end_timestamp) = (string("thread_name"), string("end_timestamp_ns"));
        let python = string("python");
        let (thread_state, running, waiting) =
            (string("thread_state"), string("running"), string("waiting"));
        let state = |is_running: bool| if is_running { running } else { waiting };
        let stack =
            |id: u32| self.stack_list.get(id as usize).map_or_else(Vec::new, |s| s.to_vec());
        let label =
            |key: i64, str: i64, num: i64, num_unit: i64| pprof::Label { key, str, num, num_unit };
        let mut samples: Vec<pprof::Sample> = self
            .timed
            .iter()
            .map(|(id, thread, at, wall, is_running)| pprof::Sample {
                location_id: stack(*id),
                value: vec![*wall],
                label: vec![
                    label(thread_name, *thread, 0, 0),
                    label(thread_state, state(*is_running), 0, 0),
                    label(end_timestamp, 0, *at as i64, nanoseconds),
                ],
            })
            .collect();
        samples.extend(self.merged.iter().map(|((id, thread, is_running), wall)| pprof::Sample {
            location_id: stack(*id),
            value: vec![*wall],
            label: vec![
                label(thread_name, *thread, 0, 0),
                label(thread_state, state(*is_running), 0, 0),
            ],
        }));
        let profile = pprof::Profile {
            sample_type: vec![pprof::ValueType { r#type: wall, unit: nanoseconds }],
            sample: samples,
            mapping: vec![pprof::Mapping {
                id: 1,
                filename: python,
                has_functions: true,
                has_filenames: true,
                has_line_numbers: true,
            }],
            location: self.location_list.clone(),
            function: self.function_list.clone(),
            string_table: strings,
            time_nanos: self.start_nanos as i64,
            duration_nanos: self.last_nanos.saturating_sub(self.start_nanos) as i64 + PERIOD_NANOS,
            period_type: Some(pprof::ValueType { r#type: wall, unit: nanoseconds }),
            period: PERIOD_NANOS,
        };
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        // Writing to a `Vec` cannot fail.
        let _ = gz.write_all(&profile.encode_to_vec());
        gz.finish().unwrap_or_default()
    }

    /// The stack's id; none when it is new and [`STACKS`] are kept already (nothing interned).
    fn stack(&mut self, frames: &[Frame<'_>]) -> Option<u32> {
        let full = self.stack_list.len() >= STACKS;
        let mut ids = Vec::with_capacity(frames.len());
        for frame in frames {
            let id = if full { self.known_location(frame)? } else { self.location(frame) };
            ids.push(id);
        }
        let ids: Arc<[u64]> = ids.into();
        if let Some(id) = self.stacks.get(&ids) {
            return Some(*id);
        }
        if full {
            return None;
        }
        let id = self.stack_list.len() as u32;
        self.stacks.insert(ids.clone(), id);
        self.stack_list.push(ids);
        Some(id)
    }

    fn thread(&mut self, name: &str) -> i64 {
        if let Some(id) = self.threads.get(name) {
            return *id;
        }
        let name = if self.threads.len() >= THREADS { "other" } else { name };
        let id = self.string(name);
        if self.threads.len() < THREADS {
            self.threads.insert(name.to_owned(), id);
        }
        id
    }

    fn string(&mut self, s: &str) -> i64 {
        if self.strings.is_empty() {
            self.strings.push(String::new());
        }
        if let Some(id) = self.string_ids.get(s) {
            return *id;
        }
        let id = self.strings.len() as i64;
        self.strings.push(s.to_owned());
        self.string_ids.insert(s.to_owned(), id);
        id
    }

    fn known_location(&self, frame: &Frame<'_>) -> Option<u64> {
        let name = *self.string_ids.get(frame.function)?;
        let file = *self.string_ids.get(frame.file)?;
        let function = *self.functions.get(&(name, file))?;
        self.locations.get(&(function, frame.line)).copied()
    }

    fn location(&mut self, frame: &Frame<'_>) -> u64 {
        let (name, file) = (self.string(frame.function), self.string(frame.file));
        let next_function = self.function_list.len() as u64 + 1;
        let function = *self.functions.entry((name, file)).or_insert(next_function);
        if function == next_function {
            let function =
                pprof::Function { id: function, name, system_name: name, filename: file };
            self.function_list.push(function);
        }
        let next_location = self.location_list.len() as u64 + 1;
        let location = *self.locations.entry((function, frame.line)).or_insert(next_location);
        if location == next_location {
            self.location_list.push(pprof::Location {
                id: location,
                mapping_id: 1,
                line: vec![pprof::Line { function_id: function, line: frame.line }],
            });
        }
        location
    }
}

/// The jobs' profiles the recording upload reads, by job.
static PROFILES: LazyLock<Mutex<HashMap<JobKey, Arc<Mutex<Profile>>>>> =
    LazyLock::new(Default::default);

/// Registers `job`'s profile, filled by the host's sampler.
pub fn register(job: JobKey, profile: Arc<Mutex<Profile>>) {
    PROFILES.lock().insert(job, profile);
}

/// Takes `job`'s profile, leaving it empty (the job is ending); none when nothing was sampled.
/// The lock is held only for the swap: encoding is the caller's, off the sampler's lock.
pub fn take(job: &JobKey) -> Option<Profile> {
    let profile = PROFILES.lock().get(job).cloned()?;
    let taken = std::mem::take(&mut *profile.lock());
    (!taken.is_empty()).then_some(taken)
}

/// Forgets `job`'s profile when it is still `profile` (its watch closed; a later watch of the
/// same job keeps its own).
pub fn forget(job: &JobKey, profile: &Arc<Mutex<Profile>>) {
    let mut profiles = PROFILES.lock();
    if profiles.get(job).is_some_and(|registered| Arc::ptr_eq(registered, profile)) {
        profiles.remove(job);
    }
}

/// The subset of pprof's `profile.proto` a job profile uses (field numbers as upstream).
pub mod pprof {
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Profile {
        #[prost(message, repeated, tag = "1")]
        pub sample_type: Vec<ValueType>,
        #[prost(message, repeated, tag = "2")]
        pub sample: Vec<Sample>,
        #[prost(message, repeated, tag = "3")]
        pub mapping: Vec<Mapping>,
        #[prost(message, repeated, tag = "4")]
        pub location: Vec<Location>,
        #[prost(message, repeated, tag = "5")]
        pub function: Vec<Function>,
        #[prost(string, repeated, tag = "6")]
        pub string_table: Vec<String>,
        #[prost(int64, tag = "9")]
        pub time_nanos: i64,
        #[prost(int64, tag = "10")]
        pub duration_nanos: i64,
        #[prost(message, optional, tag = "11")]
        pub period_type: Option<ValueType>,
        #[prost(int64, tag = "12")]
        pub period: i64,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct ValueType {
        #[prost(int64, tag = "1")]
        pub r#type: i64,
        #[prost(int64, tag = "2")]
        pub unit: i64,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Sample {
        #[prost(uint64, repeated, tag = "1")]
        pub location_id: Vec<u64>,
        #[prost(int64, repeated, tag = "2")]
        pub value: Vec<i64>,
        #[prost(message, repeated, tag = "3")]
        pub label: Vec<Label>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Label {
        #[prost(int64, tag = "1")]
        pub key: i64,
        #[prost(int64, tag = "2")]
        pub str: i64,
        #[prost(int64, tag = "3")]
        pub num: i64,
        #[prost(int64, tag = "4")]
        pub num_unit: i64,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Mapping {
        #[prost(uint64, tag = "1")]
        pub id: u64,
        #[prost(int64, tag = "5")]
        pub filename: i64,
        #[prost(bool, tag = "7")]
        pub has_functions: bool,
        #[prost(bool, tag = "8")]
        pub has_filenames: bool,
        #[prost(bool, tag = "9")]
        pub has_line_numbers: bool,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Location {
        #[prost(uint64, tag = "1")]
        pub id: u64,
        #[prost(uint64, tag = "2")]
        pub mapping_id: u64,
        #[prost(message, repeated, tag = "4")]
        pub line: Vec<Line>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Line {
        #[prost(uint64, tag = "1")]
        pub function_id: u64,
        #[prost(int64, tag = "2")]
        pub line: i64,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Function {
        #[prost(uint64, tag = "1")]
        pub id: u64,
        #[prost(int64, tag = "2")]
        pub name: i64,
        #[prost(int64, tag = "3")]
        pub system_name: i64,
        #[prost(int64, tag = "4")]
        pub filename: i64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read as _;

    fn decode(bytes: &[u8]) -> pprof::Profile {
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(bytes).read_to_end(&mut raw).unwrap();
        pprof::Profile::decode(raw.as_slice()).unwrap()
    }

    fn frames(names: &[(&'static str, i64)]) -> impl Iterator<Item = Frame<'static>> {
        names.iter().map(|(function, line)| Frame { function, file: "app.py", line: *line })
    }

    const P: i64 = PERIOD_NANOS;

    // Catches: samples merged or reordered (losing the time axis), a frame interned twice, a
    // label a viewer cannot read back, the leaf not first, or a sample's wall time lost.
    #[test]
    fn samples_keep_their_time_thread_wall_time_and_stack() {
        let mut p = Profile::default();
        p.add(1_000, P, "MainThread", true, frames(&[("busy", 7), ("main", 20)]));
        p.add(1_100, P + 3, "MainThread", true, frames(&[("busy", 7), ("main", 20)]));
        p.add(1_200, P, "worker", true, frames(&[("lookup", 3)]));
        let decoded = decode(&p.encode());
        let s = |i: i64| decoded.string_table[i as usize].as_str();
        assert_eq!(decoded.string_table[0], "");
        assert_eq!(decoded.sample.len(), 3);
        assert_eq!(decoded.function.len(), 3, "busy, main, lookup: once each");
        assert_eq!(decoded.time_nanos, 1_000);
        let first = &decoded.sample[0];
        let leaf = &decoded.location[(first.location_id[0] - 1) as usize];
        assert_eq!(s(decoded.function[(leaf.line[0].function_id - 1) as usize].name), "busy");
        assert_eq!(leaf.line[0].line, 7);
        let label = |sample: &pprof::Sample, key: &str| {
            sample.label.iter().find(|l| s(l.key) == key).cloned().unwrap()
        };
        assert_eq!(s(label(first, "thread_name").str), "MainThread");
        assert_eq!(s(label(first, "thread_state").str), "running");
        assert_eq!(label(&decoded.sample[2], "end_timestamp_ns").num, 1_200);
        assert_eq!(s(decoded.sample_type[0].r#type), "wall");
        assert_eq!(decoded.sample[1].value, [P + 3]);
        assert!(decoded.mapping[0].has_line_numbers);
    }

    // Catches: an unbounded profile on a long job, or samples past the bound lost.
    #[test]
    fn past_the_bound_samples_merge_without_time() {
        let mut p = Profile::default();
        for i in 0..TIMED_SAMPLES as u64 + 10 {
            p.add(i, P, "MainThread", true, frames(&[("busy", 7)]));
        }
        let decoded = decode(&p.encode());
        assert_eq!(decoded.sample.len(), TIMED_SAMPLES + 1);
        assert_eq!(decoded.sample[TIMED_SAMPLES].value, [P * 10]);
        assert_eq!(
            decoded.sample[TIMED_SAMPLES].label.len(),
            2,
            "the thread and its state, no time"
        );
    }

    // Catches: memory growing with distinct stacks past the bound (their frames interned all the
    // same), unbounded depth, or unbounded thread names.
    #[test]
    fn stacks_depth_and_thread_names_are_bounded() {
        let mut p = Profile::default();
        for line in 0..STACKS as i64 {
            p.add(
                1,
                P,
                "MainThread",
                true,
                [Frame { function: "f", file: "app.py", line }].into_iter(),
            );
        }
        let (strings, locations) = (p.strings.len(), p.location_list.len());
        for line in 0..1_000 {
            let frame = Frame { function: "new_function", file: "new.py", line };
            p.add(2, P, "MainThread", true, [frame].into_iter());
        }
        assert_eq!(p.dropped(), 1_000);
        assert_eq!(
            (p.strings.len(), p.location_list.len()),
            (strings, locations),
            "nothing interned"
        );
        p.add(
            3,
            P,
            "MainThread",
            true,
            [Frame { function: "f", file: "app.py", line: 0 }].into_iter(),
        );
        assert_eq!(p.dropped(), 1_000, "a known stack is still kept");

        let mut deep = Profile::default();
        let names: Vec<String> = (0..1_000).map(|i| format!("f{i}")).collect();
        let stack = names.iter().map(|name| Frame { function: name, file: "app.py", line: 1 });
        deep.add(1, P, "MainThread", true, stack);
        let decoded = decode(&deep.encode());
        assert_eq!(decoded.sample[0].location_id.len(), DEPTH + 1);
        let root = &decoded.location[(*decoded.sample[0].location_id.last().unwrap() - 1) as usize];
        let root = &decoded.function[(root.line[0].function_id - 1) as usize];
        assert_eq!(decoded.string_table[root.name as usize], "<truncated>");

        let mut threads = Profile::default();
        for i in 0..THREADS + 50 {
            threads.add(1, P, &format!("pool-{i}"), true, frames(&[("work", 1)]));
        }
        assert_eq!(threads.threads.len(), THREADS);
        let decoded = decode(&threads.encode());
        assert!(decoded.string_table.iter().any(|s| s == "other"));
    }

    // Catches: a job's profile taken twice, kept after its job, an empty one served, or a later
    // watch of the same job losing its profile when an earlier watch closes.
    #[test]
    fn the_registry_serves_a_job_s_profile_once_and_forgets_only_its_own() {
        let job = JobKey("AJ_profile".into());
        let first = Arc::new(Mutex::new(Profile::default()));
        register(job.clone(), first.clone());
        assert!(take(&job).is_none(), "nothing sampled yet");
        first.lock().add(5, P, "MainThread", true, frames(&[("busy", 7)]));
        assert!(take(&job).is_some());
        assert!(take(&job).is_none(), "taken");

        let second = Arc::new(Mutex::new(Profile::default()));
        register(job.clone(), second.clone());
        forget(&job, &first);
        second.lock().add(6, P, "MainThread", true, frames(&[("busy", 7)]));
        assert!(take(&job).is_some(), "the earlier watch did not forget the later one's");
        forget(&job, &second);
        assert!(PROFILES.lock().get(&job).is_none());
    }

    // Catches: an empty profile encoding an invalid string table.
    #[test]
    fn an_empty_profile_encodes_a_valid_string_table() {
        assert_eq!(decode(&Profile::default().encode()).string_table[0], "");
    }
}
