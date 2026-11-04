//! Telemetry contracts: the span, metric and report record declarations, the wire attribute names,
//! the policy and provider metric declarations, the session report's records.

pub mod policy;
pub mod provider;

pub mod attributes;
pub mod decl;
pub mod host;
pub mod instruments;
pub mod record;
pub mod report;
pub mod report_records;
pub mod semconv;
pub mod spans;

use std::sync::Arc;

use decl::SpanDecl;
use record::{JobKey, Record, SpanId, SpanRef, TraceContext, TraceId};

/// What records a context's spans: the process tracer (`lk_agents_net::telemetry::Tracer`), or
/// the DST's.
pub trait Recorder: Send + Sync {
    fn record(&self, record: Record);
    /// Wall-clock unix nanoseconds from the injected clock.
    fn now_nanos(&self) -> u64;
    /// A new span or trace id half, never zero.
    fn next_id(&self) -> u64;
    /// GenAI message capture: whether request and response messages are recorded on spans.
    fn capture_content(&self) -> bool;
}

/// The span work runs under, explicitly passed (`telemetry.md`, "Context across boundaries"):
/// nothing is entered into a thread-local, so the context stays correct across `.await` and
/// concurrent tasks. Every host call takes one; [`Cx::NONE`] records nothing and makes a call run
/// under no span.
#[derive(Clone)]
pub struct Cx {
    recorder: Option<Arc<dyn Recorder>>,
    current: Option<Current>,
    job: Option<JobKey>,
    conversation: Option<Arc<str>>,
}

#[derive(Clone, Copy)]
struct Current {
    span: SpanRef,
    /// The declared span's name; `None` for a span from outside the framework (an application
    /// span, a `traceparent`).
    name: Option<&'static str>,
}

impl Cx {
    /// No span: a host call made under it runs with no current span, and spans opened under it
    /// record nothing.
    pub const NONE: Cx = Cx { recorder: None, current: None, job: None, conversation: None };

    /// A context with no current span, outside any job, recording to `recorder`.
    pub fn root(recorder: Arc<dyn Recorder>) -> Cx {
        Cx { recorder: Some(recorder), ..Cx::NONE }
    }

    /// The caller's span at a host-to-framework entry (the span current in the caller's
    /// `contextvars`), recording to `recorder`.
    pub fn entry(recorder: Arc<dyn Recorder>, caller: Option<SpanRef>) -> Cx {
        Cx::root(recorder).continuing(caller)
    }

    /// The context a [`TraceContext`] carries across an API that cannot carry a `Cx`.
    pub fn resume(recorder: Arc<dyn Recorder>, trace: &TraceContext) -> Cx {
        Cx::root(recorder)
            .continuing(trace.span)
            .with_job(trace.job.clone())
            .with_conversation(trace.conversation.as_deref())
    }

    /// What records this context's spans; `None` for [`Cx::NONE`].
    pub fn recorder(&self) -> Option<&Arc<dyn Recorder>> {
        self.recorder.as_ref()
    }

    /// Wall-clock unix nanoseconds; 0 for a context that records nothing.
    pub fn now_nanos(&self) -> u64 {
        self.recorder.as_ref().map_or(0, |r| r.now_nanos())
    }

    /// Whether GenAI message content is recorded.
    pub fn capture_content(&self) -> bool {
        self.recorder.as_ref().is_some_and(|r| r.capture_content())
    }

    /// The current span, what a host call activates and a `traceparent` carries.
    pub fn span(&self) -> Option<SpanRef> {
        self.current.map(|c| c.span)
    }

    /// The current span's declared name, for the parent check; `Some(None)` for a span from
    /// outside the framework.
    pub fn current_name(&self) -> Option<Option<&'static str>> {
        self.current.map(|c| c.name)
    }

    pub fn job(&self) -> Option<&JobKey> {
        self.job.as_ref()
    }

    pub fn conversation(&self) -> Option<&str> {
        self.conversation.as_deref()
    }

    /// This context for an API that cannot carry a `Cx`; [`Cx::resume`] reopens it.
    pub fn trace_context(&self) -> TraceContext {
        TraceContext {
            span: self.span(),
            job: self.job.clone(),
            conversation: self.conversation.as_deref().map(str::to_owned),
        }
    }

    /// The same context, recording for `job`.
    pub fn with_job(mut self, job: Option<JobKey>) -> Cx {
        self.job = job;
        self
    }

    /// The same context, its spans carrying `gen_ai.conversation.id`.
    pub fn with_conversation(mut self, conversation: Option<&str>) -> Cx {
        self.conversation = conversation.map(Arc::from);
        self
    }

    /// The same context under `span`, a framework span of type `P` recorded elsewhere (a voice
    /// effect's `SpanRef`).
    pub fn under<P: SpanDecl>(mut self, span: SpanRef) -> Cx {
        self.current = Some(Current { span, name: Some(P::NAME) });
        self
    }

    /// The same context under a span from outside the framework: an application span captured
    /// at a host entry, or a `traceparent`. `None` starts new traces.
    pub fn continuing(mut self, span: Option<SpanRef>) -> Cx {
        self.current = span.map(|span| Current { span, name: None });
        self
    }

    /// A new span's context under the current span: its trace, or a new one.
    pub fn child_context(&self) -> SpanRef {
        let next = || self.recorder.as_ref().map_or(1, |r| r.next_id());
        let trace_id = match self.current {
            Some(c) => c.span.trace_id,
            None => TraceId((u128::from(next()) << 64) | u128::from(next())),
        };
        SpanRef {
            trace_id,
            span_id: SpanId(next()),
            sampled: self.current.is_none_or(|c| c.span.sampled),
        }
    }
}
