use lk_agents_abi::{
    Ts,
    errors::ApiError,
    llm::realtime::{Event, RealtimeModelError},
};
use lk_agents_inference::realtime::{Outgoing, ReplyKind, Step};

use super::{LABEL, MODEL_BUILT_BEFORE, Socket, State};
use crate::record::Tag;

impl Socket {
    /// Execute typed effects without making any client bookkeeping decisions.
    pub(super) fn settle(&self, st: &mut State, now: Ts, step: Step) {
        let step = st.machine.prepare_step(step);
        st.events.extend(step.events);
        if let Some(mut error) = step.fatal {
            error.recoverable = self.setup.fallback;
            st.events.push_back(Event::Error(error));
            self.fatal(st);
            return;
        }
        if step.recycle {
            self.hang_up(st);
            st.connections += 1;
            st.server = self.new_server(st.connections);
            let step = st.machine.reconnect(now);
            self.settle(st, now, step);
            return;
        }
        for frame in step.frames {
            self.outgoing(st, now, frame);
        }
    }

    fn outgoing(&self, st: &mut State, now: Ts, frame: Outgoing) {
        match frame {
            Outgoing::SessionUpdate { opts, .. } => {
                let vad = opts.turn_detection.as_ref().map(|td| match td {
                    lk_agents_abi::llm::realtime_options::TurnDetection::Semantic(_) => {
                        super::Vad::Semantic
                    }
                    _ => super::Vad::Server,
                });
                st.server.configure(
                    vad,
                    opts.audio_output,
                    self.setup.tool_calls,
                    self.setup.timing.clone(),
                    self.setup.delays,
                );
                st.server.instructions(opts.instructions.as_deref().unwrap_or_default());
            }
            Outgoing::InstructionsUpdate { instructions, .. } => {
                st.server.instructions(&instructions)
            }
            Outgoing::ToolsUpdate { tools, .. } => {
                st.server.tools(tools.iter().map(|t| t.name.clone()).collect())
            }
            Outgoing::OptionsUpdate { .. } => {}
            Outgoing::ItemCreate { event_id, item, previous_item_id } => {
                st.server.create_request(now, event_id, &item, previous_item_id);
            }
            Outgoing::ItemDelete { event_id, item_id } => {
                st.server.delete_request(now, event_id, &item_id);
            }
            Outgoing::ItemTruncate { item_id, audio_end_ms } => {
                st.server.truncate(now, &item_id, audio_end_ms)
            }
            Outgoing::AudioAppend { frame } => {
                if let Some(slot) = self.slot.get() {
                    let (pairs, _) = frame.data.as_chunks::<2>();
                    let tags = pairs.iter().map(|b| Tag::decode(i16::from_le_bytes(*b))).collect();
                    self.registry.record_forwarded(*slot, tags);
                }
                if frame.sample_rate > 0 {
                    st.server.append_audio(
                        u64::from(frame.samples_per_channel) * 1000 / u64::from(frame.sample_rate),
                    );
                }
            }
            Outgoing::AudioCommit => st.server.commit_audio(now),
            Outgoing::AudioClear => st.server.clear_audio(),
            Outgoing::ResponseCreate { req_id, instructions, tool_choice } => {
                st.machine.response_sent(
                    st.machine.connection_generation(),
                    now,
                    &req_id,
                    ReplyKind::Generated,
                );
                st.server.respond_to(now, req_id, instructions.as_deref(), tool_choice);
            }
            Outgoing::ResponseCancel { response_id } => {
                st.server.cancel(now, response_id.as_deref())
            }
            Outgoing::ForceMessage { .. } => {
                unreachable!("the inference dialect does not support say")
            }
        }
    }

    pub(super) fn emit_error(&self, st: &mut State, error: ApiError, recoverable: bool) {
        st.events.push_back(Event::Error(RealtimeModelError {
            timestamp: self.now().saturating_add(MODEL_BUILT_BEFORE),
            label: LABEL.into(),
            message: error.message.clone(),
            recoverable,
            req_id: None,
            code: None,
            provider_error: Some(error),
        }));
    }
}
