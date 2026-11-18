use lk_agents_abi::{AudioFrame, Ts, llm::realtime::Modalities, usage::TokenUsage};
use lk_agents_inference::realtime::protocol::{
    Incoming, OutputItemKind, ResponseDone, ServerError, TranscriptionCompleted, TranscriptionUsage,
};

use super::{
    DELTA_SAMPLES, SAMPLE_RATE, Socket, State,
    server::{Happening, function_call, message},
};
use crate::{
    fakes::Control,
    record::{Detail, Kind, Tag},
};

impl Socket {
    /// Translate provider facts only. The production machine decides which reach the caller.
    pub(super) fn told(&self, st: &mut State, now: Ts, happening: Happening) {
        let mut incoming = Vec::new();
        match happening {
            Happening::Created { r } => {
                let resp = &st.server.responses[r as usize];
                incoming.push(Incoming::ResponseCreated {
                    response_id: resp.id.clone(),
                    client_event_id: resp.request.clone(),
                });
                let slot =
                    self.registry.register(Kind::Generation, Detail::None, Control::Generation);
                st.generations.insert(r, slot);
                st.server.registered(r, slot.nth);
                st.server.told_created(r, now);
            }
            Happening::MessageAdded { r, previous } => {
                let item = st.server.responses[r as usize].item.clone().expect("a message");
                incoming.push(Incoming::OutputItemAdded {
                    item_id: item.clone(),
                    kind: OutputItemKind::Message,
                });
                incoming.push(Incoming::ItemAdded {
                    previous_item_id: previous,
                    item: message(&item, lk_agents_abi::Role::Assistant),
                });
            }
            Happening::Part { r } => {
                let item_id = st.server.responses[r as usize].item.clone().expect("a message");
                incoming.push(Incoming::ContentPartAdded {
                    item_id,
                    modalities: if self.setup.text_only {
                        Modalities::TextOnly
                    } else {
                        Modalities::AudioAndText
                    },
                });
            }
            Happening::Text { r, delta } => {
                let item_id = st.server.responses[r as usize].item.clone().expect("a message");
                incoming.push(if self.setup.text_only {
                    Incoming::TextDelta { item_id, delta }
                } else {
                    Incoming::AudioTranscriptDelta { item_id, delta, start_time: None }
                });
            }
            Happening::Audio { r, chunk } => {
                let source = st.generations.get(&r).map(|s| s.nth);
                if let Some(source) = source {
                    self.registry.record_generated(source, chunk);
                }
                let pcm =
                    Tag::generated(source.unwrap_or(0), chunk).to_le_bytes().repeat(DELTA_SAMPLES);
                let frame = AudioFrame::new(pcm, SAMPLE_RATE, 1, DELTA_SAMPLES as u32);
                let item_id = st.server.responses[r as usize].item.clone().expect("a message");
                incoming.push(Incoming::AudioDelta { item_id, frame });
            }
            Happening::CallAdded { r } => {
                let resp = &st.server.responses[r as usize];
                let (fc, name) = resp.function_call.clone().expect("a call");
                incoming.push(Incoming::OutputItemAdded {
                    item_id: fc.clone(),
                    kind: OutputItemKind::FunctionCall,
                });
                incoming.push(Incoming::ItemAdded {
                    previous_item_id: resp.item.clone(),
                    item: function_call(&fc, &name),
                });
            }
            Happening::MessageDone { r } => {
                let resp = &st.server.responses[r as usize];
                incoming.push(Incoming::OutputItemDone {
                    item: message(
                        resp.item.as_deref().expect("a message"),
                        lk_agents_abi::Role::Assistant,
                    ),
                });
            }
            Happening::CallDone { r } => {
                let (fc, name) =
                    st.server.responses[r as usize].function_call.clone().expect("a call");
                incoming.push(Incoming::OutputItemDone { item: function_call(&fc, &name) });
            }
            Happening::Done { r, status } => {
                if let Some(slot) = st.generations.remove(&r) {
                    self.registry.mark_closed(slot);
                }
                let (input_tokens, output_tokens) = st.server.usage(r);
                incoming.push(Incoming::ResponseDone(ResponseDone {
                    response_id: st.server.responses[r as usize].id.clone(),
                    status,
                    error_type: (status == lk_agents_abi::llm::realtime::GenerationStatus::Failed)
                        .then(|| {
                            st.server.responses[r as usize]
                                .failure
                                .map_or("server_error", |f| f.0)
                                .into()
                        }),
                    error_code: st.server.responses[r as usize].failure.map(|f| f.1.into()),
                    usage: TokenUsage {
                        input_tokens,
                        output_tokens,
                        input_text_tokens: Some(input_tokens),
                        output_audio_tokens: Some(30),
                        output_text_tokens: Some(10),
                        ..Default::default()
                    },
                }));
            }
            Happening::ItemAdded { item, previous } => {
                incoming.push(Incoming::ItemAdded { previous_item_id: previous, item: *item })
            }
            Happening::ItemDeleted { item } => {
                incoming.push(Incoming::ItemDeleted { item_id: item })
            }
            Happening::SpeechStarted { item } => {
                incoming.push(Incoming::SpeechStarted { item_id: Some(item) })
            }
            Happening::SpeechStopped => incoming.push(Incoming::SpeechStopped),
            Happening::Transcript { item, text, last } => {
                if last {
                    let words = text.split(' ').count() as u64;
                    incoming.push(Incoming::TranscriptionCompleted(TranscriptionCompleted {
                        item_id: item,
                        content_index: 0,
                        transcript: text,
                        confidence: None,
                        in_progress: false,
                        event_id: String::new(),
                        usage: Some(TranscriptionUsage::Tokens(TokenUsage {
                            input_tokens: 20,
                            output_tokens: words,
                            input_audio_tokens: Some(20),
                            ..Default::default()
                        })),
                    }));
                } else if self.setup.provider == lk_agents_abi::llm::realtime::RealtimeProvider::Xai
                {
                    incoming
                        .push(Incoming::TranscriptionUpdated { item_id: item, transcript: text });
                } else {
                    incoming.push(Incoming::TranscriptionDelta {
                        item_id: item,
                        content_index: 0,
                        delta: text,
                    });
                }
            }
            Happening::TranscriptFailed { item } => {
                incoming.push(Incoming::TranscriptionFailed { item_id: item, content_index: 0 });
            }
            Happening::Error { code, message, request } => {
                incoming.push(Incoming::Error(ServerError {
                    event_id: request,
                    code: code.map(str::to_owned),
                    type_: None,
                    message: message.into(),
                }))
            }
            Happening::Fatal => incoming.push(Incoming::Error(ServerError {
                event_id: None,
                code: Some("insufficient_quota".into()),
                type_: None,
                message:
                    "You exceeded your current quota, please check your plan and billing details."
                        .into(),
            })),
        }
        let generation = st.machine.connection_generation();
        for mut event in incoming {
            // The real decoder stamps items when received. Typed simulation bypasses decoding,
            // so provide that same boundary input rather than leaving fixture epoch stamps.
            if let Incoming::ItemAdded { item, .. } | Incoming::OutputItemDone { item } = &mut event
            {
                match item {
                    lk_agents_abi::ChatItem::Message(m) => m.created_at = self.clock.wall(),
                    lk_agents_abi::ChatItem::FunctionCall(c) => c.created_at = self.clock.wall(),
                    lk_agents_abi::ChatItem::FunctionCallOutput(o) => {
                        o.created_at = self.clock.wall()
                    }
                    lk_agents_abi::ChatItem::AgentHandoff(_)
                    | lk_agents_abi::ChatItem::AgentConfigUpdate(_) => {}
                }
            }
            let step = st.machine.incoming_on(generation, now, event);
            self.settle(st, now, step);
            if st.over || st.swapping {
                break;
            }
        }
    }
}
