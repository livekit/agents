//! The user's words as the realtime server reports them: its transcripts of the input audio, and
//! the items it adds to its own mirror of the conversation.

use crate::agent::{Source, Visibility};
use crate::agent_session::VoiceDriver;
use crate::audio_recognition::ServerSpeech;
use crate::events::{DiagnosticEvent, SessionEvent};
use lk_agents_abi::{ChatMessage, Role, Ts};
use std::time::Duration;

impl VoiceDriver {
    pub(super) fn on_input_transcription(
        &mut self,
        now: Ts,
        t: lk_agents_abi::llm::realtime::InputTranscription,
        started_at: Option<f64>,
    ) {
        let speech = ServerSpeech::Transcription {
            text: t.transcript.clone(),
            is_final: t.is_final,
            item_id: t.item_id.clone(),
        };
        let eff = self.activity.audio_recognition.on_server(now, speech);
        self.apply_recognition_effects(eff);
        if t.is_final {
            self.upsert_user_transcript(t, started_at);
        }
    }

    /// A final server transcript, upserted by `item_id` (a repeated final replaces in place).
    /// Placed at the turn's onset when reported, else at its existing remote placeholder.
    pub(crate) fn upsert_user_transcript(
        &mut self,
        t: lk_agents_abi::llm::realtime::InputTranscription,
        started_at: Option<f64>,
    ) {
        let started = started_at.and_then(|at| {
            let since_epoch = Duration::try_from_secs_f64(at).ok()?;
            Some((at, std::time::UNIX_EPOCH + since_epoch))
        });
        let created_at = started.map_or_else(
            || {
                self.session
                    .chat_ctx
                    .entry(&t.item_id)
                    .map_or_else(|| self.step.wall(), |entry| entry.item.created_at())
            },
            |(_, wall)| wall,
        );
        let mut item = ChatMessage::text(t.item_id.clone(), Role::User, &t.transcript, created_at);
        item.metrics.started_speaking_at = started.map(|(at, _)| at);
        let replaces_history =
            self.session.chat_ctx.entry(&t.item_id).is_some_and(|entry| entry.visibility.history());
        let id = t.item_id.clone();
        let source = Source::Remote { item_id: t.item_id };
        self.commit_item(item.clone().into(), source, Visibility::Both);
        self.route_realtime_input(item);
        if replaces_history {
            // A restored placeholder already belongs to history. Tell observers its accepted
            // final revision under that identity, as with an ordinary transcript correction.
            let entry =
                self.session.chat_ctx.entry(&id).expect("the transcript was committed").clone();
            self.publish_event(SessionEvent::ConversationItemAdded {
                item: Box::new(entry.item),
                agent: entry.agent,
                source: entry.source,
                visibility: entry.visibility,
            });
        }
    }

    /// An item the server added to its mirror, recorded once (duplicate ids ignored).
    pub(crate) fn mirror_remote_item(
        &mut self,
        previous_item_id: Option<String>,
        item: lk_agents_abi::ChatItem,
    ) {
        let id = item.id().to_owned();
        if self.session.chat_ctx.contains(&id) {
            return;
        }
        if let Some(anchor) = &previous_item_id
            && !self.session.chat_ctx.contains(anchor)
        {
            let message = "realtime server anchored an item to one this session does not hold; placed by its stamp";
            self.diagnose(DiagnosticEvent::warning(message).with("item_id", id.clone()));
        }
        // A placeholder in the model's context: the transcript that fills it reaches history.
        self.commit_item(item, Source::Remote { item_id: id }, Visibility::Model);
    }
}
