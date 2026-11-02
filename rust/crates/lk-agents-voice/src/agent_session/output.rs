//! A segment takes the session's current outputs when it activates and keeps them until it
//! settles. `OutputBindingEvent` is the single generation-to-owner lifecycle boundary:
//! reserve -> activate -> settle/release, with pending cleanup retaining the same binding.
//! Core owns admission and leases; the runtime owns physical audio/text resources.

use super::effects::{AudioOp, Effect, Out};
use super::{Driver, Outputs, RequestId};
use crate::agent_activity::scheduling::SpeechId;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::PlaybackSegment;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct SegmentBinding {
    pub segment: PlaybackSegment,
    pub outputs: Outputs,
}

impl SegmentBinding {
    fn new(segment: u64, outputs: &Outputs) -> Self {
        let attachment = outputs.audio.as_ref().map_or(0, |a| a.id.0);
        SegmentBinding {
            segment: PlaybackSegment { attachment, segment },
            outputs: outputs.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OutputBindingError {
    #[error("output binding capacity is occupied by unsettled segments")]
    Capacity,
    #[error("the speech already owns an output binding")]
    AlreadyBound,
    #[error("output segment identities are exhausted")]
    IdsExhausted,
    #[error("the output binding could not be activated")]
    ActivationUnavailable,
    #[error("the output binding reply was lost")]
    ReplyLost,
}

/// Generation requests and cleanup facts for one output binding. The generation queue
/// preserves ordering; this protocol is handled only by the session's binding owner.
#[derive(Debug)]
pub enum OutputBindingEvent {
    BindRequested {
        call: RequestId,
        speech: SpeechId,
    },
    ActivationRequested {
        call: RequestId,
        binding: RequestId,
        speech: SpeechId,
    },
    ReleaseRequested {
        binding: RequestId,
        speech: SpeechId,
    },
    /// Physical cleanup is still outstanding. Retain the lease until matched settlement.
    CleanupPending {
        binding: RequestId,
        speech: SpeechId,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BindingPhase {
    Reserved,
    Active,
}

impl BindingPhase {
    fn is_active(self) -> bool {
        !matches!(self, Self::Reserved)
    }
}

struct Bound {
    speech: SpeechId,
    binding: SegmentBinding,
    phase: BindingPhase,
}

#[derive(Default)]
pub(crate) struct OutputBindings {
    next: u64,
    bound: BTreeMap<RequestId, Bound>,
    audio: BTreeSet<ResourceId>,
    text: BTreeSet<ResourceId>,
}

impl OutputBindings {
    fn bind(
        &mut self,
        request: RequestId,
        speech: SpeechId,
        outputs: &Outputs,
    ) -> Result<SegmentBinding, OutputBindingError> {
        // Each speech forwards one segment at a time, including speculative synthesis.
        // The scheduler runs one current and one speculative generation. Unsettled cleanup
        // consumes the same slots, so a silent transport cannot accumulate retired bindings.
        if self.bound.len() >= 2 {
            return Err(OutputBindingError::Capacity);
        }
        if self.bound.values().any(|b| b.speech == speech) || self.bound.contains_key(&request) {
            return Err(OutputBindingError::AlreadyBound);
        }
        self.next = self.next.checked_add(1).ok_or(OutputBindingError::IdsExhausted)?;
        let binding = SegmentBinding::new(self.next, outputs);
        self.bound.insert(
            request,
            Bound { speech, binding: binding.clone(), phase: BindingPhase::Reserved },
        );
        Ok(binding)
    }

    /// Every output a binding names, so its resource outlives the binding.
    pub(crate) fn held(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.bound.values().flat_map(|b| b.binding.outputs.ids())
    }

    /// A speech not yet activated will take the session's current outputs.
    pub(crate) fn speech_outputs(&self, speech: SpeechId) -> Option<&Outputs> {
        self.bound
            .values()
            .find(|b| b.speech == speech && b.phase.is_active())
            .map(|b| &b.binding.outputs)
    }

    /// The captured selection becomes a physical sink only after activation.
    pub(crate) fn active_speech_audio(&self, speech: SpeechId) -> Option<ResourceId> {
        self.bound
            .values()
            .find(|bound| bound.speech == speech && bound.phase.is_active())
            .filter(|bound| bound.binding.outputs.audio_enabled)
            .and_then(|bound| bound.binding.outputs.audio.as_ref())
            .map(|output| output.id)
    }

    pub(crate) fn active_audio(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.audio.iter().copied()
    }

    /// An output stays attached while an activated segment plays on it. A segment not yet
    /// activated holds nothing: it takes the current outputs at its activation.
    fn retire(&mut self, selected: &Outputs, out: &mut Out) {
        let held: BTreeSet<_> = self
            .bound
            .values()
            .filter(|b| b.phase.is_active())
            .flat_map(|b| b.binding.outputs.ids())
            .collect();
        let audio = selected.audio.as_ref().filter(|_| selected.audio_enabled).map(|r| r.id);
        let text = selected
            .transcription
            .as_ref()
            .filter(|_| selected.transcription_enabled)
            .map(|r| r.id);
        let old_audio: Vec<_> = self
            .audio
            .iter()
            .copied()
            .filter(|id| Some(*id) != audio && !held.contains(id))
            .collect();
        let old_text: Vec<_> = self
            .text
            .iter()
            .copied()
            .filter(|id| Some(*id) != text && !held.contains(id))
            .collect();
        for id in old_audio {
            self.audio.remove(&id);
            out.machine().push(Effect::Audio(AudioOp::SetAudioOutput {
                previous: Some(id),
                next: None,
                enabled: false,
            }));
        }
        for id in old_text {
            self.text.remove(&id);
            out.machine().push(Effect::Audio(AudioOp::SetTranscriptionOutput {
                previous: Some(id),
                next: None,
                enabled: false,
            }));
        }
    }

    /// The first activation re-targets the binding to `current`: outputs replaced since it was
    /// bound apply to this segment, which has played nothing yet.
    fn activate(
        &mut self,
        request: RequestId,
        speech: SpeechId,
        current: &Outputs,
        out: &mut Out,
    ) -> Option<SegmentBinding> {
        if self.bound.iter().any(|(id, b)| *id != request && b.phase.is_active()) {
            return None;
        }
        let bound = self.bound.get_mut(&request).filter(|b| b.speech == speech)?;
        if bound.phase.is_active() {
            return Some(bound.binding.clone());
        }
        bound.phase = BindingPhase::Active;
        bound.binding = SegmentBinding::new(bound.binding.segment.segment, current);
        let outputs = &bound.binding.outputs;
        if let Some(audio) = outputs.audio.as_ref().filter(|_| outputs.audio_enabled)
            && self.audio.insert(audio.id)
        {
            out.machine().push(Effect::Audio(AudioOp::SetAudioOutput {
                previous: None,
                next: Some(audio.id),
                enabled: true,
            }));
        }
        if let Some(text) = outputs.transcription.as_ref().filter(|_| outputs.transcription_enabled)
            && self.text.insert(text.id)
        {
            out.machine().push(Effect::Audio(AudioOp::SetTranscriptionOutput {
                previous: None,
                next: Some(text.id),
                enabled: true,
            }));
        }
        Some(bound.binding.clone())
    }
}

impl Driver {
    /// One entry for output reservation, activation, cleanup retention, and final release.
    pub(crate) fn on_output_binding(&mut self, event: OutputBindingEvent, out: &mut Out) {
        match event {
            OutputBindingEvent::BindRequested { call, speech } => {
                self.bind_output(call, speech, out)
            }
            OutputBindingEvent::ActivationRequested { call, binding, speech } => {
                self.activate_output(call, binding, speech, out)
            }
            OutputBindingEvent::ReleaseRequested { binding, speech } => {
                self.release_output(binding, speech, out)
            }
            OutputBindingEvent::CleanupPending { binding, speech } => {
                self.output_unsettled(binding, speech, out)
            }
        }
    }

    /// Attach recorder observations to every admitted playback destination.
    pub fn active_audio_outputs(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.session.output_bindings.active_audio()
    }

    fn bind_output(&mut self, call: RequestId, speech: SpeechId, out: &mut Out) {
        let live =
            !self.session.closing_past_edge() && self.activity.speech_can_bind_output(speech);
        let binding = if live {
            self.session.output_bindings.bind(call, speech, &self.session.outputs).map(Some)
        } else {
            Ok(None)
        };
        if let Err(reason) = &binding {
            out.trace(crate::trace::Step::Playout(crate::trace::PlayoutStep::OutputRefused {
                speech,
                reason: *reason,
            }));
        }
        out.answer(Some(call), super::Answer::OutputBinding(binding));
    }

    fn activate_output(
        &mut self,
        call: RequestId,
        binding: RequestId,
        speech: SpeechId,
        out: &mut Out,
    ) {
        let live =
            !self.session.closing_past_edge() && self.activity.speech_can_activate_output(speech);
        let activated = live
            .then(|| {
                let current = &self.session.outputs;
                self.session.output_bindings.activate(binding, speech, current, out)
            })
            .flatten();
        let result = if live && activated.is_none() {
            let reason = OutputBindingError::ActivationUnavailable;
            out.trace(crate::trace::Step::Playout(crate::trace::PlayoutStep::OutputRefused {
                speech,
                reason,
            }));
            Err(reason)
        } else {
            Ok(activated)
        };
        out.answer(Some(call), super::Answer::OutputActivated(result));
    }

    fn release_output(&mut self, request: RequestId, speech: SpeechId, out: &mut Out) {
        if self.session.output_bindings.bound.get(&request).is_some_and(|b| b.speech == speech) {
            self.release_paused_speech(speech, out);
            self.session.output_bindings.bound.remove(&request);
            self.retire_outputs(out);
        }
    }

    fn output_unsettled(&mut self, request: RequestId, speech: SpeechId, out: &mut Out) {
        let Some(bound) =
            self.session.output_bindings.bound.get(&request).filter(|b| b.speech == speech)
        else {
            return;
        };
        // Incomplete physical cleanup retains the lease in its existing activation phase.
        // It must neither activate a reserved binding nor retire an active one.
        let diagnostic = crate::events::DiagnosticEvent::warning(
            "output cleanup is incomplete; retaining its playback binding",
        )
        .with("attachment", bound.binding.segment.attachment.to_string())
        .with("segment", bound.binding.segment.segment.to_string());
        out.machine().emit(crate::events::SessionEvent::Diagnostic(diagnostic));
    }

    pub(crate) fn retire_outputs(&mut self, out: &mut Out) {
        self.session.output_bindings.retire(&self.session.outputs, out);
    }

    pub(crate) fn speech_outputs(&self, speech: SpeechId) -> &Outputs {
        self.session.output_bindings.speech_outputs(speech).unwrap_or(&self.session.outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_session::effects::Planned;
    use lk_agents_abi::resources::AudioOutputRef;
    use lk_agents_abi::voice::io::AudioOutputCapabilities;

    fn outputs(id: u64, pause: bool) -> Outputs {
        Outputs::new(
            Some(AudioOutputRef {
                id: ResourceId(id),
                label: "speaker".into(),
                capabilities: AudioOutputCapabilities { pause },
            }),
            None,
        )
    }

    // Cleanup cannot release or activate the wrong reservation; only its matched final
    // settlement releases the old destination, even after the selected output changes.
    #[test]
    fn cleanup_retains_the_lease_until_its_correlated_release() {
        let mut driver = crate::agent_session::tests::driver();
        let speech = SpeechId(7);
        let request = RequestId(3);
        let original = outputs(1, true);
        let mut out = driver.out();
        driver.session.output_bindings.bind(request, speech, &original).unwrap();
        driver.session.output_bindings.activate(request, speech, &original, &mut out).unwrap();
        driver.session.outputs = outputs(2, false);
        out.take();
        driver.on_output_binding(
            OutputBindingEvent::CleanupPending { binding: request, speech },
            &mut out,
        );
        driver.retire_outputs(&mut out);
        assert_eq!(driver.session.output_bindings.speech_outputs(speech), Some(&original));
        assert_eq!(driver.session.output_bindings.held().collect::<Vec<_>>(), [ResourceId(1)]);
        assert!(!out.take().iter().any(|item| matches!(
            item,
            Planned::Effect(Effect::Audio(AudioOp::SetAudioOutput { previous: Some(_), .. }))
        )));
        driver.on_output_binding(
            OutputBindingEvent::ReleaseRequested { binding: request, speech: SpeechId(8) },
            &mut out,
        );
        assert!(driver.session.output_bindings.speech_outputs(speech).is_some());
        driver.on_output_binding(
            OutputBindingEvent::ReleaseRequested { binding: request, speech },
            &mut out,
        );
        assert_eq!(driver.session.output_bindings.held().count(), 0);
        assert!(matches!(
            &out.take()[..],
            [Planned::Effect(Effect::Audio(AudioOp::SetAudioOutput {
                previous: Some(ResourceId(1)),
                next: None,
                ..
            }))]
        ));
        driver.on_output_binding(
            OutputBindingEvent::ReleaseRequested { binding: request, speech },
            &mut out,
        );
        assert!(out.take().is_empty(), "late duplicate release must not repeat retirement");
    }

    #[test]
    fn unsettled_bindings_refuse_capacity_without_reusing_a_segment() {
        let mut bindings = OutputBindings::default();
        let first = bindings.bind(RequestId(1), SpeechId(1), &outputs(1, true)).unwrap();
        bindings.bind(RequestId(2), SpeechId(2), &outputs(2, false)).unwrap();
        assert!(matches!(
            bindings.bind(RequestId(3), SpeechId(3), &outputs(3, false)),
            Err(OutputBindingError::Capacity)
        ));
        bindings.bound.remove(&RequestId(1));
        let replacement = bindings.bind(RequestId(3), SpeechId(3), &outputs(3, false)).unwrap();
        assert_ne!(first.segment, replacement.segment);
    }

    #[test]
    fn selecting_a_destination_keeps_the_current_segments_controls_and_cleanup() {
        let original = outputs(1, true);
        let replacement = outputs(2, false);
        let speech = SpeechId(7);
        let request = RequestId(3);
        let mut bindings = OutputBindings::default();
        let grant = bindings.bind(request, speech, &original).unwrap();
        let mut out = Out::new(false);
        assert!(bindings.activate(request, speech, &original, &mut out).is_some());
        out.take();
        bindings.retire(&replacement, &mut out);
        assert!(out.take().is_empty(), "selection must not detach a playing segment");
        assert_eq!(bindings.speech_outputs(speech), Some(&original));
        assert!(bindings.speech_outputs(speech).unwrap().pause_facts().0.pause);
        assert_eq!(bindings.held().collect::<Vec<_>>(), vec![ResourceId(1)]);
        assert_eq!(grant.segment.attachment, 1);
        bindings.bound.remove(&request);
        bindings.retire(&replacement, &mut out);
        assert!(matches!(
            &out.take()[..],
            [Planned::Effect(Effect::Audio(AudioOp::SetAudioOutput {
                previous: Some(ResourceId(1)),
                next: None,
                ..
            }))]
        ));
    }

    #[test]
    fn reattaching_the_same_object_uses_a_new_binding_after_the_old_segment_settles() {
        // Registration assigns fresh resource IDs even when both refer to the same host object.
        let original = outputs(1, true);
        let replacement = outputs(2, true);
        let mut bindings = OutputBindings::default();
        let speech = SpeechId(7);
        let mut out = Out::new(false);
        let first = bindings.bind(RequestId(1), speech, &original).unwrap();
        assert!(bindings.activate(RequestId(1), speech, &original, &mut out).is_some());
        out.take();
        bindings.retire(&replacement, &mut out);
        assert!(out.take().is_empty());
        assert!(bindings.bind(RequestId(2), speech, &replacement).is_err());
        bindings.bound.remove(&RequestId(1));
        bindings.retire(&replacement, &mut out);
        let second = bindings.bind(RequestId(2), speech, &replacement).unwrap();
        assert!(bindings.activate(RequestId(2), speech, &replacement, &mut out).is_some());
        assert_ne!(first.segment, second.segment);
        assert!(matches!(
            &out.take()[..],
            [
                Planned::Effect(Effect::Audio(AudioOp::SetAudioOutput {
                    previous: Some(ResourceId(1)),
                    next: None,
                    ..
                })),
                Planned::Effect(Effect::Audio(AudioOp::SetAudioOutput {
                    previous: None,
                    next: Some(ResourceId(2)),
                    ..
                }))
            ]
        ));
    }

    #[test]
    fn a_segment_activated_after_a_replacement_plays_on_the_replacement() {
        let original = outputs(1, true);
        let replacement = Outputs::new(None, None);
        let mut bindings = OutputBindings::default();
        let mut out = Out::new(false);
        bindings.bind(RequestId(1), SpeechId(1), &original).unwrap();
        assert!(bindings.activate(RequestId(1), SpeechId(1), &original, &mut out).is_some());
        bindings.bound.remove(&RequestId(1));
        out.take();
        // Synthesised ahead of its authorization, bound but not yet activated.
        bindings.bind(RequestId(2), SpeechId(2), &original).unwrap();
        bindings.retire(&replacement, &mut out);
        assert!(
            matches!(
                &out.take()[..],
                [Planned::Effect(Effect::Audio(AudioOp::SetAudioOutput {
                    previous: Some(ResourceId(1)),
                    next: None,
                    ..
                }))]
            ),
            "a segment not yet activated keeps no deselected output attached"
        );
        let activated =
            bindings.activate(RequestId(2), SpeechId(2), &replacement, &mut out).unwrap();
        assert!(activated.outputs.audio.is_none());
        assert_eq!(activated.segment.attachment, 0);
        assert!(out.take().is_empty());
    }
}
