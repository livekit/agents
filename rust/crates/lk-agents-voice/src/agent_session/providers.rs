//! The providers the core holds by id: received in commands, kept by the session, a switch or an
//! inline task. One no longer held anywhere at the end of a step is released.

use crate::agent_session::Driver;
use crate::agent_session::effects::{Effect, Out, ProviderCall};
use lk_agents_abi::resources::ResourceId;
use std::collections::BTreeSet;

impl Driver {
    /// What the core holds now.
    fn providers_held(&self) -> BTreeSet<ResourceId> {
        let s = &self.session;
        let slots = [
            s.llm.as_ref().map(|r| r.id),
            s.tts.as_ref().map(|r| r.id),
            s.stt.as_ref().map(|r| r.id),
            s.vad.as_ref().map(|r| r.id),
        ];
        let updates = self.activity.handoff.updates().chain(self.activity.tasks.frames());
        slots.into_iter().flatten().chain(updates.flat_map(|u| u.resources())).collect()
    }

    /// Every provider the core held or was handed since the last step and holds no more.
    pub(crate) fn release_providers(&mut self, out: &mut Out) {
        let held = self.providers_held();
        for &id in self.session.providers.difference(&held) {
            out.machine().push(Effect::Provider(ProviderCall::Release(id)));
        }
        self.session.providers = held;
    }

    /// The loop ended: nothing is held.
    pub(crate) fn release_all_providers(&mut self) {
        self.session.providers.extend(self.providers_held());
        for id in std::mem::take(&mut self.session.providers) {
            self.shared.resources.release(id);
        }
    }
}
