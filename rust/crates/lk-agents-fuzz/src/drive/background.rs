//! Named host arguments for the production background player's binding API.
use super::*;
use lk_agents_abi::voice::background_audio::Sounds;

pub(super) struct Background {
    pub audio: BackgroundAudio,
}

pub(super) struct StartOptions {
    pub ambient: Sounds,
    pub draw: f64,
}

pub(super) struct StateOptions {
    pub sounds: Sounds,
    pub draw: f64,
    pub thinking: bool,
}

impl Background {
    pub async fn start(
        &self,
        room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
        options: StartOptions,
    ) -> Option<String> {
        self.audio
            .start(room, options.draw, Some(options.ambient), None)
            .await
            .err()
            .map(|e| e.to_string())
    }

    pub fn on_agent_state(&self, options: StateOptions) -> Option<String> {
        self.audio
            .on_agent_state(Some(options.sounds), options.draw, options.thinking)
            .err()
            .map(|e| e.to_string())
    }
}
