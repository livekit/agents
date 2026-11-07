//! Named host arguments for the production background player's binding API.
use super::*;
use lk_agents_abi::voice::background_audio::Sound;

pub(super) struct Background {
    pub audio: BackgroundAudio,
}

pub(super) struct StartOptions {
    pub ambient: Vec<Sound>,
    pub draw: f64,
    pub choose: bool,
}

pub(super) struct StateOptions {
    pub sounds: Vec<Sound>,
    pub draw: f64,
    pub thinking: bool,
    pub choose: bool,
}

impl Background {
    pub async fn start(
        &self,
        room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
        options: StartOptions,
    ) -> Option<String> {
        self.audio
            .start(room, options.draw, options.ambient, options.choose, None)
            .await
            .err()
            .map(|e| e.to_string())
    }

    pub fn on_agent_state(&self, options: StateOptions) -> Option<String> {
        self.audio
            .on_agent_state(options.sounds, options.draw, options.thinking, options.choose)
            .err()
            .map(|e| e.to_string())
    }
}
