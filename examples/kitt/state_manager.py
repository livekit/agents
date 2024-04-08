import asyncio
import json
import logging
from typing import List

from livekit import agents, rtc
from livekit.agents.llm import ChatMessage, ChatRole


class StateManager:
    """Helper class to update the UI for the Agent Playground."""

    def __init__(self, room: rtc.Room, prompt: str):
        self._room = room
        self._agent_speaking = False
        self._agent_thinking = False
        self._current_transcription = ""
        self._current_response = ""

        self._chat_history: List[agents.llm.ChatMessage] = [
            ChatMessage(role=ChatRole.SYSTEM, text=prompt)
        ]

    @property
    def agent_speaking(self):
        self._update_state()

    @agent_speaking.setter
    def agent_speaking(self, value: bool):
        self._agent_speaking = value
        self._update_state()

    @property
    def agent_thinking(self):
        self._update_state()

    @agent_thinking.setter
    def agent_thinking(self, value: bool):
        self._agent_thinking = value
        self._update_state()

    @property
    def chat_history(self):
        return self._chat_history

    def commit_user_transcription(self, transcription: str):
        logging.info("Committing user transcription: %s", transcription)
        asyncio.create_task(
            self._room.local_participant.publish_data(
                json.dumps({"transcription": transcription})
            )
        )
        self._chat_history.append(ChatMessage(role=ChatRole.USER, text=transcription))

    def commit_agent_response(self, response: str):
        logging.info("Committing agent response: %s", response)
        asyncio.create_task(
            self._room.local_participant.publish_data(
                json.dumps({"message": response}), topic="lk-chat-topic"
            )
        )
        self._chat_history.append(ChatMessage(role=ChatRole.ASSISTANT, text=response))

    def _update_state(self):
        state = "listening"
        if self._agent_speaking:
            state = "speaking"
        elif self._agent_thinking:
            state = "thinking"
        asyncio.create_task(
            self._room.local_participant.update_metadata(
                json.dumps({"agent_state": state})
            )
        )
