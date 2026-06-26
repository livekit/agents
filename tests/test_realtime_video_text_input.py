from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit import rtc

from livekit.agents import llm
from livekit.agents.voice.agent_activity import AgentActivity

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_video_frame():
    """Create a mock VideoFrame for testing."""
    frame = MagicMock(spec=rtc.VideoFrame)
    frame.width = 640
    frame.height = 480
    frame.type = rtc.VideoBufferType.RGBA
    return frame


class TestVideoFrameWithTextInput:
    """Verify that text input messages include the latest video frame when available."""

    def test_push_video_caches_frame(self, mock_video_frame):
        """push_video should cache the latest video frame."""
        activity = MagicMock(spec=AgentActivity)
        activity._started = True
        activity._rt_session = MagicMock()
        activity._latest_video_frame = None

        AgentActivity.push_video(activity, mock_video_frame)

        assert activity._latest_video_frame is mock_video_frame

    def test_push_video_skips_when_not_started(self, mock_video_frame):
        """push_video should not cache frame when activity is not started."""
        activity = MagicMock(spec=AgentActivity)
        activity._started = False
        activity._latest_video_frame = None

        AgentActivity.push_video(activity, mock_video_frame)

        assert activity._latest_video_frame is None

    @pytest.mark.asyncio
    async def test_realtime_reply_task_includes_video_frame(self, mock_video_frame):
        """_realtime_reply_task should include cached video frame as ImageContent."""
        activity = MagicMock(spec=AgentActivity)
        activity._latest_video_frame = mock_video_frame

        mock_rt_session = MagicMock()
        mock_rt_session.chat_ctx = llm.ChatContext.empty()
        mock_rt_session.update_chat_ctx = AsyncMock()
        mock_rt_session.realtime_model.capabilities.per_response_tool_choice = False
        mock_rt_session.generate_reply = MagicMock()
        mock_rt_session.generate_reply.return_value = asyncio.Future()
        activity._rt_session = mock_rt_session

        mock_speech_handle = MagicMock()
        mock_speech_handle._wait_for_authorization = AsyncMock()
        mock_speech_handle.wait_if_not_interrupted = AsyncMock()
        mock_speech_handle.interrupted = False
        mock_speech_handle.allow_interruptions = True

        activity._session = MagicMock()
        activity._agent = MagicMock()
        activity._agent._chat_ctx = llm.ChatContext.empty()
        activity._session._conversation_item_added = MagicMock()

        activity._user_silence_event = MagicMock()
        activity._user_silence_event.wait = AsyncMock()
        activity._authorization_allowed = MagicMock()
        activity._authorization_allowed.wait = AsyncMock()
        activity._tool_choice = None

        captured_chat_ctx = None

        async def mock_update_chat_ctx(chat_ctx):
            nonlocal captured_chat_ctx
            captured_chat_ctx = chat_ctx

        mock_rt_session.update_chat_ctx = mock_update_chat_ctx

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                AgentActivity._realtime_reply_task(
                    activity,
                    speech_handle=mock_speech_handle,
                    model_settings=MagicMock(tool_choice=None),
                    user_input="Can you see me?",
                ),
                timeout=0.1,
            )

        assert captured_chat_ctx is not None
        user_messages = [m for m in captured_chat_ctx.messages() if m.role == "user"]
        assert len(user_messages) > 0

        last_user_msg = user_messages[-1]
        has_image = any(isinstance(c, llm.ImageContent) for c in last_user_msg.content)
        assert has_image, "User message should include ImageContent with video frame"

    @pytest.mark.asyncio
    async def test_realtime_reply_task_no_video_frame(self):
        """_realtime_reply_task should work fine without cached video frame."""
        activity = MagicMock(spec=AgentActivity)
        activity._latest_video_frame = None

        mock_rt_session = MagicMock()
        mock_rt_session.chat_ctx = llm.ChatContext.empty()
        mock_rt_session.realtime_model.capabilities.per_response_tool_choice = False
        mock_rt_session.generate_reply = MagicMock()
        mock_rt_session.generate_reply.return_value = asyncio.Future()
        activity._rt_session = mock_rt_session

        mock_speech_handle = MagicMock()
        mock_speech_handle._wait_for_authorization = AsyncMock()
        mock_speech_handle.wait_if_not_interrupted = AsyncMock()
        mock_speech_handle.interrupted = False
        mock_speech_handle.allow_interruptions = True

        activity._session = MagicMock()
        activity._agent = MagicMock()
        activity._agent._chat_ctx = llm.ChatContext.empty()
        activity._session._conversation_item_added = MagicMock()

        activity._user_silence_event = MagicMock()
        activity._user_silence_event.wait = AsyncMock()
        activity._authorization_allowed = MagicMock()
        activity._authorization_allowed.wait = AsyncMock()
        activity._tool_choice = None

        captured_chat_ctx = None

        async def mock_update_chat_ctx(chat_ctx):
            nonlocal captured_chat_ctx
            captured_chat_ctx = chat_ctx

        mock_rt_session.update_chat_ctx = mock_update_chat_ctx

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                AgentActivity._realtime_reply_task(
                    activity,
                    speech_handle=mock_speech_handle,
                    model_settings=MagicMock(tool_choice=None),
                    user_input="Can you see me?",
                ),
                timeout=0.1,
            )

        assert captured_chat_ctx is not None
        user_messages = [m for m in captured_chat_ctx.messages() if m.role == "user"]
        assert len(user_messages) > 0

        last_user_msg = user_messages[-1]
        has_image = any(isinstance(c, llm.ImageContent) for c in last_user_msg.content)
        assert not has_image, "User message should not include ImageContent when no video frame"
