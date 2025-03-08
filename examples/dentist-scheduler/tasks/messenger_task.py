from __future__ import annotations

import os
from livekit.agents.llm import ai_function
from livekit.agents.voice import AgentTask
from livekit.plugins import openai, cartesia, deepgram, silero
from supabase import create_async_client, AsyncClient


class SupabaseClient:
    def __init__(self, supabase: AsyncClient) -> None:
        self._supabase = supabase

    @classmethod
    async def initiate_supabase(supabase):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        supabase_client: AsyncClient = await create_async_client(url, key)
        return supabase(supabase_client)

    async def insert_msg(self, name: str, message: str, phone: str) -> list:
        data = await (
            self._supabase.table("messages")
            .insert({"name": name, "message": message, "phone_number": phone})
            .execute()
        )
        return data


class Messenger(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Shimmer, an assistant taking messages for the LiveKit dental office.
            Be sure to confirm details such as phone numbers with the user. Be brief and to the point.""",
        )

    async def on_enter(self) -> None:
        self._userinfo = self.agent.userdata["userinfo"]
        self._supabase = await SupabaseClient.initiate_supabase()

        await self.agent.generate_reply(
            instructions=f"""Introduce yourself and ask {self._userinfo.name} for their phone number if not given. 
                            Their phone number is {self._userinfo.phone}. Then, ask for the message they want to leave for the office."""
        )

    @ai_function()
    async def record_message(self, phone_number: str, message: str) -> None:
        """Records the user's message to be left for the office and the user's phone number.

        Args:
            phone_number: The user's phone number
            message: The user's message to be left for the office
        """
        self.agent.userdata["userinfo"].phone = phone_number
        self.agent.userdata["userinfo"].message = message

        data = await self._supabase.insert_msg(
            name=self.agent.userdata["userinfo"].name,
            message=message,
            phone=phone_number,
        )
        if data:
            if self.agent.current_speech:
                await self.agent.current_speech.wait_for_playout()
            await self.agent.generate_reply(
                instructions="Inform the user that their message has been submitted."
            )
        else:
            raise Exception("Error sending data to Supabase")

    @ai_function()
    async def transfer_to_receptionist(self) -> AgentTask:
        """Transfers the user to the Receptionist"""
        from . import Receptionist

        return Receptionist(), "Transferring you to our receptionist!"

    @ai_function()
    async def transfer_to_scheduler(self, service: str) -> AgentTask:
        """
        Transfers the user to the Scheduler.

        Args:
            service: Either "schedule", "reschedule", or "cancel"
        """
        from . import Scheduler

        return Scheduler(service=service), "Transferring you to our scheduler!"
