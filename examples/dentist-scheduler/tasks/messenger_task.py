import os
from typing import Annotated

from global_functions import (
    transfer_to_receptionist,
    transfer_to_scheduler,
    update_information,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent
from livekit.plugins import cartesia
from pydantic import Field
from supabase import AsyncClient, create_async_client


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


class Messenger(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are an assistant taking messages for the LiveKit dental office.
            Be sure to confirm details such as phone numbers with the user. If the user's number is not known, ask for it. Otherwise return the user's number during function calls.
            Be brief and to the point.""",
            tts=cartesia.TTS(voice="156fb8d2-335b-4950-9cb3-a2d33befec77"),
            tools=[update_information, transfer_to_receptionist, transfer_to_scheduler],
        )

    async def on_enter(self) -> None:
        self._supabase = await SupabaseClient.initiate_supabase()

        await self.session.generate_reply(
            instructions=f"""Introduce yourself and ask {self.session.userdata["userinfo"].name} for their phone number if not given. 
                            Their phone number is {self.session.userdata["userinfo"].phone}. Then, ask for the message they want to leave for the office."""
        )

    @function_tool()
    async def record_message(
        self,
        phone_number: Annotated[str, Field(description="The user's phone number")],
        message: Annotated[
            str, Field(description="The user's message to be left for the office")
        ],
    ) -> None:
        """Records the user's message to be left for the office and the user's phone number."""
        self.session.userdata["userinfo"].phone = phone_number
        self.session.userdata["userinfo"].message = message
        try:
            data = await self._supabase.insert_msg(
                name=self.session.userdata["userinfo"].name,
                message=message,
                phone=phone_number,
            )
            if data:
                if self.session.current_speech:
                    await self.session.current_speech.wait_for_playout()
                await self.session.generate_reply(
                    instructions="Inform the user that their message has been submitted."
                )
        except Exception as e:
            raise Exception(f"Error sending data to Supabase: {e}")
