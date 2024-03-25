# Copyright 2024 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import random
from enum import Enum
from typing import Callable

import openai
from livekit import agents

CELEBRITIES = [
    "Tom Cruise",
    "Nicholas Cage",
    "Beyoncé",
    "Will Smith",
    "Taylor Swift",
    "Scarlett Johansson",
    "Leonardo DiCaprio",
    "Oprah Winfrey",
    "Justin Bieber",
    "Lady Gaga",
    "Robert Downey Junior",
    "Adele",
    "Chris Evans",
    "Kim Kardashian",
    "Brad Pitt",
    "Shakira",
    "Vin Diesel",
    "Jennifer Aniston",
    "Kanye West",
]

GAME_STATE = Enum("GAME_STATE", "PRE_GAME PLAYING")

PRE_GAME_PROMPT = "You are a game running agent in the pregame phase.\
                Your job is to listen to the user's input and start the game when the user expresses intent to start playing. \
                When the user has expressed that their intent to start playing (by using natural language), \
                your only job is to return the string 'start'. Do not return any other strings. Just return 'start'\
                when the user has expressed intent to start playing."

PLAYING_PROMPT_TEMPLATE = "You are an agent that can return two types of responses: A correct guess or an incorrect guess.\
    If the guess is correct, return the 'correct' string. Don't return anything else if the guess is correct.\
    If the guess is incorrect, return a short response that conveys that the guess was incorrect and give\
    a useful hint about what the answer is without revealing the name.\
    Do not return any other strings. Return 'correct' if the user has guessed the name %s correctly.\
    The spelling may be off so don't worry too much about that. Be generous."


class GameState:
    def __init__(
        self,
        ctx: agents.JobContext,
        send_message: Callable[[str], None],
    ):
        self._ctx = ctx
        self._client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._game_state = GAME_STATE.PRE_GAME
        self._send_message = send_message
        self._input_queue = asyncio.Queue()
        self._celeb_index = 0
        self._shuffle_celebrities()
        self._ctx.create_task(self._run())

    @property
    def current_celebrity(self):
        return CELEBRITIES[self._celeb_index]

    @property
    def game_state(self):
        return self._game_state

    @game_state.setter
    def game_state(self, new_state: GAME_STATE):
        self._flush_input_queue()
        self._game_state = new_state

    def add_user_input(self, user_input: str):
        self._input_queue.put_nowait(user_input)

    async def _run(self):
        while True:
            try:
                user_input = await self._input_queue.get()
                if self._game_state == GAME_STATE.PRE_GAME:
                    await self._handle_pre_game_input(user_input)
                elif self._game_state == GAME_STATE.PLAYING:
                    await self._handle_playing_input(user_input)
            except Exception as e:
                print("Error handling input: ", e)

    async def _handle_pre_game_input(self, user_input: str):
        res = await self._client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            n=1,
            stream=False,
            messages=[
                {"role": "system", "content": PRE_GAME_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )
        text = res.choices[0].message.content
        print("pre game text: ", text)
        if text == "start":
            self.game_state = GAME_STATE.PLAYING
            self._send_message(
                "Great! Let's start playing! Give the face swap engine some time to warm up."
            )
        elif text == "wait":
            pass

    async def _handle_playing_input(self, user_input: str):
        res = await self._client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            n=1,
            stream=False,
            messages=[
                {
                    "role": "system",
                    "content": PLAYING_PROMPT_TEMPLATE % self.current_celebrity,
                },
                {"role": "user", "content": user_input},
            ],
        )
        text = res.choices[0].message.content
        if text == "correct":
            self._send_message(f"Correct! It was {self.current_celebrity}!")
            self._celeb_index = (self._celeb_index + 1) % len(CELEBRITIES)
        else:
            self._send_message(text)

    def _shuffle_celebrities(self):
        random.shuffle(CELEBRITIES)

    def _flush_input_queue(self):
        while not self._input_queue.empty():
            self._input_queue.get_nowait()
