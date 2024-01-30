import asyncio
import os
import random
from typing import Callable
from enum import Enum

import openai
from livekit import agents

CELEBRITIES = [
    "Tom Cruise",
    "Nicholas Cage",
    "Beyoncé",
    "Dwayne 'The Rock' Johnson",
    "Will Smith",
    "Taylor Swift",
    "Scarlett Johansson",
    "Leonardo DiCaprio",
    "Oprah Winfrey",
    "Justin Bieber",
    "Lady Gaga",
    "Robert Downey Jr.",
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
                your only job is to return the string 'start'. \
                If the user has not expressed intent to start playing, you should return the string 'wait'. Do not return any other strings."

PLAYING_PROMPT_TEMPLATE = (
    "You are an agent that can return two types of responses: A correct guess or an incorrect guess.\
    If the guess is correct, return the 'correct' string. Don't return anything else if the guess is correct.\
    If the guess is incorrect, return a short response that conveys that the guess was incorrect and give\
    a useful hint about what the answer is without revealing the name.\
    Do not return any other strings. Return 'correct' if the user has guessed the name %s correctly.\
    The spelling may be slightly off so don't worry too much about that."
)


class GameState:
    def __init__(
        self,
        ctx: agents.JobContext,
        on_state_changed: Callable[[GAME_STATE], None],
        on_guess: Callable[[str], None],
    ):
        self._ctx = ctx
        self._client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._game_state = GAME_STATE.PRE_GAME
        self._input_queue = asyncio.Queue()
        self._on_state_changed = on_state_changed
        self._on_guess = on_guess
        self._score = 0
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
        if new_state == GAME_STATE.PRE_GAME:
            self._score = 0
        self._game_state = new_state

    @property
    def score(self):
        return self._score

    def add_user_input(self, user_input: str):
        self._input_queue.put_nowait(user_input)

    async def _run(self):
        while True:
            user_input = await self._input_queue.get()
            if self._game_state == GAME_STATE.PRE_GAME:
                await self._handle_pre_game_input(user_input)
            elif self._game_state == GAME_STATE.PLAYING:
                await self._handle_playing_input(user_input)

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
        print("playing text: ", text)
        if text == "correct":
            self._score += 1
            self._on_guess(f"Correct! It was {self.current_celebrity}!")
            self._celeb_index = (self._celeb_index + 1) % len(CELEBRITIES)
        else:
            self._on_guess(text)

    def _shuffle_celebrities(self):
        random.shuffle(CELEBRITIES)

    def _flush_input_queue(self):
        while not self._input_queue.empty():
            self._input_queue.get_nowait()
