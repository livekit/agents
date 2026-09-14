# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Oracle Corporation and/or its affiliates.
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

"""
This module implements simple audio caching used by the Oracle LiveKit TTS plug-in.

Author: Keith Schnable (at Oracle Corporation)
Date: 2025-08-12
"""

import json
import os
import time

from livekit.agents import utils

INDEX_FILE_NAME = "index.json"


class AudioCache:
    """
    The audio cache class.
    """

    def __init__(
        self, *, audio_cache_file_path: str, audio_cache_maximum_number_of_utterances: int
    ):
        self._audio_cache_file_path = audio_cache_file_path
        self._audio_cache_maximum_number_of_utterances = audio_cache_maximum_number_of_utterances

        if not os.path.exists(self._audio_cache_file_path):
            os.makedirs(self._audio_cache_file_path)

        self._index_file_spec = os.path.join(self._audio_cache_file_path, INDEX_FILE_NAME)

        if os.path.exists(self._index_file_spec):
            with open(self._index_file_spec, encoding="utf-8") as file:
                index_json_text = file.read()
            self._index_dictionary = json.loads(index_json_text)
        else:
            self._index_dictionary = {}

    def get_audio_bytes(
        self, *, text: str, voice: str, audio_rate: int, audio_channels: int, audio_bits: int
    ):
        """
        Get the audio bytes for the specified text, voice, audio rate, audio channels, and audio bits.

        Parameters:
        text (str): The text.
        voice (str): The voice.
        audio_rate (int): The audio rate (16000 for example).
        audio_channels (int): The audio channels (1 for example).
        audio_bits (int): The audio bits (16 for example).

        Returns:
        bytes: The audio bytes.
        """

        key = AudioCache.form_key(
            text=text,
            voice=voice,
            audio_rate=audio_rate,
            audio_channels=audio_channels,
            audio_bits=audio_bits,
        )

        if key in self._index_dictionary:
            dictionary = self._index_dictionary[key]
            audio_bytes_file_name = dictionary["audio_bytes_file_name"]
            audio_bytes_file_spec = os.path.join(self._audio_cache_file_path, audio_bytes_file_name)
            if os.path.exists(audio_bytes_file_spec):
                write_index_dictionary = True
                dictionary["last_accessed_milliseconds"] = int(time.time() * 1000)
                with open(audio_bytes_file_spec, "rb") as file:
                    audio_bytes = file.read()
            else:
                del self._index_dictionary[key]
                write_index_dictionary = True
                audio_bytes = None
        else:
            write_index_dictionary = False
            audio_bytes = None

        if write_index_dictionary:
            with open(self._index_file_spec, "w", encoding="utf-8") as file:
                json.dump(self._index_dictionary, file, indent=4)

        return audio_bytes

    def set_audio_bytes(
        self,
        *,
        text: str,
        voice: str,
        audio_rate: int,
        audio_channels: int,
        audio_bits: int,
        audio_bytes: bytes,
    ):
        """
        Set the audio bytes for the specified text, voice, audio rate, audio channels, audio bits, and audio bytes.

        Parameters:
        text (str): The text.
        voice (str): The voice.
        audio_rate (int): The audio rate (16000 for example).
        audio_channels (int): The audio channels (1 for example).
        audio_bits (int): The audio bits (16 for example).
        audio_bytes (bytes) : The audio bytes.

        Returns:
        (nothing)
        """

        key = AudioCache.form_key(
            text=text,
            voice=voice,
            audio_rate=audio_rate,
            audio_channels=audio_channels,
            audio_bits=audio_bits,
        )

        if key in self._index_dictionary:
            dictionary = self._index_dictionary[key]
            audio_bytes_file_name = dictionary["audio_bytes_file_name"]
            write_index_dictionary = False
        else:
            audio_bytes_file_name = str(utils.shortuuid())
            dictionary = {}
            dictionary["audio_bytes_file_name"] = audio_bytes_file_name
            dictionary["created_milliseconds"] = int(time.time() * 1000)
            dictionary["last_accessed_milliseconds"] = dictionary["created_milliseconds"]
            self._index_dictionary[key] = dictionary
            write_index_dictionary = True

        audio_bytes_file_spec = os.path.join(self._audio_cache_file_path, audio_bytes_file_name)

        with open(audio_bytes_file_spec, "wb") as file:
            file.write(audio_bytes)

        if write_index_dictionary:
            with open(self._index_file_spec, "w", encoding="utf-8") as file:
                json.dump(self._index_dictionary, file, indent=4)

        self.clean_up_old_utterances()

    def clean_up_old_utterances(self):
        """
        Clean up old utterance files based on the audio_cache_maximum_number_of_utterances parameter.
        The oldest utterances get deleted first.

        Parameters:
        (none)

        Returns:
        (nothing)
        """

        while len(self._index_dictionary) > self._audio_cache_maximum_number_of_utterances:
            oldest_key = None
            oldest_dictionary = None
            for key, dictionary in self._index_dictionary:
                if (
                    oldest_dictionary is None
                    or dictionary["last_accessed_milliseconds"]
                    < oldest_dictionary["last_accessed_milliseconds"]
                ):
                    oldest_key = key
                    oldest_dictionary = dictionary

            audio_bytes_file_name = oldest_dictionary["audio_bytes_file_name"]
            audio_bytes_file_spec = os.path.join(self._audio_cache_file_path, audio_bytes_file_name)
            if os.path.exists(audio_bytes_file_spec):
                os.remove(audio_bytes_file_spec)
            del self._index_dictionary[oldest_key]

        with open(self._index_file_spec, "w", encoding="utf-8") as file:
            json.dump(self._index_dictionary, file, indent=4)

    @staticmethod
    def form_key(*, text: str, voice: str, audio_rate: int, audio_channels: int, audio_bits: int):
        """
        Form the key for the specified text, voice, audio rate, audio channels, and audio bits.

        Parameters:
        text (str): The text.
        voice (str): The voice.
        audio_rate (int): The audio rate (16000 for example).
        audio_channels (int): The audio channels (1 for example).
        audio_bits (int): The audio bits (16 for example).

        Returns:
        (nothing)
        """

        key = (
            voice
            + "\t"
            + str(audio_rate)
            + "\t"
            + str(audio_channels)
            + "\t"
            + str(audio_bits)
            + "\t"
            + text
        )
        return key
