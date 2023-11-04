from livekit import rtc
from .processor import Processor
from dataclasses import dataclass
from enum import Enum


class TextToTextProcessor(Processor[str, str]):
    pass


VoiceActivityDetectionProcessorEventType = Enum(
    'VoiceActivityDetectionProcessorEventType', ['STARTED', 'FINISHED'])


class VoiceActivityDetectionProcessor(Processor[rtc.AudioFrame, "VoiceActivityDectionProcessor.Event"]):
    @dataclass
    class Event:
        type: VoiceActivityDetectionProcessorEventType
        frames: [rtc.AudioFrame]


SpeechToTextProcessorEventType = Enum(
    'VoiceActivityDetectionProcessorEventType', ['DELTA_RESULT'])


class SpeechToTextProcessor(Processor[VoiceActivityDetectionProcessor.Event, str]):
    @dataclass
    class Event:
        type: SpeechToTextProcessorEventType
        frames: [rtc.AudioFrame]


class TextToSpeechProcessor(Processor[str, rtc.AudioFrame]):
    pass
