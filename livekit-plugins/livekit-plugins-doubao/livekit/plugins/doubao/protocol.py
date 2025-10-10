from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from enum import IntEnum


class MsgType(IntEnum):
    Invalid = 0
    FullClientRequest = 0b1
    AudioOnlyClient = 0b10
    FullServerResponse = 0b1001
    AudioOnlyServer = 0b1011
    FrontEndResultServer = 0b1100
    Error = 0b1111

    ServerACK = AudioOnlyServer


class MsgTypeFlagBits(IntEnum):
    NoSeq = 0
    PositiveSeq = 0b1
    LastNoSeq = 0b10
    NegativeSeq = 0b11
    WithEvent = 0b100


class VersionBits(IntEnum):
    Version1 = 1
    Version2 = 2
    Version3 = 3
    Version4 = 4


class HeaderSizeBits(IntEnum):
    HeaderSize4 = 1
    HeaderSize8 = 2
    HeaderSize12 = 3
    HeaderSize16 = 4


class SerializationBits(IntEnum):
    Raw = 0
    JSON = 0b1
    Thrift = 0b11
    Custom = 0b1111


class CompressionBits(IntEnum):
    None_ = 0
    Gzip = 0b1
    Custom = 0b1111


class EventType(IntEnum):
    None_ = 0
    StartConnection = 1
    StartTask = 1
    FinishConnection = 2
    FinishTask = 2

    ConnectionStarted = 50
    TaskStarted = 50
    ConnectionFailed = 51
    TaskFailed = 51
    ConnectionFinished = 52
    TaskFinished = 52

    StartSession = 100
    CancelSession = 101
    FinishSession = 102

    SessionStarted = 150
    SessionCanceled = 151
    SessionFinished = 152
    SessionFailed = 153
    UsageResponse = 154
    ChargeData = 154

    TaskRequest = 200
    UpdateConfig = 201

    # TTS specific downstream
    TTSSentenceStart = 350
    TTSSentenceEnd = 351
    TTSResponse = 352
    TTSEnded = 359


@dataclass
class Message:
    version: VersionBits = VersionBits.Version1
    header_size: HeaderSizeBits = HeaderSizeBits.HeaderSize4
    type: MsgType = MsgType.Invalid
    flag: MsgTypeFlagBits = MsgTypeFlagBits.NoSeq
    serialization: SerializationBits = SerializationBits.JSON
    compression: CompressionBits = CompressionBits.None_

    event: EventType = EventType.None_
    session_id: str = ""
    connect_id: str = ""
    sequence: int = 0
    error_code: int = 0

    payload: bytes = b""

    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        if len(data) < 3:
            raise ValueError("data too short")

        type_and_flag = data[1]
        msg_type = MsgType(type_and_flag >> 4)
        flag = MsgTypeFlagBits(type_and_flag & 0b00001111)

        msg = cls(type=msg_type, flag=flag)
        msg.unmarshal(data)
        return msg

    def marshal(self) -> bytes:
        buf = io.BytesIO()
        header = [
            (self.version << 4) | self.header_size,
            (self.type << 4) | self.flag,
            (self.serialization << 4) | self.compression,
        ]
        header_size = 4 * self.header_size
        if padding := header_size - len(header):
            header.extend([0] * padding)
        buf.write(bytes(header))

        for writer in self._get_writers():
            writer(buf)

        return buf.getvalue()

    def unmarshal(self, data: bytes) -> None:
        buf = io.BytesIO(data)

        vh = buf.read(1)[0]
        self.version = VersionBits(vh >> 4)
        self.header_size = HeaderSizeBits(vh & 0x0F)

        # skip type/flag byte (we already parsed it)
        buf.read(1)

        sc = buf.read(1)[0]
        self.serialization = SerializationBits(sc >> 4)
        self.compression = CompressionBits(sc & 0x0F)

        header_size = 4 * self.header_size
        read_size = 3
        if padding_size := header_size - read_size:
            buf.read(padding_size)

        for reader in self._get_readers():
            reader(buf)

        remaining = buf.read()
        if remaining:
            # ignore stray padding
            pass

    def _get_writers(self):
        writers = []
        if self.flag == MsgTypeFlagBits.WithEvent:
            writers.extend([self._w_event, self._w_session_id])

        if self.type in (
            MsgType.FullClientRequest,
            MsgType.FullServerResponse,
            MsgType.FrontEndResultServer,
            MsgType.AudioOnlyClient,
            MsgType.AudioOnlyServer,
        ):
            if self.flag in (MsgTypeFlagBits.PositiveSeq, MsgTypeFlagBits.NegativeSeq):
                writers.append(self._w_sequence)
        elif self.type == MsgType.Error:
            writers.append(self._w_error)
        else:
            raise ValueError(f"unsupported msg type {self.type}")

        writers.append(self._w_payload)
        return writers

    def _get_readers(self):
        readers = []
        if self.type in (
            MsgType.FullClientRequest,
            MsgType.FullServerResponse,
            MsgType.FrontEndResultServer,
            MsgType.AudioOnlyClient,
            MsgType.AudioOnlyServer,
        ):
            if self.flag in (MsgTypeFlagBits.PositiveSeq, MsgTypeFlagBits.NegativeSeq):
                readers.append(self._r_sequence)
        elif self.type == MsgType.Error:
            readers.append(self._r_error)
        else:
            raise ValueError(f"unsupported msg type {self.type}")

        if self.flag == MsgTypeFlagBits.WithEvent:
            readers.extend([self._r_event, self._r_session_id, self._r_connect_id])

        readers.append(self._r_payload)
        return readers

    # writers
    def _w_event(self, b: io.BytesIO) -> None:
        b.write(struct.pack(">i", int(self.event)))

    def _w_session_id(self, b: io.BytesIO) -> None:
        if self.event in (
            EventType.StartConnection,
            EventType.FinishConnection,
            EventType.ConnectionStarted,
            EventType.ConnectionFailed,
        ):
            return
        sid = self.session_id.encode()
        b.write(struct.pack(">I", len(sid)))
        if sid:
            b.write(sid)

    def _w_sequence(self, b: io.BytesIO) -> None:
        b.write(struct.pack(">i", self.sequence))

    def _w_error(self, b: io.BytesIO) -> None:
        b.write(struct.pack(">I", self.error_code))

    def _w_payload(self, b: io.BytesIO) -> None:
        b.write(struct.pack(">I", len(self.payload)))
        if self.payload:
            b.write(self.payload)

    # readers
    def _r_event(self, b: io.BytesIO) -> None:
        ev_b = b.read(4)
        if ev_b:
            self.event = EventType(struct.unpack(">i", ev_b)[0])

    def _r_session_id(self, b: io.BytesIO) -> None:
        if self.event in (
            EventType.StartConnection,
            EventType.FinishConnection,
            EventType.ConnectionStarted,
            EventType.ConnectionFailed,
            EventType.ConnectionFinished,
        ):
            return
        size_b = b.read(4)
        if not size_b:
            return
        size = struct.unpack(">I", size_b)[0]
        if size:
            sid = b.read(size)
            if len(sid) == size:
                self.session_id = sid.decode()

    def _r_connect_id(self, b: io.BytesIO) -> None:
        if self.event in (
            EventType.ConnectionStarted,
            EventType.ConnectionFailed,
            EventType.ConnectionFinished,
        ):
            size_b = b.read(4)
            if size_b:
                size = struct.unpack(">I", size_b)[0]
                if size:
                    cid = b.read(size)
                    if len(cid) == size:
                        self.connect_id = cid.decode()

    def _r_sequence(self, b: io.BytesIO) -> None:
        seq_b = b.read(4)
        if seq_b:
            self.sequence = struct.unpack(">i", seq_b)[0]

    def _r_error(self, b: io.BytesIO) -> None:
        ec_b = b.read(4)
        if ec_b:
            self.error_code = struct.unpack(">I", ec_b)[0]

    def _r_payload(self, b: io.BytesIO) -> None:
        size_b = b.read(4)
        if not size_b:
            return
        size = struct.unpack(">I", size_b)[0]
        if size:
            self.payload = b.read(size)
        else:
            self.payload = b""

