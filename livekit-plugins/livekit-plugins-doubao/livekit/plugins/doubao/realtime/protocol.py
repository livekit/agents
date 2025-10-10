"""豆包 Realtime API 二进制协议编解码器"""

from __future__ import annotations

import gzip
import json
import struct
from enum import IntEnum
from typing import Optional


class MessageType(IntEnum):
    """消息类型"""
    FULL_CLIENT_REQUEST = 0b0001
    AUDIO_ONLY_REQUEST = 0b0010
    FULL_SERVER_RESPONSE = 0b1001
    AUDIO_ONLY_RESPONSE = 0b1011
    ERROR_INFORMATION = 0b1111


class SerializationMethod(IntEnum):
    """序列化方式"""
    RAW = 0b0000  # 无特殊序列化，主要针对二进制音频数据
    JSON = 0b0001  # 主要针对文本类型消息


class CompressionMethod(IntEnum):
    """压缩方式"""
    NONE = 0b0000  # 无压缩（推荐）
    GZIP = 0b0001


class MessageFlags:
    """消息标志位"""
    NO_FLAGS = 0b0000
    HAS_SEQUENCE = 0b0001  # 序号大于 0 的非终端数据包
    LAST_NO_SEQUENCE = 0b0010  # 最后一个无序号的数据包
    LAST_WITH_NEG_SEQUENCE = 0b0011  # 最后一个序号小于 0 的数据包
    HAS_EVENT = 0b0100  # 携带事件ID


class DoubaoProtocolCodec:
    """豆包协议编解码器"""

    PROTOCOL_VERSION = 0b0001
    HEADER_SIZE = 0b0001  # 4 bytes

    @staticmethod
    def encode_message(
        message_type: MessageType,
        payload: bytes,
        event_id: Optional[int] = None,
        sequence: Optional[int] = None,
        session_id: Optional[str] = None,
        serialization: SerializationMethod = SerializationMethod.JSON,
        compression: CompressionMethod = CompressionMethod.GZIP,
    ) -> bytes:
        """
        编码消息为豆包二进制协议格式

        Args:
            message_type: 消息类型
            payload: 消息负载（压缩前的数据）
            event_id: 事件ID
            sequence: 序列号
            session_id: 会话ID（StartConnection 不需要）
            serialization: 序列化方式
            compression: 压缩方式（默认 GZIP）

        Returns:
            编码后的二进制数据
        """
        # 压缩 payload
        if compression == CompressionMethod.GZIP:
            payload = gzip.compress(payload)

        # 构建header (4 bytes)
        byte0 = (DoubaoProtocolCodec.PROTOCOL_VERSION << 4) | DoubaoProtocolCodec.HEADER_SIZE

        # 确定 message_type_specific_flags
        flags = MessageFlags.NO_FLAGS
        if event_id is not None:
            flags = MessageFlags.HAS_EVENT
        elif sequence is not None:
            if sequence > 0:
                flags = MessageFlags.HAS_SEQUENCE
            elif sequence == -1:
                flags = MessageFlags.LAST_WITH_NEG_SEQUENCE

        byte1 = (message_type << 4) | flags
        byte2 = (serialization << 4) | compression
        byte3 = 0x00  # Reserved

        header = bytes([byte0, byte1, byte2, byte3])

        # 构建 optional fields（使用大端序！）
        optional = b''

        # sequence (如果有)
        if sequence is not None and flags in (MessageFlags.HAS_SEQUENCE, MessageFlags.LAST_WITH_NEG_SEQUENCE):
            optional += struct.pack('>i', sequence)

        # event_id (如果有)
        if event_id is not None:
            optional += struct.pack('>I', event_id)

        # session_id (如果有)
        # 注意：StartConnection (event_id=1) 不需要 session_id
        if session_id is not None:
            session_id_bytes = session_id.encode('utf-8')
            optional += struct.pack('>I', len(session_id_bytes))
            optional += session_id_bytes

        # payload_size + payload（使用大端序！）
        payload_size = struct.pack('>I', len(payload))

        return header + optional + payload_size + payload

    @staticmethod
    def decode_message(data: bytes) -> dict:
        """
        解码豆包二进制协议消息

        Args:
            data: 二进制数据

        Returns:
            解码后的消息字典，包含:
                - message_type: 消息类型
                - flags: 消息标志
                - serialization: 序列化方式
                - compression: 压缩方式
                - event_id: 事件ID (可选)
                - sequence: 序列号 (可选)
                - session_id: 会话ID (可选)
                - connect_id: 连接ID (可选)
                - error_code: 错误码 (可选)
                - payload: 消息负载
        """
        if len(data) < 4:
            raise ValueError("Invalid message: too short for header")

        # 解析 header
        byte0, byte1, byte2, byte3 = data[0:4]

        protocol_version = (byte0 >> 4) & 0x0F
        header_size = byte0 & 0x0F

        message_type = (byte1 >> 4) & 0x0F
        flags = byte1 & 0x0F

        serialization = (byte2 >> 4) & 0x0F
        compression = byte2 & 0x0F

        result = {
            'message_type': MessageType(message_type),
            'flags': flags,
            'serialization': SerializationMethod(serialization),
            'compression': CompressionMethod(compression),
        }

        # 解析 optional fields
        offset = 4

        # error_code (仅错误消息，使用大端序)
        if message_type == MessageType.ERROR_INFORMATION:
            if len(data) < offset + 4:
                raise ValueError("Invalid error message: missing error code")
            result['error_code'] = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4

            # 错误消息的 payload_size 直接在 error_code 后面
            if len(data) < offset + 4:
                raise ValueError("Invalid error message: missing payload size")
            payload_size = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4

            if len(data) < offset + payload_size:
                raise ValueError("Invalid error message: payload size mismatch")
            result['payload'] = data[offset:offset+payload_size]

            # 错误消息的 payload 不压缩，直接返回
            return result

        # sequence（使用大端序）
        if flags in (MessageFlags.HAS_SEQUENCE, MessageFlags.LAST_WITH_NEG_SEQUENCE):
            if len(data) < offset + 4:
                raise ValueError("Invalid message: missing sequence")
            result['sequence'] = struct.unpack('>i', data[offset:offset+4])[0]
            offset += 4

        # event_id（使用大端序）
        if flags == MessageFlags.HAS_EVENT:
            if len(data) < offset + 4:
                raise ValueError("Invalid message: missing event_id")
            result['event_id'] = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4

        # session_id (Session类事件，使用大端序)
        # 服务端返回消息如果有 session_id，格式是：session_id_len (4 bytes) + session_id
        if len(data) >= offset + 4:
            session_id_len = struct.unpack('>I', data[offset:offset+4])[0]

            # 检查 session_id_len 是否合理（36-40字节为UUID长度）
            if 0 < session_id_len < 200 and len(data) >= offset + 4 + session_id_len:
                try:
                    # 尝试解码为 UTF-8 字符串
                    session_id = data[offset+4:offset+4+session_id_len].decode('utf-8')
                    if session_id.isprintable():
                        result['session_id'] = session_id
                        offset += 4 + session_id_len
                except:
                    pass

        # payload_size + payload（使用大端序）
        if len(data) < offset + 4:
            raise ValueError("Invalid message: missing payload size")

        payload_size = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4

        if len(data) < offset + payload_size:
            raise ValueError(f"Invalid message: payload size mismatch, expected {payload_size}, remaining {len(data) - offset}")

        result['payload'] = data[offset:offset+payload_size]

        # 解压缩 payload
        if compression == CompressionMethod.GZIP:
            try:
                result['payload'] = gzip.decompress(result['payload'])
            except Exception as e:
                # 如果解压缩失败，保持原样（可能本来就没压缩）
                pass

        return result


def encode_client_event(
    event_id: int,
    payload: bytes,
    session_id: Optional[str] = None,
    is_audio: bool = False,
) -> bytes:
    """
    编码客户端事件（便捷函数）

    Args:
        event_id: 事件ID
        payload: JSON或音频数据（压缩前）
        session_id: 会话ID（StartConnection 不需要）
        is_audio: 是否为音频数据

    Returns:
        编码后的二进制数据
    """
    message_type = MessageType.AUDIO_ONLY_REQUEST if is_audio else MessageType.FULL_CLIENT_REQUEST
    # 音频用 NO_SERIALIZATION，JSON用 JSON
    serialization = SerializationMethod.RAW if is_audio else SerializationMethod.JSON
    # 都用 gzip 压缩
    compression = CompressionMethod.GZIP

    return DoubaoProtocolCodec.encode_message(
        message_type=message_type,
        payload=payload,
        event_id=event_id,
        session_id=session_id,
        serialization=serialization,
        compression=compression,
    )


def debug_binary_message(data: bytes, prefix: str = "") -> str:
    """
    调试二进制消息，返回可读格式的字符串

    Args:
        data: 二进制数据
        prefix: 日志前缀

    Returns:
        可读的调试信息
    """
    if len(data) < 4:
        return f"{prefix}Invalid message: too short ({len(data)} bytes)"

    lines = [f"{prefix}Binary message analysis:"]
    lines.append(f"  Total size: {len(data)} bytes")
    lines.append(f"  First 20 bytes (hex): {data[:min(20, len(data))].hex()}")

    # 解析 header
    byte0, byte1, byte2, byte3 = data[0:4]
    protocol_version = (byte0 >> 4) & 0x0F
    header_size = byte0 & 0x0F
    message_type = (byte1 >> 4) & 0x0F
    flags = byte1 & 0x0F
    serialization = (byte2 >> 4) & 0x0F
    compression = byte2 & 0x0F

    lines.append(f"  Header:")
    lines.append(f"    Protocol version: {protocol_version}")
    lines.append(f"    Header size: {header_size}")
    lines.append(f"    Message type: {message_type} ({MessageType(message_type).name if message_type in MessageType._value2member_map_ else 'Unknown'})")
    lines.append(f"    Flags: {flags:#06b}")
    lines.append(f"    Serialization: {serialization} ({SerializationMethod(serialization).name if serialization in SerializationMethod._value2member_map_ else 'Unknown'})")
    lines.append(f"    Compression: {compression}")

    offset = 4

    # 特殊处理错误消息
    if message_type == 15:  # ERROR_INFORMATION
        if len(data) >= offset + 4:
            error_code = struct.unpack('>I', data[offset:offset+4])[0]
            lines.append(f"  Error code: {error_code}")
            offset += 4

        if len(data) >= offset + 4:
            payload_size = struct.unpack('>I', data[offset:offset+4])[0]
            lines.append(f"  Payload size: {payload_size}")
            offset += 4

            if len(data) >= offset + payload_size:
                payload = data[offset:offset+payload_size]
                lines.append(f"  Payload (first 100 bytes): {payload[:min(100, len(payload))].hex()}")
                try:
                    payload_json = json.loads(payload.decode('utf-8'))
                    lines.append(f"  Payload (JSON): {json.dumps(payload_json, ensure_ascii=False, indent=2)}")
                except:
                    try:
                        payload_text = payload.decode('utf-8')
                        lines.append(f"  Payload (text): {payload_text}")
                    except:
                        pass

        return "\n".join(lines)

    # 解析 optional fields
    if flags in (0b0001, 0b0011):  # HAS_SEQUENCE
        if len(data) >= offset + 4:
            sequence = struct.unpack('>i', data[offset:offset+4])[0]
            lines.append(f"  Sequence: {sequence}")
            offset += 4

    if flags == 0b0100:  # HAS_EVENT
        if len(data) >= offset + 4:
            event_id = struct.unpack('>I', data[offset:offset+4])[0]
            lines.append(f"  Event ID: {event_id}")
            offset += 4

    # 尝试解析 session_id/connect_id (可能存在，使用大端序)
    if len(data) >= offset + 4:
        possible_id_len = struct.unpack('>I', data[offset:offset+4])[0]
        if 0 < possible_id_len < 100 and len(data) >= offset + 4 + possible_id_len:
            # 可能是 ID 长度
            try:
                id_bytes = data[offset+4:offset+4+possible_id_len]
                id_str = id_bytes.decode('utf-8')
                if id_str.isprintable():
                    lines.append(f"  ID (session/connect): {id_str} (length={possible_id_len})")
                    offset += 4 + possible_id_len
            except:
                pass

    # 解析 payload（使用大端序）
    if len(data) >= offset + 4:
        payload_size = struct.unpack('>I', data[offset:offset+4])[0]
        lines.append(f"  Payload size: {payload_size}")
        offset += 4

        if len(data) >= offset + payload_size:
            payload = data[offset:offset+payload_size]
            lines.append(f"  Payload (first 100 bytes): {payload[:min(100, len(payload))].hex()}")

            # 尝试解析为 JSON
            if serialization == SerializationMethod.JSON:
                try:
                    payload_json = json.loads(payload.decode('utf-8'))
                    lines.append(f"  Payload (JSON): {json.dumps(payload_json, ensure_ascii=False, indent=2)}")
                except:
                    try:
                        # 尝试直接解码为文本
                        payload_text = payload.decode('utf-8')
                        lines.append(f"  Payload (text): {payload_text}")
                    except:
                        pass
        else:
            lines.append(f"  WARNING: Payload size mismatch! Expected {payload_size}, remaining {len(data) - offset}")

    return "\n".join(lines)