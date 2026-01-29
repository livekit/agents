#!/usr/bin/env python3
"""Test script for AudioStreamEncoder.

Usage:
    python scripts/test_encoder.py input.wav output.ogg
    python scripts/test_encoder.py input.wav output.mp3
    python scripts/test_encoder.py input.wav output.ogg --bitrate 96000
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from livekit.agents.utils.codecs import AudioStreamDecoder, AudioStreamEncoder


async def encode_file(
    input_path: Path,
    output_path: Path,
    output_format: str,
    bitrate: int | None,
) -> None:
    print(f"Reading: {input_path}")
    decoder = AudioStreamDecoder(sample_rate=48000, num_channels=1)

    with open(input_path, "rb") as f:
        data = f.read()
    decoder.push(data)
    decoder.end_input()

    frames = []
    async for frame in decoder:
        frames.append(frame)
    await decoder.aclose()

    total_duration = sum(f.samples_per_channel / f.sample_rate for f in frames)
    print(f"Decoded {len(frames)} frames, {total_duration:.2f}s total")

    print(f"Encoding to {output_format} (bitrate: {bitrate or 'default'})...")
    encoder = AudioStreamEncoder(
        sample_rate=48000,
        num_channels=1,
        format=output_format,  # type: ignore[arg-type]
        bitrate=bitrate,
    )

    start_time = time.perf_counter()

    for frame in frames:
        encoder.push(frame)
    encoder.end_input()

    chunks = []
    async for chunk in encoder:
        chunks.append(chunk)
    await encoder.aclose()

    encode_time = time.perf_counter() - start_time

    output_data = b"".join(c.data for c in chunks)
    output_duration = sum(c.duration for c in chunks)

    with open(output_path, "wb") as f:
        f.write(output_data)

    print(f"Wrote {len(chunks)} chunks to {output_path}")
    print(f"  Output size: {len(output_data):,} bytes")
    print(f"  Chunk durations sum: {output_duration:.2f}s")
    print(f"  Compression ratio: {len(data) / len(output_data):.1f}x")
    print(f"  Encoding time: {encode_time:.3f}s ({total_duration / encode_time:.1f}x realtime)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test AudioStreamEncoder")
    parser.add_argument("input", type=Path, help="Input audio file (wav, mp3, ogg, etc)")
    parser.add_argument("output", type=Path, help="Output file (.ogg or .mp3)")
    parser.add_argument("--bitrate", type=int, help="Bitrate in bits/sec (e.g., 64000)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    suffix = args.output.suffix.lower()
    if suffix == ".ogg":
        output_format = "opus"
    elif suffix == ".mp3":
        output_format = "mp3"
    else:
        print(f"Error: Unsupported output format: {suffix}", file=sys.stderr)
        print("Supported: .ogg, .mp3", file=sys.stderr)
        sys.exit(1)

    asyncio.run(encode_file(args.input, args.output, output_format, args.bitrate))
    print("Done!")


if __name__ == "__main__":
    main()
