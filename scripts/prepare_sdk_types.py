#!/usr/bin/env python3
"""Prepare external SDK typing, generating Riva protobuf stubs from its installed descriptors."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from google.protobuf.compiler.plugin_pb2 import CodeGeneratorRequest, CodeGeneratorResponse
from google.protobuf.descriptor import FileDescriptor
from google.protobuf.descriptor_pb2 import FileDescriptorProto

_ROOT = Path(__file__).resolve().parent.parent
_MARKER = "sdk-types.json"


def _riva_descriptors() -> list[FileDescriptorProto]:
    descriptors: dict[str, FileDescriptorProto] = {}

    def visit(descriptor: FileDescriptor) -> None:
        if descriptor.name in descriptors:
            return
        record = FileDescriptorProto.FromString(descriptor.serialized_pb)
        # The wheel relocates its generated modules from riva.proto to riva.client.proto.
        record.name = record.name.replace("riva/proto/", "riva/client/proto/")
        for index, dependency in enumerate(record.dependency):
            record.dependency[index] = dependency.replace("riva/proto/", "riva/client/proto/")
        descriptors[descriptor.name] = record
        for dependency in descriptor.dependencies:
            visit(dependency)

    for name in ("riva_asr_pb2", "riva_tts_pb2"):
        module = importlib.import_module(f"riva.client.proto.{name}")
        descriptor = module.DESCRIPTOR
        if not isinstance(descriptor, FileDescriptor):
            raise TypeError(f"{module.__name__}.DESCRIPTOR is not a protobuf FileDescriptor")
        visit(descriptor)
    return list(descriptors.values())


def prepare(output: Path) -> None:
    """Generate protobuf/service stubs and copy the inspected SDK API annotations."""
    if output.exists() and any(output.iterdir()) and not (output / _MARKER).is_file():
        raise ValueError(f"Refusing to replace a directory without {_MARKER}: {output}")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("nvidia-riva-client", "sphn", "mypy-protobuf")
    }
    sdk = importlib.import_module("riva.client")
    opus = importlib.import_module("sphn")
    expected = {
        sdk.Auth.__init__: ("self", "ssl_cert", "use_ssl", "uri", "metadata_args"),
        sdk.ASRService.__init__: ("self", "auth"),
        sdk.ASRService.streaming_response_generator: ("self", "audio_chunks", "streaming_config"),
        sdk.SpeechSynthesisService.__init__: ("self", "auth"),
        sdk.SpeechSynthesisService.synthesize_online: (
            "self",
            "text",
            "voice_name",
            "language_code",
            "encoding",
            "sample_rate_hz",
            "audio_prompt_file",
            "audio_prompt_encoding",
            "quality",
        ),
        opus.OpusStreamReader: ("sample_rate", "flush_every_n_samples"),
        opus.OpusStreamReader.append_bytes: ("self", "data"),
        opus.OpusStreamWriter: ("sample_rate",),
        opus.OpusStreamWriter.append_pcm: ("self", "pcm"),
    }
    for function, names in expected.items():
        actual = tuple(inspect.signature(function).parameters)
        if actual != names:
            raise RuntimeError(f"SDK signature changed for {function.__qualname__}: {actual}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sdk-types-", dir=output.parent) as temporary:
        staged = Path(temporary) / "types"
        shutil.copytree(_ROOT / "scripts/type_stubs", staged)
        descriptors = _riva_descriptors()
        request = CodeGeneratorRequest(
            proto_file=descriptors,
            file_to_generate=[record.name for record in descriptors],
            parameter="quiet",
        ).SerializeToString()
        for entry in ("main", "grpc"):
            result = subprocess.run(
                [sys.executable, "-c", f"from mypy_protobuf.main import {entry}; {entry}()"],
                input=request,
                stdout=subprocess.PIPE,
            )
            if result.returncode:
                raise SystemExit(result.returncode)
            response = CodeGeneratorResponse.FromString(result.stdout)
            if response.error or not response.file:
                raise RuntimeError(response.error or "SDK type generation produced no files")
            for file in response.file:
                destination = staged / file.name
                if not destination.resolve().is_relative_to(staged.resolve()):
                    raise ValueError(
                        f"Generated filename escapes the output directory: {file.name}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(file.content)
        (staged / "riva/client/proto/__init__.pyi").touch()
        (staged / "riva/__init__.pyi").touch()
        (staged / _MARKER).write_text(json.dumps(versions, sort_keys=True, indent=2) + "\n")
        if output.exists():
            shutil.rmtree(output)
        staged.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_ROOT / ".type_stubs")
    args = parser.parse_args()
    prepare(args.output.resolve())


if __name__ == "__main__":
    main()
