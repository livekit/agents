#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "grpcio-tools>=1.72",
#     "protobuf>=6.30",
# ]
# ///
"""Regenerate livekit-protocol Python stubs from local .proto sources.

Uses grpcio-tools' bundled protoc (compatible with protobuf 6.x runtime)
instead of the system protoc to avoid gencode/runtime version mismatches.

Usage:
    uv run scripts/regenerate_protos.py [--protocol-dir PATH] [--sdk-dir PATH]

Defaults:
    --protocol-dir  ../protocol
    --sdk-dir       ../python-sdks/livekit-protocol
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path


def find_grpc_proto_include() -> str:
    """Return the include path for well-known protos bundled with grpcio-tools."""
    import grpc_tools

    return os.path.join(os.path.dirname(grpc_tools.__file__), "_proto")


def run_protoc(proto_dir: Path, out_dir: Path, proto_files: list[str]) -> None:
    """Compile .proto files using grpc_tools.protoc."""
    from grpc_tools import protoc as grpc_protoc

    include_path = find_grpc_proto_include()
    args = [
        "grpc_tools.protoc",
        f"-I={proto_dir}",
        f"-I={include_path}",
        f"--python_out={out_dir}",
        f"--pyi_out={out_dir}",
        *proto_files,
    ]
    ret = grpc_protoc.main(args)
    if ret != 0:
        print(f"protoc failed with exit code {ret}", file=sys.stderr)
        sys.exit(1)


RENAME_MAP: dict[str, str] = {
    "livekit_egress_pb2": "egress",
    "livekit_room_pb2": "room",
    "livekit_webhook_pb2": "webhook",
    "livekit_ingress_pb2": "ingress",
    "livekit_models_pb2": "models",
    "livekit_agent_pb2": "agent",
    "livekit_agent_dispatch_pb2": "agent_dispatch",
    "livekit_analytics_pb2": "analytics",
    "livekit_sip_pb2": "sip",
    "livekit_metrics_pb2": "metrics",
    "livekit_rtc_pb2": "rtc",
    "livekit_connector_whatsapp_pb2": "connector_whatsapp",
    "livekit_connector_twilio_pb2": "connector_twilio",
    "livekit_connector_pb2": "connector",
}

NESTED_MOVES: list[tuple[str, str, str]] = [
    # (src_subdir/old_name, dest_subdir, new_name)
    ("agent/livekit_agent_session_pb2", "agent_pb", "agent_session"),
    ("logger/options_pb2", "logger_pb", "options"),
]


def post_process(out_dir: Path) -> None:
    """Rename files and fix imports to match livekit-protocol conventions."""
    # 1. Rename top-level files
    for old_stem, new_name in RENAME_MAP.items():
        for ext in (".py", ".pyi"):
            src = out_dir / f"{old_stem}{ext}"
            dst = out_dir / f"{new_name}{ext}"
            if src.exists():
                shutil.move(str(src), str(dst))

    # 2. Move nested proto outputs
    for src_rel, dest_subdir, new_name in NESTED_MOVES:
        dest_dir = out_dir / dest_subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "__init__.py").touch()
        for ext in (".py", ".pyi"):
            src = out_dir / f"{src_rel}{ext}"
            dst = dest_dir / f"{new_name}{ext}"
            if src.exists():
                shutil.move(str(src), str(dst))

    # Clean up empty dirs left behind
    for subdir in ("agent", "logger"):
        d = out_dir / subdir
        if d.exists() and not any(d.iterdir()):
            d.rmdir()

    # 3. Fix imports in all .py and .pyi files
    all_files = list(out_dir.glob("*.py")) + list(out_dir.glob("*.pyi"))
    for subdir in ("agent_pb", "logger_pb"):
        d = out_dir / subdir
        if d.exists():
            all_files += list(d.glob("*.py")) + list(d.glob("*.pyi"))

    import_modules = "|".join(
        list(RENAME_MAP.keys()) + ["livekit_agent_session_pb2", "options_pb2"]
    )
    pattern_abs = re.compile(rf"^(import ({import_modules}))", re.MULTILINE)
    pattern_strip = re.compile(r"livekit_(\w+)_pb2")
    pattern_logger = re.compile(r"from logger import options_pb2 as ([^ \n]+)")
    pattern_classvar = re.compile(r"^(\w+_FIELD_NUMBER): _ClassVar\[int\]", re.MULTILINE)

    for f in all_files:
        text = f.read_text()
        text = pattern_abs.sub(r"from . \1", text)
        text = pattern_strip.sub(lambda m: m.group(1), text)
        text = pattern_logger.sub(r"from .logger_pb import options as \1", text)
        if f.name == "options.pyi" or str(f).endswith("logger_pb/options.pyi"):
            text = pattern_classvar.sub(r"\1: int", text)
        f.write_text(text)

    (out_dir / "__init__.py").touch()


def ensure_protocol_symlink(sdk_dir: Path, protocol_dir: Path) -> None:
    """Ensure sdk_dir/protocol points to the local protocol repo."""
    link = sdk_dir / "protocol"
    target = protocol_dir.resolve()

    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        print(
            f"Warning: {link} exists and is not a symlink. "
            f"Remove it manually if you want to use the local protocol.",
            file=sys.stderr,
        )
        return

    link.symlink_to(target)
    print(f"Symlinked {link} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("../protocol"),
        help="Path to the protocol repo (default: ../protocol)",
    )
    parser.add_argument(
        "--sdk-dir",
        type=Path,
        default=Path("../python-sdks/livekit-protocol"),
        help="Path to python-sdks/livekit-protocol (default: ../python-sdks/livekit-protocol)",
    )
    args = parser.parse_args()

    protocol_dir: Path = args.protocol_dir.resolve()
    sdk_dir: Path = args.sdk_dir.resolve()
    proto_src = protocol_dir / "protobufs"
    out_dir = sdk_dir / "livekit" / "protocol"

    if not proto_src.exists():
        print(f"Proto sources not found at {proto_src}", file=sys.stderr)
        sys.exit(1)

    ensure_protocol_symlink(sdk_dir, protocol_dir)

    proto_files = [
        f"{proto_src}/livekit_egress.proto",
        f"{proto_src}/livekit_room.proto",
        f"{proto_src}/livekit_webhook.proto",
        f"{proto_src}/livekit_ingress.proto",
        f"{proto_src}/livekit_models.proto",
        f"{proto_src}/livekit_agent.proto",
        f"{proto_src}/livekit_agent_dispatch.proto",
        f"{proto_src}/livekit_metrics.proto",
        f"{proto_src}/livekit_sip.proto",
        f"{proto_src}/livekit_analytics.proto",
        f"{proto_src}/livekit_rtc.proto",
        f"{proto_src}/agent/livekit_agent_session.proto",
        f"{proto_src}/logger/options.proto",
        f"{proto_src}/livekit_connector_whatsapp.proto",
        f"{proto_src}/livekit_connector_twilio.proto",
        f"{proto_src}/livekit_connector.proto",
    ]

    existing = [f for f in proto_files if Path(f).exists()]
    missing = [f for f in proto_files if not Path(f).exists()]
    if missing:
        print(f"Warning: missing proto files (skipping): {missing}", file=sys.stderr)
    if not existing:
        print("No proto files found to compile.", file=sys.stderr)
        sys.exit(1)

    print(f"Compiling {len(existing)} proto files...")
    run_protoc(proto_src, out_dir, existing)

    print("Post-processing generated stubs...")
    post_process(out_dir)

    print("Done. Now run in the agents workspace:")
    print("  uv sync")
    print("  uv run pytest tests/ -x")


if __name__ == "__main__":
    main()
