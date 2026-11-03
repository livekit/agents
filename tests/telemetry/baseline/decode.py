"""Decode captured observability requests into OTLP JSON plus the recording upload parts.

Usage: decode.py <capture dir> <out dir>
"""

import email.parser
import email.policy
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

KINDS = {
    "traces": (ExportTraceServiceRequest, "resource_spans"),
    "logs": (ExportLogsServiceRequest, "resource_logs"),
    "metrics": (ExportMetricsServiceRequest, "resource_metrics"),
}
merged: dict[str, list] = defaultdict(list)

for meta_path in sorted(SRC.glob("*.json")):
    meta = json.loads(meta_path.read_text())
    body = meta_path.with_suffix(".bin").read_bytes()
    if meta["headers"].get("Content-Encoding", meta["headers"].get("content-encoding")) == "gzip":
        body = gzip.decompress(body)
    path = meta["path"]
    kind = next((k for k in KINDS if f"/{k}/" in path), None)
    if kind:
        msg_type, field = KINDS[kind]
        msg = msg_type()
        msg.ParseFromString(body)
        merged[kind].extend(MessageToDict(msg, preserving_proto_field_name=True).get(field, []))
    elif "/recordings/" in path:
        ctype = meta["headers"].get("Content-Type") or meta["headers"].get("content-type")
        raw = f"Content-Type: {ctype}\r\n\r\n".encode() + body
        mp = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(raw)
        rec = OUT / f"recording_{meta['seq']:04d}"
        rec.mkdir(exist_ok=True)
        for part in mp.iter_parts():
            name = part.get_param("name", header="content-disposition")
            data = part.get_payload(decode=True)
            (rec / f"{name}.bin").write_bytes(data)
            if name == "header":
                from livekit.protocol import metrics as lkm

                header = lkm.MetricsRecordingHeader()
                header.ParseFromString(data)
                (rec / "header.json").write_text(
                    json.dumps(MessageToDict(header, preserving_proto_field_name=True), indent=2)
                )
        print("recording", meta["seq"], sorted(p.name for p in rec.iterdir()))

for kind, items in merged.items():
    (OUT / f"{kind}.json").write_text(json.dumps({KINDS[kind][1]: items}, indent=2))
    print(kind, len(items), "resource batches")
