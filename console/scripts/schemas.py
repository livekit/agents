"""Export SDK schemas once for browser validation and TypeScript generation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models
from a2a.types import AgentCard, Message, StreamResponse
from google.api import field_behavior_pb2
from google.protobuf.descriptor import FieldDescriptor as F


def protobuf_schema(root):
    definitions = {}

    def message(d):
        if d.full_name == "google.protobuf.Struct":
            return {"type": "object", "additionalProperties": True}
        if d.full_name in ("google.protobuf.Value", "google.protobuf.ListValue"):
            return {}
        if d.full_name == "google.protobuf.Timestamp":
            return {"type": "string"}
        name = d.name
        if name in definitions:
            return {"$ref": f"#/$defs/{name}"}
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        definitions[name] = schema
        required = []
        for f in d.fields:
            if f.message_type:
                if f.message_type.GetOptions().map_entry:
                    value = f.message_type.fields_by_name["value"]
                    item = {"type": "object", "additionalProperties": scalar(value)}
                else:
                    item = message(f.message_type)
            else:
                item = scalar(f)
            if f.is_repeated and not (f.message_type and f.message_type.GetOptions().map_entry):
                item = {"type": "array", "items": item}
            schema["properties"][f.json_name] = item
            if (
                field_behavior_pb2.REQUIRED
                in f.GetOptions().Extensions[field_behavior_pb2.field_behavior]
            ):
                required.append(f.json_name)
        if required:
            schema["required"] = required
        unions = [o for o in d.oneofs if not o.name.startswith("_")]
        if unions:
            schema["allOf"] = [
                {"oneOf": [{"required": [f.json_name]} for f in o.fields]} for o in unions
            ]
        return {"$ref": f"#/$defs/{name}"}

    def scalar(f):
        if f.message_type:
            return message(f.message_type)
        if f.enum_type:
            return {"type": "string", "enum": [v.name for v in f.enum_type.values]}
        if f.type in (F.TYPE_STRING, F.TYPE_BYTES):
            return {"type": "string"}
        if f.type == F.TYPE_BOOL:
            return {"type": "boolean"}
        return {"type": "number"}

    ref = message(root.DESCRIPTOR)
    return {**ref, "$defs": definitions}


schemas = {
    name: getattr(models, name).model_json_schema()
    for name in (
        "ConsoleConfig",
        "Grant",
        "SessionRow",
        "SessionDump",
        "History",
        "Sessions",
        "CreatedSession",
        "QueryResult",
        "Examples",
        "CallRequest",
        "CallMetadata",
        "CallToken",
        "TitleRequest",
    )
}
schemas["StreamResponse"] = protobuf_schema(StreamResponse)
schemas["AgentCard"] = protobuf_schema(AgentCard)
schemas["Message"] = protobuf_schema(Message)
# The storage schema deliberately leaves protocol payloads opaque; the browser uses
# their upstream protobuf definitions at this one decode boundary.
protocol = protobuf_schema(StreamResponse)
for schema in schemas.values():
    definitions = schema.get("$defs", {})
    if "TaskData" in definitions:
        definitions.update(protocol["$defs"])
        data = definitions["TaskData"]["properties"]
        data["history"] = {"type": "array", "items": {"$ref": "#/$defs/Message"}}
        data["artifacts"] = {
            "anyOf": [{"type": "array", "items": {"$ref": "#/$defs/Artifact"}}, {"type": "null"}]
        }
        data["status_message"] = {"anyOf": [{"$ref": "#/$defs/Message"}, {"type": "null"}]}

out = Path(__file__).resolve().parents[1] / "src/lib/generated"
out.mkdir(exist_ok=True)


def normalize(value):
    if isinstance(value, dict):
        if not value:
            value["tsType"] = "unknown"
        value.pop("title", None)
        value.pop("discriminator", None)
        properties = value.get("properties", {})
        if "const" in properties.get("type", {}):
            value["required"] = list(dict.fromkeys([*value.get("required", []), "type"]))
        for child in value.values():
            normalize(child)
    elif isinstance(value, list):
        for child in value:
            normalize(child)


for name, schema in schemas.items():
    normalize(schema)
    content = json.dumps(schema, indent=2) + "\n"
    target = out / f"{name}.json"
    if "--check" in sys.argv:
        if not target.exists() or target.read_text() != content:
            raise SystemExit(f"Schema drift: {target}. Run pnpm schema:generate in console.")
    else:
        target.write_text(content)
