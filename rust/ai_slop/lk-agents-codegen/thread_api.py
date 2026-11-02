#!/usr/bin/env python3
"""Generate the local thread HTTP contract. Builds use the checked-in JSON snapshot.

Refresh explicitly with --source ~/livekit/public-api-server --ref origin/main --write.
Only refresh needs PyYAML; ordinary --check and --write use the standard library.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "rust/crates/lk-agents-storage/api/threads.openapi.json"
OUTPUT = ROOT / "rust/crates/lk-agents-storage/src/http/generated.rs"
METHODS = {"get", "post", "patch", "delete", "put"}
ALIASES = {
    "google.protobuf.Duration": "pbjson_types::Duration",
    "google.protobuf.Timestamp": "pbjson_types::Timestamp",
    "livekit.agent.AgentThread.Thread": "livekit_protocol::agent::agent_thread::Thread",
    "livekit.agent.AgentThread.Item": "livekit_protocol::agent::agent_thread::Item",
}


def name(value: str) -> str:
    return "".join(word[:1].upper() + word[1:] for word in re.split(r"[._]", value))


def field(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).replace(".", "_").lower()


def component(value: str) -> str:
    key = value.removeprefix("#/components/schemas/")
    if ".Threads." in key:
        operation = key.split(".Threads.")[1].split(".")[0]
        operation = {
            "Items": "GetThreadItems",
            "Tasks": "GetThreadTasks",
            "List": "ListThreads",
        }.get(operation, operation + "Thread")
        return operation + "Response"
    if ".AgentTask." in key:
        return "TaskStatus" if key.endswith(".Status") else "TaskRecord"
    if ".AgentThread." in key:
        return "ThreadRecord" if key.endswith(".Thread") else "Item"
    return key.split(".")[-1]


def refresh(source: str, ref: str) -> dict:
    import yaml

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", source, *args], text=True)

    commit = git("rev-parse", ref).strip()
    spec = yaml.safe_load(git("show", f"{commit}:pkg/api/v1/openapi.yaml"))
    paths = {path: ops for path, ops in spec["paths"].items() if "/threads" in path}
    if not paths:
        raise ValueError("source has no thread operations")
    refs: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if (ref := value.get("$ref")) and ref not in refs:
                if not ref.startswith("#/components/schemas/"):
                    raise ValueError(f"unsupported reference: {ref}")
                refs.add(ref)
                visit(spec["components"]["schemas"][ref.split("/")[-1]])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(paths)
    return {
        "openapi": spec["openapi"],
        "info": spec["info"],
        "x-source": f"livekit/public-api-server@{commit}:pkg/api/v1/openapi.yaml",
        "paths": paths,
        "components": {
            "schemas": {
                ref.split("/")[-1]: spec["components"]["schemas"][ref.split("/")[-1]]
                for ref in sorted(refs)
            }
        },
    }


def rust_type(schema: dict) -> str:
    if "$ref" in schema:
        return component(schema["$ref"])
    if "oneOf" in schema:
        choices = [choice for choice in schema["oneOf"] if choice.get("type") != "null"]
        if len(choices) == 1:
            return rust_type(choices[0])
        raise ValueError("a new union requires a protocol type mapping")
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next(t for t in kind if t != "null")
    if kind == "array":
        return f"Vec<{rust_type(schema['items'])}>"
    if kind == "object":
        if isinstance(schema.get("additionalProperties"), dict):
            return (
                f"std::collections::BTreeMap<String, {rust_type(schema['additionalProperties'])}>"
            )
        raise ValueError("nested objects require named schemas")
    return {"string": "String", "boolean": "bool", "integer": "i32", "number": "f64"}[kind]


def model(model_name: str, schema: dict) -> str:
    if "enum" in schema:
        variants = []
        spellings = []
        for index, value in enumerate(schema["enum"]):
            variant = name(value.lower())
            variants.append(
                f'    #[serde(rename = "{value}")]\n'
                + ("    #[default]\n" if index == 0 else "")
                + f"    {variant},"
            )
            spellings.append(f'            Self::{variant} => "{value}",')
        return (
            "#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]\n"
            f"pub enum {model_name} {{\n"
            + "\n".join(variants)
            + "\n}\n"
            + f"impl {model_name} {{\n    pub fn as_str(self) -> &'static str {{\n        match self {{\n"
            + "\n".join(spellings)
            + "\n        }\n    }\n}\n"
        )
    if schema.get("type") != "object":
        return f"pub type {model_name} = {rust_type(schema)};\n"
    fields = []
    for key, prop in schema.get("properties", {}).items():
        kind = rust_type(prop)
        if "$ref" in prop or "oneOf" in prop or isinstance(prop.get("type"), list):
            kind = f"Option<{kind}>"
        fields.append(f'    #[serde(rename = "{key}")]\n    pub {field(key)}: {kind},')
    return (
        "#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]\n"
        "#[serde(default, deny_unknown_fields)]\n"
        f"pub struct {model_name} {{\n" + "\n".join(fields) + "\n}\n"
    )


def generate(spec: dict) -> str:
    schemas = spec["components"]["schemas"]
    operations = []
    needed: set[str] = set()

    def need(schema: dict) -> None:
        if "$ref" in schema:
            key = schema["$ref"].split("/")[-1]
            if key in needed:
                return
            needed.add(key)
            if key not in ALIASES:
                need(schemas[key])
        for value in schema.values():
            if isinstance(value, dict):
                need(value)
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, dict):
                        need(child)

    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            if method not in METHODS:
                continue
            operation = op["operationId"].removeprefix("ThreadService_")
            response = op["responses"]["200"]["content"]["application/json"]["schema"]
            need(response)
            body = (
                op.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema")
            )
            if body:
                need(body)
            for param in op.get("parameters", []):
                need(param["schema"])
            operations.append((path, method.upper(), operation, op, response, body))

    out = [
        "// @generated by rust/ai_slop/lk-agents-codegen/thread_api.py. Do not edit.",
        f"// Source: {spec['x-source']}",
        "#![allow(dead_code)]\n",
    ]
    for key in sorted(needed):
        out.append(
            f"pub type {component(key)} = {ALIASES[key]};\n"
            if key in ALIASES
            else model(component(key), schemas[key])
        )
    trait = ["#[async_trait::async_trait]", "pub trait ThreadService: Send + Sync {"]
    dispatch = []
    for path, method, operation, op, response, body in operations:
        request = operation + "Request"
        fields = []
        values = []
        allowed = []
        for param in op.get("parameters", []):
            key = param["name"]
            kind = rust_type(param["schema"])
            ident = field(key)
            fields.append(f"    pub {ident}: {kind},")
            if param["in"] == "path":
                index = path.strip("/").split("/").index("{" + key + "}")
                values.append(f"{ident}: segments[{index}].to_owned()")
            else:
                allowed.append(f'"{key}"')
                numeric = str(kind == "i32").lower()
                values.append(f'{ident}: super::parameter(query, "{key}", {numeric})?')
        if body:
            out.append(model(operation + "Body", body))
            fields.append(f"    pub body: {operation}Body,")
            values.append("body: serde_json::from_slice(body).map_err(super::ApiError::invalid)?")
        out.append(f"#[derive(Debug)]\npub struct {request} {{\n" + "\n".join(fields) + "\n}\n")
        output = component(response["$ref"])
        function = field(operation)
        trait.append(
            f"    async fn {function}(&self, context: &super::Context, request: {request}) -> Result<{output}, super::ApiError>;"
        )
        pattern = ", ".join(
            "_" if part.startswith("{") else f'"{part}"' for part in path.strip("/").split("/")
        )
        dispatch.append(f'''        ("{method}", [{pattern}]) => {{
            super::check_query(query, &[{", ".join(allowed)}])?;
            let request = {request} {{ {", ".join(values)} }};
            serde_json::to_value(service.{function}(context, request).await?).map_err(super::ApiError::internal)
        }}''')
    out.append("\n".join(trait) + "\n}\n")
    out.append(
        """pub async fn dispatch(
    service: &impl ThreadService,
    context: &super::Context,
    method: &str,
    segments: &[&str],
    query: &std::collections::BTreeMap<String, String>,
    body: &[u8],
) -> Result<serde_json::Value, super::ApiError> {
    match (method, segments) {
"""
        + "\n".join(dispatch)
        + """
        _ => Err(super::ApiError::not_found("unknown thread API route")),
    }
}
"""
    )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--ref", default="origin/main")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    spec = refresh(args.source, args.ref) if args.source else json.loads(CONTRACT.read_text())
    snapshot = json.dumps(spec, indent=2, ensure_ascii=False) + "\n"
    text = generate(spec)
    formatted = subprocess.check_output(
        [
            "rustfmt",
            "--edition",
            "2024",
            "--config-path",
            str(ROOT / "rust/rustfmt.toml"),
            "--emit",
            "stdout",
        ],
        input=text,
        text=True,
    )
    for path, content in [(CONTRACT, snapshot), (OUTPUT, formatted)]:
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        elif not path.exists() or path.read_text() != content:
            raise SystemExit(f"generated drift: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
