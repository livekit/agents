from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypeGuard

_T = TypeVar("_T")


def to_strict_json_schema(model: type[BaseModel] | TypeAdapter[Any]) -> dict[str, Any]:
    if isinstance(model, TypeAdapter):
        schema = model.json_schema()
    else:
        schema = model.model_json_schema()

    return _ensure_strict_json_schema(schema, path=(), root=schema)


# from https://platform.openai.com/docs/guides/function-calling?api-mode=responses&strict-mode=disabled#strict-mode
# Strict mode
# Setting strict to true will ensure function calls reliably adhere to the function schema,
# instead of being best effort. We recommend always enabling strict mode.
#
# Under the hood, strict mode works by leveraging our structured outputs feature and therefore
# introduces a couple requirements:
#
# additionalProperties must be set to false for each object in the parameters.
# All fields in properties must be marked as required.
# You can denote optional fields by adding null as a type option (see example below).


def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    """Mutates the given JSON schema to ensure it conforms to the `strict` standard
    that the API expects.
    """
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    defs = json_schema.get("$defs")
    if is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name), root=root)

    definitions = json_schema.get("definitions")
    if is_dict(definitions):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(
                definition_schema,
                path=(*path, "definitions", definition_name),
                root=root,
            )

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if is_dict(properties):
        json_schema["required"] = list(properties.keys())
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(prop_schema, path=(*path, "properties", key), root=root)
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(items, path=(*path, "items"), root=root)

    # unions
    any_of = json_schema.get("anyOf")
    if is_list(any_of):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i)), root=root)
            for i, variant in enumerate(any_of)
        ]

    # unions (oneOf)
    one_of = json_schema.get("oneOf")
    if is_list(one_of):
        json_schema["oneOf"] = [
            _ensure_strict_json_schema(variant, path=(*path, "oneOf", str(i)), root=root)
            for i, variant in enumerate(one_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if is_list(all_of):
        if len(all_of) == 1:
            json_schema.update(
                _ensure_strict_json_schema(all_of[0], path=(*path, "allOf", "0"), root=root)
            )
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_json_schema(entry, path=(*path, "allOf", str(i)), root=root)
                for i, entry in enumerate(all_of)
            ]

    # strict mode doesn't support default
    if "default" in json_schema:
        json_schema.pop("default", None)

        # Treat any parameter with a default value as optional. If the parameter’s type doesn't
        # support None, the default will be used instead.
        t = json_schema.get("type")
        if isinstance(t, str):
            json_schema["type"] = [t, "null"]

        elif isinstance(t, list):
            types = t.copy()
            if "null" not in types:
                types.append("null")

            json_schema["type"] = types

    json_schema.pop("title", None)
    json_schema.pop("discriminator", None)

    # we can't use `$ref`s if there are also other properties defined, e.g.
    # `{"$ref": "...", "description": "my description"}`
    #
    # so we unravel the ref
    # `{"type": "string", "description": "my description"}`
    ref = json_schema.get("$ref")
    if ref and has_more_than_n_keys(json_schema, 1):
        assert isinstance(ref, str), f"Received non-string $ref - {ref}"

        resolved = resolve_ref(root=root, ref=ref)
        if not is_dict(resolved):
            raise ValueError(
                f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}"
            )

        # properties from the json schema take priority over the ones on the `$ref`
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
        # Since the schema expanded from `$ref` might not have `additionalProperties: false` applied,  # noqa: E501
        # we call `_ensure_strict_json_schema` again to fix the inlined schema and ensure it's valid.  # noqa: E501
        return _ensure_strict_json_schema(json_schema, path=path, root=root)

    # simplify nullable unions (“anyOf” or “oneOf”)
    for union_key in ("anyOf", "oneOf"):
        variants = json_schema.get(union_key)
        if is_list(variants) and len(variants) == 2 and {"type": "null"} in variants:
            # pick out the non-null branch
            non_null = next(
                (item for item in variants if item != {"type": "null"}),
                None,
            )
            assert is_dict(non_null)

            t = non_null["type"]
            if isinstance(t, str):
                non_null["type"] = [t, "null"]

            merged = {k: v for k, v in json_schema.items() if k not in ("anyOf", "oneOf")}
            merged.update(non_null)
            json_schema = merged
            break

    return json_schema


def resolve_ref(*, root: dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert is_dict(value), (
            f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        )
        resolved = value

    return resolved


def is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    # just pretend that we know there are only `str` keys
    # as that check is not worth the performance cost
    return isinstance(obj, dict)


def is_list(obj: object) -> TypeGuard[list[object]]:
    return isinstance(obj, list)


def has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    i = 0
    for _ in obj.keys():
        i += 1
        if i > n:
            return True
    return False
