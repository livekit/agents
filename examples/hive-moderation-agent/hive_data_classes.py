from dataclasses import dataclass, is_dataclass
from typing import List, get_type_hints


def from_dict(cls, data):
    if is_dataclass(cls) and isinstance(data, dict):
        # Get type hints for all fields in the dataclass
        field_types = get_type_hints(cls)
        # Special handling for reserved words like 'class'
        reserved_word_mappings = {"class": "class_"}  # Map 'class' to 'class_'
        processed_data = {}
        for key, value in data.items():
            # Check if the key is a reserved word and map it accordingly
            field_name = reserved_word_mappings.get(key, key)
            # Only include keys that have corresponding fields in the dataclass
            if field_name in field_types:
                field_type = field_types[field_name]
                # Determine if the field_type is itself a dataclass
                if is_dataclass(field_type):
                    processed_value = from_dict(field_type, value)
                elif hasattr(field_type, "__origin__") and issubclass(field_type.__origin__, List):
                    # Handle List fields, assuming all elements are of the same type
                    item_type = field_type.__args__[0]
                    processed_value = [from_dict(item_type, item) for item in value]
                else:
                    processed_value = value
                processed_data[field_name] = processed_value
        return cls(**processed_data)
    elif isinstance(data, list):
        # This assumes that the function was called with a list type as `cls`,
        # which might not work as expected without context on the list's element type.
        # A better approach might be needed for handling lists of dataclasses.
        return [
            from_dict(cls.__args__[0], item) if hasattr(cls, "__args__") else item for item in data
        ]
    else:
        return data


@dataclass
class Status:
    code: str
    message: str


@dataclass
class ModInput:
    id: str
    charge: float
    config_tag: SyntaxWarning
    config_version: float
    created_on: str
    model: str
    model_type: str
    model_version: float
    project_id: int
    user_id: int


@dataclass
class ModClass:
    class_: str
    score: float


@dataclass
class ModOutput:
    time: int
    classes: List[ModClass]


@dataclass
class Response:
    input: ModInput
    output: List[ModOutput]


@dataclass
class ModResponse:
    status: Status
    response: Response


@dataclass
class HiveResponse:
    id: str
    code: int
    project_id: int
    user_id: int
    created_on: str
    status: List[ModResponse]
    from_cache: bool
