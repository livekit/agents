from typing import Dict, Any
from slack_sdk.models.basic_objects import JsonObject


class Metadata(JsonObject):
    """Message metadata

    https://api.slack.com/metadata
    """

    attributes = {
        "event_type",
        "event_payload",
    }

    def __init__(
        self,
        event_type: str,
        event_payload: Dict[str, Any],
        **kwargs,
    ):
        self.event_type = event_type
        self.event_payload = event_payload
        self.additional_attributes = kwargs

    def __str__(self):
        return str(self.get_non_null_attributes())

    def __repr__(self):
        return self.__str__()
