from typing import Any, Callable

from realtime.connection import Socket
from realtime.transformers import convert_change_data


class SupabaseRealtimeClient:
    def __init__(self, socket: Socket, schema: str, table_name: str):
        topic = (
            f"realtime:{schema}"
            if table_name == "*"
            else f"realtime:{schema}:{table_name}"
        )
        self.subscription = socket.set_channel(topic)

    @staticmethod
    def get_payload_records(payload: Any):
        records: dict = {"new": {}, "old": {}}
        if payload.type in ["INSERT", "UPDATE"]:
            records["new"] = payload.record
            convert_change_data(payload.columns, payload.record)
        if payload.type in ["UPDATE", "DELETE"]:
            records["old"] = payload.record
            convert_change_data(payload.columns, payload.old_record)
        return records

    def on(self, event, callback: Callable[..., Any]):
        def cb(payload):
            enriched_payload = {
                "schema": payload.schema,
                "table": payload.table,
                "commit_timestamp": payload.commit_timestamp,
                "event_type": payload.type,
                "new": {},
                "old": {},
            }
            enriched_payload = {**enriched_payload, **self.get_payload_records(payload)}
            callback(enriched_payload)

        self.subscription.join().on(event, cb)
        return self

    def subscribe(self, callback: Callable[..., Any]):
        # TODO: Handle state change callbacks for error and close
        self.subscription.join().on("ok", callback("SUBSCRIBED"))
        self.subscription.join().on(
            "error", lambda x: callback("SUBSCRIPTION_ERROR", x)
        )
        self.subscription.join().on(
            "timeout", lambda: callback("RETRYING_AFTER_TIMEOUT")
        )
        return self.subscription
