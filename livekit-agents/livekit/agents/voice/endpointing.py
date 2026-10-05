from ..core.endpointing import BaseEndpointing, DynamicEndpointing
from .turn import EndpointingOptions


def create_endpointing(options: EndpointingOptions) -> BaseEndpointing:
    match options.get("mode", "fixed"):
        case "dynamic":
            return DynamicEndpointing(
                min_delay=options["min_delay"],
                max_delay=options["max_delay"],
                alpha=options["alpha"],
            )
        case _:
            return BaseEndpointing(
                min_delay=options["min_delay"],
                max_delay=options["max_delay"],
            )
