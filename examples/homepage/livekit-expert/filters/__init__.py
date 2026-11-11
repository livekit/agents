"""The agent's pipeline filters — transformations wrapped around node streams.

Where a behavior subscribes to session events (see ``behaviors``), a filter
transforms data flowing through a pipeline node: an async-iterable in, an
async-iterable out. Each filter is one module in this directory; the agent
composes them where it overrides the node (e.g. ``Assistant.tts_node``).
"""
