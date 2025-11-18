# plugin.py
from .handler import InterruptHandler

def attach_interrupt_handler(session, **kwargs):
    """
    Simple helper to create and start the handler.
    Usage:
        from voice_interrupt import attach_interrupt_handler
        attach_interrupt_handler(session, ignored_words={"uh","umm"})
    """
    handler = InterruptHandler(session, **kwargs)
    handler.start()
    # Optionally return handler for further control (dynamic updates)
    return handler
