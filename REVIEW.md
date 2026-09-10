# Review guidelines

## PII-sensitive logs and telemetry

- Review each added or modified log call, exception path, and telemetry payload for values that
  can contain personal data or customer content. This includes participant identities, room
  names, speech and transcripts, prompts and instructions, chat context, tool arguments and
  output, DTMF digits, and provider request, response, or event payloads.
- Sensitive values must not appear in log message bodies, span names, or event names. Use a
  static message and move the value to a structured attribute instead.
- Treat exception messages, `str(e)`, `repr(e)`, tracebacks, and `__cause__` or `__context__`
  chains as possible sensitive data. Review `logger.exception`, `exc_info`, exception
  interpolation, and automatic `Task exception was never retrieved` logs. If an exception can
  contain headers, URL credentials, or payloads, require a safe wrapper without the original
  cause (`raise ... from None`), log only safe metadata such as its type, and retrieve
  background-task failures.
- Each structured attribute that can contain a sensitive value must have a key with a whole,
  dot-delimited `pii` segment. Use `lk.pii.<name>` or the matching constant from
  `telemetry/trace_types.py`. Keys such as `lk.chatpii` or `lk.pii_value` are not valid markers.
- Apply the marker at every emission path, including logger `extra` fields, span attributes,
  event attributes, tag metadata, nested session data, and provider debug dumps. Check the value
  source instead of relying only on the field name.
- Flag an untagged sensitive value as a security issue, even if a similar path is tagged or a
  static test does not recognize the field name.

```python
# Wrong: the message body cannot be redacted.
logger.debug(f"received provider event: {event}")

# Wrong: the structured key has no dot-delimited pii segment.
logger.debug("received provider event", extra={"event": event})

# Correct: the collector can remove the sensitive attribute.
logger.debug("received provider event", extra={"lk.pii.event": event})
```
