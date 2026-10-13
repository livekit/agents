# json_repair reference

Native Rust recovery is translated from https://github.com/mangiucugna/json_repair,
MIT licensed, copyright Stefano Baccianella. See LICENSE in this directory.

Reference: v0.60.1, commit 63992e94b9cfec79a3d6671c9eceebfe596f55e3,
the previously pinned Python dependency. The clone used for translation is at
/private/tmp/agents-json-repair-reference (not required to build or test).

cases.json contains 236 inputs exercised by upstream test_parse_string,
test_parse_object, test_parse_array, test_parse_number, test_parse_comment,
and test_json_repair. Expected values were captured by running the reference's
repair_json(input, return_objects=True), i.e. the production default decoder mode.
Cases that upstream additionally exercises with skip_json_loads=True are normalized
to that default mode. Python is needed only to regenerate this reference corpus.

The port provides schema-free, non-streaming, in-memory recovery over serde_json
values. It does not expose upstream's CLI, file wrappers, schema repair, diagnostic
log format, or optional strict/stream-stable modes. Those are not used by tool
argument parsing. Depth overflow is an explicit error, not a partial repair.
Valid JSON uses serde_json directly; malformed input uses the translated parser.
Non-finite numbers are outside the shared JSON value contract.
