# JSON repair reference

The Rust recovery parser is translated from [json_repair](https://github.com/mangiucugna/json_repair),
MIT licensed, copyright Stefano Baccianella. See LICENSE in this directory.
The reference is v0.60.1, commit 63992e94b9cfec79a3d6671c9eceebfe596f55e3.

cases.json contains 236 inputs from the upstream string, object, array, number,
comment and repair tests. Expected values use `repair_json(input, return_objects=True)`,
including cases upstream also tests with `skip_json_loads=True`.

Run the corpus from `rust/`:

```sh
cargo test -p lk-agents-core upstream_corpus
```

Valid JSON uses serde_json directly; malformed input uses the recovery parser.
Depth overflow produces an error. Non-finite numbers are outside the shared JSON contract.
