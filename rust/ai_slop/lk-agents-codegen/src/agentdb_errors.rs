//! `livekit.AgentDB.ErrorCode` as Rust: every agent-db error code with its `applied` and `retry`
//! enum value options, read from the protocol the workspace's `livekit-protocol` is generated
//! from. prost drops custom options, so this is their one Rust reading.

use std::fmt::Write as _;
use std::path::Path;

/// The proto, in the protocol submodule of the `rust-sdks` checkout beside this repository.
const PROTO: &str = "../rust-sdks/livekit-protocol/protocol/protobufs/livekit_agentdb.proto";
/// Where the Rust enum is written.
pub const PATH: &str = "rust/crates/lk-agents-abi/src/storage/agentdb_error.rs";

struct Code {
    name: String,
    applied: &'static str,
    retry: &'static str,
}

pub fn emit(root: &Path) -> Result<String, String> {
    let path = root.join(PROTO);
    let proto = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let codes = parse(&proto).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(render(&codes))
}

fn parse(proto: &str) -> Result<Vec<Code>, String> {
    let start = proto.find("enum ErrorCode {").ok_or("no `enum ErrorCode`")?;
    let body = &proto[start..];
    let body = &body[body.find('{').expect("found above") + 1..body.find('}').ok_or("unclosed")?];
    let mut codes = Vec::new();
    for line in body.lines() {
        let line = line.split("//").next().unwrap_or_default().trim();
        if line.is_empty() || line.starts_with("reserved") {
            continue;
        }
        let (name, rest) = line.split_once('=').ok_or_else(|| format!("unexpected `{line}`"))?;
        let option = |key: &str, values: &[(&str, &'static str)], zero: &'static str| {
            let Some(at) = rest.find(&format!("({key})")) else { return Ok(zero) };
            let value = rest[at..]
                .split_once('=')
                .map(|(_, v)| v.trim_start())
                .ok_or_else(|| format!("`{line}`: ({key}) has no value"))?;
            values
                .iter()
                .find(|(proto, _)| value.starts_with(proto))
                .map(|(_, rust)| *rust)
                .ok_or_else(|| format!("`{line}`: unknown ({key}) value"))
        };
        codes.push(Code {
            name: name.trim().to_owned(),
            // The zero values: a code without the option may have applied and is never retried.
            applied: option(
                "applied",
                &[("APPLIED_MAYBE", "Maybe"), ("APPLIED_NO", "No")],
                "Maybe",
            )?,
            retry: option(
                "retry",
                &[
                    ("RETRY_NEVER", "Never"),
                    ("RETRY_HERE", "Here"),
                    ("RETRY_ELSEWHERE", "Elsewhere"),
                ],
                "Never",
            )?,
        });
    }
    if codes.first().is_none_or(|c| c.name != "UNKNOWN") {
        return Err("ErrorCode's zero value is not UNKNOWN".into());
    }
    Ok(codes)
}

fn camel(name: &str) -> String {
    name.split('_').map(|w| w[..1].to_owned() + &w[1..].to_lowercase()).collect()
}

fn render(codes: &[Code]) -> String {
    let mut s = String::from(
        "//! GENERATED FILE -- DO NOT EDIT.\n\
         //! `livekit.AgentDB.ErrorCode` (`livekit_agentdb.proto`), written by `lk-agents-codegen`: \
         every agent-db error code, on the data plane and the management services, with what a \
         client may conclude from it.\n\n\
         /// Whether a request answered with an error may have taken effect.\n\
         #[derive(Debug, Clone, Copy, PartialEq, Eq)]\n\
         pub enum Applied {\n\
         /// It may have committed, whole: resend only under its idempotency key.\n\
         Maybe,\n\
         /// Nothing took effect.\n\
         No,\n\
         }\n\n\
         /// Whether the same request can succeed if sent again.\n\
         #[derive(Debug, Clone, Copy, PartialEq, Eq)]\n\
         pub enum Retry {\n\
         /// The same request fails the same way.\n\
         Never,\n\
         /// On the same connection, after a backoff.\n\
         Here,\n\
         /// On a new connection, which may reach another node.\n\
         Elsewhere,\n\
         }\n\n\
         /// An agent-db error code. It travels as its name in lowercase; one this build does not \
         know is `Unknown`, which may have applied and is never retried.\n\
         #[derive(Debug, Clone, Copy, PartialEq, Eq)]\n\
         pub enum ErrorCode {\n",
    );
    for c in codes {
        let _ = writeln!(s, "{},", camel(&c.name));
    }
    s.push_str("}\n\nimpl ErrorCode {\n/// The code as it travels.\npub const fn as_str(self) -> &'static str {\nmatch self {\n");
    for c in codes {
        let _ = writeln!(s, "Self::{} => \"{}\",", camel(&c.name), c.name.to_lowercase());
    }
    s.push_str("}\n}\n\n/// A code as it travels; anything else is `Unknown`.\npub fn from_wire(code: &str) -> Self {\nmatch code {\n");
    for c in &codes[1..] {
        let _ = writeln!(s, "\"{}\" => Self::{},", c.name.to_lowercase(), camel(&c.name));
    }
    s.push_str("_ => Self::Unknown,\n}\n}\n\n/// Whether a request answered with this code may have taken effect.\npub const fn applied(self) -> Applied {\nmatch self {\n");
    for c in codes {
        let _ = writeln!(s, "Self::{} => Applied::{},", camel(&c.name), c.applied);
    }
    s.push_str("}\n}\n\n/// Whether the same request may succeed if sent again, and where.\npub const fn retry(self) -> Retry {\nmatch self {\n");
    for c in codes {
        let _ = writeln!(s, "Self::{} => Retry::{},", camel(&c.name), c.retry);
    }
    s.push_str("}\n}\n}\n");
    s
}
