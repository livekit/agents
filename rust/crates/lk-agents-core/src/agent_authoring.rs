//! One authoring authority for agents used by text runs and voice sessions. The host registry
//! keeps actual callables under these IDs so no native configuration retains Python self.

use lk_agents_abi::agent::Retries;
use lk_agents_abi::agent::authoring::{
    AgentConfiguration, AuthoringError, CallbackKind, DEFAULT_AGENT_NAME,
};
use serde_json::Value;
use std::collections::BTreeMap;

pub struct AgentConfig {
    name: Option<String>,
    instructions: String,
    retries: Retries,
    frozen: bool,
    next: u64,
    callbacks: BTreeMap<CallbackKind, Vec<u64>>,
}
impl lk_agents_abi::agent::authoring::AgentConfig for AgentConfig {
    fn new(
        instructions: String,
        name: Option<String>,
        retries: Option<Value>,
    ) -> Result<Self, AuthoringError> {
        Ok(Self {
            name,
            instructions,
            retries: resolve_retries(retries, Retries::default())?,
            frozen: false,
            next: 0,
            callbacks: BTreeMap::new(),
        })
    }
    fn resolve_name(&mut self, discovered: Option<String>) -> String {
        if self.name.is_none() {
            self.name = discovered;
        }
        self.name()
    }
    fn instructions(&self) -> String {
        self.instructions.clone()
    }
    fn freeze(&mut self) {
        self.frozen = true;
    }
    fn check_mutable(&self, what: String) -> Result<(), AuthoringError> {
        if self.frozen { Err(AuthoringError::Frozen { what, agent: self.name() }) } else { Ok(()) }
    }
    fn register(&mut self, kind: CallbackKind) -> Result<(u64, Option<u64>), AuthoringError> {
        let (what, single) = match kind {
            CallbackKind::Tool => ("a tool", false),
            CallbackKind::Instructions => ("an instructions function", false),
            CallbackKind::Validator => ("an output validator", false),
            CallbackKind::OnEnter => ("an on_enter function", true),
            CallbackKind::OnExit => ("an on_exit function", true),
        };
        self.check_mutable(what.into())?;
        self.next += 1;
        let ids = self.callbacks.entry(kind).or_default();
        let retired = if single { ids.pop() } else { None };
        ids.push(self.next);
        Ok((self.next, retired))
    }
    fn callbacks(&self, kind: CallbackKind) -> Vec<u64> {
        self.callbacks.get(&kind).cloned().unwrap_or_default()
    }
    fn hook(&self, name: String) -> Option<u64> {
        self.callbacks
            .get(&match name.as_str() {
                "on_enter" => CallbackKind::OnEnter,
                "on_exit" => CallbackKind::OnExit,
                _ => return None,
            })
            .and_then(|ids| ids.first().copied())
    }
    fn has_enter(&self, overridden: bool) -> bool {
        overridden || self.hook("on_enter".into()).is_some()
    }
    fn dynamic_instructions(&self) -> bool {
        self.callbacks.get(&CallbackKind::Instructions).is_some_and(|ids| !ids.is_empty())
    }
    fn validated_output(&self) -> bool {
        self.callbacks.get(&CallbackKind::Validator).is_some_and(|ids| !ids.is_empty())
    }
    fn configuration(&self, retries: Option<Value>) -> Result<AgentConfiguration, AuthoringError> {
        Ok(AgentConfiguration {
            name: self.name(),
            instructions: self.instructions.clone(),
            dynamic_instructions: self.dynamic_instructions(),
            retries: resolve_retries(retries, self.retries)?,
        })
    }
    fn check_tool_options(&self, given: Vec<String>) -> Result<(), AuthoringError> {
        if given.is_empty() { Ok(()) } else { Err(AuthoringError::BuiltToolOptions(given)) }
    }
    fn join_instructions(&self, parts: Vec<String>) -> String {
        parts.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n")
    }
    fn retries(&self) -> Retries {
        self.retries
    }
}
impl AgentConfig {
    fn name(&self) -> String {
        self.name.as_deref().filter(|s| !s.is_empty()).unwrap_or(DEFAULT_AGENT_NAME).to_owned()
    }
}
fn resolve_retries(value: Option<Value>, base: Retries) -> Result<Retries, AuthoringError> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(base);
    };
    let budget = |value: &Value| value.as_u64().and_then(|n| u32::try_from(n).ok());
    let invalid = || AuthoringError::Retries(value.to_string());
    match &value {
        Value::Number(_) => {
            let count = budget(&value).ok_or_else(invalid)?;
            Ok(Retries { tools: count, output: count })
        }
        Value::Object(map) => Ok(Retries {
            tools: map
                .get("tools")
                .map(|v| budget(v).ok_or_else(invalid))
                .transpose()?
                .unwrap_or(base.tools),
            output: map
                .get("output")
                .map(|v| budget(v).ok_or_else(invalid))
                .transpose()?
                .unwrap_or(base.output),
        }),
        _ => Err(invalid()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::agent::authoring::AgentConfig as _;
    use serde_json::json;
    #[test]
    fn retry_overrides_preserve_omitted_budgets_and_reject_invalid_values() {
        let base = Retries { tools: 4, output: 2 };
        assert_eq!(
            resolve_retries(Some(json!({"output": 0})), base).unwrap(),
            Retries { tools: 4, output: 0 }
        );
        assert_eq!(resolve_retries(Some(json!(3)), base).unwrap(), Retries { tools: 3, output: 3 });
        for value in [
            json!(-1),
            json!(true),
            json!(1.0),
            json!({"tools": null}),
            json!({"output": false}),
            json!(u64::MAX),
        ] {
            assert!(resolve_retries(Some(value), base).is_err());
        }
    }
    #[test]
    fn registration_order_hook_replacement_and_freeze_are_shared_across_entrypoints() {
        let mut agent = AgentConfig::new("Help".into(), None, None).unwrap();
        assert_eq!(agent.resolve_name(None), DEFAULT_AGENT_NAME);
        assert_eq!(agent.resolve_name(Some("desk".into())), "desk");
        let (first, _) = agent.register(CallbackKind::Instructions).unwrap();
        let (second, _) = agent.register(CallbackKind::Instructions).unwrap();
        assert_eq!(agent.callbacks(CallbackKind::Instructions), vec![first, second]);
        let (old, _) = agent.register(CallbackKind::OnEnter).unwrap();
        let (new, retired) = agent.register(CallbackKind::OnEnter).unwrap();
        assert_eq!(retired, Some(old));
        assert_eq!(agent.hook("on_enter".into()), Some(new));
        agent.freeze();
        assert!(matches!(agent.register(CallbackKind::Tool), Err(AuthoringError::Frozen { .. })));
        assert_eq!(agent.configuration(None).unwrap().name, "desk");
    }
}
