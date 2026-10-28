//! Scope identity, schema compatibility and revision-checked context staging.

use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::storage::context::{
    ContextField, ContextManifest, ContextScope, ContextScopeWrite, ContextValues,
};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::StorageError;
use crate::db::STATE_LIMIT;

pub mod db;

/// One activation's mutable scope state. Runtime owns serialization between callers.
pub struct ContextActivation {
    scopes: BTreeMap<String, ContextScope>,
    committed_scopes: BTreeMap<String, ContextScope>,
    base_revisions: BTreeMap<String, i64>,
    writes: BTreeMap<String, ContextScopeWrite>,
    routing_state: Option<Value>,
    read_only: bool,
    closed: bool,
}

impl ContextActivation {
    pub fn new(
        scopes: Vec<ContextScope>,
        routing_state: Option<Value>,
        read_only: bool,
    ) -> Result<Self, StorageError> {
        let mut by_id = BTreeMap::new();
        let mut base_revisions = BTreeMap::new();
        for scope in scopes {
            if scope.scope_id.is_empty() || scope.revision < 0 {
                return Err(StorageError::Decode(
                    "invalid context scope identity or revision".into(),
                ));
            }
            base_revisions.insert(scope.scope_id.clone(), scope.revision);
            if by_id.insert(scope.scope_id.clone(), scope).is_some() {
                return Err(StorageError::Decode("duplicate context scope".into()));
            }
        }
        Ok(Self {
            committed_scopes: by_id.clone(),
            scopes: by_id,
            base_revisions,
            writes: BTreeMap::new(),
            routing_state,
            read_only,
            closed: false,
        })
    }

    fn ensure_open(&self) -> Result<(), StorageError> {
        if self.closed {
            return Err(StorageError::Invalid("the context store's activation has ended".into()));
        }
        Ok(())
    }

    pub fn load(
        &mut self,
        scope_id: &str,
        identity: &str,
        fields: &[ContextField],
    ) -> Result<ContextValues, StorageError> {
        self.ensure_open()?;
        if scope_id.is_empty() || identity.is_empty() {
            return Err(StorageError::Invalid(
                "context scope and schema identities are required".into(),
            ));
        }
        let mut scope = self.scopes.get(scope_id).cloned().unwrap_or_else(|| ContextScope {
            scope_id: scope_id.to_owned(),
            ..Default::default()
        });
        check_manifest(&mut scope, identity, fields)?;
        let values = fields
            .iter()
            .filter_map(|field| {
                scope.fields.get(&field.name).map(|value| (field.name.clone(), value.clone()))
            })
            .collect();
        let revision = scope.revision;
        self.scopes.insert(scope_id.to_owned(), scope);
        Ok(ContextValues { revision, fields: values })
    }

    pub fn save(
        &mut self,
        scope_id: &str,
        identity: &str,
        revision: i64,
        fields: &[ContextField],
    ) -> Result<ContextValues, StorageError> {
        self.ensure_open()?;
        if self.read_only {
            return Err(StorageError::Invalid(
                "context writes require an exclusive managed turn".into(),
            ));
        }
        let mut scope = self.scopes.get(scope_id).cloned().ok_or_else(|| {
            StorageError::Invalid("load_context must be called before save_context".into())
        })?;
        if scope.revision != revision {
            return Err(StorageError::VersionConflict {
                lane: scope_id.into(),
                expected: revision,
            });
        }
        if !scope.manifests.iter().any(|manifest| manifest.identity == identity) {
            return Err(StorageError::Invalid(
                "load_context must be called before save_context".into(),
            ));
        }
        check_manifest(&mut scope, identity, fields)?;
        scope.fields.extend(fields.iter().map(|field| (field.name.clone(), field.value.clone())));
        scope.revision = scope.revision.checked_add(1).ok_or_else(|| {
            StorageError::Invalid(format!("context scope {scope_id} exhausted its revision"))
        })?;
        check_state_size(
            self.routing_state.as_ref(),
            self.scopes.values().filter(|other| other.scope_id != scope_id).chain([&scope]),
        )?;
        let values = ContextValues { revision: scope.revision, fields: scope.fields.clone() };
        let write = ContextScopeWrite {
            expected_revision: *self.base_revisions.get(scope_id).unwrap_or(&0),
            scope: scope.clone(),
        };
        self.scopes.insert(scope_id.to_owned(), scope);
        self.writes.insert(scope_id.to_owned(), write);
        Ok(values)
    }

    pub fn writes(&self) -> Vec<ContextScopeWrite> {
        self.writes.values().cloned().collect()
    }

    pub fn acknowledge(&mut self) {
        for (id, write) in &self.writes {
            self.base_revisions.insert(id.clone(), write.scope.revision);
            self.committed_scopes.insert(id.clone(), write.scope.clone());
        }
        self.writes.clear();
    }

    pub fn rebase(
        &mut self,
        scopes: Vec<ContextScope>,
        routing_state: Option<Value>,
    ) -> Result<(), StorageError> {
        self.ensure_open()?;
        if !self.writes.is_empty() {
            return Err(StorageError::Invalid("context saves are still pending".into()));
        }
        let incoming = Self::new(scopes, routing_state, self.read_only)?;
        if incoming.scopes == self.committed_scopes {
            self.routing_state = incoming.routing_state;
        } else {
            *self = incoming;
        }
        Ok(())
    }

    /// Discard staged fields after an abandoned activation without claiming a durable save.
    pub fn abandon(&mut self) {
        self.scopes = self.committed_scopes.clone();
        self.base_revisions =
            self.committed_scopes.iter().map(|(id, scope)| (id.clone(), scope.revision)).collect();
        self.writes.clear();
    }

    pub fn close(&mut self) {
        self.closed = true;
    }
}

fn check_manifest(
    scope: &mut ContextScope,
    identity: &str,
    fields: &[ContextField],
) -> Result<(), StorageError> {
    let names: BTreeSet<_> = fields.iter().map(|field| field.name.as_str()).collect();
    if names.len() != fields.len() || names.contains("") {
        return Err(StorageError::Invalid(
            "persisted field names must be unique and nonempty".into(),
        ));
    }
    if let Some(previous) = scope.manifests.iter().find(|manifest| manifest.identity == identity) {
        for name in &previous.fields {
            if !names.contains(name.as_str()) {
                return Err(StorageError::Schema(format!(
                    "context {identity} removed or renamed persisted field {name}; a migration is required"
                )));
            }
        }
    }
    for field in fields {
        if let Some(previous) = scope.contracts.get(&field.name)
            && previous != &field.schema
        {
            return Err(StorageError::Schema(format!(
                "context scope {} has an incompatible contract for field {}",
                scope.scope_id, field.name
            )));
        }
    }
    for field in fields {
        scope.contracts.insert(field.name.clone(), field.schema.clone());
    }
    let manifest = ContextManifest {
        identity: identity.to_owned(),
        fields: names.into_iter().map(str::to_owned).collect(),
    };
    if let Some(previous) = scope.manifests.iter_mut().find(|m| m.identity == identity) {
        *previous = manifest;
    } else {
        scope.manifests.push(manifest);
    }
    Ok(())
}

/// The complete keyed data snapshot, counting every scope once and excluding schema metadata.
pub fn check_state_size<'a>(
    routing: Option<&Value>,
    scopes: impl IntoIterator<Item = &'a ContextScope>,
) -> Result<(), StorageError> {
    #[derive(Serialize)]
    struct Snapshot<'a> {
        scopes: BTreeMap<&'a str, &'a Map<String, Value>>,
        routing: Option<&'a Value>,
    }
    let snapshot = Snapshot {
        scopes: scopes.into_iter().map(|scope| (scope.scope_id.as_str(), &scope.fields)).collect(),
        routing,
    };
    struct Budget {
        bytes: usize,
        exceeded: bool,
    }
    impl std::io::Write for Budget {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.bytes = self.bytes.saturating_add(bytes.len());
            if self.bytes >= STATE_LIMIT {
                self.exceeded = true;
                return Err(std::io::Error::other("state exceeds its byte budget"));
            }
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut budget = Budget { bytes: 0, exceeded: false };
    let result = serde_json::to_writer(&mut budget, &snapshot);
    if budget.exceeded {
        return Err(StorageError::TooLarge { what: "state".into(), limit: STATE_LIMIT });
    }
    result.map_err(|error| StorageError::NotSerializable("state".into(), error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn field(name: &str, value: Value) -> ContextField {
        ContextField { name: name.into(), schema: json!({"type": "string"}), value }
    }

    #[test]
    fn scope_revisions_and_contracts_survive_restart_without_colliding() {
        let mut activation = ContextActivation::new(Vec::new(), None, false).unwrap();
        let first = vec![field("draft", json!("first"))];
        let second = vec![field("draft", json!("second"))];
        for (id, fields) in [("booking_a", &first), ("booking_b", &second)] {
            let loaded = activation.load(id, "Booking", fields).unwrap();
            activation.save(id, "Booking", loaded.revision, fields).unwrap();
        }
        let scopes = activation.writes().into_iter().map(|write| write.scope).collect();
        let mut restored = ContextActivation::new(scopes, None, false).unwrap();
        let a = restored.load("booking_a", "Booking", &second).unwrap();
        let b = restored.load("booking_b", "Booking", &first).unwrap();
        assert_eq!((a.revision, a.fields["draft"].clone()), (1, json!("first")));
        assert_eq!((b.revision, b.fields["draft"].clone()), (1, json!("second")));
        restored.save("booking_a", "Booking", a.revision, &second).unwrap();
        assert!(matches!(
            restored.save("booking_a", "Booking", a.revision, &first),
            Err(StorageError::VersionConflict { .. })
        ));
        assert_eq!(restored.writes()[0].expected_revision, 1);
        assert!(matches!(
            restored.load("booking_a", "Booking", &[field("renamed", json!("default"))]),
            Err(StorageError::Schema(_))
        ));
        let mut incompatible = field("draft", json!(0));
        incompatible.schema = json!({"type": "integer"});
        assert!(matches!(
            restored.load("booking_a", "Shared", &[incompatible]),
            Err(StorageError::Schema(_))
        ));
        assert_eq!(restored.writes()[0].scope.fields["draft"], json!("second"));
    }

    #[test]
    fn aggregate_size_refusal_preserves_every_staged_scope() {
        let mut activation = ContextActivation::new(Vec::new(), None, false).unwrap();
        let fields = vec![field("draft", json!("é\\\"".repeat(6000)))];
        let a = activation.load("a", "Booking", &fields).unwrap();
        activation.save("a", "Booking", a.revision, &fields).unwrap();
        let before = activation.writes();
        let b = activation.load("b", "Booking", &fields).unwrap();
        assert!(matches!(
            activation.save("b", "Booking", b.revision, &fields),
            Err(StorageError::TooLarge { .. })
        ));
        assert_eq!(activation.writes(), before);
        activation.close();
        assert!(activation.load("a", "Booking", &fields).is_err());
    }

    #[test]
    fn the_complete_encoded_state_must_be_strictly_under_the_limit() {
        let mut scope = ContextScope { scope_id: "root".into(), ..Default::default() };
        let empty = serde_json::to_vec(&json!({
            "scopes": {"root": {"draft": ""}}, "routing": null,
        }))
        .unwrap()
        .len();
        scope.fields.insert("draft".into(), json!("x".repeat(STATE_LIMIT - empty - 1)));
        assert!(check_state_size(None, [&scope]).is_ok());
        scope.fields.insert("draft".into(), json!("x".repeat(STATE_LIMIT - empty)));
        assert!(matches!(check_state_size(None, [&scope]), Err(StorageError::TooLarge { .. })));

        scope.fields.insert("draft".into(), json!("x".repeat(32_000)));
        let routing = json!({"frames": "x".repeat(34_000)});
        assert!(matches!(
            check_state_size(Some(&routing), [&scope]),
            Err(StorageError::TooLarge { .. })
        ));
    }
}
