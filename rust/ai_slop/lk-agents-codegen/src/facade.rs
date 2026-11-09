//! `#[host(facade(..))]` traits as Python classes (docs/internals/rust-core.md "Facades"): one
//! module `_generated/<family>_facades.py` per declaring family. A class subclasses its `base`,
//! takes its `#[host(new)]` parameters with their ABI defaults and builds its native object from
//! them, then its base from the native getters `base_args` names. A method forwards to the native
//! method of its name, or `via` a delegate; a stream's `_run` is its base's native pump over the
//! owner's operation.

use crate::class_stub::{Dir, Imports, py_literal};
use crate::decl::{Class, Declarations, FacadeClass, FacadeMethod, Method, MethodKind, Param, Ty};

#[path = "facade_with.rs"]
mod with;
use with::{bound_path, known_roots};
#[cfg(test)]
#[path = "facade_tests.rs"]
mod facade_tests;
use crate::python::Out;

pub fn emit(declared: &Declarations, bases: &[crate::bases::Bases]) -> Result<Vec<Out>, String> {
    let mut modules = std::collections::BTreeMap::<&str, Vec<&FacadeClass>>::new();
    for facade in &declared.facades {
        modules.entry(&facade.module).or_default().push(facade);
    }
    let placed = crate::exports::placed(declared, bases);
    let mut out = Vec::new();
    for (module, facades) in modules {
        let mut x = Module {
            declared,
            placed: &placed,
            module,
            runtime: Imports::new(declared),
            typing: Imports::new(declared),
            stdlib: std::collections::BTreeSet::new(),
            casts: false,
            known: Vec::new(),
        };
        let mut body = String::new();
        let mut identities = String::new();
        for facade in &facades {
            body.push_str(&x.class(facade).map_err(|e| format!("{}: {e}", facade.source))?);
            if !facade.extended {
                identities.push_str(&format!("{}.__module__ = {:?}\n", facade.name, facade.path));
            }
        }
        if !identities.is_empty() {
            body.push_str(&format!("\n\n{identities}"));
        }
        let mut text = "# GENERATED FILE -- DO NOT EDIT.\n# From #[host(facade)] traits in lk-agents-abi.\n\nfrom __future__ import annotations\n\n".to_owned();
        let mut header = String::new();
        for module in &x.stdlib {
            header.push_str(&format!("import {module}\n"));
        }
        let typing = if x.casts { "TYPE_CHECKING, Any, cast" } else { "TYPE_CHECKING, Any" };
        header
            .push_str(&format!("from typing import {typing}\n\nfrom .. import _bindings, _core\n"));
        let header = crate::imports::merge(&header, &x.runtime.render(&[]))?;
        text.push_str(header.trim_start_matches('\n'));
        // A name bound at runtime is not imported again for the type checker.
        let bound = |import: &str| import.rsplit(" as ").next().unwrap_or(import).to_owned();
        let runtime: Vec<String> = x.runtime.entries().iter().map(|(_, n)| bound(n)).collect();
        let shadowed: Vec<(String, String)> = x
            .typing
            .entries()
            .iter()
            .filter(|(_, n)| runtime.contains(&bound(n)))
            .cloned()
            .collect();
        for (module, name) in shadowed {
            x.typing.remove(&module, &name);
        }
        x.typing.remove_module("typing");
        let typing = x.typing.render(&[]);
        if !typing.is_empty() {
            text.push_str("\nif TYPE_CHECKING:\n");
            for line in typing.lines() {
                match line.is_empty() {
                    true => text.push('\n'),
                    false => text.push_str(&format!("    {line}\n")),
                }
            }
        }
        text.push_str(&body);
        out.push(Out { path: format!("{module}.py"), text });
    }
    Ok(out)
}

struct Module<'a> {
    declared: &'a Declarations,
    placed: &'a std::collections::BTreeMap<String, Vec<(String, String)>>,
    module: &'a str,
    runtime: Imports,
    typing: Imports,
    stdlib: std::collections::BTreeSet<&'static str>,
    /// A provider method hands its stream the owner through `typing.cast`.
    casts: bool,
    /// The attributes the class itself declares, which a bound path may start from.
    known: Vec<String>,
}

impl Module<'_> {
    fn native(&self, name: &str) -> Result<&Class, String> {
        self.declared
            .classes
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| format!("no native class {name}"))
    }

    /// `module.Name` relative to `livekit.agents`, bound in the generated module at runtime: a
    /// generated class from its generated module, so a public module never imports itself.
    fn runtime_name(&mut self, dotted: &str) -> Result<String, String> {
        let (source, name) = self.placed_source(dotted);
        if source.trim_start_matches('.') != self.module {
            self.runtime.add(&source, &name);
        }
        Ok(name)
    }

    /// Where `module.Name` (or a top-level `Name`) is imported from: its generated module when
    /// placed, else its public module.
    fn placed_source(&self, dotted: &str) -> (String, String) {
        let (module, name) = dotted.rsplit_once('.').unwrap_or(("", dotted));
        let public = match module {
            "" => "livekit.agents".to_owned(),
            module => format!("livekit.agents.{module}"),
        };
        let source = self
            .placed
            .get(&public)
            .and_then(|names| names.iter().find(|(_, n)| n == name))
            .map_or_else(|| format!("..{module}"), |(source, _)| source.clone());
        (source, name.to_owned())
    }

    /// A host callable `module.fn` written relative to `_generated` (`.._bindings.core_legs`).
    fn callable(&mut self, dotted: &str) -> Result<String, String> {
        let (module, name) =
            dotted.rsplit_once('.').ok_or_else(|| format!("`{dotted}` names module.item"))?;
        if module != "builtins" {
            self.runtime.add(module, name);
        }
        Ok(name.to_owned())
    }

    fn class(&mut self, facade: &FacadeClass) -> Result<String, String> {
        let native = match facade.native.as_str() {
            "" => Class::default(),
            name => self.native(name)?.clone(),
        };
        let mut base = match (&facade.base, facade.foreign) {
            (Some(b), true) => Some(self.callable(b)?),
            (b, false) => b.as_deref().map(|b| self.runtime_name(b)).transpose()?,
            (None, true) => return Err("`foreign` qualifies a `base`".into()),
        };
        // A facade named as its base binds the base under another name.
        if let (Some(b), Some(dotted)) = (&mut base, &facade.base)
            && *b == facade.name
            && !facade.foreign
        {
            let (source, name) = self.placed_source(dotted);
            self.runtime.remove(&source, &name);
            self.runtime.add(&source, &format!("{name} as Base{name}"));
            *b = format!("Base{name}");
        }
        // An adapter observing availability emits its event besides its base's.
        if let (Some(b), Some(availability)) = (&mut base, &facade.availability) {
            self.runtime.add("typing", "Literal");
            *b = format!("{b}[Literal[{:?}]]", availability.event);
        }
        let mut text = String::new();
        let mut bases: Vec<String> = base.iter().cloned().collect();
        if let Some(t) = &facade.generic {
            self.runtime.add("typing", "Generic");
            self.runtime.add("typing", "TypeVar");
            text.push_str(&format!("\n\n{t} = TypeVar(\"{t}\")\n"));
            bases.push(format!("Generic[{t}]"));
        }
        let bases =
            if bases.is_empty() { String::new() } else { format!("({})", bases.join(", ")) };
        text.push_str(&format!("\n\nclass {}{bases}:\n", facade.name));
        let mut body = String::new();
        if crate::python::docstring(&mut body, &facade.doc, 4) {
            body.push('\n');
        }
        body.push_str(&self.host_attributes(facade, &native)?);
        let new = facade.methods.iter().find(|m| m.method.kind == MethodKind::New);
        let mut declared = Vec::new();
        if let Some(stream) = &facade.stream {
            let pumped = base.as_deref().ok_or("a stream names its provider stream `base`")?;
            body.push_str(&self.stream(facade, &native, stream, pumped)?);
            text.push_str(&body);
            return Ok(text);
        }
        for delegate in facade.delegates.iter().filter(|d| d.held) {
            let in_new =
                new.is_some_and(|n| n.delegated.iter().any(|(_, a)| *a == delegate.attribute));
            if !in_new {
                let ty = annotation(&delegate.ty, Dir::In, &mut self.typing);
                declared.push(property(&delegate.attribute, &ty));
            }
        }
        let given = new.is_some() || facade.from_native || native.name.is_empty();
        let build = self.build(facade, &native, given, &mut declared)?;
        declared.extend(self.read_paths(facade, new));
        self.known = known_roots(facade, new);
        if !declared.is_empty() {
            body.push_str("    if TYPE_CHECKING:\n");
            for line in declared.join("\n").lines() {
                match line.is_empty() {
                    true => body.push('\n'),
                    false => body.push_str(&format!("        {line}\n")),
                }
            }
            body.push('\n');
        }
        let mut members = Vec::new();
        if facade.from_native {
            members.push(self.from_native(facade, &native, new)?);
        } else if let Some(new) = new {
            members.push(self.constructor(facade, &native, new)?);
        }
        members.extend(build);
        for method in facade.methods.iter().filter(|m| m.method.kind != MethodKind::New) {
            members.push(
                self.method(&native, method)
                    .map_err(|e| format!("{}.{}: {e}", facade.name, method.method.name))?,
            );
        }
        let declared = self.declared;
        let owned = |f: &&FacadeClass| !f.native.is_empty() && f.native == facade.native;
        for opened in declared.facades.iter().filter(owned) {
            if let Some(stream) = opened.stream.as_ref().filter(|s| s.method.is_some()) {
                members.push(self.provider_method(facade, &native, opened, stream)?);
            }
        }
        body.push_str(&members.join("\n"));
        if body.trim().is_empty() {
            body.push_str("    pass\n");
        }
        text.push_str(&body);
        Ok(text)
    }

    /// `host_args(p = "self._x")`: the class attribute a subclass sets, `None` until it does,
    /// typed as the native constructor's optional parameter.
    fn host_attributes(&mut self, facade: &FacadeClass, native: &Class) -> Result<String, String> {
        let mut text = String::new();
        for (param, object) in &facade.host_args {
            let Some(attribute) = object.strip_prefix("self.") else { continue };
            let p = native
                .methods
                .iter()
                .find(|m| m.kind == MethodKind::New)
                .and_then(|new| new.params.iter().find(|p| &p.name == param))
                .ok_or_else(|| {
                    format!("host_args({param}) names no native constructor parameter")
                })?;
            if !matches!(p.ty, Ty::Optional(_)) {
                return Err(format!(
                    "host_args({param} = \"{object}\") fills an optional parameter"
                ));
            }
            let ty = annotation(&p.ty, Dir::In, &mut self.typing);
            text.push_str(&format!("    {attribute}: {ty} = None\n"));
        }
        if !text.is_empty() {
            text.push('\n');
        }
        Ok(text)
    }

    /// The `_native` declaration (when no `#[host(new)]` builds it), the leg and availability
    /// members.
    fn build(
        &mut self,
        facade: &FacadeClass,
        native: &Class,
        has_new: bool,
        declared: &mut Vec<String>,
    ) -> Result<Vec<String>, String> {
        let name = &native.name;
        let mut members = Vec::new();
        if !has_new {
            // The handwritten subclass sets it.
            declared.push(format!("_native: _core.{name}\n"));
        }
        if !facade.legs.is_empty() {
            let leg = match &facade.leg {
                Some(leg) => {
                    declared.push(format!("def {leg}(self) -> _core.{name}: ...\n"));
                    format!("self.{leg}()")
                }
                None => "self._native".to_owned(),
            };
            let mut arguments = vec!["self".to_owned(), facade.name.clone()];
            arguments.extend(facade.legs.iter().map(|l| format!("{l:?}")));
            members.push(format!(
                "    def _core_leg(self) -> Any:\n        \"\"\"This engine, unless a subclass replaces what it would bypass.\"\"\"\n{}        return self if replaced else {leg}\n",
                call("replaced = _bindings.overrides", &arguments, 8)
            ));
        }
        if let Some(availability) = &facade.availability {
            members.extend(self.availability(facade, native, availability, declared)?);
        }
        Ok(members)
    }

    /// An availability observer's members: the chain's events as the host's, and the serving
    /// entry as the adapter's identity.
    fn availability(
        &mut self,
        facade: &FacadeClass,
        native: &Class,
        availability: &crate::decl::Availability,
        declared: &mut Vec<String>,
    ) -> Result<Vec<String>, String> {
        let name = &native.name;
        let constructor = native
            .methods
            .iter()
            .find(|m| m.kind == MethodKind::New)
            .ok_or_else(|| format!("_core.{name} has no #[host(new)]"))?;
        let parameter = constructor
            .params
            .iter()
            .find(|p| p.name == availability.observer)
            .ok_or_else(|| "availability observes through no parameter".to_owned())?;
        // The members as the facade takes them, else as its native does.
        let facade_new = facade.methods.iter().find(|m| m.method.kind == MethodKind::New);
        let entries = facade_new
            .and_then(|n| n.method.params.iter().find(|p| p.name == availability.members))
            .or_else(|| constructor.params.iter().find(|p| p.name == availability.members))
            .ok_or_else(|| format!("_core.{name}() takes no {}", availability.members))?;
        let Ty::List(entry) = &entries.ty else {
            return Err(format!("availability members `{}` is a list", availability.members));
        };
        let entry = annotation(entry, Dir::Out, &mut self.typing);
        let (module, record) = availability
            .record
            .rsplit_once('.')
            .ok_or_else(|| format!("`{}` names module.item", availability.record))?;
        let (package, module) = module.rsplit_once('.').unwrap_or(("", module));
        let package = match package.trim_start_matches('.') {
            "" => format!("{package}."),
            _ => package.to_owned(),
        };
        self.runtime.add(&package, module);
        self.runtime.add("..log", "logger");
        self.typing.add(".llm_fallback", "AvailabilityChanged");
        self.typing.add(".chat_items", "MetricsMetadata");
        declared.push(format!("_members: list[{entry}]\n"));
        let observed = self.observed(parameter)?;
        if !observed.trim().is_empty() {
            declared.push(observed);
        }
        if facade.base.is_none() {
            declared.push("def emit(self, event: str, *args: Any) -> None: ...\n".into());
        }
        let event = format!("{:?}", availability.event);
        let value = format!("{module}.{record}(self._members[ev.index], ev.available)");
        let mut members = vec![format!(
            "    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:\n{}",
            call("self.emit", &[event, value], 8)
        )];
        members.push(format!(
            "    def _core_all_unavailable(self) -> None:\n        logger.error(\"all {}s are unavailable, retrying..\")\n",
            availability.kind
        ));
        members.push(format!(
            "    @property\n    def _active(self) -> {entry}:\n        return self._members[self._native.active_index]\n"
        ));
        for (property, ty) in
            [("model", "str"), ("provider", "str"), ("metrics_metadata", "MetricsMetadata")]
        {
            members.push(format!(
                "    @property\n    def {property}(self) -> {ty}:\n        return self._active.{property}\n"
            ));
        }
        let mut close = match facade.base {
            Some(_) => "        await super().aclose()\n".to_owned(),
            None => String::new(),
        };
        // A chain holding no native resources closes its members.
        match native.methods.iter().any(|m| m.name == "aclose") {
            true => close.push_str("        await self._native.aclose()\n"),
            false => close.push_str(
                "        for member in self._members:\n            await member.aclose()\n",
            ),
        }
        members.push(format!("    async def aclose(self) -> None:\n{close}"));
        Ok(members)
    }

    /// The observer proxy's other calls, which the host class answers itself.
    fn observed(&mut self, parameter: &Param) -> Result<String, String> {
        let declared = match &parameter.ty {
            Ty::Optional(inner) => inner.as_ref(),
            other => other,
        };
        let Ty::Proxy { declared: Some((family, name)), .. } = declared else {
            return Err("availability observer is no proxy".into());
        };
        let proxy = self
            .declared
            .proxies
            .iter()
            .find(|p| &p.family == family && &p.rust_name == name)
            .ok_or_else(|| format!("no proxy {name}"))?;
        let mut text = String::new();
        let generated = ["_core_availability_changed", "_core_all_unavailable"];
        for call in proxy.methods.iter().filter(|m| !generated.contains(&m.name.as_str())) {
            let mut params = vec!["self".to_owned()];
            for p in call.params.iter().filter(|p| !p.ty.hidden()) {
                let ty = annotation(&p.ty, Dir::Out, &mut self.typing);
                params.push(format!("{}: {ty}", p.name));
            }
            text.push_str(&format!("def {}({}) -> None: ...\n\n", call.name, params.join(", ")));
        }
        Ok(text.trim_end().to_owned() + "\n")
    }

    fn constructor(
        &mut self,
        facade: &FacadeClass,
        native: &Class,
        new: &FacadeMethod,
    ) -> Result<String, String> {
        let (signature, values) = self.parameters(native, new)?;
        let mut body = String::new();
        // A base built from the native object's getters is built after it.
        let base_first = facade.base.is_some() && facade.base_args.is_empty();
        if base_first {
            body.push_str("        super().__init__()\n");
        }
        for p in new.method.params.iter().filter(|p| !p.ty.hidden()) {
            let n = p.name.trim_start_matches("r#");
            if let Some((_, attribute)) = new.delegated.iter().find(|(d, _)| d == n) {
                body.push_str(&format!("        self.{attribute} = {n}\n"));
            }
        }
        let constructor = native
            .methods
            .iter()
            .find(|m| m.kind == MethodKind::New)
            .ok_or_else(|| format!("_core.{} has no #[host(new)]", native.name))?;
        let mut facade = facade.clone();
        for (_, object) in &mut facade.host_args {
            if !object.starts_with("self.") {
                *object = self.callable(object)?;
            }
        }
        let facade = &facade;
        let observer = facade.availability.as_ref().map(|a| a.observer.as_str());
        let arguments = sent(&constructor.params, |p| {
            let n = p.name.trim_start_matches("r#");
            if observer == Some(n) {
                return Some("self".to_owned());
            }
            if let Some((_, object)) = facade.host_args.iter().find(|(a, _)| a == n) {
                return Some(object.clone());
            }
            values.iter().find(|(v, _)| v == n).map(|(_, value)| value.clone())
        })?;
        body.push_str(&call(&format!("self._native = _core.{}", native.name), &arguments, 8));
        for n in &facade.keep {
            body.push_str(&format!("        self._{n} = {n}\n"));
        }
        if let Some(availability) = &facade.availability {
            body.push_str(&format!("        self._members = list({})\n", availability.members));
        }
        if !base_first && (facade.base.is_some() || !facade.base_args.is_empty()) {
            for n in &facade.base_args {
                native
                    .methods
                    .iter()
                    .find(|m| &m.name == n && m.kind == MethodKind::Getter)
                    .ok_or_else(|| format!("_core.{} has no getter {n}", native.name))?;
            }
            let arguments: Vec<String> =
                facade.base_args.iter().map(|n| format!("{n}=self._native.{n}")).collect();
            body.push_str(&call("super().__init__", &arguments, 8));
        }
        let mut text = def("__init__", false, &signature, "None");
        crate::python::docstring(&mut text, &new.method.doc, 8);
        text.push_str(&body);
        Ok(text)
    }

    /// A method's signature parameters and, by name, the value each sends on.
    fn parameters(
        &mut self,
        native: &Class,
        method: &FacadeMethod,
    ) -> Result<(Vec<String>, Vec<(String, String)>), String> {
        let adapt = &method.adapt;
        let shown = crate::class_stub::parameters(native, &method.method, &mut self.typing);
        let mut signature = vec!["self".to_owned()];
        let mut values = Vec::new();
        let mut star = false;
        let visible = method.method.params.iter().filter(|p| !p.ty.hidden());
        for (p, (shown, _)) in visible.zip(shown) {
            let n = p.name.trim_start_matches("r#").to_owned();
            if p.kw_only && !star {
                signature.push("*".into());
                star = true;
            }
            let annotation_of =
                |x: &mut Self, inner: &Ty| match adapt.annotations.iter().find(|(a, _)| *a == n) {
                    Some((_, text)) => x.typing.annotation(text, ""),
                    None => annotation(inner, Dir::In, &mut x.typing),
                };
            let resolve = adapt.resolve.iter().find(|(a, _)| *a == n).map(|(_, f)| f.clone());
            let resolve = resolve.map(|f| self.callable(&f)).transpose()?;
            let sent = match &resolve {
                Some(f) => format!("{f}({n})"),
                None => n.clone(),
            };
            let defaults_none = p.default.as_ref().map(py_literal).as_deref() == Some("None");
            let (text, value) = match &p.ty {
                Ty::Maybe(inner) => {
                    self.runtime.add("..types", "NOT_GIVEN");
                    self.typing.add("..types", "NotGivenOr");
                    let value = match &resolve {
                        Some(f) => {
                            self.runtime.add("..utils", "is_given");
                            format!("{f}({n}) if is_given({n}) else {n}")
                        }
                        None => n.clone(),
                    };
                    (format!("{n}: NotGivenOr[{}] = NOT_GIVEN", annotation_of(self, inner)), value)
                }
                Ty::Optional(inner) if adapt.not_given && defaults_none => {
                    self.runtime.add("..types", "NOT_GIVEN");
                    self.typing.add("..types", "NotGivenOr");
                    self.runtime.add("..utils", "is_given");
                    let text =
                        format!("{n}: NotGivenOr[{}] = NOT_GIVEN", annotation_of(self, inner));
                    (text, format!("{sent} if is_given({n}) else None"))
                }
                ty => {
                    let text = match adapt.annotations.iter().find(|(a, _)| *a == n) {
                        Some(_) => {
                            let annotation = annotation_of(self, ty);
                            match &p.default {
                                Some(d) => format!("{n}: {annotation} = {}", py_literal(d)),
                                None => format!("{n}: {annotation}"),
                            }
                        }
                        None => shown,
                    };
                    (text, sent)
                }
            };
            signature.push(text);
            values.push((n, value));
        }
        Ok((signature, values))
    }

    fn method(&mut self, native: &Class, method: &FacadeMethod) -> Result<String, String> {
        let m = &method.method;
        let adapt = &method.adapt;
        let (signature, values) = self.parameters(native, method)?;
        // A setter assigns the native property of its name.
        let name = adapt.setter.as_ref().unwrap_or(&method.native);
        let wanted = |n: &&Method| {
            &n.name == name
                && n.kind != MethodKind::New
                && (adapt.setter.is_some() == (n.kind == MethodKind::Setter))
        };
        let (target, called) = match &method.via {
            Some(delegate) => {
                let called = match &delegate.ty {
                    Ty::Instance { class, .. } => {
                        Some(self.native(class)?.methods.iter().find(wanted).cloned().ok_or_else(
                            || format!("_core.{class} has no method {}", method.native),
                        )?)
                    }
                    _ => None,
                };
                (format!("self.{}", delegate.attribute), called)
            }
            None => {
                let called = native.methods.iter().find(wanted).cloned().ok_or_else(|| {
                    format!("_core.{} has no method {}", native.name, method.native)
                })?;
                ("self._native".to_owned(), Some(called))
            }
        };
        let origin = adapt.origin.as_deref().map(|o| self.callable(o)).transpose()?;
        self.casts |=
            adapt.bind.iter().any(|(_, p)| bound_path(&self.known, p).starts_with("cast"));
        let arguments = match &called {
            Some(called) => sent(&called.params, |p| {
                let n = p.name.trim_start_matches("r#");
                if n == "origin"
                    && let Some(origin) = &origin
                {
                    return Some(format!("{origin}()"));
                }
                if let Some((_, path)) = adapt.bind.iter().find(|(b, _)| b == n) {
                    return Some(bound_path(&self.known, path));
                }
                values.iter().find(|(v, _)| v == n).map(|(_, value)| value.clone())
            })?,
            None => {
                let mut keyword = false;
                m.params
                    .iter()
                    .filter(|p| !p.ty.hidden())
                    .zip(&values)
                    .map(|(p, (n, value))| {
                        keyword |= p.kw_only;
                        if keyword { format!("{n}={value}") } else { value.clone() }
                    })
                    .collect()
            }
        };
        for (n, _) in &values {
            let taken = match &called {
                Some(called) => called.params.iter().any(|p| p.name.trim_start_matches("r#") == n),
                None => true,
            };
            if !taken {
                return Err(format!("the native method takes no {n}"));
            }
        }
        let getter =
            called.as_ref().map_or(m.kind == MethodKind::Getter, |c| c.kind == MethodKind::Getter);
        let head = format!("{target}.{}", method.native);
        let invocation = |indent: usize| {
            if getter { head.clone() } else { inline_call(&head, &arguments, indent) }
        };
        let awaited = m.kind == MethodKind::Await;
        let mut ret = self.returned(method);
        let mut body = String::new();
        if let Some(exit) = &method.exit {
            let awaited = called.as_ref().map_or(awaited, |c| c.kind == MethodKind::Await);
            let enter = format!("{}{}", if awaited { "await " } else { "" }, invocation(8));
            let called = called.as_ref();
            return self.context_manager(native, method, called, exit, &signature, &target, &enter);
        }
        if let Some(property) = &adapt.setter {
            let (value, _) = values.first().ok_or("a setter takes the new value")?;
            let text =
                format!("    @{property}.setter\n{}", def(property, false, &signature, "None"));
            let body = format!("        {target}.{property} = {value}\n");
            return Ok(with_doc(text, &m.doc, &body));
        }
        if method.none_ends_iteration {
            let Ty::Optional(inner) = &m.ret.ok else {
                return Err("none_ends_iteration returns an `Option`".into());
            };
            ret = annotation(inner, Dir::Out, &mut self.typing);
            let value = match &method.wrap {
                Some((module, name, field)) => {
                    let name = self.record(module, name);
                    format!("{name}({field}=value)")
                }
                None => "value".into(),
            };
            body.push_str(&format!(
                "        if (value := await {}) is None:\n            raise StopAsyncIteration\n        return {value}\n",
                invocation(8)
            ));
            return Ok(with_doc(def(&m.name, true, &signature, &ret), &m.doc, &body));
        }
        let statement = if let Some(returns) = &adapt.returns {
            ret = self.callable(returns)?;
            let prefix = if awaited { "await " } else { "" };
            match &adapt.each {
                Some(field) => {
                    let value = format!("{prefix}{}", invocation(8));
                    format!("return {ret}([x.{field} for x in {value}])")
                }
                None => {
                    let flat = format!("return {ret}({prefix}{head}({}))", arguments.join(", "));
                    match 8 + flat.len() <= 100 {
                        true => flat,
                        false => {
                            let lines: String = arguments
                                .iter()
                                .map(|a| format!("                {a},\n"))
                                .collect();
                            format!(
                                "return {ret}(\n            {prefix}{head}(\n{lines}            )\n        )"
                            )
                        }
                    }
                }
            }
        } else if adapt.chain {
            self.runtime.add("typing", "Self");
            ret = "Self".into();
            let prefix = if awaited { "await " } else { "" };
            format!("{prefix}{}\n        return self", invocation(8 + prefix.len()))
        } else if let Some((module, name, field)) = &method.wrap {
            let name = self.record(module, name);
            let prefix = if awaited { "await " } else { "" };
            format!("return {name}({field}={prefix}{})", invocation(8))
        } else if adapt.future {
            self.stdlib.insert("asyncio");
            ret = format!("asyncio.Future[{ret}]");
            format!("return asyncio.ensure_future({})", invocation(36))
        } else if awaited && adapt.discard {
            ret = "None".into();
            format!("await {}", invocation(14))
        } else if m.ret.ok != Ty::Unit && adapt.annotations.iter().any(|(a, _)| a == "ret") {
            // `annotation(ret = ..)` narrows what the native's stub declares (a JSON `object`).
            self.casts = true;
            let prefix = if awaited { "await " } else { "" };
            format!("return cast({ret:?}, {prefix}{})", invocation(22 + ret.len() + prefix.len()))
        } else {
            let prefix = match (awaited, m.ret.ok == Ty::Unit) {
                (true, true) => "await ",
                (true, false) => "return await ",
                (false, true) => "",
                (false, false) => "return ",
            };
            format!("{prefix}{}", invocation(8 + prefix.len()))
        };
        body.push_str(&format!("        {statement}\n"));
        let asynchronous = awaited && !adapt.future;
        let mut text = def(&m.name, asynchronous, &signature, &ret);
        if m.kind == MethodKind::Getter {
            text = format!("    @property\n{text}");
        }
        Ok(with_doc(text, &m.doc, &body))
    }

    /// A method's return annotation: `annotation(ret = "..")`, else its declared type's.
    fn returned(&mut self, method: &FacadeMethod) -> String {
        match method.adapt.annotations.iter().find(|(a, _)| a == "ret") {
            Some((_, text)) => self.typing.annotation(text, ""),
            None => annotation(&method.method.ret.ok, Dir::Out, &mut self.typing),
        }
    }

    /// The per-call parameters of a stream's native operation: each one's provider-method
    /// parameter, the annotation the stream stores it under and the value the method passes.
    fn per_call(
        &mut self,
        op: &Method,
        stream: &crate::decl::StreamForm,
    ) -> Result<Vec<PerCall>, String> {
        let mut out = Vec::new();
        let mut star = false;
        for p in op.params.iter().filter(|p| !p.ty.hidden()) {
            let n = p.name.trim_start_matches("r#").to_owned();
            if stream.state.contains(&n) {
                continue;
            }
            let given = stream.annotations.iter().find(|(a, _)| *a == n);
            let shown = |x: &mut Self, ty: &Ty| match given {
                Some((_, text)) => x.typing.annotation(text, ""),
                None => annotation(ty, Dir::In, &mut x.typing),
            };
            let defaults_none = p.default.as_ref().map(py_literal).as_deref() == Some("None");
            let host_default = stream.defaults.iter().find(|(a, _)| *a == n);
            let public = stream
                .rename
                .iter()
                .find(|(a, _)| *a == n)
                .map_or_else(|| n.clone(), |(_, public)| public.clone());
            let (signature, stored, value) = match (&p.ty, host_default) {
                (Ty::Optional(inner), None) if stream.provider_default.contains(&n) => {
                    self.runtime.add("..types", "NOT_GIVEN");
                    self.typing.add("..types", "NotGivenOr");
                    self.runtime.add("..utils", "is_given");
                    let a = shown(self, inner);
                    let value = format!("{n} if is_given({n}) else self._native.{n}");
                    (format!("{n}: NotGivenOr[{a}] = NOT_GIVEN"), a, value)
                }
                (Ty::Optional(inner), Some((_, value))) => {
                    let value = self.callable(value)?;
                    let a = shown(self, inner);
                    (format!("{n}: {a} = {value}"), a, n.clone())
                }
                (Ty::Optional(inner), None) if stream.not_given && defaults_none => {
                    self.runtime.add("..types", "NOT_GIVEN");
                    self.typing.add("..types", "NotGivenOr");
                    self.runtime.add("..utils", "is_given");
                    let a = shown(self, inner);
                    let value = format!("{n} if is_given({n}) else None");
                    (format!("{n}: NotGivenOr[{a}] = NOT_GIVEN"), format!("{a} | None"), value)
                }
                (_, Some(_)) => return Err(format!("default({n}) is for an `Option` parameter")),
                (ty, None) => {
                    let a = shown(self, ty);
                    let signature = match &p.default {
                        Some(d) => format!("{public}: {a} = {}", py_literal(d)),
                        None => format!("{public}: {a}"),
                    };
                    (signature, a, public.clone())
                }
            };
            let keyword = p.kw_only && !star;
            star |= keyword;
            out.push(PerCall { name: n, signature, stored, value, keyword });
        }
        let renamed = stream.rename.iter().map(|(n, _)| n);
        for n in stream.state.iter().chain(&stream.base_args).chain(&stream.provider_default) {
            if !op.params.iter().any(|p| &p.name == n) {
                return Err(format!("{} takes no {n}", stream.op));
            }
        }
        for n in renamed {
            if !op.params.iter().any(|p| &p.name == n) {
                return Err(format!("{} takes no {n}", stream.op));
            }
        }
        Ok(out)
    }

    /// The owner's provider method (`stream()`, `chat()`) that opens the stream facade `opened`.
    fn provider_method(
        &mut self,
        owner: &FacadeClass,
        native: &Class,
        opened: &FacadeClass,
        stream: &crate::decl::StreamForm,
    ) -> Result<String, String> {
        if opened.module != owner.module {
            return Err(format!("{} opens {} of another module", owner.name, opened.name));
        }
        let op = native
            .methods
            .iter()
            .find(|m| m.name == stream.op)
            .ok_or_else(|| format!("_core.{} has no operation {}", native.name, stream.op))?
            .clone();
        let owner_param = stream.owner.trim_start_matches('_');
        let public = owner.path.strip_prefix("livekit.agents.").unwrap_or(&owner.path);
        let module = public.rsplit('.').next().unwrap_or(public);
        self.casts = true;
        let mut params = vec!["self".to_owned()];
        let mut arguments = vec![format!("{owner_param}=cast(\"{module}.{}\", self)", owner.name)];
        for p in self.per_call(&op, stream)? {
            if p.keyword {
                params.push("*".into());
            }
            params.push(p.signature);
            arguments.push(format!("{}={}", p.name, p.value));
        }
        let name = stream.method.as_deref().unwrap_or_default();
        // Annotated as the provider stream, which a plugin overriding the method returns.
        let base = opened.base.as_deref().ok_or("a stream names its provider stream `base`")?;
        let returned = self.runtime_name(base)?;
        let mut text = def(name, false, &params, &returned);
        text.push_str(&call(&format!("return {}", opened.name), &arguments, 8));
        Ok(text)
    }

    fn record(&mut self, module: &str, name: &str) -> String {
        self.runtime.add(&format!(".{module}"), name);
        name.to_owned()
    }

    /// A stream facade: its owner's native operation, pumped by its base.
    fn stream(
        &mut self,
        facade: &FacadeClass,
        native: &Class,
        stream: &crate::decl::StreamForm,
        base: &str,
    ) -> Result<String, String> {
        let owner = self
            .declared
            .facades
            .iter()
            .find(|f| f.native == facade.native && f.stream.is_none())
            .ok_or_else(|| format!("no facade owns _core.{}", facade.native))?;
        // The public class, the handwritten subclass when the generated one is extended: named
        // through its module, beside the generated class of its name.
        let public = owner.path.strip_prefix("livekit.agents.").unwrap_or(&owner.path);
        let (package, module) = match public.rsplit_once('.') {
            Some((package, module)) => (format!("..{package}"), module),
            None => ("..".to_owned(), public),
        };
        self.typing.add(&package, module);
        let owner_class = format!("{module}.{}", owner.name);
        let op = native
            .methods
            .iter()
            .find(|m| m.name == stream.op)
            .ok_or_else(|| format!("_core.{} has no operation {}", native.name, stream.op))?;
        let mut arguments = Vec::new();
        let mut keyword = false;
        for p in op.params.iter().filter(|p| !p.ty.hidden()) {
            let n = p.name.trim_start_matches("r#");
            keyword |= p.kw_only;
            let value = match stream.resolve.iter().find(|(a, _)| a == n) {
                Some((_, f)) => format!("{}(self._{n})", self.callable(f)?),
                None => format!("self._{n}"),
            };
            arguments.push(if keyword { format!("{n}={value}") } else { value });
        }
        let opened = annotation(&op.ret.ok, Dir::Out, &mut self.typing);
        let target = match &owner.leg {
            Some(leg) => format!("self.{}.{leg}()", stream.owner),
            None => format!("self.{}._native", stream.owner),
        };
        let awaited = if op.kind == MethodKind::Await { "await " } else { "" };
        let mut text = String::new();
        if stream.native_retries {
            text.push_str("    _core_native_retries = True\n");
        }
        text.push_str(&format!("    {}: {owner_class}\n\n", stream.owner));
        if stream.method.is_some() {
            let owner_param = stream.owner.trim_start_matches('_');
            let mut params = vec!["self".to_owned(), "*".to_owned()];
            params.push(format!("{owner_param}: {owner_class}"));
            let mut base = vec![format!("{owner_param}={owner_param}")];
            let mut kept = String::new();
            for p in self.per_call(op, stream)? {
                params.push(format!("{}: {}", p.name, p.stored));
                match stream.base_args.contains(&p.name) {
                    true => base.push(format!("{0}={0}", p.name)),
                    // Declared, so a checker does not widen a literal it holds.
                    false => {
                        kept.push_str(&format!("        self._{0}: {1} = {0}\n", p.name, p.stored))
                    }
                }
            }
            text.push_str(&def("__init__", false, &params, "None"));
            text.push_str(&call("super().__init__", &base, 8));
            text.push_str(&kept);
            text.push('\n');
        }
        text.push_str(&format!("    async def _core_open(self) -> {opened}:\n"));
        text.push_str(&call(&format!("return {awaited}{target}.{}", stream.op), &arguments, 8));
        text.push_str(&format!("\n    _run = {base}._core_run_native\n"));
        Ok(text)
    }
}

/// One per-call parameter of a stream's native operation (`Module::per_call`).
struct PerCall {
    name: String,
    /// In the provider method's signature.
    signature: String,
    /// The stream constructor's annotation for it.
    stored: String,
    /// What the provider method passes the stream.
    value: String,
    /// The first keyword-only parameter.
    keyword: bool,
}

/// A host annotation, a native class named through `_core` (a generated class may share its name).
fn annotation(ty: &Ty, dir: Dir, imports: &mut Imports) -> String {
    match ty {
        Ty::Instance { class, .. }
        | Ty::Handle { class, .. }
        | Ty::Shared { class, .. }
        | Ty::Source { class, .. }
        | Ty::Sink { class, .. }
        | Ty::Duplex { class, .. } => format!("_core.{class}"),
        Ty::Optional(inner) => format!("{} | None", annotation(inner, dir, imports)),
        other => crate::class_stub::annotation("", other, dir, imports),
    }
}

/// A TYPE_CHECKING declaration of a member the subclass provides.
fn property(name: &str, ty: &str) -> String {
    format!("@property\ndef {name}(self) -> {ty}: ...\n")
}

/// The arguments a native call takes, by its parameters: positional until one is left to its
/// default, then by keyword. A required parameter nothing gives is an error.
fn sent(params: &[Param], value: impl Fn(&Param) -> Option<String>) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut keyword = false;
    for p in params.iter().filter(|p| !p.ty.hidden()) {
        let n = p.name.trim_start_matches("r#");
        keyword |= p.kw_only;
        match value(p) {
            Some(v) if keyword => out.push(format!("{n}={v}")),
            Some(v) => out.push(v),
            None if p.default.is_some() || matches!(p.ty, Ty::Maybe(_)) => keyword = true,
            None => return Err(format!("nothing gives the native's required {n}")),
        }
    }
    Ok(out)
}

/// `def name(params) -> ret:` at class indentation, one parameter a line past 100 columns.
fn def(name: &str, asynchronous: bool, params: &[String], ret: &str) -> String {
    let prefix = if asynchronous { "async " } else { "" };
    let flat = format!("    {prefix}def {name}({}) -> {ret}:\n", params.join(", "));
    if flat.len() <= 101 {
        return flat;
    }
    let lines: String = params.iter().map(|p| format!("        {p},\n")).collect();
    format!("    {prefix}def {name}(\n{lines}    ) -> {ret}:\n")
}

fn with_doc(mut text: String, doc: &[String], body: &str) -> String {
    crate::python::docstring(&mut text, doc, 8);
    text.push_str(body);
    text
}

/// `head(arguments)` as an expression starting at column `indent`, wrapped one argument a line.
fn inline_call(head: &str, arguments: &[String], indent: usize) -> String {
    let flat = format!("{head}({})", arguments.join(", "));
    if indent + flat.len() <= 100 {
        return flat;
    }
    let pad = " ".repeat(8);
    let lines: String = arguments.iter().map(|a| format!("{pad}    {a},\n")).collect();
    format!("{head}(\n{lines}{pad})")
}

/// `head(arguments)` as one statement at `indent`, wrapped one argument a line past 100 columns.
fn call(head: &str, arguments: &[String], indent: usize) -> String {
    let pad = " ".repeat(indent);
    let flat = format!("{pad}{head}({})\n", arguments.join(", "));
    if flat.len() <= 101 {
        return flat;
    }
    let lines: String = arguments.iter().map(|a| format!("{pad}    {a},\n")).collect();
    format!("{pad}{head}(\n{lines}{pad})\n")
}

#[cfg(test)]
mod tests {
    fn emit(source: &str) -> Result<String, String> {
        let declarations = crate::decl::tests::read(source)?;
        Ok(super::emit(&declarations, &[])?.remove(0).text)
    }

    const PACER: &str = r#"
        use std::sync::Arc;
        #[crate::host(proxy(host = "livekit.agents.tokenize.SentenceStream"))]
        pub trait Sentences: Send + Sync {
            #[host(notify, name = "push_text")]
            fn push_text(&self, text: String);
        }
        #[crate::host(record(module = "tokenizer", extract, emit))]
        pub struct Token { pub token: String }
        #[crate::host(class(name = "_core_Paced"))]
        pub trait Paced: Sized {
            #[host(env)]
            type Env;
            #[host(new)]
            fn new(env: &Self::Env, sent_stream: Arc<dyn Sentences>, #[host(kw, default = 300)] limit: u32) -> Self;
            #[host(await)]
            async fn next(&self) -> Result<Option<String>, crate::errors::ApiError>;
            #[host(await)]
            async fn aclose(&self);
            #[host(call)]
            fn begin(&self, text: String) -> u64;
            #[host(call)]
            fn end(&self, token: u64);
        }
        /// A paced stream.
        #[crate::host(facade(path = "livekit.agents.tts", base = "tokenize.SentenceStream", native = Paced))]
        pub trait Wrapper {
            #[host(delegate)]
            type SentStream: Sentences;
            #[host(new)]
            fn new(sent_stream: Self::SentStream, #[host(kw, default = 300)] limit: u32) -> Self;
            #[host(via = SentStream)]
            fn push_text(&self, text: String);
            #[host(none_ends_iteration, wrap = Token::token)]
            async fn __anext__(&self) -> Option<Token>;
            async fn aclose(&self);
            #[host(native = begin, exit = end)]
            fn hold(&self, text: String) -> u64;
        }
    "#;

    // Catches: the constructor's ABI default retyped, the delegate not kept, or the native built
    // from positional arguments past a keyword.
    #[test]
    fn the_constructor_takes_the_abi_defaults_keeps_the_delegate_and_builds_the_native() {
        let text = emit(PACER).unwrap();
        assert!(text.contains("class Wrapper(SentenceStream):\n    \"\"\"A paced stream.\"\"\""));
        assert!(text.contains("from ..tokenize import SentenceStream\n"));
        assert!(text.contains(
            "    def __init__(self, sent_stream: SentenceStream, *, limit: int = 300) -> None:\n        super().__init__()\n        self._sent_stream = sent_stream\n        self._native = _core._core_Paced(sent_stream, limit=limit)\n"
        ));
    }

    #[test]
    fn a_delegate_method_forwards_and_none_ends_iteration_wraps_the_value() {
        let text = emit(PACER).unwrap();
        assert!(text.contains(
            "    def push_text(self, text: str) -> None:\n        self._sent_stream.push_text(text)\n"
        ));
        assert!(text.contains(
            "    async def __anext__(self) -> Token:\n        if (value := await self._native.next()) is None:\n            raise StopAsyncIteration\n        return Token(token=value)\n"
        ));
        assert!(text.contains("from .tokenizer import Token\n"));
        assert!(text.contains(
            "    async def aclose(self) -> None:\n        await self._native.aclose()\n"
        ));
    }

    #[test]
    fn a_begin_end_pair_is_an_async_with() {
        let text = emit(PACER).unwrap();
        assert!(text.contains(
            "    @contextlib.asynccontextmanager\n    async def hold(self, text: str) -> AsyncIterator[int]:\n        value = self._native.begin(text)\n        try:\n            yield value\n        finally:\n            self._native.end(value)\n"
        ));
        assert!(text.contains("import contextlib\n"));
    }

    #[test]
    fn a_facade_method_the_native_lacks_is_refused() {
        let source = PACER.replace(
            "async fn aclose(&self);\n            #[host(native",
            "async fn close(&self);\n            #[host(native",
        );
        assert!(emit(&source).unwrap_err().contains("has no method close"));
    }

    const STREAM: &str = r#"
        #[crate::host(class(name = "Chain"))]
        pub trait Chain: Sized {
            #[host(env)]
            type Env;
            #[host(new)]
            fn new(env: &Self::Env, members: Vec<String>, #[host(kw, default = 1.0)] timeout: f64) -> Self;
            #[host(getter)]
            fn model(&self) -> String;
            #[host(await)]
            async fn synthesize(&self, input_text: String, #[host(default = None)] conn_options: Option<u32>) -> Result<u64, crate::errors::ApiError>;
        }
        #[crate::host(facade(path = "livekit.agents.tts", extended, native = Chain, legs(synthesize), keep(members)))]
        pub trait Adapter {
            #[host(new, resolve(members = ".._bindings.core_legs"))]
            fn new(members: Vec<String>, #[host(kw, default = 1.0)] timeout: f64) -> Self;
            #[host(getter)]
            fn model(&self) -> String;
        }
        #[crate::host(facade(path = "livekit.agents.tts", base = "tts.ChunkedStream", native = Chain, stream(op = synthesize, owner = "_tts", native_retries)))]
        pub trait AdapterStream {}
    "#;

    // Catches: an engine not built by the constructor, or a stream retried on top of a native
    // chain.
    #[test]
    fn a_stream_opens_its_owners_native_operation_and_the_base_pumps_it() {
        let text = emit(STREAM).unwrap();
        assert!(text.contains(
            "class AdapterStream(ChunkedStream):\n    _core_native_retries = True\n    _tts: tts.Adapter\n\n    async def _core_open(self) -> int:\n        return await self._tts._native.synthesize(self._input_text, self._conn_options)\n\n    _run = ChunkedStream._core_run_native\n"
        ));
        assert!(text.contains("    from .. import tts\n"));
        assert!(text.contains(
            "    def __init__(self, members: Sequence[str], *, timeout: float = 1.0) -> None:\n        self._native = _core.Chain(core_legs(members), timeout=timeout)\n        self._members = members\n"
        ));
        assert!(text.contains("replaced = _bindings.overrides(self, Adapter, \"synthesize\")\n"));
        assert!(text.contains(
            "    @property\n    def model(self) -> str:\n        return self._native.model\n"
        ));
    }

    // Catches: a per-call argument retyped beside its native operation, or a stream retried on
    // top of a native operation that retries itself.
    #[test]
    fn a_provider_method_takes_the_operations_per_call_arguments_and_the_stream_keeps_them() {
        let source = STREAM.replace(
            "async fn synthesize(&self, input_text: String,",
            "async fn listen(&self, #[host(kw, default = None)] language: Option<String>, #[host(default = None)] offset: Option<f64>, #[host(default = None)] conn_options: Option<u32>) -> Result<u64, crate::errors::ApiError>;\n            #[host(await)]\n            async fn synthesize(&self, input_text: String,",
        ) + r#"
        #[crate::host(facade(path = "livekit.agents.tts", base = "tts.ChunkedStream", native = Chain, stream(op = listen, owner = "_tts", native_retries, method = "listen", base_args(conn_options), state(offset), default(conn_options = "..types.DEFAULT"), not_given)))]
        pub trait Listening {}
        "#;
        let text = emit(&source).unwrap();
        assert!(text.contains(
            "    def listen(\n        self,\n        *,\n        language: NotGivenOr[str] = NOT_GIVEN,\n        conn_options: int = DEFAULT,\n    ) -> ChunkedStream:\n        return Listening(\n            tts=cast(\"tts.Adapter\", self),\n            language=language if is_given(language) else None,\n            conn_options=conn_options,\n        )\n"
        ));
        assert!(text.contains(
            "    def __init__(self, *, tts: tts.Adapter, language: str | None, conn_options: int) -> None:\n        super().__init__(tts=tts, conn_options=conn_options)\n        self._language: str | None = language\n"
        ));
        assert!(text.contains(
            "return await self._tts._native.listen(\n            language=self._language,\n            offset=self._offset,\n            conn_options=self._conn_options,\n        )\n"
        ));
        assert!(text.contains("from typing import TYPE_CHECKING, Any, cast\n"));
    }

    // Catches: the provider method taking the operation's name for a renamed parameter, a
    // parameter not given defaulting to `None` rather than the owner's policy, or a facade named
    // as its base subclassing itself.
    #[test]
    fn a_provider_method_renames_and_takes_the_owners_default() {
        let source = STREAM
            .replace("extended, native = Chain", "extended, name = \"TTS\", base = \"tts.TTS\", native = Chain")
            .replace(
                "stream(op = synthesize, owner = \"_tts\", native_retries)",
                "stream(op = synthesize, owner = \"_tts\", method = \"synthesize\", base_args(input_text, conn_options), rename(input_text = \"text\"), provider_default(conn_options))",
            );
        let text = emit(&source).unwrap();
        assert!(text.contains("class TTS(BaseTTS):\n"));
        assert!(text.contains(
            "    def synthesize(self, text: str, conn_options: NotGivenOr[int] = NOT_GIVEN) -> ChunkedStream:\n        return AdapterStream(\n            tts=cast(\"tts.TTS\", self),\n            input_text=text,\n            conn_options=conn_options if is_given(conn_options) else self._native.conn_options,\n        )\n"
        ));
    }
}
