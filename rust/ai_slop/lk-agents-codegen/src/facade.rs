//! `#[host(facade(..))]` traits as Python classes (docs/internals/rust-core.md "Facades"): one
//! module `_generated/<family>_facades.py` per declaring family. A class subclasses its `base`,
//! takes its `#[host(new)]` parameters with their ABI defaults and builds its native object from
//! them (`cached`: once, on first use; `per_loop`: once per running loop, from its `self._<param>`
//! attributes). A method forwards to the native method of its name, or `via` a delegate; a
//! stream's `_run` is its base's native pump over the owner's operation.

use crate::class_stub::{Dir, Imports, py_literal};
use crate::decl::{Build, Class, Declarations, FacadeClass, FacadeMethod, MethodKind, Param, Ty};
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
        header
            .push_str("from typing import TYPE_CHECKING, Any\n\nfrom .. import _bindings, _core\n");
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
        let (module, name) =
            dotted.rsplit_once('.').ok_or_else(|| format!("`{dotted}` names module.Name"))?;
        let public = format!("livekit.agents.{module}");
        let source = self
            .placed
            .get(&public)
            .and_then(|names| names.iter().find(|(_, n)| n == name))
            .map(|(source, _)| source.clone());
        match source {
            Some(source) if source.trim_start_matches('.') == self.module => {}
            Some(source) => self.runtime.add(&source, name),
            None => self.runtime.add(&format!("..{module}"), name),
        }
        Ok(name.to_owned())
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
        let native = self.native(&facade.native)?.clone();
        let base = facade.base.as_deref().map(|b| self.runtime_name(b)).transpose()?;
        let mut text = format!(
            "\n\nclass {}{}:\n",
            facade.name,
            base.map(|b| format!("({b})")).unwrap_or_default()
        );
        let mut body = String::new();
        if crate::python::docstring(&mut body, &facade.doc, 4) {
            body.push('\n');
        }
        let new = facade.methods.iter().find(|m| m.method.kind == MethodKind::New);
        let mut declared = Vec::new();
        if let Some(stream) = &facade.stream {
            body.push_str(&self.stream(facade, &native, stream)?);
            text.push_str(&body);
            return Ok(text);
        }
        for delegate in &facade.delegates {
            let in_new =
                new.is_some_and(|n| n.delegated.iter().any(|(_, a)| *a == delegate.attribute));
            if !in_new {
                let ty = annotation(&delegate.ty, Dir::In, &mut self.typing);
                declared.push(property(&delegate.attribute, &ty));
            }
        }
        let build = self.build(facade, &native, new.is_some(), &mut declared)?;
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
        if let Some(new) = new {
            members.push(self.constructor(facade, &native, new)?);
        }
        members.extend(build);
        for method in facade.methods.iter().filter(|m| m.method.kind != MethodKind::New) {
            members.push(
                self.method(&native, method)
                    .map_err(|e| format!("{}.{}: {e}", facade.name, method.method.name))?,
            );
        }
        body.push_str(&members.join("\n"));
        if body.trim().is_empty() {
            body.push_str("    pass\n");
        }
        text.push_str(&body);
        Ok(text)
    }

    /// The `_native` member (when not assigned by `__init__`), the leg and availability members.
    fn build(
        &mut self,
        facade: &FacadeClass,
        native: &Class,
        has_new: bool,
        declared: &mut Vec<String>,
    ) -> Result<Vec<String>, String> {
        let name = &native.name;
        let mut members = Vec::new();
        let constructor = native.methods.iter().find(|m| m.kind == MethodKind::New);
        let observer = facade.availability.as_ref().map(|a| a.observer.as_str());
        match facade.build {
            Build::Eager if has_new => {}
            // The handwritten subclass sets it.
            Build::Eager => declared.push(format!("_native: _core.{name}\n")),
            Build::Cached | Build::PerLoop => {
                let constructor =
                    constructor.ok_or_else(|| format!("_core.{name} has no #[host(new)]"))?;
                let mut fields = String::new();
                let mut arguments = Vec::new();
                let mut keyword = false;
                for p in constructor.params.iter().filter(|p| !p.ty.hidden()) {
                    let n = p.name.trim_start_matches("r#");
                    keyword |= p.kw_only;
                    let value = if observer == Some(n) {
                        "self".to_owned()
                    } else {
                        let ty = match facade.annotations.iter().find(|(a, _)| a == n) {
                            Some((_, text)) => self.typing.annotation(text, ""),
                            None => annotation(&p.ty, Dir::Out, &mut self.typing),
                        };
                        fields.push_str(&format!("_{n}: {ty}\n"));
                        match facade.resolve.iter().find(|(a, _)| a == n) {
                            Some((_, function)) => {
                                format!("{}(self._{n})", self.callable(function)?)
                            }
                            None => format!("self._{n}"),
                        }
                    };
                    arguments.push(if keyword { format!("{n}={value}") } else { value });
                }
                for (n, _) in facade.annotations.iter().chain(&facade.resolve) {
                    if !constructor.params.iter().any(|p| &p.name == n) {
                        return Err(format!("_core.{name}() takes no {n}"));
                    }
                }
                declared.push(fields);
                members.push(match facade.build {
                    Build::Cached => {
                        self.stdlib.insert("functools");
                        format!(
                            "    @functools.cached_property\n    def _native(self) -> _core.{name}:\n{}",
                            call(&format!("return _core.{name}"), &arguments, 8)
                        )
                    }
                    _ => {
                        let lines: String =
                            arguments.iter().map(|a| format!("                {a},\n")).collect();
                        format!(
                            "    @property\n    def _native(self) -> _core.{name}:\n        return _bindings.on_loop(\n            self,\n            lambda: _core.{name}(\n{lines}            ),\n        )\n"
                        )
                    }
                });
            }
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
            let constructor =
                constructor.ok_or_else(|| format!("_core.{name} has no #[host(new)]"))?;
            let parameter = constructor
                .params
                .iter()
                .find(|p| p.name == availability.observer)
                .ok_or_else(|| "availability observes through no parameter".to_owned())?;
            if !constructor.params.iter().any(|p| p.name == "members") {
                return Err("availability names entries of `members`".into());
            }
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
            self.typing.add(".llm_fallback", "AvailabilityChanged");
            declared.push(self.observed(parameter)?);
            declared.push("def emit(self, event: str, *args: Any) -> None: ...\n".into());
            let event = format!("{:?}", availability.event);
            let value = format!("{module}.{record}(self._members[ev.index], ev.available)");
            members.push(format!(
                "    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:\n{}",
                call("self.emit", &[event, value], 8)
            ));
        }
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
        for call in proxy.methods.iter().filter(|m| m.name != "_core_availability_changed") {
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
        if facade.base.is_some() {
            body.push_str("        super().__init__()\n");
        }
        for p in new.method.params.iter().filter(|p| !p.ty.hidden()) {
            let n = p.name.trim_start_matches("r#");
            match new.delegated.iter().find(|(d, _)| d == n) {
                Some((_, attribute)) => body.push_str(&format!("        self.{attribute} = {n}\n")),
                None if facade.build != Build::Eager => {
                    body.push_str(&format!("        self._{n} = {n}\n"))
                }
                None => {}
            }
        }
        if facade.build == Build::Eager {
            let constructor = native
                .methods
                .iter()
                .find(|m| m.kind == MethodKind::New)
                .ok_or_else(|| format!("_core.{} has no #[host(new)]", native.name))?;
            let observer = facade.availability.as_ref().map(|a| a.observer.as_str());
            let arguments = sent(&constructor.params, |p| {
                let n = p.name.trim_start_matches("r#");
                if observer == Some(n) {
                    return Some("self".to_owned());
                }
                values.iter().find(|(v, _)| v == n).map(|(_, value)| value.clone())
            })?;
            body.push_str(&call(&format!("self._native = _core.{}", native.name), &arguments, 8));
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
        let (target, called) = match &method.via {
            Some(delegate) => {
                let called = match &delegate.ty {
                    Ty::Instance { class, .. } => Some(
                        self.native(class)?
                            .methods
                            .iter()
                            .find(|n| n.name == method.native)
                            .cloned()
                            .ok_or_else(|| {
                                format!("_core.{class} has no method {}", method.native)
                            })?,
                    ),
                    _ => None,
                };
                (format!("self.{}", delegate.attribute), called)
            }
            None => {
                let called = native
                    .methods
                    .iter()
                    .find(|n| n.name == method.native && n.kind != MethodKind::New)
                    .cloned()
                    .ok_or_else(|| {
                        format!("_core.{} has no method {}", native.name, method.native)
                    })?;
                ("self._native".to_owned(), Some(called))
            }
        };
        let origin = adapt.origin.as_deref().map(|o| self.callable(o)).transpose()?;
        let arguments = match &called {
            Some(called) => sent(&called.params, |p| {
                let n = p.name.trim_start_matches("r#");
                if n == "origin"
                    && let Some(origin) = &origin
                {
                    return Some(format!("{origin}()"));
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
        let mut ret = annotation(&m.ret.ok, Dir::Out, &mut self.typing);
        let mut body = String::new();
        if let Some(exit) = &method.exit {
            self.stdlib.insert("contextlib");
            self.typing.add("collections.abc", "AsyncIterator");
            let exit_method = native
                .methods
                .iter()
                .find(|n| &n.name == exit)
                .ok_or_else(|| format!("_core.{} has no method {exit}", native.name))?;
            let takes = exit_method.params.iter().any(|p| !p.ty.hidden());
            let unit = m.ret.ok == Ty::Unit;
            let enter = format!("{}{}", if awaited { "await " } else { "" }, invocation(8));
            body.push_str(&match unit {
                true => format!("        {enter}\n"),
                false => format!("        value = {enter}\n"),
            });
            let close = format!(
                "{}self._native.{exit}({})",
                if exit_method.kind == MethodKind::Await { "await " } else { "" },
                if takes { "value" } else { "" }
            );
            body.push_str(&format!(
                "        try:\n            yield{}\n        finally:\n            {close}\n",
                if unit { "" } else { " value" }
            ));
            let text = format!(
                "    @contextlib.asynccontextmanager\n{}",
                def(&m.name, true, &signature, &format!("AsyncIterator[{ret}]"))
            );
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
            let flat = format!("return {ret}({head}({}))", arguments.join(", "));
            match 8 + flat.len() <= 100 {
                true => flat,
                false => {
                    let lines: String =
                        arguments.iter().map(|a| format!("                {a},\n")).collect();
                    format!("return {ret}(\n            {head}(\n{lines}            )\n        )")
                }
            }
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
        if getter || m.kind == MethodKind::Getter {
            text = format!("    @property\n{text}");
        }
        Ok(with_doc(text, &m.doc, &body))
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
    ) -> Result<String, String> {
        let base = facade.base.as_deref().ok_or("a stream names its provider stream `base`")?;
        let base = base.rsplit('.').next().unwrap_or(base);
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
        let mut text = String::new();
        if stream.native_retries {
            text.push_str("    _core_native_retries = True\n");
        }
        text.push_str(&format!("    {}: {owner_class}\n\n", stream.owner));
        text.push_str(&format!("    async def _core_open(self) -> {opened}:\n"));
        text.push_str(&call(
            &format!("return await self.{}._native.{}", stream.owner, stream.op),
            &arguments,
            8,
        ));
        text.push_str(&format!("\n    _run = {base}._core_run_native\n"));
        Ok(text)
    }
}

/// A host annotation, a native class named through `_core` (a generated class may share its name).
fn annotation(ty: &Ty, dir: Dir, imports: &mut Imports) -> String {
    match ty {
        Ty::Instance { class, .. } | Ty::Handle { class, .. } | Ty::Shared { class, .. } => {
            format!("_core.{class}")
        }
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
        #[crate::host(facade(path = "livekit.agents.tts", extended, native = Chain, per_loop, resolve(members = ".._bindings.core_legs"), legs(synthesize)))]
        pub trait Adapter {
            #[host(getter)]
            fn model(&self) -> String;
        }
        #[crate::host(facade(path = "livekit.agents.tts", base = "tts.ChunkedStream", native = Chain, stream(op = synthesize, owner = "_tts", native_retries)))]
        pub trait AdapterStream {}
    "#;

    // Catches: a per-loop engine built once, or a stream retried on top of a native chain.
    #[test]
    fn a_stream_opens_its_owners_native_operation_and_the_base_pumps_it() {
        let text = emit(STREAM).unwrap();
        assert!(text.contains(
            "class AdapterStream(ChunkedStream):\n    _core_native_retries = True\n    _tts: tts.Adapter\n\n    async def _core_open(self) -> int:\n        return await self._tts._native.synthesize(self._input_text, self._conn_options)\n\n    _run = ChunkedStream._core_run_native\n"
        ));
        assert!(text.contains("    from .. import tts\n"));
        assert!(text.contains(
            "\n\nclass Adapter:\n    if TYPE_CHECKING:\n        _members: list[str]\n        _timeout: float\n"
        ));
        assert!(text.contains("lambda: _core.Chain(\n                core_legs(self._members),\n                timeout=self._timeout,\n            ),"));
        assert!(text.contains("replaced = _bindings.overrides(self, Adapter, \"synthesize\")\n"));
        assert!(text.contains(
            "    @property\n    def model(self) -> str:\n        return self._native.model\n"
        ));
    }
}
