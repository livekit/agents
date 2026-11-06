//! A facade on a class's `#[host(new)]`: the host mixin that builds that native class from its
//! host class's own attributes, one per running event loop (a native engine calls the host
//! objects it holds on the loop it was built on), or with `cached` once per host object, on first
//! use. Each native parameter `p` is read from `self._p`, declared for the type checker and set by
//! the host class's `__init__`, and passed through `resolve(p = "module.fn")` when given. The
//! class's other facades with the same module and class are the mixin's methods.
//!
//! `legs(a, b)`: the mixin's `_core_leg` hands the core this native engine (`leg = "m"`: the
//! value of the host's method `m()`) in place of the host object, unless the host class's
//! subclass replaces one of those provider methods.
//! `availability(observer = "p", event = "..", record = "module.Name")`: the host object is the
//! chain observer the constructor takes as `p`; an entry of `members` going unavailable or
//! recovering is emitted as `event`, a `record(entry, available)`.

use crate::class_stub::{Dir, Imports};
use crate::decl::{Class, Declarations, Facade, Method, MethodKind, Ty};
use crate::python::Out;

/// Whether `facade` is a method of a constructor facade's host mixin.
pub fn hosts(declarations: &Declarations, facade: &Facade) -> bool {
    declarations.classes.iter().flat_map(|class| &class.methods).any(|method| {
        method.kind == MethodKind::New
            && method
                .facade
                .as_ref()
                .is_some_and(|f| f.module == facade.module && f.class == facade.class)
    })
}

pub fn emit(declarations: &Declarations) -> Result<Vec<Out>, String> {
    let mut modules = std::collections::BTreeMap::<&str, Vec<_>>::new();
    for class in &declarations.classes {
        for method in &class.methods {
            if let Some(facade) = method.facade.as_ref().filter(|_| method.kind == MethodKind::New)
            {
                modules.entry(&facade.module).or_default().push((class, method, facade));
            }
        }
    }
    let mut out = Vec::new();
    for (module, hosts) in modules {
        let mut runtime = Imports::new(declarations);
        let mut typing = Imports::new(declarations);
        let mut body = String::new();
        for (class, method, facade) in hosts {
            body.push_str(&host(declarations, class, method, facade, &mut runtime, &mut typing)?);
        }
        let mut text = "# GENERATED FILE -- DO NOT EDIT.\n# From #[host(new, facade)] declarations in lk-agents-abi.\n\nfrom __future__ import annotations\n\nimport functools\nfrom typing import TYPE_CHECKING, Any\n\nfrom .. import _bindings, _core\n".to_owned();
        text.push_str(&runtime.render(&[]));
        // `Any` is imported at runtime: the declarations below name it too.
        typing.remove_module("typing");
        text.push_str("\nif TYPE_CHECKING:\n");
        for line in typing.render(&[]).lines() {
            match line.is_empty() {
                true => text.push('\n'),
                false => text.push_str(&format!("    {line}\n")),
            }
        }
        text.push_str(&body);
        out.push(Out { path: format!("{module}.py"), text });
    }
    Ok(out)
}

/// One host mixin class.
fn host(
    declarations: &Declarations,
    class: &Class,
    method: &Method,
    facade: &Facade,
    runtime: &mut Imports,
    typing: &mut Imports,
) -> Result<String, String> {
    let at = &method.rust_name;
    let observer = facade.availability.as_ref().map(|a| a.observer.as_str());
    for (name, _) in facade.annotations.iter().chain(&facade.resolve) {
        if !method.params.iter().any(|p| &p.name == name) {
            return Err(format!("{at}: facade names no parameter {name}"));
        }
    }
    let mut declared = String::new();
    let mut arguments = Vec::new();
    let mut keyword = false;
    for parameter in method.params.iter().filter(|p| !p.ty.hidden()) {
        let name = parameter.name.trim_start_matches("r#");
        keyword |= parameter.kw_only;
        let value = if observer == Some(name) {
            "self".to_owned()
        } else {
            let annotation = match facade.annotations.iter().find(|(p, _)| p == name) {
                Some((_, text)) => typing.annotation(text, ""),
                None => crate::class_stub::annotation(&class.name, &parameter.ty, Dir::Out, typing),
            };
            declared.push_str(&format!("        _{name}: {annotation}\n"));
            match facade.resolve.iter().find(|(p, _)| p == name) {
                Some((_, function)) => {
                    format!("{}(self._{name})", imported(runtime, function, at)?)
                }
                None => format!("self._{name}"),
            }
        };
        arguments.push(if keyword { format!("{name}={value}") } else { value });
    }
    let (target, native, mixin) = (&facade.target, &class.name, &facade.class);
    let mut members = String::new();
    if let Some(availability) = &facade.availability {
        if !method.params.iter().any(|p| p.name == availability.observer) {
            return Err(format!("{at}: facade availability observes through no parameter"));
        }
        if !method.params.iter().any(|p| p.name == "members") {
            return Err(format!("{at}: facade availability names entries of `members`"));
        }
        // Qualified by its module: the adapters' events share one name.
        let (module, name) = availability
            .record
            .rsplit_once('.')
            .ok_or_else(|| format!("{at}: `{}` names module.item", availability.record))?;
        let (package, module) = module.rsplit_once('.').unwrap_or(("", module));
        let package = match package.trim_start_matches('.') {
            "" => format!("{package}."),
            _ => package.to_owned(),
        };
        runtime.add(&package, module);
        typing.add(".llm_fallback", "AvailabilityChanged");
        declared.push('\n');
        declared.push_str(&observed(declarations, class, method, &availability.observer, typing)?);
        declared.push_str("        def emit(self, event: str, *args: Any) -> None: ...\n");
        let event = format!("{:?}", availability.event);
        let record = format!("{module}.{name}(self._members[ev.index], ev.available)");
        members.push_str(&format!(
            "\n    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:\n{}",
            call("self.emit", &[event, record], 8)
        ));
    }
    let build = match facade.cached {
        true => {
            let lines: String = arguments.iter().map(|a| format!("            {a},\n")).collect();
            format!(
                "    @functools.cached_property\n    def {target}(self) -> _core.{native}:\n        return _core.{native}(\n{lines}        )\n"
            )
        }
        false => {
            let lines: String =
                arguments.iter().map(|a| format!("                {a},\n")).collect();
            format!(
                "    @property\n    def {target}(self) -> _core.{native}:\n        return _bindings.on_loop(\n            self,\n            lambda: _core.{native}(\n{lines}            ),\n        )\n"
            )
        }
    };
    let leg = match &facade.leg {
        Some(leg) => {
            declared.push_str(&format!("\n        def {leg}(self) -> _core.{native}: ...\n"));
            format!("self.{leg}()")
        }
        None => format!("self.{target}"),
    };
    let mut text = format!("\n\nclass {mixin}:\n    if TYPE_CHECKING:\n{declared}\n{build}");
    if !facade.legs.is_empty() {
        let mut arguments = vec!["self".to_owned(), mixin.clone()];
        arguments.extend(facade.legs.iter().map(|l| format!("{l:?}")));
        text.push_str(&format!(
            "\n    def _core_leg(self) -> Any:\n        \"\"\"This engine, unless a subclass replaces what it would bypass.\"\"\"\n{}        return self if replaced else {leg}\n",
            call("replaced = _bindings.overrides", &arguments, 8)
        ));
    }
    text.push_str(&members);
    for other in &class.methods {
        if let Some(delegated) = other.facade.as_ref().filter(|f| {
            other.kind != MethodKind::New && f.module == facade.module && f.class == *mixin
        }) {
            if delegated.target != *target {
                return Err(format!("{}: facade targets {target}", other.rust_name));
            }
            text.push_str(&crate::facade::delegate(class, other, delegated, runtime)?);
        }
    }
    Ok(text)
}

/// The observer proxy's other calls, which the host class answers itself.
fn observed(
    declarations: &Declarations,
    class: &Class,
    method: &Method,
    observer: &str,
    typing: &mut Imports,
) -> Result<String, String> {
    let parameter = method.params.iter().find(|p| p.name == observer).expect("checked");
    let declared = match &parameter.ty {
        Ty::Optional(inner) => inner.as_ref(),
        other => other,
    };
    let Ty::Proxy { declared: Some((family, name)), .. } = declared else {
        return Err(format!("{}: facade availability observer is no proxy", method.rust_name));
    };
    let proxy = declarations
        .proxies
        .iter()
        .find(|p| &p.family == family && &p.rust_name == name)
        .ok_or_else(|| format!("{}: no proxy {name}", method.rust_name))?;
    let mut text = String::new();
    for call in proxy.methods.iter().filter(|m| m.name != "_core_availability_changed") {
        let mut params = vec!["self".to_owned()];
        for p in call.params.iter().filter(|p| !p.ty.hidden()) {
            let ty = crate::class_stub::annotation(&class.name, &p.ty, Dir::Out, typing);
            params.push(format!("{}: {ty}", p.name));
        }
        text.push_str(&format!(
            "        def {}({}) -> None: ...\n\n",
            call.name,
            params.join(", ")
        ));
    }
    Ok(text)
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

/// `module.Name` imported into the generated module, as the name it is bound to.
fn imported(imports: &mut Imports, dotted: &str, at: &str) -> Result<String, String> {
    let (module, name) =
        dotted.rsplit_once('.').ok_or_else(|| format!("{at}: `{dotted}` names module.item"))?;
    imports.add(module, name);
    Ok(name.to_owned())
}

#[cfg(test)]
mod tests {
    fn declarations() -> crate::decl::Declarations {
        crate::decl::tests::read(
            r#"
            #[crate::host(proxy(host = "livekit.agents.FallbackListener"))]
            pub trait Observer: Send + Sync {
                #[host(notify, name = "_core_availability_changed")]
                fn changed(&self, index: usize);
                #[host(notify, name = "_core_gone")]
                fn gone(&self);
            }
            #[crate::host(class(name = "Chain"))]
            pub trait Chain: Sized {
                #[host(env)]
                type Env;
                #[host(new, facade(module = "hosts", class = "ChainHost", target = "_native", legs(chat), resolve(members = "..m.legs"), annotation(rate = "int"), availability(observer = "on_availability", event = "changed", record = ".events.Changed")))]
                fn new(env: &Self::Env, members: Vec<String>, #[host(kw, default = 1.0)] timeout: f64, #[host(default = None)] rate: Option<u32>, #[host(default = None)] on_availability: Option<std::sync::Arc<dyn Observer>>) -> Self;
            }
            "#,
        )
        .unwrap()
    }

    #[test]
    fn the_host_builds_its_engine_per_loop_from_its_attributes() {
        let text = super::emit(&declarations()).unwrap().remove(0).text;
        assert!(text.contains("class ChainHost:\n    if TYPE_CHECKING:\n        _members: list[str]\n        _timeout: float\n        _rate: int\n"));
        assert!(text.contains(
            "            lambda: _core.Chain(\n                legs(self._members),\n                timeout=self._timeout,\n                rate=self._rate,\n                on_availability=self,\n            ),"
        ));
        assert!(text.contains("from ..m import legs\n"));
    }

    #[test]
    fn a_replaced_leg_keeps_the_host_object_and_availability_is_emitted() {
        let text = super::emit(&declarations()).unwrap().remove(0).text;
        assert!(text.contains("replaced = _bindings.overrides(self, ChainHost, \"chat\")\n"));
        assert!(text.contains("return self if replaced else self._native"));
        assert!(text.contains(
            "self.emit(\"changed\", events.Changed(self._members[ev.index], ev.available))\n"
        ));
        assert!(text.contains("from . import events\n"));
        assert!(text.contains("        def _core_gone(self) -> None: ...\n"));
    }

    #[test]
    fn a_cached_engine_is_built_once_and_its_methods_join_the_mixin() {
        let declarations = crate::decl::tests::read(
            r#"
            #[crate::host(class(name = "Client"))]
            pub trait Client: Sized {
                #[host(env)]
                type Env;
                #[host(new, facade(module = "hosts", class = "ClientHost", target = "_native", cached, legs(chat), leg = "_core_native"))]
                fn new(env: &Self::Env, model: String) -> Self;
                #[host(getter, facade(module = "hosts", class = "ClientHost", target = "_native"))]
                fn model(&self) -> String;
            }
            "#,
        )
        .unwrap();
        let text = super::emit(&declarations).unwrap().remove(0).text;
        assert!(text.contains(
            "    @functools.cached_property\n    def _native(self) -> _core.Client:\n        return _core.Client(\n            self._model,\n        )\n"
        ));
        assert!(text.contains("        def _core_native(self) -> _core.Client: ...\n"));
        assert!(text.contains("return self if replaced else self._core_native()\n"));
        assert!(text.contains(
            "    @property\n    def model(self) -> str:\n        return self._native.model\n"
        ));
        assert!(crate::facade::emit(&declarations).unwrap().is_empty());
    }
}
