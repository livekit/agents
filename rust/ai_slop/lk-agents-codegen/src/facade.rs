//! Explicit Python forwarding signatures over a facade's declared native target.

use crate::decl::{Class, Declarations, Facade, Method, MethodKind, Param, Ty};
use crate::python::Out;

pub fn emit(declarations: &Declarations) -> Result<Vec<Out>, String> {
    let mut groups = std::collections::BTreeMap::<(String, String), Vec<_>>::new();
    for class in &declarations.classes {
        for method in &class.methods {
            if let Some(facade) = &method.facade {
                groups
                    .entry((facade.module.clone(), facade.class.clone()))
                    .or_default()
                    .push((class, method, facade));
            }
        }
    }
    let mut outputs = Vec::new();
    for ((module, name), methods) in groups {
        let mut imports = crate::class_stub::Imports::new(declarations);
        let mut body = format!("class {name}:\n");
        let mut targets = std::collections::BTreeMap::new();
        for (class, method, facade) in methods {
            targets.insert(facade.target.clone(), class.name.clone());
            body.push_str(&delegate(class, method, facade, &mut imports)?);
        }
        let mut header = "# GENERATED FILE -- DO NOT EDIT.\n# From #[host(facade)] declarations in lk-agents-abi.\n\nfrom __future__ import annotations\n\nimport asyncio\nfrom typing import TYPE_CHECKING\n\nfrom .. import _core\n".to_owned();
        for (module, item) in imports.entries() {
            header.push_str(&format!("from {module} import {item}\n"));
        }
        let mut declared = String::new();
        for (target, class) in targets {
            if !declared.is_empty() {
                declared.push('\n');
            }
            declared.push_str(&format!(
                "    @property\n    def {target}(self) -> _core.{class}: ...\n"
            ));
        }
        let marker = format!("class {name}:\n");
        body = body.replacen(
            &marker,
            &format!(
                "{marker}    if TYPE_CHECKING:\n\n{}",
                declared
                    .lines()
                    .map(|line| if line.is_empty() {
                        "\n".to_owned()
                    } else {
                        format!("    {line}\n")
                    })
                    .collect::<String>()
            ),
            1,
        );
        outputs.push(Out { path: format!("{module}.py"), text: format!("{header}\n\n{body}") });
    }
    Ok(outputs)
}

/// `module.Name` imported into the generated module, as the name it is bound to.
fn imported(
    imports: &mut crate::class_stub::Imports,
    dotted: &str,
    at: &str,
) -> Result<String, String> {
    let (module, name) =
        dotted.rsplit_once('.').ok_or_else(|| format!("{at}: `{dotted}` names module.item"))?;
    imports.add(module, name);
    Ok(name.to_owned())
}

/// One parameter as the delegate takes it and as it sends it on.
struct Argument {
    name: String,
    /// `None`: not in the delegate's signature.
    parameter: Option<String>,
    keyword: bool,
    value: String,
}

fn argument(
    class: &Class,
    parameter: &Param,
    shown: Option<(String, bool)>,
    facade: &Facade,
    imports: &mut crate::class_stub::Imports,
) -> Result<Argument, String> {
    let at = &parameter.name;
    let name = parameter.name.trim_start_matches("r#").to_owned();
    let keyword = parameter.kw_only;
    if facade.origin.is_some() && name == "origin" {
        let origin = imported(imports, facade.origin.as_deref().expect("checked"), at)?;
        return Ok(Argument { name, parameter: None, keyword, value: format!("{origin}()") });
    }
    let lookup = |list: &[(String, String)]| {
        list.iter().find(|(p, _)| *p == name).map(|(_, value)| value.clone())
    };
    let annotation = lookup(&facade.annotations).map(|a| imports.annotation(&a, ""));
    let resolve = lookup(&facade.resolve).map(|f| imported(imports, &f, at)).transpose()?;
    let sent = match &resolve {
        Some(resolve) => format!("{resolve}({name})"),
        None => name.clone(),
    };
    if let Ty::Maybe(inner) = &parameter.ty {
        // Omission is the native argument's own state: `NOT_GIVEN` passes through unchanged.
        imports.add("..types", "NOT_GIVEN");
        imports.add("..types", "NotGivenOr");
        let inner = annotation.unwrap_or_else(|| {
            crate::class_stub::annotation(&class.name, inner, crate::class_stub::Dir::In, imports)
        });
        let parameter = Some(format!("{name}: NotGivenOr[{inner}] = NOT_GIVEN"));
        let value = match &resolve {
            Some(resolve) => {
                imports.add("..utils", "is_given");
                format!("{resolve}({name}) if is_given({name}) else {name}")
            }
            None => name.clone(),
        };
        return Ok(Argument { name, parameter, keyword, value });
    }
    let defaults_none =
        parameter.default.as_ref().map(crate::class_stub::py_literal).as_deref() == Some("None");
    if let (true, true, Ty::Optional(inner)) = (facade.not_given, defaults_none, &parameter.ty) {
        imports.add("..types", "NOT_GIVEN");
        imports.add("..types", "NotGivenOr");
        imports.add("..utils", "is_given");
        let inner = annotation.unwrap_or_else(|| {
            crate::class_stub::annotation(&class.name, inner, crate::class_stub::Dir::In, imports)
        });
        let parameter = Some(format!("{name}: NotGivenOr[{inner}] = NOT_GIVEN"));
        let value = format!("{sent} if is_given({name}) else None");
        return Ok(Argument { name, parameter, keyword, value });
    }
    let parameter = match (annotation, &parameter.default) {
        (Some(annotation), Some(default)) => {
            format!("{name}: {annotation} = {}", crate::class_stub::py_literal(default))
        }
        (Some(annotation), None) => format!("{name}: {annotation}"),
        (None, _) => shown.map(|(text, _)| text).expect("a shown parameter"),
    };
    Ok(Argument { name, parameter: Some(parameter), keyword, value: sent })
}

/// The Python delegate of one facade method.
fn delegate(
    class: &Class,
    method: &Method,
    facade: &Facade,
    imports: &mut crate::class_stub::Imports,
) -> Result<String, String> {
    if !matches!(method.kind, MethodKind::Call | MethodKind::Await | MethodKind::Getter) {
        return Err(format!("{}: facade needs a call, await or getter", method.rust_name));
    }
    if facade.future && method.kind != MethodKind::Await {
        return Err(format!("{}: facade future needs an await", method.rust_name));
    }
    if facade.returns.is_some() && method.kind != MethodKind::Call {
        return Err(format!("{}: facade returns needs a call", method.rust_name));
    }
    // Only what the delegate's signature shows is imported: a replaced annotation's native type,
    // the filled origin's and a wrapped result's are not.
    let shown = |p: &Param| {
        facade.omit.as_ref() != Some(&p.name)
            && !(facade.origin.is_some() && p.name == "origin")
            && !facade.annotations.iter().any(|(name, _)| *name == p.name)
    };
    let mut visible = method.clone();
    visible.params.retain(|p| !p.ty.hidden() && shown(p));
    let (params, mut ret) = match &facade.returns {
        Some(returns) => (
            crate::class_stub::parameters(class, &visible, imports),
            imported(imports, returns, &method.name)?,
        ),
        None => crate::class_stub::signature_parts(class, &visible, imports),
    };
    let mut shown: std::collections::BTreeMap<_, _> =
        visible.params.iter().map(|p| p.name.clone()).zip(params).collect();
    let mut arguments = Vec::new();
    for native in method.params.iter().filter(|p| !p.ty.hidden()) {
        if facade.omit.as_ref() == Some(&native.name) {
            if native.default.is_none() {
                return Err(format!(
                    "{}: facade cannot omit required {}",
                    method.name, native.name
                ));
            }
            continue;
        }
        let part = shown.remove(&native.name);
        arguments.push(argument(class, native, part, facade, imports)?);
    }
    if let Some(omit) = &facade.omit
        && !method.params.iter().any(|p| &p.name == omit)
    {
        return Err(format!("{}: facade omits no parameter {omit}", method.name));
    }
    for (name, _) in facade.annotations.iter().chain(&facade.resolve) {
        if !method.params.iter().any(|p| &p.name == name) {
            return Err(format!("{}: facade names no parameter {name}", method.name));
        }
    }
    let mut signature = vec!["self".to_owned()];
    let mut sent = Vec::new();
    let mut star = false;
    for Argument { name, parameter, keyword, value } in arguments {
        if keyword && !star && parameter.is_some() {
            signature.push("*".into());
            star = true;
        }
        sent.push(if keyword { format!("{name}={value}") } else { value });
        signature.extend(parameter);
    }
    let target = format!("self.{}.{}", facade.target, method.name);
    let call = if method.kind == MethodKind::Getter {
        target.clone()
    } else {
        format!("{target}({})", sent.join(", "))
    };
    let prefix = if method.kind == MethodKind::Await && !facade.future { "async " } else { "" };
    let statement = if facade.returns.is_some() {
        if call.len() + ret.len() + 17 <= 100 {
            format!("return {ret}({call})")
        } else {
            let lines = sent.iter().map(|a| format!("                {a},\n")).collect::<String>();
            format!("return {ret}(\n            {target}(\n{lines}            )\n        )")
        }
    } else if facade.future {
        ret = format!("asyncio.Future[{ret}]");
        if call.len() + 38 <= 100 {
            format!("return asyncio.ensure_future({call})")
        } else {
            let lines = sent.iter().map(|a| format!("                {a},\n")).collect::<String>();
            format!(
                "return asyncio.ensure_future(\n            {target}(\n{lines}            )\n        )"
            )
        }
    } else if method.kind == MethodKind::Await && facade.discard {
        ret = "None".into();
        format!("await {call}")
    } else {
        let prefix = if method.kind == MethodKind::Await { "return await" } else { "return" };
        if 8 + prefix.len() + 1 + call.len() <= 100 {
            format!("{prefix} {call}")
        } else {
            let lines = sent.iter().map(|a| format!("            {a},\n")).collect::<String>();
            format!("{prefix} {target}(\n{lines}        )")
        }
    };
    let signature_line =
        format!("    {prefix}def {}({}) -> {ret}:\n", method.name, signature.join(", "));
    let mut body = String::new();
    if method.kind == MethodKind::Getter {
        body.push_str("\n    @property\n");
        body.push_str(&signature_line);
    } else if signature_line.len() <= 101 {
        body.push_str(&format!("\n{signature_line}"));
    } else {
        body.push_str(&format!("\n    {prefix}def {}(\n", method.name));
        for parameter in signature {
            body.push_str(&format!("        {parameter},\n"));
        }
        body.push_str(&format!("    ) -> {ret}:\n"));
    }
    crate::python::docstring(&mut body, &method.doc, 8);
    body.push_str(&format!("        {statement}\n"));
    Ok(body)
}

#[cfg(test)]
mod tests {
    fn declarations(method: &str) -> crate::decl::Declarations {
        crate::decl::tests::read(&format!(
            "#[crate::host(class(name = \"Native\"))] pub trait Native: Sized {{ {method} }}"
        ))
        .unwrap()
    }

    #[test]
    fn public_future_keeps_keyword_defaults_and_eager_native_call() {
        let declarations = declarations(
            r#"
            #[host(await, facade(module = "commands", class = "Commands", target = "_running", future))]
            async fn commit(&self, #[host(kw, default = 2.0)] timeout: f64) -> Result<String, crate::errors::ApiError>;
        "#,
        );
        let output = super::emit(&declarations).unwrap().remove(0).text;
        assert!(
            output.contains("def commit(self, *, timeout: float = 2.0) -> asyncio.Future[str]")
        );
        assert!(
            output.contains("return asyncio.ensure_future(self._running.commit(timeout=timeout))")
        );
        assert!(!output.contains("async def commit"));
    }

    #[test]
    fn a_private_optional_argument_is_not_added_to_the_public_signature() {
        let declarations = declarations(
            r#"
            #[host(await, facade(module = "commands", class = "Commands", target = "_running", omit = "hold"))]
            async fn idle(&self, #[host(kw, default = None)] hold: Option<u64>) -> Result<(), crate::errors::ApiError>;
        "#,
        );
        let output = super::emit(&declarations).unwrap().remove(0).text;
        assert!(output.contains("async def idle(self) -> None"));
        assert!(output.contains("return await self._running.idle()"));
        assert!(!output.contains("hold:"));
    }

    #[test]
    fn a_required_native_argument_cannot_be_silently_dropped() {
        let declarations = declarations(
            r#"
            #[host(call, facade(module = "commands", class = "Commands", target = "_running", omit = "hold"))]
            fn idle(&self, hold: u64);
        "#,
        );
        assert!(super::emit(&declarations).err().unwrap().contains("cannot omit required hold"));
    }

    #[test]
    fn a_native_getter_is_read_as_a_property_instead_of_called() {
        let declarations = declarations(
            r#"
            #[host(getter, facade(module = "commands", class = "Commands", target = "_owner"))]
            fn usage(&self) -> String;
        "#,
        );
        let output = super::emit(&declarations).unwrap().remove(0).text;
        assert!(output.contains("@property\n    def usage(self) -> str"));
        assert!(output.contains("return self._owner.usage\n"));
        assert!(!output.contains("self._owner.usage()"));
    }

    #[test]
    fn a_not_given_option_is_sent_as_none_when_the_caller_leaves_it_out() {
        let declarations = declarations(
            r#"
            #[host(call, facade(module = "commands", class = "Commands", target = "_running", not_given))]
            fn reply(&self, #[host(kw, default = None)] instructions: Option<String>, #[host(default = true)] add: bool);
        "#,
        );
        let output = super::emit(&declarations).unwrap().remove(0).text;
        assert!(output.contains(
            "def reply(self, *, instructions: NotGivenOr[str] = NOT_GIVEN, add: bool = True)"
        ));
        assert!(output.contains(
            "return self._running.reply(\n            instructions=instructions if is_given(instructions) else None,\n            add=add,\n        )"
        ));
        assert!(output.contains("from ..types import NOT_GIVEN\n"));
        assert!(output.contains("from ..utils import is_given\n"));
    }

    #[test]
    fn the_origin_is_the_callers_and_absent_from_the_signature() {
        let declarations = declarations(
            r#"
            #[host(call, facade(module = "commands", class = "Commands", target = "_running", origin = "..voice.calls._origin"))]
            fn say(&self, text: String, #[host(kw, default = None)] origin: Option<String>);
        "#,
        );
        let output = super::emit(&declarations).unwrap().remove(0).text;
        assert!(output.contains("def say(self, text: str) -> None"));
        assert!(output.contains("return self._running.say(text, origin=_origin())"));
        assert!(output.contains("from ..voice.calls import _origin\n"));
    }

    // Catches: the native result's own class imported beside the wrapper, shadowing it.
    #[test]
    fn a_returned_wrapper_replaces_the_native_result_type() {
        let declarations = declarations(
            r#"
            #[host(call, facade(module = "commands", class = "Commands", target = "_running", returns = "..voice.io.SpeechHandle"))]
            fn say(&self, text: String) -> crate::llm::chat::Chunk;
        "#,
        );
        let output = super::emit(&declarations).unwrap().remove(0).text;
        assert!(output.contains("def say(self, text: str) -> SpeechHandle:"));
        assert!(output.contains("return SpeechHandle(self._running.say(text))"));
        assert!(output.contains("from ..voice.io import SpeechHandle\n"));
        assert!(!output.contains("ChatChunk"));
    }

    #[test]
    fn a_resolved_argument_takes_its_host_annotation_and_passes_through_the_function() {
        let declarations = declarations(
            r#"
            #[host(call, facade(module = "commands", class = "Commands", target = "_running", not_given, annotation(context = "..llm.ChatContext"), resolve(context = "..voice.calls._items")))]
            fn reply(&self, #[host(kw, default = None)] context: Option<Vec<String>>);
        "#,
        );
        let output = super::emit(&declarations).unwrap().remove(0).text;
        assert!(
            output.contains("def reply(self, *, context: NotGivenOr[ChatContext] = NOT_GIVEN)")
        );
        assert!(output.contains("context=_items(context) if is_given(context) else None"));
        assert!(output.contains("from ..llm import ChatContext\n"));
        assert!(!output.contains("Sequence"));
    }

    // Catches: an omittable argument rendered with the stub's `...` default, or resolved while
    // omitted.
    #[test]
    fn an_omittable_argument_defaults_to_not_given_and_passes_through() {
        let declarations = declarations(
            r#"
            #[host(call, facade(module = "commands", class = "Commands", target = "_owner", annotation(voice = "..tts.TTS | str | None"), resolve(voice = "..voice.calls._voice")))]
            fn update(&self, #[host(kw)] voice: crate::utils::maybe::Maybe<Option<crate::utils::host_ref::HostRef>>, speech_delay_seconds: crate::utils::maybe::Maybe<f64>);
        "#,
        );
        let output = super::emit(&declarations).unwrap().remove(0).text;
        assert!(output.contains("voice: NotGivenOr[TTS | str | None] = NOT_GIVEN,"));
        assert!(output.contains("speech_delay_seconds: NotGivenOr[float] = NOT_GIVEN,"));
        assert!(output.contains(
            "return self._owner.update(\n            voice=_voice(voice) if is_given(voice) else voice,\n            speech_delay_seconds=speech_delay_seconds,\n        )"
        ));
        assert!(output.contains("from ..tts import TTS\n"));
        assert!(!output.contains("= ..."));
    }

    #[test]
    fn a_facade_cannot_name_a_parameter_the_method_lacks() {
        let declarations = declarations(
            r#"
            #[host(call, facade(module = "commands", class = "Commands", target = "_running", resolve(other = "m.f")))]
            fn reply(&self, text: String);
        "#,
        );
        assert!(super::emit(&declarations).err().unwrap().contains("names no parameter other"));
    }

    #[test]
    fn a_str_or_its_stream_is_one_argument() {
        let declarations = crate::decl::tests::read(
            r#"#[crate::host(class(name = "Native"))] pub trait Native: Sized {
                #[host(call, facade(module = "commands", class = "Commands", target = "_running"))]
                fn say(&self, text: crate::utils::stream::OneOrStream<String>);
            }"#,
        )
        .unwrap();
        let output = super::emit(&declarations).unwrap().remove(0).text;
        assert!(output.contains("def say(self, text: str | AsyncIterable[str]) -> None"));
        let refused = crate::decl::tests::read(
            r#"#[crate::host(class(name = "Native"))] pub trait Native: Sized {
                #[host(call)] fn say(&self, text: crate::utils::stream::OneOrStream<u32>);
            }"#,
        );
        assert!(refused.err().unwrap().contains("told apart as a `str`"));
    }
}
