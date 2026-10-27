//! The plugin base classes: each `#[python(provider(..))]` trait as the Python class plugins
//! subclass, written after its module's records.
//!
//! The provider class is read off the trait, method by method: its constructor, properties and
//! hooks come from each method's role ([`Role`]), and every method gets the `_core_<method>` the
//! proxy calls, so a trait method the base class does not answer is a codegen error rather than a
//! runtime one. The classes a method returns (`stream(kind = ..)`) are the host-stream idioms
//! below: the channels, the task running the plugin's `_run`, and the `_core_push`/`_core_flush`/
//! `_core_end_input` surface the core feeds them through.

use crate::ir::*;

/// What one module gains: its classes, the imports they need, and their public names.
pub struct Bases {
    pub module: String,
    pub sources: Vec<String>,
    pub body: String,
    /// `(module, name)`; `module` empty is `import name`.
    pub imports: Vec<(String, String)>,
    /// `(class, public module)`: the `__module__` fixups.
    pub paths: Vec<(String, String)>,
}

/// One entry per module, in first-seen order.
pub fn emit(decls: &[Decl], x: &Exposed) -> Result<Vec<Bases>, String> {
    let mut out: Vec<Bases> = Vec::new();
    for p in &x.providers {
        let mut w =
            Writer { decls, x, p, imports: Vec::new(), paths: Vec::new(), b: String::new() };
        w.provider()?;
        let i = match out.iter().position(|b| b.module == p.module) {
            Some(i) => i,
            None => {
                out.push(Bases {
                    module: p.module.clone(),
                    sources: Vec::new(),
                    body: String::new(),
                    imports: Vec::new(),
                    paths: Vec::new(),
                });
                out.len() - 1
            }
        };
        let m = &mut out[i];
        if !m.sources.contains(&p.source) {
            m.sources.push(p.source.clone());
        }
        if !m.body.is_empty() {
            m.body.push_str("\n\n");
        }
        m.body.push_str(&w.b);
        for imp in w.imports {
            if !m.imports.contains(&imp) {
                m.imports.push(imp);
            }
        }
        m.paths.extend(w.paths);
    }
    Ok(out)
}

struct Writer<'a> {
    decls: &'a [Decl],
    x: &'a Exposed,
    p: &'a Provider,
    imports: Vec<(String, String)>,
    paths: Vec<(String, String)>,
    b: String,
}

/// A Rust doc comment as a docstring's lines: Rust's `code` is reST's ``code``, and a Markdown
/// list under a Google section header (`Args:`) is that section's indented entries.
fn py_doc(doc: &[String]) -> Vec<String> {
    let mut section = false;
    doc.iter()
        .map(|l| {
            if l.ends_with(':') && ["Args:", "Returns:", "Raises:", "Yields:"].contains(&l.as_str())
            {
                section = true;
            } else if l.is_empty() {
                section = false;
            }
            let l = match (section, l.strip_prefix("- ")) {
                (true, Some(entry)) => format!("    {entry}"),
                (true, None) if l.starts_with("  ") => format!("        {}", l.trim_start()),
                _ => l.clone(),
            };
            let mut out = String::new();
            let mut rest = l.as_str();
            while let Some(i) = rest.find('`') {
                out.push_str(&rest[..i]);
                let tail = &rest[i..];
                if tail.starts_with("``") {
                    let end = tail[2..].find("``").map_or(tail.len(), |j| j + 4);
                    out.push_str(&tail[..end]);
                    rest = &tail[end..];
                } else if let Some(j) = tail[1..].find('`') {
                    out.push_str(&format!("``{}``", &tail[1..=j]));
                    rest = &tail[j + 2..];
                } else {
                    out.push_str(tail);
                    rest = "";
                }
            }
            out.push_str(rest);
            out
        })
        .collect()
}

/// `"""..."""` at `indent`: each paragraph and section entry refilled to 100 columns, the first
/// line past the opening quotes.
fn docstring(b: &mut String, doc: &[String], indent: usize) {
    let mut lines: Vec<String> = Vec::new();
    // (lead of the first line, lead of the rest, words)
    let mut para: Option<(String, String, Vec<String>)> = None;
    let flush = |para: &mut Option<(String, String, Vec<String>)>, lines: &mut Vec<String>| {
        let Some((first, rest, words)) = para.take() else { return };
        let mut line = first;
        let mut empty = true;
        for w in words {
            let width = if lines.is_empty() { 100 - indent - 3 } else { 100 - indent };
            if !empty && line.len() + 1 + w.len() > width {
                lines.push(std::mem::replace(&mut line, rest.clone()));
                empty = true;
            }
            if !empty {
                line.push(' ');
            }
            line.push_str(&w);
            empty = false;
        }
        lines.push(line);
    };
    for l in py_doc(doc) {
        let words = || l.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
        if l.is_empty() || l.ends_with(':') && !l.starts_with(' ') {
            flush(&mut para, &mut lines);
            lines.push(l.clone());
        } else if l.starts_with("    ") && !l.starts_with("     ") {
            flush(&mut para, &mut lines);
            para = Some(("    ".to_owned(), "        ".to_owned(), words()));
        } else if let Some(p) = para.as_mut() {
            p.2.extend(words());
        } else {
            para = Some((String::new(), String::new(), words()));
        }
    }
    flush(&mut para, &mut lines);
    crate::python::docstring(b, &lines, indent);
}

/// The leading type of `Option<T>` / `Result<T, E>` / `&T`, recursively peeled by the callers.
fn generic<'t>(t: &'t syn::Type, head: &str) -> Option<&'t syn::Type> {
    let syn::Type::Path(p) = t else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != head {
        return None;
    }
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return None };
    a.args.iter().find_map(|g| if let syn::GenericArgument::Type(t) = g { Some(t) } else { None })
}

fn last_ident(t: &syn::Type) -> Option<String> {
    match t {
        syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
        syn::Type::Reference(r) => last_ident(&r.elem),
        _ => None,
    }
}

impl Writer<'_> {
    fn need(&mut self, module: &str, name: &str) {
        let imp = (module.to_owned(), name.to_owned());
        if !self.imports.contains(&imp) {
            self.imports.push(imp);
        }
    }

    /// The Python class a declared record is, importing it when another module holds it.
    fn record(&mut self, rust: &str) -> Option<String> {
        let d = self.decls.iter().find(|d| d.rust_name == rust)?;
        let t = d.target(Host::Python)?;
        if t.module != self.p.module {
            self.need(&format!(".{}", t.module), &t.name);
        }
        Some(t.name.clone())
    }

    /// A Rust type as the core hands it to (or reads it from) a `_core_` method.
    fn core_ty(&mut self, m: &ProviderMethod, rust: &str) -> Result<String, String> {
        let t: syn::Type = syn::parse_str(rust).map_err(|e| format!("{rust}: {e}"))?;
        self.ty(m, &t)
    }

    fn ty(&mut self, m: &ProviderMethod, t: &syn::Type) -> Result<String, String> {
        if let syn::Type::Reference(r) = t {
            return self.ty(m, &r.elem);
        }
        if let syn::Type::Slice(s) = t {
            return Ok(format!("list[{}]", self.ty(m, &s.elem)?));
        }
        if let Some(inner) = generic(t, "Option") {
            // `Option<Result<stream>>`: no stream, a stream, or the failure raised.
            let inner = generic(inner, "Result").unwrap_or(inner);
            return Ok(format!("{} | None", self.ty(m, inner)?));
        }
        if let Some(inner) = generic(t, "Result") {
            return self.ty(m, inner);
        }
        if let Some(inner) = generic(t, "Vec") {
            return Ok(format!("list[{}]", self.ty(m, inner)?));
        }
        if matches!(t, syn::Type::Tuple(_)) || generic(t, "Box").is_some() {
            return m
                .stream
                .as_ref()
                .map(|s| s.name.clone())
                .ok_or_else(|| format!("{}: a stream return needs stream(name = ..)", m.rust));
        }
        let name = last_ident(t).ok_or_else(|| format!("{}: unreadable type", m.rust))?;
        Ok(match name.as_str() {
            "String" | "str" | "LanguageCode" => "str".to_owned(),
            "bool" => "bool".to_owned(),
            "f64" | "f32" => "float".to_owned(),
            "u8" | "u16" | "u32" | "u64" | "usize" | "i32" | "i64" => "int".to_owned(),
            "AudioFrame" => {
                self.need("livekit", "rtc");
                "rtc.AudioFrame".to_owned()
            }
            "ApiConnectOptions" => {
                self.need("typing", "Any");
                "dict[str, Any]".to_owned()
            }
            "ProviderId" => "dict[str, str]".to_owned(),
            "ChatRequest" => {
                self.need("..", "_core");
                "_core.ChatRequest".to_owned()
            }
            "LlmStream" | "SttSource" | "TtsSource" => m
                .stream
                .as_ref()
                .map(|s| s.name.clone())
                .ok_or_else(|| format!("{}: a stream return needs stream(name = ..)", m.rust))?,
            other => self
                .record(other)
                .ok_or_else(|| format!("{}: `{other}` has no Python name", m.rust))?,
        })
    }

    fn conn(&self) -> Option<&ProviderMethod> {
        self.p.methods.iter().find(|m| matches!(m.role, Role::Conn { .. }))
    }

    /// The methods a `conn_options` reaches: every awaited one, on a provider with a `conn`.
    fn carries_conn(&self, m: &ProviderMethod) -> bool {
        m.is_async && self.conn().is_some()
    }

    fn provider(&mut self) -> Result<(), String> {
        let p = self.p;
        let class = p.class.clone();
        self.need("abc", "ABC");
        let mut b = String::new();
        if let Some((name, kind)) = &p.error {
            for (m, n) in [
                ("pydantic", "BaseModel"),
                ("pydantic", "ConfigDict"),
                ("pydantic", "Field"),
                ("typing", "Literal"),
            ] {
                self.need(m, n);
            }
            b.push_str(&format!(
                "class {name}(BaseModel):\n    model_config = ConfigDict(arbitrary_types_allowed=True)\n    type: Literal[\"{kind}\"] = \"{kind}\"\n    timestamp: float\n    label: str\n    error: Exception = Field(..., exclude=True)\n    recoverable: bool\n\n\n"
            ));
            self.paths.push((name.clone(), p.path.clone()));
        }
        let generic = !p.events.is_empty();
        if generic {
            self.need("typing", "TypeVar");
            self.need("typing", "Generic");
            self.need("typing", "Literal");
            self.need("livekit", "rtc");
            b.push_str("TEvent = TypeVar(\"TEvent\")\n\n\n");
            let events = p.events.iter().map(|e| format!("\"{e}\"")).collect::<Vec<_>>().join(", ");
            b.push_str(&format!(
                "class {class}(\n    ABC,\n    rtc.EventEmitter[Literal[{events}] | TEvent],\n    Generic[TEvent],\n):\n"
            ));
        } else {
            b.push_str(&format!("class {class}(ABC):\n"));
        }
        docstring(&mut b, &p.doc, 4);
        // The constructor: the `init` keywords, then what every instance holds.
        let mut params = Vec::new();
        let mut lines = Vec::new();
        if generic {
            lines.push("super().__init__()".to_owned());
        }
        for m in &p.methods {
            match &m.role {
                Role::Init => {
                    let ret = m.ret.clone().ok_or_else(|| format!("{}: init returns", m.rust))?;
                    let ty = self.core_ty(m, &ret)?;
                    params.push(format!("{}: {ty}", m.rust));
                    lines.push(format!("self._{0} = {0}", m.rust));
                }
                Role::Label => lines.push(
                    "self._label = f\"{type(self).__module__}.{type(self).__name__}\"".to_owned(),
                ),
                Role::Hook { store: Some((attr, default)), .. } => {
                    lines.push(format!("self.{attr} = {default}"))
                }
                Role::Markup => lines.push("self._markup = self.Markup(self)".to_owned()),
                _ => {}
            }
        }
        let sig =
            if params.is_empty() { String::new() } else { format!(", *, {}", params.join(", ")) };
        signature(&mut b, "def", "__init__", &format!("self{sig}"), "None");
        for l in &lines {
            b.push_str(&format!("        {l}\n"));
        }
        if lines.is_empty() {
            b.push_str("        pass\n");
        }
        for m in &p.methods {
            self.face(&mut b, m)?;
            if m.stream.as_ref().is_some_and(|s| s.kind == "synthesize")
                && let Some(chunked) = self.stream_of("chunked")
            {
                self.need("..types", "DEFAULT_API_CONNECT_OPTIONS");
                signature(
                    &mut b,
                    "def",
                    "_synthesize_with_stream",
                    "self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS",
                    &chunked,
                );
                b.push_str("        \"\"\"``synthesize()`` for a provider that only streams: one push, then the end of\n        input.\"\"\"\n        return _ChunkedStreamFromStream(tts=self, input_text=text, conn_options=conn_options)\n");
            }
        }
        self.idiom(&mut b)?;
        for m in &p.methods {
            self.core(&mut b, m)?;
        }
        if let Some(Role::Conn { retry: Some(_) }) = self.conn().map(|m| m.role.clone()) {
            b.push_str("\n    def _core_failed(self, error: Exception, recoverable: bool) -> None:\n        self._emit_error(error, recoverable)\n");
        }
        self.paths.push((class.clone(), p.path.clone()));
        if self.conn().is_some() {
            b.push_str("\n\n");
            self.conn_view(&mut b)?;
        }
        for m in &p.methods {
            if let Some(s) = &m.stream {
                b.push_str("\n\n");
                self.stream(&mut b, m, s)?;
                self.paths.push((s.name.clone(), p.path.clone()));
            }
        }
        self.b = b;
        Ok(())
    }

    /// The hook's parameters (after `self`): positional ones, then the keyword-only optional
    /// ones and `conn_options`. `required_conn`: the abstract method takes it without default.
    fn hook_params(&mut self, m: &ProviderMethod, required_conn: bool) -> Result<String, String> {
        let mut pos = Vec::new();
        let mut kw = Vec::new();
        for p in m.params.iter().filter(|p| !p.attr) {
            let t: syn::Type = syn::parse_str(&p.rust_ty).map_err(|e| e.to_string())?;
            let optional = generic(&t, "Option").is_some();
            if let Some((module, name)) = p.import.as_deref().and_then(|i| i.split_once(':')) {
                self.need(module, name);
            }
            let ty = match &p.py_ty {
                Some(ty) => ty.clone(),
                None if optional => {
                    let inner = generic(&t, "Option").expect("checked");
                    self.ty(m, inner)?
                }
                None => self.ty(m, &t)?,
            };
            if optional {
                self.need("..types", "NOT_GIVEN");
                self.need("..types", "NotGivenOr");
                kw.push(format!("{}: NotGivenOr[{ty}] = NOT_GIVEN", p.name));
            } else {
                pos.push(format!("{}: {ty}", p.name));
            }
        }
        if self.carries_conn(m) {
            self.need("..types", "APIConnectOptions");
            if required_conn {
                kw.push("conn_options: APIConnectOptions".to_owned());
            } else {
                self.need("..types", "DEFAULT_API_CONNECT_OPTIONS");
                kw.push("conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS".to_owned());
            }
        }
        let mut all = pos;
        if !kw.is_empty() {
            all.push("*".to_owned());
            all.extend(kw);
        }
        Ok(all.join(", "))
    }

    /// The call into the hook from its `_core_` method: optional values `None` are `NOT_GIVEN`.
    fn hook_args(&self, m: &ProviderMethod, conn: &str) -> String {
        let mut pos = Vec::new();
        let mut kw = Vec::new();
        for p in m.params.iter().filter(|p| !p.attr) {
            if p.rust_ty.starts_with("Option<") {
                kw.push(format!("{0}=NOT_GIVEN if {0} is None else {0}", p.name));
            } else {
                pos.push(p.name.clone());
            }
        }
        if self.carries_conn(m) {
            kw.push(format!("conn_options={conn}"));
        }
        pos.extend(kw);
        pos.join(", ")
    }

    /// The hook's return annotation; a stream class is returned by a plain method.
    fn hook_ret(&mut self, m: &ProviderMethod) -> Result<String, String> {
        if let Some(s) = &m.stream {
            return Ok(s.name.clone());
        }
        match &m.ret {
            None => Ok("None".to_owned()),
            Some(r) => self.core_ty(m, r),
        }
    }

    /// One method's public face.
    fn face(&mut self, b: &mut String, m: &ProviderMethod) -> Result<(), String> {
        match &m.role {
            Role::Label => {
                property(b, "label", "str", &m.doc, "self._label");
            }
            Role::Init => {
                let ret = m.ret.clone().expect("checked in __init__");
                let ty = self.core_ty(m, &ret)?;
                property(b, &m.rust, &ty, &m.doc, &format!("self._{}", m.rust));
            }
            Role::Identity => {
                let id = self
                    .decls
                    .iter()
                    .find(|d| d.rust_name == "ProviderId")
                    .ok_or("ProviderId is not declared")?;
                for f in id.fields().iter().filter(|f| f.name != "label") {
                    property(b, &f.name, "str", &f.doc, "\"unknown\"");
                }
                self.need(".chat_items", "MetricsMetadata");
                property(
                    b,
                    "metrics_metadata",
                    "MetricsMetadata",
                    &m.doc,
                    "{\"model_name\": self.model, \"model_provider\": self.provider}",
                );
            }
            Role::Hook { name, implementation, unsupported, store } => {
                // A stream is opened, not awaited.
                let is_async = m.is_async && m.stream.is_none();
                let def = if is_async { "async def" } else { "def" };
                let ret = self.hook_ret(m)?;
                if let Some(imp) = implementation {
                    let params = self.hook_params(m, true)?;
                    let sep = if params.is_empty() { "" } else { ", " };
                    if !m.is_async && self.impl_is_task(m) {
                        return self.task_hook(b, m, name, imp);
                    }
                    self.need("abc", "abstractmethod");
                    b.push_str("\n    @abstractmethod");
                    signature(b, def, imp, &format!("self{sep}{params}"), &ret);
                    b.truncate(b.len() - 1);
                    b.push_str(" ...\n");
                    let params = self.hook_params(m, false)?;
                    let sep = if params.is_empty() { "" } else { ", " };
                    signature(b, def, name, &format!("self{sep}{params}"), &ret);
                    docstring(b, &m.doc, 8);
                    let args = self.forward_args(m);
                    let aw = if is_async { "await " } else { "" };
                    b.push_str(&format!("        return {aw}self.{imp}({args})\n"));
                    return Ok(());
                }
                let params = match &m.py_params {
                    Some(params) => {
                        for (module, name) in m.imports.iter().filter_map(|i| i.split_once(':')) {
                            if params.contains(name) {
                                self.need(&format!("?{module}"), name);
                            }
                        }
                        params.clone()
                    }
                    None => self.hook_params(m, false)?,
                };
                let sep = if params.is_empty() { "" } else { ", " };
                let abstract_ = !m.has_default && unsupported.is_none() && store.is_none();
                if abstract_ {
                    self.need("abc", "abstractmethod");
                    b.push_str("\n    @abstractmethod");
                }
                signature(b, def, name, &format!("self{sep}{params}"), &ret);
                docstring(b, &m.doc, 8);
                if let Some(msg) = unsupported {
                    let line = format!("        raise NotImplementedError(\"{msg}\")");
                    if line.len() <= 100 {
                        b.push_str(&format!("{line}\n"));
                    } else {
                        b.push_str(&format!(
                            "        raise NotImplementedError(\n            \"{msg}\"  # noqa: E501\n        )\n"
                        ));
                    }
                } else if let Some((attr, _)) = store {
                    let arg = &m
                        .params
                        .first()
                        .ok_or_else(|| format!("{}: store takes one argument", m.rust))?
                        .name;
                    b.push_str(&format!("        self.{attr} = {arg}\n"));
                } else if abstract_ && m.doc.is_empty() {
                    b.push_str("        ...\n");
                } else if m.doc.is_empty() {
                    b.push_str("        pass\n");
                }
            }
            Role::Markup => self.markup(b, m)?,
            Role::Chat { name } => self.chat_hook(b, m, name)?,
            Role::Conn { .. } | Role::Value(_) => {}
        }
        Ok(())
    }

    /// `name(..)` forwarding its own parameters to `imp`.
    fn forward_args(&self, m: &ProviderMethod) -> String {
        let mut pos = Vec::new();
        let mut kw = Vec::new();
        for p in m.params.iter().filter(|p| !p.attr) {
            if p.rust_ty.starts_with("Option<") {
                kw.push(format!("{0}={0}", p.name));
            } else {
                pos.push(p.name.clone());
            }
        }
        if self.carries_conn(m) {
            kw.push("conn_options=conn_options".to_owned());
        }
        pos.extend(kw);
        pos.join(", ")
    }

    /// The idioms every provider has: the error event, closing, the async context manager.
    fn idiom(&mut self, b: &mut String) -> Result<(), String> {
        if let Some((name, _)) = self.p.error.clone() {
            self.need("", "time");
            b.push_str(&format!(
                "\n    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:\n        self.emit(\n            \"error\",\n            {name}(\n                timestamp=time.time(),\n                label=self._label,\n                error=api_error,\n                recoverable=recoverable,\n            ),\n        )\n"
            ));
        }
        self.need("typing_extensions", "Self");
        self.need("types", "TracebackType");
        let task = self.p.methods.iter().any(|m| self.impl_is_task(m));
        if task {
            self.need("..utils", "aio");
            b.push_str("\n    async def aclose(self) -> None:\n        \"\"\"Close the provider and everything it opened.\"\"\"\n        if self._prewarm_task is not None:\n            await aio.cancel_and_wait(self._prewarm_task)\n");
        } else {
            b.push_str("\n    async def aclose(self) -> None:\n        \"\"\"Close the provider and everything it opened.\"\"\"\n");
        }
        b.push_str(ACONTEXT);
        Ok(())
    }

    /// One method's `_core_` surface.
    fn core(&mut self, b: &mut String, m: &ProviderMethod) -> Result<(), String> {
        let def = if m.is_async { "async def" } else { "def" };
        let mut params = Vec::new();
        for p in &m.params {
            params.push(format!("{}: {}", p.name, self.core_ty(m, &p.rust_ty)?));
        }
        let routed = self.carries_conn(m);
        if routed {
            self.need("..types", "APIConnectOptions");
            self.need("..types", "DEFAULT_API_CONNECT_OPTIONS");
            params.push("conn: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS".to_owned());
        }
        let ret = match (&m.role, &m.ret) {
            (Role::Conn { .. }, _) => {
                self.need("typing", "Any");
                "Any".to_owned()
            }
            (_, None) => "None".to_owned(),
            (_, Some(r)) => self.core_ty(m, r)?,
        };
        let name = format!("_core_{}", m.rust);
        let all = std::iter::once("self".to_owned()).chain(params).collect::<Vec<_>>().join(", ");
        signature(b, def, &name, &all, &ret);
        let body = match &m.role {
            Role::Label => "return self.label".to_owned(),
            Role::Init => {
                let ret = m.ret.as_deref().unwrap_or_default();
                if self.from_items(ret) {
                    self.need("", "dataclasses");
                    format!("return dataclasses.asdict(self.{})", m.rust)
                } else {
                    format!("return self.{}", m.rust)
                }
            }
            Role::Identity => {
                "return {\"label\": self.label, \"model\": self.model, \"provider\": self.provider}"
                    .to_owned()
            }
            Role::Hook { name, .. } => {
                let aw = if m.is_async && m.stream.is_none() { "await " } else { "" };
                let args = m.py_call.clone().unwrap_or_else(|| self.hook_args(m, "conn"));
                let call = format!("self.{name}({args})");
                let guard = m.when.as_ref().map(|cap| {
                    format!("if not self.capabilities.{cap}:\n            return None\n        ")
                });
                let attrs: Vec<&ProviderParam> = m.params.iter().filter(|p| p.attr).collect();
                let mut s = guard.unwrap_or_default();
                if m.py_call.is_some() {
                    let mut local: Vec<(&str, &str)> =
                        m.imports.iter().filter_map(|i| i.split_once(':')).collect();
                    local.sort();
                    for (module, name) in local {
                        s.push_str(&format!("from {module} import {name}\n        "));
                    }
                    s.truncate(s.len() - 8);
                    s.push_str("\n        ");
                }
                if attrs.is_empty() {
                    let ret = if m.ret.is_some() { "return " } else { "" };
                    s.push_str(&wrap_call(&format!("{ret}{aw}"), &call, 8));
                } else {
                    s.push_str(&wrap_call(&format!("out = {aw}"), &call, 8));
                    for a in attrs {
                        s.push_str(&format!(
                            "\n        if {0} is not None:\n            out.{0} = {0}",
                            a.name
                        ));
                    }
                    s.push_str("\n        return out");
                }
                s
            }
            Role::Conn { retry } => {
                self.need("..types", "APIConnectOptions");
                let view = format!("_{}Conn(self, APIConnectOptions(**conn))", self.p.class);
                match retry {
                    Some(r) => {
                        self.need("..", "_core");
                        format!("return _core.{r}({view}, self, **conn)")
                    }
                    None => format!("return {view}"),
                }
            }
            Role::Markup => "return self.markup._provider_key()".to_owned(),
            Role::Chat { name } => self.chat_core(m, name)?,
            Role::Value(expr) => format!("return {expr}"),
        };
        b.push_str(&format!("        {body}\n"));
        Ok(())
    }

    /// Whether the core reads the record as a dict (`#[binding(from_items)]`).
    fn from_items(&self, rust: &str) -> bool {
        let Ok(t) = syn::parse_str::<syn::Type>(rust) else { return false };
        let Some(name) = last_ident(&t) else { return false };
        self.decls.iter().any(|d| d.rust_name == name && d.targets.iter().any(|t| t.from_items))
    }

    /// `_<Class>Conn`: the provider as the core holds it under one `conn_options`.
    fn conn_view(&mut self, b: &mut String) -> Result<(), String> {
        let class = self.p.class.clone();
        self.need("typing", "Any");
        b.push_str(&format!(
            "class _{class}Conn:\n    \"\"\"``{class}`` as the framework holds it under one ``conn_options``.\"\"\"\n\n    def __init__(self, provider: {class}, conn: APIConnectOptions) -> None:\n        self._provider = provider\n        self._conn = conn\n\n    def __getattr__(self, name: str) -> Any:\n        return getattr(self._provider, name)\n"
        ));
        let routed: Vec<ProviderMethod> =
            self.p.methods.iter().filter(|m| self.carries_conn(m)).cloned().collect();
        for m in &routed {
            let mut params = Vec::new();
            let mut args = Vec::new();
            for p in &m.params {
                params.push(format!("{}: {}", p.name, self.core_ty(m, &p.rust_ty)?));
                args.push(p.name.clone());
            }
            args.push("self._conn".to_owned());
            let ret = match &m.ret {
                None => "None".to_owned(),
                Some(r) => self.core_ty(m, r)?,
            };
            let all =
                std::iter::once("self".to_owned()).chain(params).collect::<Vec<_>>().join(", ");
            signature(b, "async def", &format!("_core_{}", m.rust), &all, &ret);
            b.push_str("        ");
            b.push_str(&wrap_call(
                "return await ",
                &format!("self._provider._core_{}({})", m.rust, args.join(", ")),
                8,
            ));
            b.push('\n');
        }
        Ok(())
    }

    fn stream(
        &mut self,
        b: &mut String,
        m: &ProviderMethod,
        s: &StreamClass,
    ) -> Result<(), String> {
        let text = match s.kind.as_str() {
            "recognize" => self.recognize_stream(s)?,
            "chunked" => self.synthesis_stream(s, CHUNKED)?,
            "synthesize" => self.synthesis_stream(s, SYNTHESIZE)?,
            other => return Err(format!("{}: no stream idiom `{other}`", m.rust)),
        };
        b.push_str(&text);
        b.push_str(ACONTEXT);
        // What the core feeds a stream with input through: an end of input after the core closed
        // it is a no-op, not the plugin's "input ended".
        if let Some((push, arg, ty)) = match s.kind.as_str() {
            "recognize" | "vad" => Some(("push_frame", "frame", "rtc.AudioFrame")),
            "synthesize" => Some(("push_text", "token", "str")),
            _ => None,
        } {
            b.push_str(&format!(
                "\n    def _core_push(self, {arg}: {ty}) -> None:\n        self.{push}({arg})\n\n    def _core_flush(self) -> None:\n        if not self._input_ch.closed:\n            self.flush()\n\n    def _core_end_input(self) -> None:\n        if not self._input_ch.closed:\n            self.end_input()\n"
            ));
        }
        if let Some(alias) = &s.alias {
            b.push_str(&format!("\n\n{alias} = {}\n", s.name));
        }
        // A provider with both syntheses has `synthesize()` over its stream for the plugins that
        // only stream.
        if s.kind == "synthesize"
            && let Some(chunked) = self.stream_of("chunked")
        {
            b.push_str(
                &FROM_STREAM.replace("{Chunked}", &chunked).replace("{Provider}", &self.p.class),
            );
        }
        Ok(())
    }

    /// The class a `stream(kind = ..)` method of this provider returns.
    fn stream_of(&self, kind: &str) -> Option<String> {
        self.p
            .methods
            .iter()
            .filter_map(|m| m.stream.as_ref())
            .find(|s| s.kind == kind)
            .map(|s| s.name.clone())
    }

    fn impl_is_task(&self, _m: &ProviderMethod) -> bool {
        false
    }

    fn task_hook(
        &mut self,
        _b: &mut String,
        m: &ProviderMethod,
        _name: &str,
        _imp: &str,
    ) -> Result<(), String> {
        Err(format!("{}: a task hook is not written yet", m.rust))
    }

    /// `markup`: the nested `Markup` class plugins override `_provider_key` on, forwarding the
    /// rest to the core's `Markup` object of that key, and the `markup` property.
    fn markup(&mut self, b: &mut String, m: &ProviderMethod) -> Result<(), String> {
        let o = self
            .x
            .objects
            .iter()
            .find(|o| o.name == "Markup")
            .ok_or_else(|| format!("{}: markup forwards to the core's `Markup`", m.rust))?;
        self.need("..", "_core");
        let info = self.record("MarkupInfo").ok_or("MarkupInfo is not declared")?;
        let class = self.p.class.clone();
        b.push_str("\n    class Markup:\n");
        docstring(b, &o.doc, 8);
        b.push_str(&format!(
            "\n        def __init__(self, tts: {class}) -> None:\n            self._tts = tts\n"
        ));
        let key = o
            .methods
            .iter()
            .find(|x| x.name == "_provider_key")
            .ok_or("Markup has no _provider_key")?;
        b.push_str("\n        def _provider_key(self) -> str:\n");
        docstring(b, &key.doc, 12);
        b.push_str(&format!(
            "            return \"\"\n\n        def _core(self) -> _core.Markup:\n            return _core.Markup(self._provider_key())\n\n        @property\n        def info(self) -> {info}:\n            \"\"\"The queryable markup matrix for this voice.\"\"\"\n            return {info}(nonverbals=_core.supported_nonverbals(self._provider_key()))\n"
        ));
        for f in
            o.methods.iter().filter(|x| x.kind == MethodKind::Method && x.name != "_provider_key")
        {
            let (params, ret) = crate::stub::signature_parts(self.decls, self.x, &f.sig)?;
            let mut decl = vec!["self".to_owned()];
            let mut call = Vec::new();
            let mut kw = false;
            for (param, star) in &params {
                if *star {
                    decl.push("*".to_owned());
                    kw = true;
                }
                decl.push(param.clone());
                let name = param.split(':').next().unwrap_or(param).trim();
                call.push(if kw { format!("{name}={name}") } else { name.to_owned() });
            }
            if ret.contains("Any") || decl.iter().any(|d| d.contains("Any")) {
                self.need("typing", "Any");
            }
            b.push_str(&format!("\n        def {}({}) -> {ret}:\n", f.name, decl.join(", ")));
            docstring(b, &f.doc, 12);
            b.push_str(&format!(
                "            return self._core().{}({})\n",
                f.name,
                call.join(", ")
            ));
        }
        property(b, "markup", "Markup", &m.doc, "self._markup");
        Ok(())
    }

    fn chat_hook(
        &mut self,
        _b: &mut String,
        m: &ProviderMethod,
        _name: &str,
    ) -> Result<(), String> {
        Err(format!("{}: chat is not written yet", m.rust))
    }

    fn chat_core(&mut self, m: &ProviderMethod, _name: &str) -> Result<String, String> {
        Err(format!("{}: chat is not written yet", m.rust))
    }

    // ------------------------------------------------------------------- the stream idioms

    /// `stream(kind = "chunked" | "synthesize")`: one attempt run against an `AudioEmitter`, its
    /// audio checked against the text it was given.
    fn synthesis_stream(&mut self, s: &StreamClass, idiom: &str) -> Result<String, String> {
        for (m, n) in [
            ("", "asyncio"),
            ("", "time"),
            ("abc", "abstractmethod"),
            ("collections.abc", "AsyncIterator"),
            ("livekit", "rtc"),
            ("..", "_core"),
            (".._exceptions", "APIError"),
            (".._exceptions", "APIStatusError"),
            ("..log", "logger"),
            ("..types", "APIConnectOptions"),
            ("..types", "USERDATA_TIMED_TRANSCRIPT"),
            ("..types", "USERDATA_TTS_STARTED_TIME"),
            ("..utils", "aio"),
            ("..utils", "shortuuid"),
        ] {
            self.need(m, n);
        }
        let audio = self.record("SynthesizedAudio").ok_or("SynthesizedAudio is not declared")?;
        let mut b = format!("class {}(ABC):\n", s.name);
        docstring(&mut b, &s.doc, 4);
        b.push_str(
            &idiom
                .replace("{SYNTHESIS}", SYNTHESIS)
                .replace("{Stream}", &s.name)
                .replace("{Provider}", &self.p.class)
                .replace("{Audio}", &audio),
        );
        Ok(b)
    }

    /// `stream(kind = "recognize")`: audio frames in, `SpeechEvent`s out, resampled to the rate
    /// the plugin asks for.
    fn recognize_stream(&mut self, s: &StreamClass) -> Result<String, String> {
        for (m, n) in [
            ("", "asyncio"),
            ("", "time"),
            ("abc", "abstractmethod"),
            ("collections.abc", "AsyncIterator"),
            ("pydantic", "BaseModel"),
            ("livekit", "rtc"),
            ("..types", "APIConnectOptions"),
            ("..types", "NOT_GIVEN"),
            ("..types", "NotGivenOr"),
            ("..utils", "aio"),
            ("..utils", "is_given"),
        ] {
            self.need(m, n);
        }
        let event = self.record("SpeechEvent").ok_or("SpeechEvent is not declared")?;
        let mut b = format!("class {}(ABC):\n", s.name);
        docstring(&mut b, &s.doc, 4);
        b.push_str(
            &RECOGNIZE
                .replace("{Stream}", &s.name)
                .replace("{Provider}", &self.p.class)
                .replace("{Event}", &event),
        );
        Ok(b)
    }
}

/// `def name(self, ..) -> ret:`, wrapped at 100 columns one parameter per line.
fn signature(b: &mut String, def: &str, name: &str, params: &str, ret: &str) {
    let flat = format!("    {def} {name}({params}) -> {ret}:");
    if flat.len() <= 100 {
        b.push_str(&format!("\n{flat}\n"));
        return;
    }
    let inner =
        params.strip_prefix("self, ").map(|r| format!("self, {r}")).unwrap_or(params.to_owned());
    let one = format!("    {def} {name}(\n        {inner}\n    ) -> {ret}:");
    if one.lines().all(|l| l.len() <= 100) {
        b.push_str(&format!("\n{one}\n"));
        return;
    }
    b.push_str(&format!("\n    {def} {name}(\n"));
    for p in split_params(params) {
        b.push_str(&format!("        {p},\n"));
    }
    b.push_str(&format!("    ) -> {ret}:\n"));
}

/// Top-level commas only.
fn split_params(params: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in params.chars() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(cur.trim().to_owned());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_owned());
    }
    out
}

/// `<lead>self.f(a, b)` at `indent`, its arguments on their own line past 100 columns.
fn wrap_call(lead: &str, call: &str, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let flat = format!("{pad}{lead}{call}");
    if flat.len() <= 100 {
        return flat.trim_start().to_owned();
    }
    let (head, args) = call.split_once('(').expect("a call");
    let args = args.strip_suffix(')').expect("a call");
    let one = format!("{pad}    {args}");
    if one.len() <= 100 {
        return format!("{lead}{head}(\n{one}\n{pad})");
    }
    let mut s = format!("{lead}{head}(\n");
    for a in split_params(args) {
        s.push_str(&format!("{pad}    {a},\n"));
    }
    s.push_str(&format!("{pad})"));
    s
}

fn property(b: &mut String, name: &str, ty: &str, doc: &[String], value: &str) {
    b.push_str(&format!("\n    @property\n    def {name}(self) -> {ty}:\n"));
    docstring(b, doc, 8);
    b.push_str(&format!("        return {value}\n"));
}

const ACONTEXT: &str = r#"
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()
"#;

/// The recognition stream's idiom. `{Stream}`, `{Provider}` and `{Event}` are filled in.
const RECOGNIZE: &str = r#"
    class _FlushSentinel:
        """Marks the end of a segment in ``_input_ch``."""

    def __init__(
        self,
        *,
        stt: {Provider},
        conn_options: APIConnectOptions,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            stt: The provider this stream recognizes for.
            conn_options: The timeout and retries of the connection; the stream runs one attempt,
                which the framework retries inside a session.
            sample_rate: The rate the audio is resampled to before recognition; the input's own
                rate when not given.
        """
        self._stt = stt
        self._conn_options = conn_options
        self._input_ch = aio.Chan[rtc.AudioFrame | {Stream}._FlushSentinel]()
        self._event_ch = aio.Chan[{Event}]()
        self._start_time = time.time()
        self._start_time_offset = 0.0
        self._context: BaseModel | None = None
        self._needed_sr = sample_rate if is_given(sample_rate) else None
        self._pushed_sr = 0
        self._resampler: rtc.AudioResampler | None = None
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @property
    def context(self) -> BaseModel | None:
        """Speaker metadata the plugin updates as it learns about the speaker."""
        return self._context

    @context.setter
    def context(self, value: BaseModel | None) -> None:
        self._context = value

    @property
    def start_time_offset(self) -> float:
        """Where this stream's timestamps sit on the caller's input timeline, in seconds."""
        return self._start_time_offset

    @start_time_offset.setter
    def start_time_offset(self, value: float) -> None:
        if value < 0:
            raise ValueError("start_time_offset must be non-negative")
        self._start_time_offset = value

    @property
    def start_time(self) -> float:
        """Wall-clock anchor of the stream; plugins may move it to when the first audio reached
        the provider, so stream-relative times convert accurately."""
        return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        if value < 0:
            raise ValueError("start_time must be non-negative")
        self._start_time = value

    def _report_connection_acquired(self, acquire_time: float, connection_reused: bool) -> None:
        """Connection timing; kept on the stream, the framework measures its own."""
        self._acquire_time, self._connection_reused = acquire_time, connection_reused

    @abstractmethod
    async def _run(self) -> None: ...

    async def _main_task(self) -> None:
        await self._run()

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._stt._emit_error(api_error, recoverable)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push audio to be recognized."""
        self._check_input_not_ended()
        self._check_not_closed()
        if self._pushed_sr and self._pushed_sr != frame.sample_rate:
            raise ValueError("the sample rate of the input frames must be consistent")
        self._pushed_sr = frame.sample_rate
        if self._needed_sr and self._needed_sr != frame.sample_rate and not self._resampler:
            self._resampler = rtc.AudioResampler(
                frame.sample_rate, self._needed_sr, quality=rtc.AudioResamplerQuality.HIGH
            )
        for f in self._resampler.push(frame) if self._resampler else [frame]:
            self._input_ch.send_nowait(f)

    def flush(self) -> None:
        """Mark the end of the current segment."""
        self._check_input_not_ended()
        self._check_not_closed()
        if self._resampler:
            for frame in self._resampler.flush():
                self._input_ch.send_nowait(frame)
        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Mark the end of input: no more audio will be pushed."""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Close the stream immediately."""
        self._input_ch.close()
        await aio.cancel_and_wait(self._task)

    async def __anext__(self) -> {Event}:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        return val

    def __aiter__(self) -> AsyncIterator[{Event}]:
        return self

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self._input_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")
"#;

/// What both syntheses share: one attempt run against an `AudioEmitter`; retries are the core's.
const SYNTHESIS: &str = r#"
    def _set_token_usage(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._tts._emit_error(api_error, recoverable)

    @abstractmethod
    async def _run(self, output_emitter: _core.AudioEmitter) -> None: ...

    async def _attempt(self) -> None:
        output_emitter = _core.AudioEmitter(label=self._tts.label, dst_ch=self._event_ch)
        try:
            await self._run(output_emitter)
            output_emitter.end_input()
            # wait for all audio frames to be pushed & propagate errors
            await output_emitter.join()
            if problem := self._invalid(output_emitter):
                raise APIError(problem)
        except APIError as e:
            # 499 (Client Closed Request) - close gracefully without raising
            if isinstance(e, APIStatusError) and e.status_code == 499:
                return
            self._emit_error(e, recoverable=False)
            raise
        finally:
            await output_emitter.aclose()
"#;

/// The one-shot synthesis's idiom.
const CHUNKED: &str = r#"
    def __init__(self, *, tts: {Provider}, input_text: str, conn_options: APIConnectOptions) -> None:
        self._input_text = input_text
        self._tts = tts
        self._conn_options = conn_options
        # the full text is submitted to the provider at creation time
        self._started_time: float = time.perf_counter()
        self._event_ch = aio.Chan[{Audio}]()
        self._input_tokens = 0
        self._output_tokens = 0
        self._acquire_time: float = 0.0
        self._connection_reused: bool = False
        self._synthesize_task = asyncio.create_task(self._attempt(), name="TTS._synthesize_task")
        self._synthesize_task.add_done_callback(lambda _: self._event_ch.close())

    @property
    def input_text(self) -> str:
        return self._input_text

    @property
    def done(self) -> bool:
        return self._synthesize_task.done()

    @property
    def exception(self) -> BaseException | None:
        return self._synthesize_task.exception()
{SYNTHESIS}
    def _invalid(self, output_emitter: _core.AudioEmitter) -> str | None:
        return _core.validate_synthesis(self._input_text, output_emitter.pushed_duration(), None)  # type: ignore[no-any-return]

    async def collect(self) -> rtc.AudioFrame:
        """Utility method to collect every frame in a single call"""
        return rtc.combine_audio_frames([ev.frame async for ev in self])

    async def aclose(self) -> None:
        """Close is automatically called if the stream is completely collected"""
        await aio.cancel_and_wait(self._synthesize_task)
        self._event_ch.close()

    async def __anext__(self) -> {Audio}:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._synthesize_task.cancelled() and (exc := self._synthesize_task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        val.frame.userdata[USERDATA_TTS_STARTED_TIME] = self._started_time
        return val

    def __aiter__(self) -> AsyncIterator[{Audio}]:
        return self
"#;

/// The live synthesis's idiom.
const SYNTHESIZE: &str = r#"
    class _FlushSentinel:
        """Marks the end of a segment in ``_input_ch``."""

    def __init__(self, *, tts: {Provider}, conn_options: APIConnectOptions) -> None:
        super().__init__()
        self._tts = tts
        self._conn_options = conn_options
        self._input_ch = aio.Chan[str | {Stream}._FlushSentinel]()
        self._event_ch = aio.Chan[{Audio}]()
        self._task = asyncio.create_task(self._attempt(), name="TTS._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._started_time: float = 0
        self._pushed_text: str = ""
        # driven by plugins that manage their own segments (Rime)
        self._input_buffer: list[str | {Stream}._FlushSentinel] = []
        self._mtc_text = ""
        self._mtc_pending_texts: list[str] = []
        self._input_ended = False
        self._segment_open = False
        self._num_segments = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._acquire_time: float = 0.0
        self._connection_reused: bool = False
{SYNTHESIS}
    def _invalid(self, output_emitter: _core.AudioEmitter) -> str | None:
        return _core.validate_synthesis(  # type: ignore[no-any-return]
            self._pushed_text,
            output_emitter.pushed_duration(),
            (self._num_segments, output_emitter.num_segments),
        )

    def _mark_started(self) -> None:
        if self._started_time == 0:
            self._started_time = time.perf_counter()

    def push_text(self, token: str) -> None:
        """Push some text to be synthesized"""
        if not token or self._input_ch.closed:
            return
        if not self._segment_open:
            if self._num_segments >= 1:
                logger.warning(
                    "SynthesizeStream: handling multiple segments in a single instance is "
                    "deprecated. Please create a new SynthesizeStream instance for each segment."
                )
                return
            self._segment_open = True
            self._num_segments += 1
        self._pushed_text += token
        self._input_ch.send_nowait(token)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        if self._input_ch.closed:
            return
        self._segment_open = False
        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Mark the end of input, no more text will be pushed"""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Close ths stream immediately"""
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()
        self._input_ch.close()

    async def __anext__(self) -> {Audio}:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        if self._started_time:
            val.frame.userdata[USERDATA_TTS_STARTED_TIME] = self._started_time
        return val

    def __aiter__(self) -> AsyncIterator[{Audio}]:
        return self
"#;

/// `synthesize()` over the provider's stream: one push, then the end of input.
const FROM_STREAM: &str = r#"

class _ChunkedStreamFromStream({Chunked}):
    """``synthesize()`` over a provider that only streams."""

    async def _run(self, output_emitter: _core.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=False,
        )
        async with self._tts.stream(conn_options=self._conn_options) as stream:
            stream.push_text(self._input_text)
            stream.end_input()
            async for ev in stream:
                output_emitter.push(ev.frame.data.tobytes())
                if timed_transcripts := ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT):
                    output_emitter.push_timed_transcript(timed_transcripts)
        output_emitter.flush()
"#;
