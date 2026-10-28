//! The plugin base classes: each `#[python(provider(..))]` trait as the Python class plugins
//! subclass, written after its module's records.
//!
//! The provider class is read off the trait, method by method: its constructor, properties and
//! hooks come from each method's role ([`Role`]), and every method gets the `_core_<method>` the
//! proxy calls, so a trait method the base class does not answer is a codegen error rather than a
//! runtime one. The classes a method returns (`stream(kind = ..)`) are the host-stream idioms
//! below: the channels, the task running the plugin's `_run`, and the `_core_push`/`_core_flush`/
//! `_core_end_input` surface the core feeds them through.

mod idioms;

use crate::ir::*;
use idioms::*;

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
                if let Some(quoted) = tail.strip_prefix("``") {
                    let end = quoted.find("``").map_or(tail.len(), |j| j + 4);
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
        if matches!(t, syn::Type::Tuple(_))
            || generic(t, "Box").is_some()
            || generic(t, "Arc").is_some()
        {
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
            "ProviderError" => "Exception".to_owned(),
            "ChatRequest" => {
                self.need("..", "_core");
                "_core.ChatRequest".to_owned()
            }
            // Not a record: a stream alias (`TtsSource`), the class the method returns.
            other => self
                .record(other)
                .or_else(|| m.stream.as_ref().map(|s| s.name.clone()))
                .ok_or_else(|| format!("{}: `{other}` has no Python name", m.rust))?,
        })
    }

    fn conn(&self) -> Option<&ProviderMethod> {
        self.p.methods.iter().find(|m| matches!(m.role, Role::Conn))
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
            self.need("typing_extensions", "TypeVar");
            self.need("typing", "Never");
            self.need("typing", "Generic");
            self.need("typing", "Literal");
            self.need("livekit", "rtc");
            b.push_str("TEvent = TypeVar(\"TEvent\", default=Never, contravariant=True)\n\n\n");
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
                Role::Hook { name, .. } if self.impl_is_task(m) => {
                    self.need("", "asyncio");
                    lines.push(format!(
                        "self.{}: asyncio.Task[None] | None = None",
                        self.task_attr(name)
                    ));
                }
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
        let realtime =
            p.methods.iter().find(|m| m.stream.as_ref().is_some_and(|x| x.kind == "realtime"));
        if realtime.is_some() {
            self.need("typing", "Any");
            b.push_str(&format!(
                "\n    def _core_with_tools(self, tools: Callable[[], Sequence[Tool | Toolset]]) -> Any:\n        \"\"\"The model as a session hands it to the framework: its sessions offer the tools\n        ``tools`` names.\"\"\"\n        return _{class}Tools(self, tools)\n"
            ));
        }
        self.paths.push((class.clone(), p.path.clone()));
        if let Some(m) = realtime {
            let session = m.stream.as_ref().map(|x| x.name.clone()).unwrap_or_default();
            b.push_str(&format!(
                "\n\nclass _{class}Tools:\n    \"\"\"``{class}`` as a session holds it: the tools its sessions offer.\"\"\"\n\n    def __init__(\n        self, provider: {class}, tools: Callable[[], Sequence[Tool | Toolset]]\n    ) -> None:\n        self._provider = provider\n        self._tools = tools\n\n    def __getattr__(self, name: str) -> Any:\n        return getattr(self._provider, name)\n\n    async def _core_{rust}(self, turn_detection_disabled: bool) -> {session}:\n        return await self._provider._core_{rust}(turn_detection_disabled, self._tools)\n",
                rust = m.rust,
            ));
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
            if let Some(default) = &p.default {
                kw.push(format!("{}: {ty} = {default}", p.name));
            } else if optional {
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
            if p.default.is_some() {
                kw.push(format!("{0}={0}", p.name));
            } else if p.rust_ty.starts_with("Option<") {
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
            Role::Hook { name, implementation, retry, unsupported, store } => {
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
                    let call = format!("self.{imp}({args})");
                    match retry {
                        Some(helper) if is_async && self.carries_conn(m) => {
                            self.need("..", "_bindings");
                            let call = format!("_bindings.{helper}(self, {args})");
                            b.push_str("        ");
                            b.push_str(&wrap_call("return await ", &call, 8));
                            b.push('\n');
                        }
                        Some(_) => {
                            return Err(format!(
                                "{}: retry needs an awaited call with conn_options",
                                m.rust
                            ));
                        }
                        None => b.push_str(&format!("        return {aw}{call}\n")),
                    }
                    return Ok(());
                }
                let params = self.hook_params(m, false)?;
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
            Role::Conn | Role::Value(_) | Role::Report => {}
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
        if let Some((name, _)) =
            self.p.error.clone().filter(|_| self.p.events.iter().any(|e| e == "error"))
        {
            self.need("", "time");
            b.push_str(&format!(
                "\n    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:\n        self.emit(\n            \"error\",\n            {name}(\n                timestamp=time.time(),\n                label=self._label,\n                error=api_error,\n                recoverable=recoverable,\n            ),\n        )\n"
            ));
        }
        self.need("typing_extensions", "Self");
        self.need("types", "TracebackType");
        let tasks: Vec<String> = self
            .p
            .methods
            .iter()
            .filter(|m| self.impl_is_task(m))
            .filter_map(|m| match &m.role {
                Role::Hook { name, .. } => Some(self.task_attr(name)),
                _ => None,
            })
            .collect();
        // With nothing of its own to close it is a no-op a provider overrides, not an omission.
        let noqa = if tasks.is_empty() { "  # noqa: B027" } else { "" };
        b.push_str(&format!(
            "\n    async def aclose(self) -> None:{noqa}\n        \"\"\"Close the provider and everything it opened.\"\"\"\n"
        ));
        for task in tasks {
            self.need("..utils", "aio");
            b.push_str(&format!(
                "        if self.{task} is not None:\n            await aio.cancel_and_wait(self.{task})\n"
            ));
        }
        b.push_str(ACONTEXT);
        Ok(())
    }

    /// One method's `_core_` surface.
    fn core(&mut self, b: &mut String, m: &ProviderMethod) -> Result<(), String> {
        if matches!(m.role, Role::Conn) {
            return Ok(());
        }
        let def = if m.is_async { "async def" } else { "def" };
        let mut params = Vec::new();
        for p in &m.params {
            params.push(format!("{}: {}", p.name, self.core_ty(m, &p.rust_ty)?));
        }
        if m.stream.as_ref().is_some_and(|x| x.kind == "realtime") {
            self.tools_types();
            params.push("tools: Callable[[], Sequence[Tool | Toolset]] = list".to_owned());
        }
        let routed = self.carries_conn(m);
        if routed {
            self.need("..types", "APIConnectOptions");
            self.need("..types", "DEFAULT_API_CONNECT_OPTIONS");
            params.push("conn: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS".to_owned());
        }
        let ret = match (&m.role, &m.ret) {
            (Role::Init, Some(r)) if self.read_as_dict(r) => {
                self.need("typing", "Any");
                "dict[str, Any]".to_owned()
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
                if self.read_as_dict(ret) {
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
            Role::Hook { name, implementation, retry, .. } => {
                let aw = if m.is_async && m.stream.is_none() { "await " } else { "" };
                let args = self.hook_args(m, "conn");
                // The framework's call is one attempt: it retries it itself.
                let target = match (retry, implementation) {
                    (Some(_), Some(imp)) => imp,
                    _ => name,
                };
                let call = format!("self.{target}({args})");
                // A stream the framework opens runs one attempt and reports nothing (`_core_attempts`).
                let driven = m.stream.as_ref().is_some_and(|x| {
                    matches!(x.kind.as_str(), "recognize" | "synthesize" | "chunked")
                });
                let guard = m.when.as_ref().map(|cap| {
                    format!("if not self.capabilities.{cap}:\n            return None\n        ")
                });
                let mut attrs: Vec<&ProviderParam> = m.params.iter().filter(|p| p.attr).collect();
                let mut s = guard.unwrap_or_default();
                let attach = m.stream.as_ref().is_some_and(|x| x.kind == "realtime");
                if attach {
                    attrs.clear();
                }
                if attach {
                    s.push_str(&wrap_call(&format!("out = {aw}"), &call, 8));
                    s.push_str("\n        out._core_attach(tools)\n        return out");
                } else if attrs.is_empty() && !driven {
                    let ret = if m.ret.is_some() { "return " } else { "" };
                    s.push_str(&wrap_call(&format!("{ret}{aw}"), &call, 8));
                } else {
                    s.push_str(&wrap_call(&format!("out = {aw}"), &call, 8));
                    if driven {
                        s.push_str("\n        out._core_attempts.core = True");
                    }
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
            Role::Conn => unreachable!("a `conn` has no `_core_` surface"),
            Role::Markup => "return self.markup._provider_key()".to_owned(),
            Role::Report => "self._emit_error(error, recoverable)".to_owned(),
            Role::Chat { name } => self.chat_core(m, name)?,
            Role::Value(expr) => format!("return {expr}"),
        };
        b.push_str(&format!("        {body}\n"));
        Ok(())
    }

    /// Whether the core reads the record as a dict (`#[binding(from_items)]`).
    fn read_as_dict(&self, rust: &str) -> bool {
        let Ok(t) = syn::parse_str::<syn::Type>(rust) else { return false };
        let Some(name) = last_ident(&t) else { return false };
        self.decls.iter().any(|d| d.rust_name == name && d.targets.iter().any(|t| t.from_items))
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
            "chat" => self.chat_stream(s)?,
            "vad" => self.vad_stream(s)?,
            "realtime" => self.realtime_session(m, s)?,
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
                "\n    def _core_push(self, {arg}: {ty}) -> None:\n        self.{push}({arg})\n\n    def _core_flush(self) -> None:\n        if not self.input.closed:\n            self.flush()\n\n    def _core_end_input(self) -> None:\n        if not self.input.closed:\n            self.end_input()\n"
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

    /// A plain hook calling an `impl`: the coroutine the hook schedules once.
    fn impl_is_task(&self, m: &ProviderMethod) -> bool {
        !m.is_async && matches!(&m.role, Role::Hook { implementation: Some(_), .. })
    }

    /// The task idiom: `name()` schedules `imp()` once on the loop, fire and forget, unless the
    /// plugin left `imp` as it is; `aclose` cancels it.
    fn task_hook(
        &mut self,
        b: &mut String,
        m: &ProviderMethod,
        name: &str,
        imp: &str,
    ) -> Result<(), String> {
        self.need("", "asyncio");
        let class = &self.p.class;
        b.push_str(&format!(
            "\n    def {name}(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:\n"
        ));
        docstring(b, &m.doc, 8);
        b.push_str(&format!(
            "        if type(self).{imp} is {class}.{imp}:\n            return  # nothing to warm\n        if self.{task} is not None:\n            return\n        if loop is None:\n            loop = asyncio.get_event_loop()\n\n        async def _run() -> None:\n            try:\n                await self.{imp}()\n            except Exception:\n                pass\n\n        self.{task} = loop.create_task(_run())\n\n    async def {imp}(self) -> None:\n        \"\"\"The provider's own ``{name}``: overriding it enables ``{name}()``.\n\n        A cheap, token-free request (e.g. listing models) on the client that serves the\n        requests, so DNS and TLS are established and a connection is left in the pool. Its\n        exceptions are swallowed; a subclass overriding ``aclose`` must call\n        ``await super().aclose()`` to cancel one in flight.\n        \"\"\"\n",
            task = self.task_attr(name),
        ));
        Ok(())
    }

    /// The task a task hook keeps: `_<name>_task`.
    fn task_attr(&self, name: &str) -> String {
        format!("_{}_task", name.trim_start_matches('_'))
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

    /// `chat`: the abstract method a `_core.ChatRequest` is read into.
    fn chat_hook(&mut self, b: &mut String, m: &ProviderMethod, name: &str) -> Result<(), String> {
        for (module, n) in [
            ("abc", "abstractmethod"),
            ("typing", "Any"),
            ("..types", "APIConnectOptions"),
            ("..types", "DEFAULT_API_CONNECT_OPTIONS"),
            ("..types", "NOT_GIVEN"),
            ("..types", "NotGivenOr"),
            ("?..llm.chat_context", "ChatContext"),
            ("?..llm.tool_context", "Tool"),
            ("?..llm.tool_context", "ToolChoice"),
        ] {
            self.need(module, n);
        }
        let ret = self.hook_ret(m)?;
        b.push_str(&format!(
            "\n    @abstractmethod\n    def {name}(\n        self,\n        *,\n        chat_ctx: ChatContext,\n        tools: list[Tool] | None = None,\n        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,\n        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,\n        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,\n        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,\n    ) -> {ret}:\n"
        ));
        docstring(b, &m.doc, 8);
        Ok(())
    }

    /// The request's context, offered tools and options as the hook's keywords.
    fn chat_core(&mut self, _m: &ProviderMethod, name: &str) -> Result<String, String> {
        Ok(format!(
            "from .._core_view import chat_request\n\n        chat_ctx, tools = chat_request(req)\n        out = self.{name}(\n            chat_ctx=chat_ctx,\n            conn_options=conn,\n            tools=tools,\n            tool_choice=NOT_GIVEN if req.tool_choice is None else req.tool_choice,\n            parallel_tool_calls=(\n                NOT_GIVEN if req.parallel_tool_calls is None else req.parallel_tool_calls\n            ),\n            extra_kwargs=req.extra or NOT_GIVEN,\n        )\n        out._core_attempts.core = True\n        return out"
        ))
    }

    /// The annotations of a tool pool.
    fn tools_types(&mut self) {
        for (m, n) in [
            ("collections.abc", "Callable"),
            ("collections.abc", "Sequence"),
            ("?..llm.tool_context", "Tool"),
            ("?..llm.tool_context", "Toolset"),
        ] {
            self.need(m, n);
        }
    }

    // ------------------------------------------------------------------- the stream idioms

    /// `stream(kind = "realtime", proxy = "Trait")`: the session plugins subclass, and the
    /// flattened surface the core drives it through, which must answer every method of `Trait`.
    fn realtime_session(&mut self, m: &ProviderMethod, s: &StreamClass) -> Result<String, String> {
        for (module, n) in [
            ("", "asyncio"),
            ("", "time"),
            ("abc", "abstractmethod"),
            ("collections.abc", "AsyncIterable"),
            ("typing", "Any"),
            ("typing", "Generic"),
            ("typing", "Literal"),
            ("typing_extensions", "TypeVar"),
            ("typing", "Never"),
            ("livekit", "rtc"),
            ("..log", "logger"),
            ("..types", "NOT_GIVEN"),
            ("..types", "NotGivenOr"),
            ("..utils", "is_given"),
            ("?..llm.chat_context", "ChatContext"),
            ("?..llm.realtime", "GenerationCreatedEvent"),
            ("?..llm.tool_context", "Tool"),
            ("?..llm.tool_context", "ToolChoice"),
            ("?..llm.tool_context", "ToolContext"),
        ] {
            self.need(module, n);
        }
        self.tools_types();
        let caps = m
            .ret
            .as_ref()
            .and(self.p.methods.iter().find(|x| x.role == Role::Init))
            .and_then(|x| x.ret.clone())
            .ok_or_else(|| format!("{}: a realtime model has init capabilities", m.rust))?;
        let caps = self.core_ty(m, &caps)?;
        let event = if self.p.events.is_empty() { "TEvent = TypeVar(\"TEvent\", default=Never, contravariant=True)\n\n" } else { "" };
        let text = REALTIME
            .replace("{TEVENT}", event)
            .replace("{Stream}", &s.name)
            .replace("{Provider}", &self.p.class)
            .replace("{Caps}", &caps);
        let mut b = String::new();
        let (head, body) = text.split_once("{DOC}").expect("the template has its docstring slot");
        b.push_str(head);
        docstring(&mut b, &s.doc, 4);
        b.push_str(body);
        // Every method of the session's proxy trait has its `_core_` here.
        let trait_name =
            s.proxy.as_ref().ok_or_else(|| format!("{}: stream(proxy = \"Trait\")", m.rust))?;
        let proxy = self
            .x
            .proxies
            .iter()
            .find(|x| x.trait_name == *trait_name)
            .ok_or_else(|| format!("{}: no #[host(proxy)] trait `{trait_name}`", m.rust))?;
        for method in &proxy.methods {
            let core = format!("def _core_{method}(");
            if !b.contains(&core) {
                return Err(format!(
                    "{}: `{}` answers no `{trait_name}::{method}` (`_core_{method}`)",
                    m.rust, s.name
                ));
            }
        }
        Ok(b)
    }

    /// `stream(kind = "vad")`: audio frames in, `VADEvent`s out; the plugin's `_main_task` is the
    /// detection.
    fn vad_stream(&mut self, s: &StreamClass) -> Result<String, String> {
        for (m, n) in [
            ("", "asyncio"),
            ("abc", "abstractmethod"),
            ("collections.abc", "AsyncIterator"),
            ("livekit", "rtc"),
            ("..utils", "aio"),
        ] {
            self.need(m, n);
        }
        let event = self.record("VadEvent").ok_or("VadEvent is not declared")?;
        let mut b = format!("class {}(ABC):\n", s.name);
        docstring(&mut b, &s.doc, 4);
        b.push_str(
            &VAD.replace("{Stream}", &s.name)
                .replace("{Provider}", &self.p.class)
                .replace("{Event}", &event),
        );
        Ok(b)
    }

    /// `stream(kind = "chat")`: one completion's chunks; a failure after output reached the
    /// reader is final (the core never retries produced output).
    fn chat_stream(&mut self, s: &StreamClass) -> Result<String, String> {
        for (m, n) in [
            ("", "asyncio"),
            ("abc", "abstractmethod"),
            ("collections.abc", "AsyncIterable"),
            ("collections.abc", "AsyncIterator"),
            ("", "time"),
            ("typing", "Any"),
            ("..", "_bindings"),
            ("..", "_core"),
            ("..types", "APIConnectOptions"),
            ("..utils", "aio"),
            ("?..llm.chat_context", "ChatContext"),
            ("?..llm.tool_context", "Tool"),
        ] {
            self.need(m, n);
        }
        let chunk = self.record("LlmChunk").ok_or("LlmChunk is not declared")?;
        let usage = self.record("CompletionUsage").ok_or("CompletionUsage is not declared")?;
        let call = self.record("FunctionToolCall").ok_or("FunctionToolCall is not declared")?;
        let collected =
            self.record("CollectedResponse").ok_or("CollectedResponse is not declared")?;
        let mut b = format!(
            "class _{}EventChannel(aio.Chan[{chunk}]):\n    def __init__(self) -> None:\n        super().__init__()\n        self.output_sent = False\n\n    def send_nowait(self, value: {chunk}) -> None:\n        super().send_nowait(value)\n        # a provider can raise before the consumer reads the chunk\n        self.output_sent = self.output_sent or value.has_response()\n\n\nclass {}(ABC):\n",
            self.p.class, s.name
        );
        docstring(&mut b, &s.doc, 4);
        b.push_str(
            &CHAT
                .replace("{Stream}", &s.name)
                .replace("{Provider}", &self.p.class)
                .replace("{Chunk}", &chunk)
                .replace("{Usage}", &usage)
                .replace("{Call}", &call)
                .replace("{Collected}", &collected),
        );
        Ok(b)
    }

    /// `stream(kind = "chunked" | "synthesize")`: one attempt run against an `AudioEmitter`, its
    /// audio checked against the text it was given.
    fn synthesis_stream(&mut self, s: &StreamClass, idiom: &str) -> Result<String, String> {
        for (m, n) in [
            ("", "asyncio"),
            ("", "time"),
            ("abc", "abstractmethod"),
            ("collections.abc", "AsyncIterator"),
            ("livekit", "rtc"),
            ("..", "_bindings"),
            ("..", "_core"),
            (".._exceptions", "APIError"),
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
            ("..", "_bindings"),
            ("..", "_core"),
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
