use super::{Module, def, with_doc};
use crate::class_stub::Dir;
use crate::decl::{Class, Exit, ExitCall, FacadeClass, FacadeMethod, Method, MethodKind, Ty};

impl Module<'_> {
    /// `from_native`: `__init__(self, native, <#[host(new)] parameters>)`, keeping each.
    pub(super) fn from_native(
        &mut self,
        facade: &FacadeClass,
        native: &Class,
        new: Option<&FacadeMethod>,
    ) -> Result<String, String> {
        let mut signature = vec!["self".to_owned(), format!("native: _core.{}", native.name)];
        let mut body = String::new();
        if facade.base.is_some() {
            body.push_str("        super().__init__()\n");
        }
        body.push_str("        self._native = native\n");
        if let Some(new) = new {
            let (params, values) = self.parameters(native, new)?;
            signature.extend(params.into_iter().skip(1));
            for (n, _) in values {
                let attribute = match new.delegated.iter().find(|(d, _)| *d == n) {
                    Some((_, attribute)) => attribute.clone(),
                    None => format!("_{n}"),
                };
                body.push_str(&format!("        self.{attribute} = {n}\n"));
            }
        }
        let mut text = def("__init__", false, &signature, "None");
        if let Some(new) = new {
            crate::python::docstring(&mut text, &new.method.doc, 8);
        }
        text.push_str(&body);
        Ok(text)
    }

    /// `exit = m`: an `async with` entered by the native call `enter` on `target`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn context_manager(
        &mut self,
        native: &Class,
        method: &FacadeMethod,
        called: Option<&Method>,
        exit: &Exit,
        signature: &[String],
        target: &str,
        enter: &str,
    ) -> Result<String, String> {
        let m = &method.method;
        self.stdlib.insert("contextlib");
        self.typing.add("collections.abc", "AsyncIterator");
        let mut body = String::new();
        // A delegate is read once: the exit reaches the object that entered.
        let (target, enter) = match &method.via {
            Some(_) => {
                body.push_str(&format!("        target = {target}\n"));
                let enter = enter.replacen(&format!("{target}."), "target.", 1);
                ("target".to_owned(), enter)
            }
            None => (target.to_owned(), enter.to_owned()),
        };
        let entered = called.map_or(m.ret.ok != Ty::Unit, |c| c.ret.ok != Ty::Unit);
        body.push_str(&match entered {
            true => format!("        value = {enter}\n"),
            false => format!("        {enter}\n"),
        });
        let mut ret = self.returned(method);
        let yielded = match (&exit.yields, &method.adapt.returns) {
            (Some(path), _) => format!(" self.{path}"),
            (None, Some(returns)) => {
                ret = self.module_attribute(returns)?;
                format!(" {ret}(value)")
            }
            (None, None) if entered && m.ret.ok != Ty::Unit => " value".to_owned(),
            (None, None) => String::new(),
        };
        let target_class = match &method.via {
            Some(delegate) => match &delegate.ty {
                Ty::Instance { class, .. } => Some(self.native(class)?.clone()),
                _ => None,
            },
            None => Some(native.clone()),
        };
        let close = self.exit_call(&exit.method, &target, target_class.as_ref(), entered)?;
        match &exit.abort {
            Some(abort) => {
                let abort = self.exit_call(abort, &target, target_class.as_ref(), entered)?;
                body.push_str(&format!(
                    "        try:\n            yield{yielded}\n        except BaseException:\n            {abort}\n            raise\n        {close}\n"
                ));
            }
            None => body.push_str(&format!(
                "        try:\n            yield{yielded}\n        finally:\n            {close}\n"
            )),
        }
        let text = format!(
            "    @contextlib.asynccontextmanager\n{}",
            def(&m.name, true, signature, &format!("AsyncIterator[{ret}]"))
        );
        Ok(with_doc(text, &m.doc, &body))
    }

    /// The TYPE_CHECKING declarations of what the subclass provides along the paths a facade
    /// reads: a per-call delegate's root (an attribute, or a method when called) and a bound
    /// argument's root (a property).
    pub(super) fn read_paths(
        &mut self,
        facade: &FacadeClass,
        new: Option<&FacadeMethod>,
    ) -> Vec<String> {
        let kept: Vec<String> = new
            .map(|n| n.method.params.iter().map(|p| format!("_{}", p.name)).collect())
            .unwrap_or_default();
        let mut roots = Vec::<String>::new();
        let mut out = Vec::new();
        let mut declare = |root: &str, text: String| {
            if root != "_native"
                && !kept.iter().any(|k| k == root)
                && !roots.iter().any(|r| r == root)
            {
                roots.push(root.to_owned());
                out.push(text);
            }
        };
        for delegate in facade.delegates.iter().filter(|d| !d.held) {
            let path = &delegate.attribute;
            let root = path.split(['.', '(']).next().unwrap_or(path);
            let text = match (path.as_str() == root, path[root.len()..].starts_with("()")) {
                (true, _) => {
                    let ty = super::annotation(&delegate.ty, Dir::In, &mut self.typing);
                    format!("{root}: {ty}\n")
                }
                (false, true) => format!("def {root}(self) -> Any: ...\n"),
                (false, false) => format!("{root}: Any\n"),
            };
            declare(root, text);
        }
        // A called root is a method the subclass defines; an attribute root is read through
        // `Any` (`bound_path`), the subclass free to make it a property or an attribute.
        let binds = facade.methods.iter().flat_map(|m| m.adapt.bind.iter().map(|(_, p)| p));
        let yields = facade.methods.iter().filter_map(|m| m.exit.as_ref()?.yields.as_ref());
        for path in binds.chain(yields) {
            let root = path.split(['.', '(']).next().unwrap_or(path);
            if path[root.len()..].starts_with("()") {
                declare(root, format!("def {root}(self) -> Any: ...\n"));
            }
        }
        out
    }

    /// `module.Name` read through its module at the call: the class a handwritten module
    /// defines, which may itself import this generated module.
    fn module_attribute(&mut self, dotted: &str) -> Result<String, String> {
        let (module, name) =
            dotted.rsplit_once('.').ok_or_else(|| format!("`{dotted}` names module.item"))?;
        let (package, leaf) = module.rsplit_once('.').unwrap_or(("", module));
        let package = if package.trim_start_matches('.').is_empty() {
            format!("{package}.")
        } else {
            package.to_owned()
        };
        self.runtime.add(&package, leaf);
        Ok(format!("{leaf}.{name}"))
    }

    /// One exit statement: the entered value's own method, or the target's, passed the value
    /// when it takes one.
    fn exit_call(
        &self,
        exit: &ExitCall,
        target: &str,
        target_class: Option<&Class>,
        entered: bool,
    ) -> Result<String, String> {
        let (receiver, class) = match &exit.on_value {
            Some(class) => ("value", self.native(class)?),
            None => (target, target_class.ok_or("the exit's target is no native class")?),
        };
        let called = class
            .methods
            .iter()
            .find(|n| n.name == exit.method && n.kind != MethodKind::New)
            .ok_or_else(|| format!("_core.{} has no method {}", class.name, exit.method))?;
        let takes = called.params.iter().any(|p| !p.ty.hidden());
        if takes && (exit.on_value.is_some() || !entered) {
            return Err(format!("{} takes what nothing enters", exit.method));
        }
        let awaited = if called.kind == MethodKind::Await { "await " } else { "" };
        let value = if takes { "value" } else { "" };
        Ok(format!("{awaited}{receiver}.{}({value})", exit.method))
    }
}

/// The attributes a facade declares itself: `_native`, its kept constructor parameters and its
/// held delegates.
pub(super) fn known_roots(facade: &FacadeClass, new: Option<&FacadeMethod>) -> Vec<String> {
    let mut known = vec!["_native".to_owned()];
    known.extend(new.iter().flat_map(|n| n.method.params.iter().map(|p| format!("_{}", p.name))));
    known.extend(facade.delegates.iter().filter(|d| d.held).map(|d| d.attribute.clone()));
    known
}

/// `self.<path>`, read through `Any` when its root is an attribute the subclass provides.
pub(super) fn bound_path(known: &[String], path: &str) -> String {
    let root = path.split(['.', '(']).next().unwrap_or(path);
    match known.iter().any(|k| k == root) || path[root.len()..].starts_with("()") {
        true => format!("self.{path}"),
        false => format!("cast(Any, self).{path}"),
    }
}
