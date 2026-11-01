//! Stream contracts are resolved by their ABI declarations, never by short type names.
use super::*;

impl Resolver<'_> {
    pub(super) fn is_provider_stream(&self, module: usize, item: usize) -> bool {
        let path = syn::parse_quote!(crate::utils::stream::ProviderStream);
        self.krate.resolve(0, &path).is_ok_and(|f| f == Found::Item { module, item })
    }

    pub(super) fn is_stream_source(&self, module: usize, item: usize) -> bool {
        let path = syn::parse_quote!(crate::utils::stream::StreamOutput);
        self.krate.resolve(0, &path).is_ok_and(|f| f == Found::Item { module, item })
            || self.is_provider_stream(module, item)
    }

    pub(super) fn declared_duplex(
        &self,
        module: usize,
        item: usize,
        args: &[Shape],
        at: &str,
    ) -> Result<Option<Ty>, String> {
        let wanted =
            self.krate.resolve(0, &syn::parse_quote!(crate::utils::stream::DuplexStream)).ok();
        if wanted != Some(Found::Item { module, item }) {
            return Ok(None);
        }
        if let Some((input, output, provider)) = self.duplex(args) {
            return Ok(Some(Ty::HostDuplex {
                input: Box::new(self.classify(input, at)?),
                output: Box::new(self.classify(output, at)?),
                provider,
            }));
        }
        let types = args.iter().map(|arg| self.classify(arg, at)).collect::<Result<Vec<_>, _>>()?;
        match &types[..] {
            [Ty::Sink { item: input, class }, Ty::Source { item: output, class: other }]
                if class == other =>
            {
                Ok(Some(Ty::Duplex {
                    input: input.clone(),
                    output: output.clone(),
                    class: class.clone(),
                }))
            }
            _ => Err(format!("{at}: a duplex declares its input and output halves")),
        }
    }

    pub(super) fn duplex<'s>(&self, items: &'s [Shape]) -> Option<(&'s Shape, &'s Shape, bool)> {
        let [sink, source] = items else { return None };
        let Shape::Named { found: Found::External(b), args } = sink else { return None };
        let [Shape::Dyn { found, args }] = &args[..] else { return None };
        let wanted =
            self.krate.resolve(0, &syn::parse_quote!(crate::utils::stream::StreamInput)).ok()?;
        if !is(b, &["std", "boxed", "Box"]) || *found != wanted {
            return None;
        }
        let [input] = &args[..] else { return None };
        let Shape::Named { found: Found::External(b), args } = source else { return None };
        let [Shape::Dyn { found: Found::Item { module, item }, args }] = &args[..] else {
            return None;
        };
        let [output] = &args[..] else { return None };
        (is(b, &["std", "boxed", "Box"]) && self.is_stream_source(*module, *item))
            .then(|| (input, output, self.is_provider_stream(*module, *item)))
    }
}
