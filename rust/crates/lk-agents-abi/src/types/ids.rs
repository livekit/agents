// The protocol's `guid` format: a prefix then twelve characters of the shortuuid alphabet, as
// LiveKit Cloud mints its own.

#[crate::host(functions)]
pub trait Ids {
    /// A new id: `prefix` then twelve random characters.
    #[host(function)]
    fn shortuuid(#[host(default = "")] prefix: String) -> String;
}

// Seeded ids, for code that may draw no randomness (the sans-I/O cores, the DST): `<prefix>_<12
// hex>` as v1's `shortuuid`, from a seed its caller drew at random.

/// The bits after an id's prefix: 12 hex digits, as v1's `shortuuid`.
pub const ID_MASK: u64 = (1 << 48) - 1;

/// The `n`th id of `seed`'s stream. An odd step makes `n -> id` one-to-one over 48 bits: one
/// stream never repeats an id.
pub fn nth(seed: u64, n: u64) -> u64 {
    const STEP: u64 = 0x9E37_79B9_7F4B; // odd: the 48-bit golden ratio, scattering ids
    seed.wrapping_add(n.wrapping_mul(STEP)) & ID_MASK
}

/// `<prefix>_<12 hex>`, the `n`th id of `seed`'s stream.
pub fn short(prefix: &str, seed: u64, n: u64) -> String {
    format!("{prefix}_{:012x}", nth(seed, n))
}

/// The `n`th value of splitmix64 from `seed`: the seed of an independent stream. Streams are
/// [`nth`]'s arithmetic progressions, so a stream seeded from another's [`nth`] would run along
/// it and repeat its ids; a mixed seed lands anywhere.
pub fn splitmix64(seed: u64, n: u64) -> u64 {
    let mut z = seed.wrapping_add(n.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    // Catches: one stream minting an id twice; two items then share every map entry and one
    // provider group.
    #[test]
    fn a_stream_never_repeats_an_id() {
        let ids: BTreeSet<u64> = (0..100_000).map(|n| nth(u64::MAX, n)).collect();
        assert_eq!(ids.len(), 100_000);
        assert_eq!(short("item", 0, 1).len(), "item_".len() + 12);
    }

    // Catches: streams forked from one session overlapping, as `nth`-derived seeds did: a later
    // reply's message took an earlier one's id and replaced it in the history.
    #[test]
    fn forked_streams_do_not_overlap() {
        let session = 7;
        let mut seen = BTreeSet::new();
        for fork in 0..200 {
            let seed = splitmix64(session, fork);
            for n in 0..50 {
                assert!(seen.insert(nth(seed, n)), "fork {fork} repeated an id at {n}");
            }
        }
    }
}
