//! The `BoundedDict` stand-in (`inference/interruption.py:433`, `maxsize=10`). Not an LRU: a hit
//! never promotes, so eviction is FIFO by insertion, and the keys are chronological. A linear
//! `Vec` of pairs, since it never holds more than ten entries.

use std::num::NonZeroUsize;

#[derive(Debug, Clone)]
pub struct BoundedCache<K, V> {
    entries: Vec<(K, V)>,
    maxsize: NonZeroUsize,
}

impl<K: PartialEq, V> BoundedCache<K, V> {
    /// `BoundedDict(maxsize=...)`; `NonZeroUsize` makes the reference's `maxsize <= 0` error
    /// unrepresentable.
    pub fn new(maxsize: NonZeroUsize) -> Self {
        BoundedCache { entries: Vec::new(), maxsize }
    }

    /// Oldest first -- `OrderedDict.items()`.
    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.entries.iter()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// `d[key] = value`, including the eviction loop. An existing key is overwritten in place.
    pub fn insert(&mut self, key: K, value: V) {
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
            return;
        }
        self.entries.push((key, value));
        // `while`, as `BoundedDict` writes it; runs at most once here.
        while self.entries.len() > self.maxsize.get() {
            self.entries.remove(0);
        }
    }

    /// `BoundedDict.set_or_update` (`utils/bounded_dict.py:52`): create on miss, apply the update
    /// either way, return the live entry.
    pub fn set_or_update(
        &mut self,
        key: K,
        factory: impl FnOnce() -> V,
        apply: impl FnOnce(&mut V),
    ) -> &mut V
    where
        K: Copy,
    {
        if !self.entries.iter().any(|(k, _)| *k == key) {
            self.insert(key, factory());
        }
        let slot = self
            .entries
            .iter_mut()
            .find(|(k, _)| *k == key)
            .expect("just inserted, and maxsize >= 1 so it cannot have been evicted");
        apply(&mut slot.1);
        &mut slot.1
    }

    /// `BoundedDict.pop_if(predicate)` (`utils/bounded_dict.py:73`), newest first
    /// (`reversed(...)`).
    pub fn pop_if(&mut self, predicate: impl Fn(&V) -> bool) -> Option<(K, V)> {
        let idx = self.entries.iter().rposition(|(_, v)| predicate(v))?;
        Some(self.entries.remove(idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(maxsize: usize) -> BoundedCache<u64, i32> {
        BoundedCache::new(NonZeroUsize::new(maxsize).unwrap())
    }

    fn keys(c: &BoundedCache<u64, i32>) -> Vec<u64> {
        c.iter().map(|(k, _)| *k).collect()
    }

    // Catches: eviction from the wrong end.
    #[test]
    fn overflow_evicts_the_oldest_insertion() {
        let mut c = cache(3);
        for k in 1..=5 {
            c.insert(k, k as i32);
        }
        assert_eq!(keys(&c), vec![3, 4, 5]);
    }

    // Catches: implementing this as an LRU.
    #[test]
    fn updating_an_entry_does_not_promote_it() {
        let mut c = cache(3);
        for k in 1..=3 {
            c.insert(k, 0);
        }
        c.set_or_update(1, || 0, |v| *v = 99);
        assert_eq!(keys(&c), vec![1, 2, 3], "an update must not reorder");
        c.insert(4, 0);
        assert_eq!(keys(&c), vec![2, 3, 4], "and the updated entry is still the one evicted");
    }

    // Catches: `set_or_update` skipping the update when it had to create the entry.
    #[test]
    fn set_or_update_applies_the_update_on_the_miss_path_too() {
        let mut c = cache(2);
        let created = *c.set_or_update(7, || 0, |v| *v = 42);
        assert_eq!(created, 42);
        assert_eq!(keys(&c), vec![7]);
    }

    // Catches: `pop_if` scanning forwards instead of taking the last match.
    #[test]
    fn pop_if_takes_the_newest_match_and_removes_only_that_one() {
        let mut c = cache(5);
        for (k, v) in [(1, 10), (2, 0), (3, 30), (4, 0)] {
            c.insert(k, v);
        }
        assert_eq!(c.pop_if(|v| *v > 0), Some((3, 30)));
        assert_eq!(keys(&c), vec![1, 2, 4]);
        assert_eq!(c.pop_if(|v| *v > 0), Some((1, 10)));
        assert_eq!(c.pop_if(|v| *v > 0), None, "no match must leave the cache untouched");
        assert_eq!(keys(&c), vec![2, 4]);
    }
}
