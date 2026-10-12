//! Connection lifetime and reuse policy. The host owns handles and performs I/O.
use std::collections::{BTreeMap, BTreeSet};

#[crate::class(name = "ConnectionPoolState", mutable)]
pub struct ConnectionPool {
    maximum_age: Option<f64>,
    refresh_on_get: bool,
    connections: BTreeMap<u64, f64>,
    available: BTreeSet<u64>,
    closing: Vec<u64>,
}
#[crate::host(object(name = "ConnectionPoolState", pid = false))]
impl ConnectionPool {
    #[host(new)]
    pub fn new(maximum_age: Option<f64>, refresh_on_get: bool) -> Self {
        Self {
            maximum_age,
            refresh_on_get,
            connections: BTreeMap::new(),
            available: BTreeSet::new(),
            closing: vec![],
        }
    }
    #[host(method)]
    pub fn connected(&mut self, id: u64, now: f64) {
        self.connections.insert(id, now);
    }
    #[host(method)]
    pub fn empty(&self) -> bool {
        self.connections.is_empty()
    }
    #[host(method)]
    pub fn get(&mut self, now: f64) -> Option<u64> {
        while let Some(id) = self.available.pop_first() {
            let connected = self.connections[&id];
            if self.maximum_age.is_none_or(|age| now - connected <= age) {
                if self.refresh_on_get {
                    self.connections.insert(id, now);
                }
                return Some(id);
            }
            self.remove(id);
        }
        None
    }
    #[host(method)]
    pub fn put(&mut self, id: u64) {
        if self.connections.contains_key(&id) {
            self.available.insert(id);
        }
    }
    #[host(method)]
    pub fn remove(&mut self, id: u64) {
        self.available.remove(&id);
        if self.connections.remove(&id).is_some() {
            self.closing.push(id);
        }
    }
    #[host(method)]
    pub fn invalidate(&mut self) {
        self.closing.extend(self.connections.keys());
        self.connections.clear();
        self.available.clear();
    }
    #[host(method)]
    pub fn next_closing(&mut self) -> Option<u64> {
        self.closing.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refresh_expiration_and_invalidated_checkout() {
        let mut p = ConnectionPool::new(Some(10.0), true);
        p.connected(1, 0.0);
        p.put(1);
        assert_eq!(p.get(10.0), Some(1));
        p.put(1);
        assert_eq!(p.get(20.0), Some(1));
        p.put(1);
        assert_eq!(p.get(31.0), None);
        assert_eq!(p.next_closing(), Some(1));
        p.connected(2, 31.0);
        p.invalidate();
        p.put(2);
        assert_eq!(p.get(31.0), None);
        assert_eq!(p.next_closing(), Some(2));
        assert_eq!(p.next_closing(), None);
    }
}
