//! `utils.ConnectionPool`'s bookkeeping: which connection is reused, retired, or closed. The host
//! owns the handles (by id) and performs the I/O: the handshakes and the closes this queues.
use std::collections::{BTreeMap, BTreeSet};

pub struct ConnectionPool {
    maximum_age: Option<f64>,
    refresh_on_get: bool,
    /// Each pooled connection's connect (or last refresh) time.
    connections: BTreeMap<u64, f64>,
    available: BTreeSet<u64>,
    /// Invalidated while checked out: usable by their holder, closed once returned.
    retired: BTreeSet<u64>,
    closing: Vec<u64>,
    /// Bumped by `invalidate`: a handshake started before it carries the old options.
    generation: u64,
    last_acquire_time: f64,
    last_connection_reused: bool,
}

impl lk_agents_abi::utils::connection_pool::ConnectionPoolState for ConnectionPool {
    fn new(maximum_age: Option<f64>, refresh_on_get: bool) -> Self {
        Self {
            maximum_age,
            refresh_on_get,
            connections: BTreeMap::new(),
            available: BTreeSet::new(),
            retired: BTreeSet::new(),
            closing: vec![],
            generation: 0,
            last_acquire_time: 0.0,
            last_connection_reused: false,
        }
    }

    /// Read before a handshake, handed back to `connected`.
    fn generation(&self) -> u64 {
        self.generation
    }

    /// Seconds the last `get` spent connecting; 0 when it reused a connection.
    fn last_acquire_time(&self) -> f64 {
        self.last_acquire_time
    }

    fn last_connection_reused(&self) -> bool {
        self.last_connection_reused
    }

    /// A handshake begun at `generation` completed. False when the pool was invalidated during it:
    /// the connection carries the old options and is queued for closing instead. `acquire_time` is
    /// the time a `get` spent connecting, `None` for a prewarm.
    fn connected(&mut self, id: u64, now: f64, generation: u64, acquire_time: Option<f64>) -> bool {
        if generation != self.generation {
            self.closing.push(id);
            return false;
        }
        self.connections.insert(id, now);
        if let Some(elapsed) = acquire_time {
            self.last_acquire_time = elapsed;
            self.last_connection_reused = false;
        }
        true
    }

    fn empty(&self) -> bool {
        self.connections.is_empty()
    }

    /// An available connection that has not expired; the expired ones are queued for closing.
    fn get(&mut self, now: f64) -> Option<u64> {
        while let Some(id) = self.available.pop_first() {
            let connected = self.connections[&id];
            if self.maximum_age.is_none_or(|age| now - connected <= age) {
                if self.refresh_on_get {
                    self.connections.insert(id, now);
                }
                self.last_acquire_time = 0.0;
                self.last_connection_reused = true;
                return Some(id);
            }
            self.remove(id);
        }
        None
    }

    /// Back for reuse, or queued for closing when it was retired while checked out.
    fn put(&mut self, id: u64) {
        if self.retired.contains(&id) {
            self.remove(id);
        } else if self.connections.contains_key(&id) {
            self.available.insert(id);
        }
    }

    /// Out of the pool, queued for closing.
    fn remove(&mut self, id: u64) {
        self.available.remove(&id);
        if self.retired.remove(&id) || self.connections.remove(&id).is_some() {
            self.closing.push(id);
        }
    }

    /// No existing connection is reused: the idle ones are queued for closing, the checked-out
    /// ones retired (closing them would cut a request in flight).
    fn invalidate(&mut self) {
        self.generation += 1;
        for id in std::mem::take(&mut self.connections).into_keys() {
            if self.available.contains(&id) {
                self.closing.push(id);
            } else {
                self.retired.insert(id);
            }
        }
        self.available.clear();
    }

    /// The pool is going away: every connection is queued for closing, the retired ones too.
    fn close_all(&mut self) {
        self.invalidate();
        self.closing.extend(std::mem::take(&mut self.retired));
    }

    /// The next connection to close; it stays queued until `closed`, so a close the host abandons
    /// (cancelled) is not lost.
    fn next_closing(&self) -> Option<u64> {
        self.closing.last().copied()
    }

    fn closed(&mut self, id: u64) {
        self.closing.retain(|c| *c != id);
    }
}

impl ConnectionPool {
    /// `next_closing` and `closed` in one, for a caller whose close cannot be abandoned.
    pub fn pop_closing(&mut self) -> Option<u64> {
        self.closing.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::utils::connection_pool::ConnectionPoolState;

    fn drain(p: &mut ConnectionPool) -> Vec<u64> {
        let mut out = vec![];
        while let Some(id) = p.next_closing() {
            p.closed(id);
            out.push(id);
        }
        out
    }

    #[test]
    fn refresh_and_expiration() {
        let mut p = ConnectionPool::new(Some(10.0), true);
        assert!(p.connected(1, 0.0, p.generation(), Some(0.5)));
        assert!(!p.last_connection_reused());
        assert_eq!(p.last_acquire_time(), 0.5);
        p.put(1);
        assert_eq!(p.get(10.0), Some(1));
        assert!(p.last_connection_reused());
        p.put(1);
        assert_eq!(p.get(20.0), Some(1));
        p.put(1);
        assert_eq!(p.get(31.0), None);
        assert_eq!(drain(&mut p), vec![1]);
    }

    #[test]
    fn invalidate_retires_checked_out_and_closes_idle() {
        let mut p = ConnectionPool::new(None, false);
        let g = p.generation();
        p.connected(1, 0.0, g, None);
        p.connected(2, 0.0, g, None);
        p.put(1);
        p.invalidate();
        assert_eq!(drain(&mut p), vec![1]);
        assert_eq!(p.get(0.0), None);
        // the holder of 2 keeps it until it returns it
        p.put(2);
        assert_eq!(drain(&mut p), vec![2]);
        assert_eq!(p.get(0.0), None);
    }

    #[test]
    fn a_handshake_across_an_invalidation_is_closed() {
        let mut p = ConnectionPool::new(None, false);
        let g = p.generation();
        p.invalidate();
        assert!(!p.connected(7, 0.0, g, Some(1.0)));
        assert!(p.empty());
        assert_eq!(drain(&mut p), vec![7]);
        assert!(p.connected(8, 0.0, p.generation(), Some(1.0)));
    }

    #[test]
    fn close_all_takes_the_retired_and_an_abandoned_close_stays_queued() {
        let mut p = ConnectionPool::new(None, false);
        p.connected(1, 0.0, 0, None);
        p.connected(2, 0.0, 0, None);
        p.put(2);
        p.close_all();
        let first = p.next_closing().unwrap();
        // not marked closed: still queued
        assert_eq!(p.next_closing(), Some(first));
        let mut all = drain(&mut p);
        all.sort();
        assert_eq!(all, vec![1, 2]);
        // a retired connection removed by its holder is closed once
        p.connected(3, 0.0, p.generation(), None);
        p.invalidate();
        p.remove(3);
        p.remove(3);
        assert_eq!(drain(&mut p), vec![3]);
    }
}
