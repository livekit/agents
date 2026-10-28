//! Connection pool bookkeeping exposed to host transports.

#[crate::host(class(name = "ConnectionPoolState", pid = false))]
pub trait ConnectionPoolState: Send + Sized + 'static {
    #[host(new)]
    fn new(maximum_age: Option<f64>, refresh_on_get: bool) -> Self;
    #[host(getter)]
    fn generation(&self) -> u64;
    #[host(getter)]
    fn last_acquire_time(&self) -> f64;
    #[host(getter)]
    fn last_connection_reused(&self) -> bool;
    #[host(call)]
    fn connected(&mut self, id: u64, now: f64, generation: u64, acquire_time: Option<f64>) -> bool;
    #[host(call)]
    fn empty(&self) -> bool;
    #[host(call)]
    fn get(&mut self, now: f64) -> Option<u64>;
    #[host(call)]
    fn put(&mut self, id: u64);
    #[host(call)]
    fn remove(&mut self, id: u64);
    #[host(call)]
    fn invalidate(&mut self);
    #[host(call)]
    fn close_all(&mut self);
    #[host(call)]
    fn next_closing(&self) -> Option<u64>;
    #[host(call)]
    fn closed(&mut self, id: u64);
}
