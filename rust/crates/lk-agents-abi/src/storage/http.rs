//! The local project's ThreadService listener, owned for the worker's lifetime.

#[crate::host(class(name = "_core_ThreadServer"))]
pub trait ThreadServer: Send + Sync + Sized + 'static {
    #[host(getter)]
    fn port(&self) -> u16;
    #[host(await)]
    async fn aclose(&self);
}
