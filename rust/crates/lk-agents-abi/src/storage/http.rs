#[crate::host(class(name = "_core_ThreadServer"))]
pub trait ThreadServer: Send + Sync + Sized + 'static {
    #[host(getter)]
    fn port(&self) -> u16;
    #[host(await)]
    async fn aclose(&self);
}
