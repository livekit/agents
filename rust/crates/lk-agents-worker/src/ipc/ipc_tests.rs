#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "tests drive tokio's sync primitives directly"
)]
/// Every `log` line the code under test writes, with its key-values as text.
pub(super) mod logs {
    use parking_lot::Mutex;
    use std::sync::LazyLock;

    #[derive(Clone, Debug)]
    pub(crate) struct Line {
        pub(crate) level: log::Level,
        pub(crate) message: String,
        pub(crate) fields: Vec<(String, String)>,
    }

    impl Line {
        pub(crate) fn field(&self, key: &str) -> Option<&str> {
            self.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
        }
    }

    static LINES: Mutex<Vec<Line>> = Mutex::new(Vec::new());
    /// Woken on every line.
    pub(crate) static LOGGED: LazyLock<tokio::sync::Notify> =
        LazyLock::new(tokio::sync::Notify::new);

    struct Fields(Vec<(String, String)>);

    impl<'kvs> log::kv::VisitSource<'kvs> for Fields {
        fn visit_pair(
            &mut self,
            key: log::kv::Key<'kvs>,
            value: log::kv::Value<'kvs>,
        ) -> Result<(), log::kv::Error> {
            self.0.push((key.to_string(), value.to_string()));
            Ok(())
        }
    }

    struct Capture;

    impl log::Log for Capture {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, record: &log::Record<'_>) {
            let mut fields = Fields(Vec::new());
            let _ = record.key_values().visit(&mut fields);
            let message = record.args().to_string();
            LINES.lock().push(Line { level: record.level(), message, fields: fields.0 });
            LOGGED.notify_waiters();
        }
        fn flush(&self) {}
    }

    /// Makes the capture the test process's logger.
    pub(crate) fn capture() {
        if log::set_logger(&Capture).is_ok() {
            log::set_max_level(log::LevelFilter::Trace);
        }
    }

    pub(crate) fn lines() -> Vec<Line> {
        LINES.lock().clone()
    }
}
