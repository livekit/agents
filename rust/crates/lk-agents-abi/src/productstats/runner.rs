use crate::telemetry::decl::{Attrs, Bounded, SpanValue};

/// The command that started the process, which its host names when it installs telemetry:
/// `none` when no `lk agent` command did (a script, a test).
#[crate::host(native_enum(name = "RunnerCmd"))]
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RunnerCmd {
    Dev,
    Console,
    Simulation,
    Start,
    #[wire = "none"]
    NoCommand,
}

/// What kind of runner the process is, derived once at its start.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerKind {
    Production,
    Development,
    Ci,
    Unknown,
}

/// The process's runner, as every telemetry consumer reads it.
#[derive(Attrs, Bounded, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Runner {
    /// The command that started the process.
    #[attr("lk.product.agents.runner.cmd")]
    pub cmd: RunnerCmd,
    /// Production, development, CI or unknown.
    #[attr("lk.product.agents.runner.kind")]
    pub kind: RunnerKind,
}

impl Runner {
    /// The first rule that matches: `CI=true` is CI; a LiveKit Cloud host (`LK_HOSTED_AGENTS`) is
    /// production; `dev`, `console` and `simulation` are development, `start` too on macOS and
    /// Windows; `start` elsewhere is production; no command is unknown. `os` is
    /// `std::env::consts::OS`.
    pub fn detect(var: impl Fn(&str) -> Option<String>, os: &str, cmd: RunnerCmd) -> Runner {
        let set = |name: &str| var(name).is_some_and(|v| !v.is_empty());
        let kind = if var("CI").is_some_and(|v| v.trim().eq_ignore_ascii_case("true")) {
            RunnerKind::Ci
        } else if set("LK_HOSTED_AGENTS") {
            RunnerKind::Production
        } else {
            match cmd {
                RunnerCmd::Dev | RunnerCmd::Console | RunnerCmd::Simulation => {
                    RunnerKind::Development
                }
                RunnerCmd::Start if matches!(os, "macos" | "windows") => RunnerKind::Development,
                RunnerCmd::Start => RunnerKind::Production,
                RunnerCmd::NoCommand => RunnerKind::Unknown,
            }
        };
        Runner { cmd, kind }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a rule applied out of order (CI or Cloud hosting losing to the command), a laptop
    // `start` counted as production, or a script counted as anything but unknown.
    #[test]
    fn the_first_matching_rule_decides_the_kind() {
        use RunnerCmd::{Console, Dev, NoCommand, Simulation, Start};
        use RunnerKind::*;
        type Vars = &'static [(&'static str, &'static str)];
        let cases: &[(Vars, &str, RunnerCmd, RunnerKind)] = &[
            (&[("CI", "true"), ("LK_HOSTED_AGENTS", "1")], "linux", Start, Ci),
            (&[("CI", "TRUE")], "macos", Dev, Ci),
            (&[("CI", "false")], "linux", NoCommand, Unknown),
            (&[("LK_HOSTED_AGENTS", "1")], "linux", Dev, Production),
            (&[("LK_HOSTED_AGENTS", "")], "linux", NoCommand, Unknown),
            (&[], "linux", Dev, Development),
            (&[], "linux", Console, Development),
            (&[], "linux", Simulation, Development),
            (&[], "macos", Start, Development),
            (&[], "windows", Start, Development),
            (&[], "linux", Start, Production),
            (&[], "linux", NoCommand, Unknown),
        ];
        for (vars, os, cmd, kind) in cases {
            let var =
                |name: &str| vars.iter().find(|(k, _)| *k == name).map(|(_, v)| (*v).to_owned());
            assert_eq!(Runner::detect(var, os, *cmd), Runner { cmd: *cmd, kind: *kind });
        }
    }
}
