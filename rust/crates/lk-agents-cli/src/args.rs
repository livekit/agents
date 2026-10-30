//! Typed launcher arguments. Clap owns syntax, help, aliases and diagnostics.

use clap::{Parser, Subcommand, ValueEnum};
use lk_agents_abi::cli::{CliArgs, CliParseResult};

#[derive(Parser)]
#[command(about = "Run a LiveKit agent server")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start an agent worker.
    Start(StartArgs),
    /// Connect a local agent to the external lk console.
    Console(ConsoleArgs),
}

#[derive(clap::Args)]
struct StartArgs {
    #[arg(long, value_enum, ignore_case = true, default_value_t = LogLevel::Info)]
    log_level: LogLevel,
    #[arg(long)]
    dev: bool,
    #[arg(long, alias = "reload-addr", requires = "dev")]
    cli_addr: Option<String>,
    #[arg(long)]
    simulation: bool,
    #[arg(long)]
    url: Option<String>,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long)]
    api_secret: Option<String>,
    /// Total seconds for draining, resource closure and final process cleanup.
    #[arg(long, value_parser = seconds)]
    shutdown_timeout: Option<f64>,
    /// Maximum seconds to wait for active work; must leave time for cleanup.
    #[arg(long, value_parser = seconds)]
    drain_timeout: Option<f64>,
    // Older Go launchers pass this flag. Presentation now belongs to that launcher.
    #[arg(long = "log-format", hide = true, value_enum)]
    _legacy_log_format: Option<LegacyLogFormat>,
}

#[derive(clap::Args)]
struct ConsoleArgs {
    #[arg(long)]
    connect_addr: String,
    #[arg(long)]
    record: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum LegacyLogFormat {
    Json,
    Colored,
}

fn seconds(value: &str) -> Result<f64, String> {
    let seconds: f64 = value.parse().map_err(|_| "expected seconds".to_owned())?;
    if !seconds.is_finite() || seconds < 0.0 {
        return Err("seconds must be finite and nonnegative".into());
    }
    Ok(seconds)
}

#[derive(Clone, Copy, ValueEnum)]
enum LogLevel {
    Trace,
    Debug,
    Info,
    #[value(alias = "warning")]
    Warn,
    Error,
    #[value(alias = "critical")]
    Fatal,
    Off,
}

impl LogLevel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Fatal => "fatal",
            Self::Off => "off",
        }
    }
}

pub fn parse_cli(argv: Vec<String>) -> CliParseResult {
    let parsed = Args::try_parse_from(argv);
    let command = match parsed {
        Ok(args) => args.command,
        Err(error) => return diagnostic(error),
    };
    let args = match command {
        Command::Start(args) => CliArgs::Start {
            log_level: args.log_level.as_str().into(),
            dev: args.dev,
            cli_addr: args.cli_addr,
            simulation: args.simulation,
            ws_url: args.url.filter(|s| !s.is_empty()),
            api_key: args.api_key.filter(|s| !s.is_empty()),
            api_secret: args.api_secret.filter(|s| !s.is_empty()),
            shutdown_timeout: args.shutdown_timeout,
            drain_timeout: args.drain_timeout,
        },
        Command::Console(args) => {
            CliArgs::Console { connect_addr: args.connect_addr, record: args.record }
        }
    };
    CliParseResult::Parsed { args }
}

fn diagnostic(error: clap::Error) -> CliParseResult {
    CliParseResult::Display {
        exit_code: error.exit_code(),
        stderr: error.use_stderr(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> CliParseResult {
        parse_cli(std::iter::once("agent".into()).chain(args.iter().map(|s| (*s).into())).collect())
    }

    #[test]
    fn launcher_contract_and_deadline_overrides() {
        let CliParseResult::Parsed {
            args: CliArgs::Start { cli_addr, dev, shutdown_timeout, drain_timeout, .. },
        } = parse(&[
            "start",
            "--dev",
            "--reload-addr",
            "127.0.0.1:9",
            "--shutdown-timeout",
            "55",
            "--drain-timeout",
            "45",
        ])
        else {
            panic!("expected start")
        };
        assert_eq!(cli_addr.as_deref(), Some("127.0.0.1:9"));
        assert!(dev);
        assert_eq!((shutdown_timeout, drain_timeout), (Some(55.0), Some(45.0)));
        assert!(matches!(
            parse(&["console", "--connect-addr=localhost:9", "--record"]),
            CliParseResult::Parsed { args: CliArgs::Console { record: true, .. } }
        ));
    }

    #[test]
    fn help_and_errors_are_data_not_process_exits() {
        for args in [&["--help"][..], &["start", "--help"], &["console", "--help"]] {
            assert!(matches!(
                parse(args),
                CliParseResult::Display { exit_code: 0, stderr: false, .. }
            ));
        }
        for args in [
            &["start", "--url", "--dev"][..],
            &["download-files"],
            &["console"],
            &["start", "--cli-addr", "x"],
            &["start", "--log-format", "xml"],
            &["start", "--shutdown-timeout", "NaN"],
        ] {
            assert!(
                matches!(parse(args), CliParseResult::Display { exit_code: 2, stderr: true, .. }),
                "{args:?}"
            );
        }
    }
}
