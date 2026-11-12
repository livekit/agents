# Processes and IPC

Job supervision, process lifetime, fork safety, preloading and logging.

[Contributor index](README.md) · [Execution paths](architecture.md#execution-paths)

## Processes and IPC

`lk-agents-worker` owns socket/pipe creation, ownership, framing, protocol
messages, reads, writes, shutdown, exit monitoring, termination, supervision and
pool policy, in three layers:

- `ipc/ipc_channel`: the only code touching the connection's bytes. A typed
  `Channel` (`send`/`try_send`/`recv`/`close`) over the platform's best
  transport: an AF_UNIX socketpair on Linux and macOS (close-on-exec, the job's
  end too once adopted; `SO_NOSIGPIPE` on macOS), a single-instance named pipe on
  Windows (random name, current-user-only DACL, remote clients rejected) that the
  worker connects itself and refuses unless the connected client is itself, and a
  bounded in-memory duplex in tests. Each side writes a magic and its protocol
  version, then length-prefixed postcard frames, then an empty end frame (a named
  pipe cannot half-close); lengths are checked before allocating, in both
  directions.
  `recv` ends with `Closed` (EOF, reset, aborted, broken pipe, not connected),
  `Truncated` (EOF inside a frame), `TooLarge`, `Malformed` or `Version`. A
  refused oversized send is a typed error and the channel stays usable; senders
  bound free text. `close()` stops sending, flushes and finishes the stream, then
  drains the peer to its end, so closing with unread data never resets the peer;
  a side whose peer finished finishes too.
- `ipc/process`: the process executor, the only code touching the OS process:
  the start method's spawn and reaping receipts, exit observation, termination,
  memory samples.
- Supervisor, pool and dispatch: policy over a channel and a runner, never
  sockets, framing or the OS.

What a job runs in is a `RunnerExecutor`: `launch()` creates the channel, has the
host's launcher start the runner on the job's end and returns the `Runner` with the
worker's end. A `Runner` has three operations: `finished()` resolves with its
`Exit` (code, and whether it was killed) once nothing of it is left, `stop()`
ends it where it can be forced and `memory_mb()` samples its own footprint.
The process executor's runner is killed by `stop()` and finishes once reaped;
a launch that fails after the OS created the child still returns the runner,
which kills it. The thread executor runs the job on a thread of the worker
(every platform): `stop()` cannot force it, it has no footprint of its own, and
it finishes when the thread reports its return. Supervision ends a runner the
same way whichever it is: it asks it to shut down over the channel, then calls
`stop()`, and waits for `finished()` within its deadlines.
Each child has one Rust-owned control connection. Malformed/truncated input
and unexpected EOF settle pending operations as failures and enter supervision
cleanup; they cannot appear as successful exit. The peer closing after it said
it was shutting down is the clean end.

The host-language process adapter launches and bootstraps the child with the
Rust-created IPC endpoint. Python uses `multiprocessing`; Node supplies its own
launch adapter. The child runs application initialization and job callbacks.
Host adapters do not implement IPC loops, supervision or pool policy.

The host's port is the launcher (`ipc.JobLauncher`): `_core_launch(endpoint, launch)`
starts the process (`multiprocessing`, the forkserver's preloads) or the job thread
and reports it on `launch` (`_core.JobLaunch`: a process's receipt, a thread's
return); `_core_completed` finishes the start method's bookkeeping. The
supervisor's diagnostics are ordinary log lines naming the pid and the job
(`pid`, `job_id`, `room`, `room_id`). Everything else is the native `JobSupervisor` the pool creates and drives:

| State | Meaning |
| --- | --- |
| `started`, `closing` | `start()` was called; `aclose()`/`kill()` was called, after which nothing starts and no job is taken |
| `initialized`, `initialize_time` | the runner answered its `Hello`, and how long it took |
| `pid` | the process's id once the OS created it; `None` for a thread |
| `status` | `None` until a job was launched, `RUNNING` until the runner ended, then `SUCCESS` (exit 0, clean channel) or `FAILED` |
| `running_job` | the worker's `RunningJobInfo`, set once the runner took it, kept after it ended |
| `exitcode`, `killed` | set once supervision is over; `killed`: stopped by force |

`aclose()` asks an initialized runner to shut down and stops it past
`close_timeout`; `kill()` stops it at once. A thread cannot be stopped: both wait
for it to return, and it has no pong deadline nor memory limit. A runner takes
one job: a second `launch_job` is refused. Supervision settings and their
production defaults are `JobRunnerOptions`. The pool answers `active_jobs`
(`status` `RUNNING`) and `get_by_job_id`, notifies the host of each
process's step through one `emit`, and records the worker's jobs and warm
processes as process instruments on every change.

Provider clients and model execution run in the job process
(`lk-agents-inference`). There is no separate inference-process service.

The worker retains cleanup responsibility for every launched child, including
cancelled startup and failed initialization. Each start-method protocol defines
how exit is observed and who reaps the process; a forkserver child's OS parent
is the forkserver, not the logical worker owner.

Detect logical worker loss and initiate termination/escalation within configured
deadlines independently of a blocked host loop. Observe OS exit and completed
reaping separately; deadline expiry reports incomplete cleanup and retains its
owner. IPC queue saturation cannot disable supervision or shutdown.

### IPC stress and leak tests

Test IPC with generated event sequences against the worker and at the real OS
boundary:

- Generate supervision/pool event sequences covering startup, assignment,
  cancellation, timeout, drain and close races, as property tests of
  `lk-agents-worker` (the DST does not simulate the worker). Every accepted request settles;
  stale events cannot revive a process or lose cleanup ownership.
- Stress the real transport with fragmented/truncated frames, malformed and
  oversized messages, slow readers/writers, full queues and disconnects during
  pending operations. Assert bounded buffering and responsive supervision.
- Run repeated real-process launch/job/close cycles through each host launch
  adapter and supported start method. Mix concurrent jobs, initialization
  failure, child crashes, abrupt parent death and blocked host loops. Verify
  process exit and reaping separately; treating a zombie as gone does not prove
  successful reaping. Check inherited endpoint cleanup as well as live children.
  On Windows the job's pipe end is duplicated straight into the child by
  multiprocessing's spawning `Popen`; the earlier socket path
  (`resource_sharer.DupSocket`) kept the parent's copy open until a child that
  died before unpickling would have fetched it, leaking a socket per failed
  launch. Count the parent's handles across failed launches to keep that gone.
- Measure resource retention after warm-up and after each settled batch:
  processes, descriptors/OS handles, threads, tasks, pending requests and host
  references. Baselines include declared caches and launch helpers, such as the
  forkserver/resource tracker. Counts return to their baseline; memory remains
  bounded under repeated fixed workloads. RSS alone does not establish a leak or its
  absence.

Use reproducible fault schedules and bounded watchdogs. Fixtures clean up the
whole test-owned process tree after failure, while preserving evidence of
orphans or leaked resources before cleanup. Cleanup must not turn a failed
assertion into a pass. Run a bounded stress campaign in the merge gate and
extended soak campaigns separately, retaining failing schedules and resource
measurements. Simulated resource counters do not replace real-process checks.

### Fork safety

Importing `_core` starts no runtime, thread or socket. Linux defaults to
`forkserver`; macOS and Windows use `spawn`. An explicit `forkserver` choice
outside Linux falls back to `spawn`.

Bindings use a lazy process-wide Tokio runtime after process initialization.
Python tests run on the real clock; deterministic time is the DST's.

`_core.init(role)` declares the process. A process that never declares a role
is a job. Entry points check the PID before using inherited native state;
a forked child must reinitialize before using state declared by its parent.
A stale declaration raises `RuntimeError`. No `pthread_atfork` handler is
installed; fork followed by exec remains allowed.

### Python preloading

```python
server = AgentServer(preload_modules=["myapp.preload"])
```

`preload_modules` contains importable module names in execution order. Importing
a module may synchronously prepare immutable catalogs, fork-safe weights or
other module-held data. The Python launch adapter owns this configuration.

| Start method | Execution |
| --- | --- |
| Linux `forkserver` | Import in the forkserver after framework/plugin preloads, before the final thread check, `gc.freeze()` and any job forks |
| `spawn` | Import in each child before reporting readiness |

Forked children inherit prepared objects through copy-on-write; their mutations
are private. Spawned children load independent copies. Per-process clients and
active runtimes are created by the job that uses them, not at process launch.

The Linux forkserver also imports the session report implementation before
freezing. Clients and exporter tasks initialize in each child rather than in the
forkserver.

User preload failures stop initialization and report the module name and
original error. Detected background threads prevent forking. This check does
not prove arbitrary native-library state is fork-safe. Preloads must not start
threads, event loops, connections, GPU contexts or the Rust async runtime.

### Logging

Each process writes its own logs to inherited stdout. Logs do not travel over
the IPC control socket. Rust uses the structured `log` facade:

```rust
log::warn!(target: lk_agents_abi::LOG_TARGET, partition = partition.as_str(); "the lease was lost");
```

Every line names the one `LOG_TARGET` (Python's `livekit.agents` logger) and carries its
fields as key-values, never formatted into the message.

`lk-agents-py` queues each record without taking the GIL (bounded; a full queue
drops records and the drain reports how many) and one drain thread per process
forwards them, in order and stamped when they were logged, to the Python logger
of their target, with structured keys as `extra` (integers, floats, booleans and strings keep
their type; a `None` value is left out; an `exc_text` key becomes the record's formatted
exception). So code anywhere, sans-I/O
cores included, logs freely, even under a lock. `_core.flush_logs()` waits, without polling, until
the drain forwarded every line queued before it (tests at their boundaries; the interpreter at
exit). This is the one
host entry outside HostInvoker: a record carries no invocation and settles nothing. Logs are observations, not protocol messages or
inputs to decision policy.
