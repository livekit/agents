# NetEq feature coverage for ordered PCM

Audited against WebRTC `a545de4`: `api/neteq/neteq.h` operation/mode enums,
`decision_logic`, `delay_manager`, `packet_arrival_history`, `buffer_level_filter`,
`time_stretch`, `accelerate`, `preemptive_expand`, `expand`, `merge`, `normal`,
`background_noise`, `comfort_noise` and their unit tests. This is feature coverage,
not a claim of algorithm, output, or perceptual equivalence. The product remains pure Rust.

## Applicable processing

| NetEq feature | Rust handling | Regression evidence |
| --- | --- | --- |
| Normal decoded playout / sync buffer | Ordered PCM queue plus bounded pending output | Clean PCM preserved with stretching off; source accounting |
| Arrival history and clock mismatch | Actual first-sample arrival envelope; sample-precision relative delay | Tiny chunks, sub-ms reserve, batching, permanent shift, deterministic drift |
| Underrun optimizer / delay constraints | Q30 95th-percentile histograms, 500 ms peaks, configurable target/startup bounds | Histogram normalization, repeated stalls, bounded target |
| Buffer-level smoothing | Target-dependent Q8 filter on a 10 ms render-sample cadence; correct for time-scale changes | Spike rejection, correction, render-size cadence, protected startup budget |
| Preemptive expansion | Prefer quiet splices; adjacent-period/overlap validation; borrow history for 20 ms chunks | A later 20 ms stall is absorbed with reserve; transient rejection |
| Acceleration | Strong two-window splice; per-epoch insertion/removal budget; 50 ms minimum ordinary operation spacing | Tone pitch/energy, source accounting, budget |
| Fast acceleration | Four-times-high-limit trigger, bypass ordinary cooldown, verified pitch multiples and lower correlation threshold | Severe consumer backlog, transient rejection, same hard rate budget |
| Expand / PLC | Multiple pitch candidates/lags, voiced/unvoiced mix, stabilized LPC residual excitation, signal-dependent muting | Harmonics, voiced gap energy, unvoiced spectrum, DC/full-scale/impulses |
| Merge | Two-window match; bounded synthetic prefix to align returning voiced audio; up to 20 ms blend and gain recovery | Phase reversals, onset energy, no prefix for unvoiced/EOF, complete source drain |
| Normal transition after noise/PLC | Merge handles the return to real PCM | Short/long returns, no hard silence edges |
| Background noise and passive speech detection | Conservative source-only LPC room-tone model; eight-times-noise energy criterion for passive time scaling | Room tone, eventual mute, loud consonants not learned as noise |
| Postpone return after faded PLC | Wait for half the adaptive target, bypass waiting at EOF | Repeated returns, deterministic bounded replay, EOF |
| Statistics / test instrumentation | Source/output accounting, target/queue/pending depth, underruns, stretch/acceleration, fast and merge subsets | Accounting on every simulated render; time-series CSV and operation timing |
| Resource and reset behavior | Preallocated queues/scratch, explicit whole-chunk rejection and epoch reset | Zero allocations including fast/merge paths; atomic errors; reset |

The filter stabilizes Rust time-scale decisions; it is not a literal transcription of
NetEq decision logic (which also uses the filter for other packet/noise decisions).
Regular voiced time scaling requires correlation above 0.9; fast mode uses 0.5, like
NetEq. History-assisted expansion additionally validates upcoming source amplitude and
correlation so a new consonant is not replaced by an old vowel. All modes share the
configured `maximum_stretch` fraction; fast mode can spend accumulated budget sooner,
but cannot exceed it. This intentionally differs from unbounded emergency catch-up.

## Transport / decoder features outside this layer

| NetEq feature | Why absent here |
| --- | --- |
| RTP timestamps, sequence wrap, packet sorting, duplicate/obsolete rejection, reorder optimizer | PCM arrives ordered over a reliable transport; source positions are cumulative samples |
| RED, FEC, RTX, codec decoding and decoder PLC | They require encoded packets and codec state; PCM conceals a delivery stall instead |
| RFC 3389 CNG / `kRfc3389CngNoPacket` | No SID/noise-parameter packets; locally learned room tone covers temporary PCM starvation |
| Codec-internal DTX/CNG / `kCodecInternalCng` | Provider adapter must distinguish end-of-turn from a temporary gap; no decoder DTX state exists |
| RTP DTMF events / `kDtmf` | No telephony event payloads at this boundary; already rendered tones pass through as PCM |
| Codec switching, stereo / multichannel coupling, sample-rate switching | Fixed mono 24 kHz format by contract; media conversion belongs before this controller |
| Packet-loss/discard rates, RTCP reporting, codec VAD, UMA | No corresponding packet/codec state; expose actual PCM operations and queue metrics instead |
| NetEq packet-buffer flush / timestamp fast-forward after a large discontinuity | Accepted source speech is preserved; a caller explicitly resets an epoch or handles atomic capacity rejection |
| `kUndefined`, decoder error modes | Typed validation/capacity/time errors; no partially changed state |

No relevant operation enum is silently unaccounted for. Listening quality across wider
speech/noise content and real provider arrival traces remains an evaluation requirement;
passing these regressions does not establish NetEq quality parity.

## Startup and multi-answer qualification

Startup uses an 80 ms configurable cold prior, measured first-500ms coverage deficits,
early release for burst coverage/EOF and a slowly relaxing, bounded per-answer budget.
`begin_utterance()` retains only normalized timing; `reset()` clears the connection profile.
Regressions cover ten answers with long idle gaps, cancellation, single-chunk replies,
clean low priors and severe 300 ms startup deficits under 100/400 ms caller budgets.

Actual chunk arrivals replace imaginary intra-payload arrivals. The sustained 100 ms speech
fixture previously had 20 Rust underruns; the revised control path has two tiny events in
the recorded run. This does not establish universal gap-free playback or human quality
parity. The primary native NetEq audio remains unchanged. Native-default and explicitly
capped targets are recorded separately, with measured startup and queue depth.
