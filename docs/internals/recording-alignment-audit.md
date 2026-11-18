# Recording alignment audit

The recording represents the session's input and the agent output's reported playout. It is
not an exact observation of a remote listener's speakers. RTC source queue drainage is local
evidence; downstream network jitter, packet loss, receiver buffering, volume and device playback
are outside that evidence. Microphone frames are placed using arrival time minus frame duration,
with sample-count continuity and drift correction, rather than a remote capture timestamp.

## What the implementation records

- `VoiceRuntime::on_frame` taps enabled microphone input, including the pre-connect head. Muted
  input is excluded. The native recording worker places a pre-connect head before the live
  audio that follows it.
- `RecorderFeed` retains accepted agent PCM and observes playback independently of generation.
  `UnplayedAudio` retains unfinished segments, including for a recorder attached mid-reply.
- `SegmentPlacer` places only reported stretches at their playback timestamps and offsets.
  Captured but unplayed tails are discarded on completion. Outputs that report no progress
  use their start and final playback position as a fallback; pauses cannot be reconstructed
  accurately if the output never reports where they occurred.
- `RecordingFile` mixes both channels on one timeline and delays flushing while output
  placement or input resampling is pending. Finalization writes the remaining settled tail.
- Background audio publishes a separate track and bypasses `RecorderFeed`. Ambient and thinking
  sounds therefore are not included in this recording. Other participants' tracks and sounds
  produced by the receiving application are likewise not a complete room mix.

Recording is bounded. A microphone backlog beyond the native worker's catch-up budget becomes
silence of the same duration. Output queue/retention overflow and file/encoder failures fail the
recording; a failed recording is not offered as complete audio by the session reporter. This
protects the conversation but prevents an unconditional completeness guarantee.

## Reproduced and fixed

A recorder attached between two room playback reports retained the unconfirmed prefix of
the segment, as it should. The next report could describe audio spanning both sides of the
attachment. Forwarding its entire duration placed that prefix at the attachment boundary,
overlapping subsequent PCM and changing the recorded audio.

`UnplayedAudio` now retains the attachment timestamp. On receiving a report, it excludes only
the prefix confirmed to have played before attachment and preserves the original sample
offsets for the rest. It never infers playout from elapsed time alone. Replayed start timestamps
also respect attachment. The independent PCM regression fails when this clipping is removed.

## DST coverage

The earlier `recording_is_what_played` check derived its expected intervals from callbacks to
the recorder and checked only for silence and late placement. It could not establish sample
identity or detect extra unplayed audio.

Room-only runs now compare the real `RecordingFile`'s PCM against PCM retained by the fake RTC
source at its simulated speaker, including partial frames cut by clear/close. This reference
does not use framework playback reports, the production playhead ledger or recorder callbacks.
It compares both sound and silence, allowing one PCM quantization unit and excluding 2 ms at
resampling boundaries. Oracle tests deliberately substitute wrong, missing, shifted and extra
audio to establish that those corruptions are rejected.

The regressions execute the production session and room output with:

- normal buffered playout;
- interruption and immediate session close inside a frame;
- recorder attachment between progress reports, at several times and scheduler orders;
- an audio stream that runs dry, remains open, and later resumes.

The harness retains and finalizes attached recording files after session/output closure,
checking the final tail as well as periodic flushes. Recording-file errors are retained and
reported instead of silently treated as successful flushes.

## Remaining boundaries

Custom-output, avatar and console runs retain the older callback-based checks. The independent
room PCM check applies only while the room source is the sole audio output, with no background
audio track; it does not claim to judge a mixture containing other speakers. Input timing,
pre-connect placement and recording-worker pressure have their existing runtime/media tests,
but this new oracle judges the agent channel only. It runs the real timeline with an in-memory
PCM encoder, not the blocking recording worker or Ogg/Opus codec. Codec round-trip tests remain
separate and do not establish sample-exact end-to-end alignment.

An exact remote-listener guarantee would require timestamped receiver-side playout observations
and an explicit policy for background/other tracks, microphone capture clocks, gaps and missing
receipts. The current local recording cannot supply that evidence by itself.

## Cloud and web time origins

The nominal upload contract agrees across the inspected checkouts:

1. `SessionRecorder.started_at()` is the real file's sample-zero timestamp.
2. `SessionOwner` copies that value to `audio_recording_started_at`.
3. The upload builder converts it to `RecordingHeader.start_time_millis`; transport encodes
   protobuf `MetricsRecordingHeader.start_time`, with millisecond precision.
4. Cloud's recording collector stores that start as the index's `CreatedAt` and the waveform's
   `created_at`. The object URL service returns the index's timestamp too.
5. Web positions waveform peaks at `createdAt - timeline.from` and computes audio playback time
   from `timeline.from + audioOffset + audio.currentTime`. Transcript rows prefer the absolute
   `started_speaking_at` metric, with absolute creation/log time as fallbacks. Traces use their
   absolute Unix timestamps.

The new DST export regression starts audio 400 ms after session creation, derives the upload
origin from the finalized recording file, and checks the real final transcript's speaking time
and exported `agent_speaking` span against the independent speaker. Both agree within 1 ms.
This catches substituting session creation time for recording sample zero. The existing
transport regression additionally checks the protobuf timestamp in the multipart upload.

There is a separate, reproduced downstream decoding issue. The shared backend's
`observability/recording/WaveformFromOGG` reads `OpusHead.PreSkip` but does not discard those
decoder samples. It also does not trim end padding against the final Ogg granule. A synthetic
48 kHz Opus file decoded through the actual waveform source produced:

| Observation | Value |
| --- | ---: |
| Opus pre-skip | 312 samples (6.5 ms) |
| Waveform decoder's sample count | 24,960 |
| FFmpeg playback decoder's sample count | 24,000 |
| Measured waveform lag at 500 peaks/s | 3 bins (6 ms) |

The isolated probe is in `/tmp/recording-waveform-probe`, with its result in
`/tmp/recording-waveform-probe.log`. The shared backend decoder and its consuming services have
not been changed by this SDK patch. Their pre-skip and end-trim handling must be corrected
before claiming full waveform/playback alignment; changing the SDK's upload start to compensate
would instead move the correctly anchored audio/transcript timestamps.

## Follow-up: clock corrections

Fault injection reproduced a second SDK alignment bug. A 10-second forward correction of the
host's wall clock moved the next transcript's speaking timestamp by 10 seconds, although the
speaker's PCM and core trace remained on their original timeline. Recording flushes also used
the newly shifted wall reading. This establishes a code defect, not the cause of a particular
customer incident.

`Timebase` now samples one monotonic/wall pair and lends an anchored `SessionClock` to its
consumers. Scheduling still delegates to the original monotonic clock. Transcript timestamps,
recording flushes and recorder attachment therefore retain one time origin through wall-clock
corrections. The session report and avatar item notifications also use that session clock.

Scoped provider work previously retained the process tracer's clock even when the session had
its own clock. Session trace contexts now time that work on the session clock while delegating
records, ID allocation and live content-capture policy to the original process recorder. An
integration regression captures the actual context passed to the production executor's provider
opening; it fails when the session clock adapter is removed. The telemetry regression checks
parent/trace identity, unique IDs and changes to the shared content-capture policy.

The DST correction regression exercises both forward and backward 10-second changes, with a
recorder attached either before or after correction. It checks speaking timestamps, exported
core spans, the recording's sample-zero anchor, and that the file does not flush silence into
the future. A separate report regression failed against the old process-clock timestamp and
now checks the public report against its session's audio clock. External services and spans
with explicitly supplied application timestamps still require their own clock alignment.

## Follow-up: Gladia's optional energy filter

A [November 15 merge notification](https://live-kit.slack.com/archives/C01R3SZUGTT/p1794784590142349)
links the upstream removal of Gladia's energy filter (PR #8577). This checkout still offered
the option. With `energy_filter=True`, silence was discarded and a local energy transition
could send `stop_recording` before the caller ended input. Dropping samples compresses the
provider's timestamp axis relative to the recording.

The filter now replaces quiet audio with zero PCM of exactly the same byte length. Only an
explicit flush or end-of-input finalizes the provider session. The option and its default remain
unchanged. A local-peer regression sends two seconds of silence followed by speech, awaits
each wire receipt to exclude ingress pressure, and checks every uploaded sample plus exactly
one finalization. Against the old implementation it fails on premature `stop_recording`.
This is a reproduced conditional defect, not a confirmed explanation of the Vivoroom report.

## Verification limits

The full DST regression run also fails two storage-only cases with no recorder attached:
`an_interrupted_turn_waits_on_its_checkpoint` and
`bound_storage_faults_close_the_call_and_successful_receipts_finish_the_speech` (seed 24).
Both reproduce after temporarily removing this recording work and restoring the prior source.
They are baseline failures, not recording regressions.

Targeted recording regressions, the independent oracle's corruption tests, voice/media unit
tests and upload-builder tests pass. Formatting, Clippy and architecture checks pass. Web's
existing waveform/audio-clock/transcript tests were attempted but could not run because
`vitest` is not installed in that checkout; the consumer contract above was verified by source
inspection. The synthetic decoder probe ran successfully in its isolated Go module.

## Reported incidents from Slack

These reports establish several distinct failure families. They do not establish that the
Opus pre-skip issue explains a particular customer session.

- [Vivoroom feedback, November 17, 2026](https://live-kit.slack.com/archives/C03SPAWBY3W/p1794953388942549):
  Jack reports audio waveform playback sometimes misaligned with both transcripts and traces.
  The thread contains no recording-specific reproduction or confirmed cause.
- [Delphi, August 18, 2026](https://live-kit.slack.com/archives/C078MN6GT0V/p1787080772972429):
  the same words appeared around 4 s in audio/waveform playback and 19 s in the transcript.
  Jon associated this with no participant tracks being published and gaps being removed from
  the recording, and linked agents PR #6921 as the fix. This seconds-scale case is not explained
  by the measured millisecond-scale Opus offset.
- [Late microphone and ghost audio, December 1, 2025](https://live-kit.slack.com/archives/C07TWVC0W4A/p1764624438982699):
  Chenghao and Long reproduced lost silence between an agent greeting and microphone input
  enabled roughly 20 s later. The same thread discusses audio captured before an interruption
  being recorded even though it never played, and links PRs #4131/#4149 for playback signals.
- [OpenAI realtime/Node redaction, August 25, 2026](https://live-kit.slack.com/archives/C0ADZF1PTPC/p1787704796688749):
  NTT reported user timestamps rendered in year 1677 and turns pinned to zero, breaking audio
  redaction alignment. Chenghao identified milliseconds versus seconds conversion and a
  separate realtime speaking-boundary limitation. The thread reports the conversion fix and
  client warning released in 1.7.1; it does not claim accurate realtime boundaries were solved.
- [Per-STT-stream origins, December 8, 2025](https://live-kit.slack.com/archives/C07TWVC0W4A/p1765207021561289):
  duration versus end-timestamp semantics differed by provider, and reconnects/handoffs reset
  stream-relative timestamps. The agreed normalization tracked each stream's origin relative
  to the recording; the thread links PR #4155.
- [Traces outside the room range, December 2–3, 2025](https://live-kit.slack.com/archives/C09CKR3N4LR/p1764707715977589):
  traces began before the server-reported room start and rendered off-screen. Clock skew was
  suggested, not confirmed. Web PR #2530 extended the displayed range to include the data.

The new `a_late_microphone_does_not_collapse_twenty_seconds_of_silence` DST regression
specifically exercises the reported missing-input pattern: an agent greeting, 20 s with no
microphone frames, late microphone frames, and another reply. Independent speaker PCM checks
the intervening silence and the later reply's position. It checks audio-frame absence rather
than browser permission handling or RTC track negotiation.
