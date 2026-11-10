Agent Observability in LiveKit Cloud, including transcripts, session traces and metrics, logs, audio recordings, data retention, and pricing.

OVERVIEW:
Agent Observability lets you monitor and analyze your voice AI agent's behavior in LiveKit Cloud. It includes transcripts, session traces and metrics, logs, and audio recordings. It works for agents deployed to LiveKit Cloud and for self-hosted agents connecting to LiveKit Cloud media servers. It does not work with entirely self-hosted deployments.

GETTING STARTED:
To get started, sign up for a LiveKit Cloud account, build an agent using the LiveKit Agents SDK or Agent Builder, then deploy to LiveKit Cloud. Once deployed, enable Agent Observability at the project level in the Data and Privacy section of your project settings. You need Python SDK version 1.3.0 or higher, or Node.js SDK version 1.0.18 or higher. Agent insights are found in the Agent Insights tab on the sessions dashboard.

WHAT'S INCLUDED:
Transcripts: Turn-by-turn transcripts for user and agent, including tool calls and handoffs, with additional metadata and metrics in the detail pane.
Session traces and metrics: Traces capture the execution flow broken into spans for every pipeline stage, enriched with token counts, durations, and speech identifiers. Includes user and agent turns, STT-LLM-TTS pipeline steps, tool calls, and more.
Logs: Runtime logs from the agent server, collected according to your configured log level.
Audio recordings: Recorded for each session, available for playback and download. Includes both agent and user audio. If noise cancellation is enabled, user audio is recorded after noise cancellation is applied.

DATA STORAGE AND RETENTION:
All observability data is stored in the United States with a 30-day retention window. Data older than 30 days is automatically deleted. Projects on the free Build plan are included in LiveKit's model improvement program where some anonymized data may be retained longer. Paid plans like Ship, Scale, and Enterprise are not included in that program.

PRICING:
Observability events such as transcripts, trace spans, and logs are billed by the event. Audio session recordings are billed by the minute. Refer users to the LiveKit pricing page for specifics.

ENABLING AND DISABLING:
Agent Observability can be enabled or disabled at the project level in LiveKit Cloud settings under Data and Privacy. You can also disable recording for individual sessions in your agent code using the record parameter on AgentSession.start. You can pass True for everything, False for nothing, or a dictionary with granular options for audio, transcript, traces, and logs individually.

DATA HOOKS:
The Agents SDK provides data hooks for collecting session data locally and integrating with external systems. You can access session.history for full conversation history, subscribe to events like conversation_item_added for live dashboards, and call ctx.make_session_report for a structured JSON report with identifiers, history, events, and recording metadata.

METRICS:
AgentSession emits a metrics_collected event with detailed metrics including VAD metrics like idle time and inference duration, STT metrics like audio duration, EOU metrics like end of utterance delay and transcription delay, LLM metrics like token counts and time to first token, and TTS metrics like audio duration and time to first byte. Total conversation latency can be approximated as end_of_utterance_delay plus LLM time to first token plus TTS time to first byte. You can also use UsageCollector to aggregate LLM, TTS, and STT usage for cost estimation.

OPENTELEMETRY:
The Python SDK supports OpenTelemetry integration. You can set a tracer provider to export spans to any OpenTelemetry-compatible backend like LangFuse.

SHARING WITH SUPPORT:
You can share specific session insights with LiveKit support on Ship plan or higher. Enable sharing from the session's Agent Insights tab to generate a link you can email to support.
