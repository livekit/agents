Deploying and running agents on LiveKit Cloud, including CLI deployment, builds, secrets, rollbacks, regions, scaling, plans, quotas, and billing.

OVERVIEW:
LiveKit Cloud is a fully managed, globally distributed platform for building, hosting, and operating voice AI agent applications at scale. It extends the open-source LiveKit server with managed agent hosting, built-in inference, native telephony, a browser-based Agent Builder, and production-grade observability. You can deploy your agent with a single CLI command and LiveKit handles scaling, lifecycle management, container isolation, and upgrades. LiveKit Cloud runs the same open-source server and supports the same APIs and SDKs, so your code stays portable between cloud and self-hosted environments.

WHAT LIVEKIT CLOUD INCLUDES:
Realtime communication core provides a fully managed, globally distributed mesh of LiveKit servers for low-latency audio, video, and data streaming. Agent Builder lets you prototype and deploy simple voice agents in your browser without writing code. Managed agent hosting lets you deploy and run agents directly on LiveKit Cloud without managing servers or orchestration. Built-in inference through LiveKit Inference gives you access to AI models from OpenAI, Google, Deepgram, Cartesia, ElevenLabs, and more without needing separate API keys. Native telephony with LiveKit Phone Numbers lets you provision phone numbers and connect calls directly into LiveKit rooms. Observability and operations give you analytics, logs, quality metrics, transcripts, traces, and audio recordings through the dashboard. Background voice cancellation through Krisp and ai-coustics models ensures agents receive crystal-clear audio regardless of where the agent runs.

PLANS:
LiveKit Cloud offers four plan tiers. Build is the free tier with usage quotas that act as hard limits, shared across all of a user's free projects. Ship, Scale, and Enterprise are paid plans with higher quotas and additional features. Paid plans get incremental billing beyond their included quotas rather than hard cutoffs. The Scale plan allows customers to request limit increases in project settings. Enterprise plans offer custom rates and capacity. Refer users to the LiveKit pricing page at livekit.io/pricing for current plan details and pricing.

FREE BUILD PLAN QUOTAS:
The free Build plan includes 1000 agent session minutes, 100000 agent observability events, 1000 minutes of agent audio recordings, 2.50 dollars in LiveKit Inference credits, 1 US local phone number, 50 US local inbound minutes, 1000 third-party SIP minutes, 5000 WebRTC participant minutes, 50 GB downstream data transfer, and 60 minutes each of transcode and track egress. Quotas reset monthly and do not roll over. Free plan quotas are shared across all free projects for a given user. Projects on the free Build plan are included in the model improvement program where some anonymized data may be retained longer. Paid plans are not included in that program.

CONCURRENCY LIMITS ON FREE PLAN:
The free Build plan allows 5 concurrent agent sessions, 5 concurrent STT connections, 5 concurrent TTS connections, 100 total participants, 2 ingress requests, and 2 egress requests. LLM rate limits on the free plan are 100 requests per minute and 600000 tokens per minute.

DEPLOYING AGENTS:
To deploy an agent to LiveKit Cloud, you need the latest LiveKit CLI, a LiveKit Cloud project, and a working agent. First authenticate with lk cloud auth. Then run lk agent create from your project directory. This registers your agent, assigns a unique ID, writes a livekit.toml configuration file, creates a Dockerfile if you do not have one, uploads your code to the LiveKit Cloud build service, builds a container image, and deploys it. After that, deploy new versions with lk agent deploy. The CLI uploads your code, builds a container image from your Dockerfile, and performs a rolling deployment. New instances serve new sessions while old instances get up to 1 hour to finish active sessions.

LIVEKIT.TOML:
The livekit.toml file stores your agent's deployment configuration including the project subdomain and agent ID. The CLI automatically looks for this file in the current directory. You can generate a new one with lk agent config.

BUILD PROCESS:
LiveKit Cloud builds container images on its build service. The CLI gathers files from your working directory, excludes dot-env files and anything matched by dockerignore or gitignore, uploads the context, and builds using your Dockerfile. Builds have a maximum duration of 10 minutes. Use glibc-based images like Debian or Ubuntu, not Alpine. Run as an unprivileged user. Download models during build, not at runtime, using the download-files command. Do not include secrets in the image. LiveKit Cloud injects LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET automatically at runtime.

SECRETS MANAGEMENT:
Secrets are secure variables stored encrypted and injected into agent containers at runtime as environment variables. Set initial secrets during lk agent create and update them with lk agent update-secrets. The CLI looks for dot-env, dot-env-local, or dot-env-production files automatically. You can also pass secrets inline with the secrets flag. Secret names must be letters, numbers, and underscores up to 70 characters. Values have a 16KB maximum. You can also mount files as secrets using secret-mount, which places them at etc/secrets/filename in the container. LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are auto-generated and cannot be overridden.

ROLLING DEPLOYMENTS AND ROLLBACKS:
When you deploy a new version, LiveKit Cloud uses a rolling deployment strategy. New instances must pass health checks within 5 minutes before old instances are removed. Old instances stop accepting new sessions but continue serving active ones for up to 1 hour. You can roll back to a previous version without rebuilding using lk agent rollback. Instant rollback is only available on paid plans. Free plan users must revert code and redeploy.

COLD STARTS:
On the free Build plan, agents can be scaled down to zero replicas after all sessions end. When a new user connects, a cold start occurs, which can take 10 to 20 seconds before the agent joins the room. The agent status shows as Sleeping when scaled down and Waking when starting back up.

REGIONS AND AVAILABILITY:
Agent deployments are assigned to a specific region during creation, and this cannot be changed afterward. Currently available regions are us-east in Ashburn Virginia, eu-central in Frankfurt Germany, and ap-south in Mumbai India. For global apps, deploy the same agent to multiple regions using lk agent create once per region with different config files, for example livekit.us-east.toml and livekit.eu-central.toml. By default, users connect to the closest region. You can use explicit agent dispatch for fine-grained control over routing. LiveKit Cloud's global mesh architecture connects each user to the nearest edge for minimal latency.

SCALING AND LOAD BALANCING:
LiveKit Cloud provides automatic scaling and load balancing for deployed agents. New instances are automatically scaled up and down to meet demand. LiveKit server includes built-in balanced job distribution using round-robin with single-assignment, ensuring each job goes to one agent server. LiveKit Cloud also uses geographic affinity to match users with the closest agent servers. For self-hosted deployments, you configure autoscaling yourself using CPU utilization or custom load functions.

LIVEKIT INFERENCE:
LiveKit Inference provides access to AI models included in LiveKit Cloud without requiring separate API keys. It supports LLMs from OpenAI including GPT-4o, GPT-4.1, GPT-5, and GPT-5.1, plus Google Gemini models, DeepSeek, and Kimi. STT providers include AssemblyAI, Cartesia, Deepgram, and ElevenLabs. TTS providers include Cartesia, Deepgram, ElevenLabs, Inworld, and Rime. You use the inference module classes in your AgentSession, for example inference.STT, inference.LLM, and inference.TTS. Inference billing is usage-based, metered by tokens for LLMs, seconds for STT, and characters for TTS. The free plan includes 2.50 dollars in monthly inference credits.

AGENT BUILDER:
The Agent Builder is a browser-based tool for prototyping and deploying simple voice agents without writing code. It produces best-practice Python code using the LiveKit Agents SDK and deploys directly to LiveKit Cloud. You can configure instructions, welcome greetings, models, HTTP tools, client tools, MCP servers, metadata variables, secrets, and call summaries. You can test your agent live in the builder and convert it to downloadable code at any time. Access it by selecting Deploy new agent in your project's Agents dashboard.

DASHBOARD:
The LiveKit Cloud dashboard at cloud.livekit.io provides realtime metrics including session count and agent status, error tracking, usage and billing information, build logs, agent observability with transcripts, traces, logs, and audio recordings. You can monitor deployed and self-hosted agents, manage secrets, configure telephony, and view project settings including current limits.

LOG COLLECTION:
LiveKit Cloud collects runtime logs from your agent's stdout and stderr, and build logs from the container build process. Use lk agent logs to tail runtime logs from the latest instance. Use lk agent logs with log-type build to view Docker build logs. You can forward runtime logs to external services including Datadog, CloudWatch, Sentry, and New Relic by adding the appropriate API keys as secrets. For example, add a DATADOG_TOKEN secret to enable Datadog forwarding automatically.

CLI COMMANDS:
The primary CLI commands for agent management are lk agent create to register and deploy a new agent, lk agent deploy to build and deploy a new version, lk agent status to check current status, lk agent logs to stream runtime logs, lk agent rollback to revert to a prior version, lk agent delete to remove an agent, lk agent list to show all agents in a project, lk agent versions to list available versions, lk agent secrets to view secret names, lk agent update-secrets to modify secrets, lk agent config to generate a livekit.toml file, and lk agent dockerfile to generate a Dockerfile. All commands use the livekit.toml file in the working directory by default.

AGENT BILLING:
Agents deployed to LiveKit Cloud are metered by agent session minutes, which is the time the agent is actively connected to a WebRTC or SIP session. Metering starts after the agent connects to the room and stops when the room ends or the agent disconnects. If an agent receives a job but never connects, no metering occurs. You can explicitly stop metering by calling ctx.shutdown in your entrypoint function.

CLOUD VS SELF-HOSTED:
Both options support the full LiveKit feature set for realtime media, egress, ingress, SIP, telephony, and the agents framework. LiveKit Cloud adds managed agent hosting, Agent Builder, built-in inference, LiveKit Phone Numbers, the global mesh SFU architecture with no participant limits, the cloud dashboard with analytics, and a 99.99 percent uptime guarantee. Self-hosted uses a single-home SFU with up to roughly 3000 users per room and requires you to manage your own infrastructure, scaling, and monitoring. Your code remains portable between the two since the connection endpoint is the primary difference.

ENTERPRISE AND UPTIME:
LiveKit Cloud provides a 99.99 percent uptime guarantee with redundant infrastructure. Enterprise plans offer custom pricing, capacity, and support. Scale plan customers can request limit increases through project settings. Enterprise customers should contact the LiveKit sales team for custom arrangements.
