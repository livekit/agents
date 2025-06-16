# AWS plugin for LiveKit Agents

Support for AWS AI including Bedrock, Polly, Transcribe and optionally Nova Sonic (realtime STS model).

See [https://docs.livekit.io/agents/integrations/aws/](https://docs.livekit.io/agents/integrations/aws/) for more information.

## Installation

```bash
pip install livekit-plugins-aws

# for access to Nova Sonic
pip install livekit-plugins-aws[realtime]
```

## Pre-requisites

You'll need to specify an AWS Access Key and a Deployment Region. They can be set as environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION`, respectively.
