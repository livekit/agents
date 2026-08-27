from livekit.plugins.aws.llm import _supports_inference_config


def test_claude_opus_47_does_not_receive_temperature():
    assert not _supports_inference_config("anthropic.claude-opus-4-7-v1:0")
    assert not _supports_inference_config(
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-4-7"
    )


def test_claude_opus_47_does_not_receive_top_p():
    assert not _supports_inference_config("anthropic.claude-opus-4-7-v1:0")


def test_other_models_keep_temperature_support():
    assert _supports_inference_config("anthropic.claude-sonnet-4-20250514-v1:0")
