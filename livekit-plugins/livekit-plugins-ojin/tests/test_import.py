"""Smoke test: verify the ojin plugin is importable."""


def test_ojin_plugin_importable():
    from livekit.plugins import ojin

    assert hasattr(ojin, "AvatarSession")
    assert hasattr(ojin, "OjinException")
    assert hasattr(ojin, "__version__")
