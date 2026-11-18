"""Installs the E2E provider mode in every interpreter of an app, job processes included."""

import os

if "E2E_PROVIDERS" in os.environ:
    from tests.e2e.framework.providers import install

    install()
