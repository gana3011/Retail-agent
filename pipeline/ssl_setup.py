"""
SSL configuration for environments with self-signed certificates (corporate proxies).

Provides a single setup point instead of copy-pasting monkey-patches across files.
"""

import os
import ssl

import httpx


def configure_ssl():
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        return

    ssl._create_default_https_context = _create_unverified_https_context

    _original_init = httpx.Client.__init__

    def _ssl_patched_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _original_init(self, *args, **kwargs)

    httpx.Client.__init__ = _ssl_patched_init
