"""
core/apps.py

FIX: Override ready() to pre-load ML artefacts at Django startup time
instead of lazily on the first request. This prevents the first student
submission from blocking a worker thread for 10-30s during a HuggingFace
download on a Render cold start.
"""
import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class CoreConfig(AppConfig):
    name = "core"

    def ready(self):
        # Import here to avoid AppRegistryNotReady errors during startup.
        # _load_artefacts is lru_cache'd so this single call warms the cache;
        # all subsequent request-time calls return instantly.
        try:
            from .inference import _load_artefacts
            _load_artefacts()
            logger.info("ML artefacts loaded successfully at startup.")
        except Exception as exc:
            # Don't crash the entire app if artefacts are missing —
            # individual prediction requests will surface the error gracefully.
            logger.warning("ML artefacts could not be loaded at startup: %s", exc)