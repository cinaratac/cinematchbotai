import os
import logging

from logging_config import configure_logging

configure_logging()

from aiohttp import web

from voice.server import create_app


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    logger.info(
        "CineMatch API ve Voice Agent baslatiliyor.",
        extra={"event": "service_starting", "status": "starting"},
    )
    web.run_app(create_app(), host="0.0.0.0", port=port)
