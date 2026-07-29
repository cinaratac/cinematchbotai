import asyncio
import os

from aiohttp import web
from aiohttp_wsgi import WSGIHandler

from voice.config import PEER_CONNECTIONS
from voice.legacy_webrtc import offer, options_handler
from voice.streaming import voice_stream


async def close_peer_connections(app):
    await asyncio.gather(
        *(pc.close() for pc in tuple(PEER_CONNECTIONS)),
        return_exceptions=True,
    )
    PEER_CONNECTIONS.clear()


def create_app():
    os.environ["CINEMATCH_DEFER_SERVICE_INITIALIZATION"] = "1"
    import main as main_module

    app = web.Application()
    app.router.add_get("/api/voice/stream", voice_stream)
    app.router.add_post("/api/voice/offer", offer)
    app.router.add_options("/api/voice/offer", options_handler)
    app.router.add_route("*", "/{path_info:.*}", WSGIHandler(main_module.app))

    async def start_external_services(aiohttp_app):
        # Flask/WSGI üzerinden gelen admin yeniden-değerlendirme isteği kendi
        # thread'inde çalışır. QA taskını voice bağlantılarının kullandığı bu
        # ana aiohttp event loop'unda başlatmak için loop'u kaydediyoruz.
        from evaluation_service import set_voice_evaluation_loop

        set_voice_evaluation_loop(asyncio.get_running_loop())
        task = asyncio.create_task(
            asyncio.to_thread(main_module.initialize_services),
            name="cinematch-service-initialization",
        )
        aiohttp_app["service_initialization_task"] = task

    app.on_startup.append(start_external_services)
    app.on_shutdown.append(close_peer_connections)
    return app
