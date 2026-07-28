import os

from aiohttp import web

from voice.server import create_app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    print(f"CineMatch API ve WebRTC Voice Agent {port} portunda başlatılıyor...")
    web.run_app(create_app(), host="0.0.0.0", port=port)
