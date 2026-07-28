import asyncio


_active_connections = 0
_idle_event = None


def _get_idle_event():
    global _idle_event
    if _idle_event is None:
        _idle_event = asyncio.Event()
        if _active_connections == 0:
            _idle_event.set()
    return _idle_event


def voice_connection_started():
    global _active_connections
    _active_connections += 1
    _get_idle_event().clear()
    print("Aktif voice bağlantısı:", _active_connections)


def voice_connection_finished():
    global _active_connections
    _active_connections = max(0, _active_connections - 1)
    if _active_connections == 0:
        _get_idle_event().set()
    print("Aktif voice bağlantısı:", _active_connections)


async def wait_for_voice_idle(grace_seconds=15):
    """Hiçbir canlı görüşme yokken ve sakinlik süresi dolunca devam et."""
    idle_event = _get_idle_event()
    while True:
        await idle_event.wait()
        await asyncio.sleep(grace_seconds)
        if _active_connections == 0:
            return

