import asyncio
import json
import time
from fractions import Fraction

import aiohttp
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame
from av.audio.resampler import AudioResampler

from database import get_or_create_session, log_chat, touch_session
from voice.config import (
    DEEPGRAM_API_KEY,
    PEER_CONNECTIONS,
    VOICE_AGENT_PROMPT,
    cors_headers as build_cors_headers,
    request_is_authorized,
    rtc_configuration,
)


class AgentAudioTrack(MediaStreamTrack):
    """Deepgram'ın 24 kHz linear16 sesini WebRTC audio track olarak yayınlar."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._buffer = bytearray()
        self._condition = asyncio.Condition()
        self._pts = 0
        self._closed = False
        self._playback_started = False
        self._clock_start = None
        self._utterance_done = False

    async def write(self, data):
        async with self._condition:
            self._buffer.extend(data)
            self._condition.notify_all()

    async def clear(self):
        async with self._condition:
            self._buffer.clear()
            self._utterance_done = True
            self._condition.notify_all()

    async def mark_utterance_done(self):
        async with self._condition:
            self._utterance_done = True
            self._condition.notify_all()

    async def begin_utterance(self):
        async with self._condition:
            self._playback_started = False
            self._utterance_done = False

    async def recv(self):
        # 24 kHz'de 20 ms = 480 sample = 960 byte linear16 mono.
        frame_size = 960

        # İlk sesi yaklaşık 80 ms biriktirdikten sonra başlat. Küçük ağ/TTS
        # dalgalanmaları böylece kelime ortasında tamponun boşalmasına yol açmaz.
        async with self._condition:
            if (
                self._playback_started
                and self._utterance_done
                and not self._buffer
            ):
                self._playback_started = False

            if not self._playback_started:
                await self._condition.wait_for(
                    lambda: (
                        len(self._buffer) >= frame_size * 4
                        or (self._utterance_done and len(self._buffer) >= frame_size)
                        or self._closed
                    )
                )
                self._playback_started = True
                self._utterance_done = False
                self._clock_start = time.time() - (self._pts / 24000)

            if self._closed and not self._buffer:
                self.stop()
                raise asyncio.CancelledError

        # WebRTC'ye kesintisiz 20 ms kare ver. TTS paketi birkaç milisaniye
        # gecikirse recv'i durdurmak yerine sessiz kare üret; zaman çizelgesi
        # ve sonraki kelimeler korunur.
        target_time = self._clock_start + (self._pts / 24000)
        await asyncio.sleep(max(0, target_time - time.time()))

        async with self._condition:
            if len(self._buffer) >= frame_size:
                chunk = bytes(self._buffer[:frame_size])
                del self._buffer[:frame_size]
            elif self._buffer:
                chunk = bytes(self._buffer)
                self._buffer.clear()
                chunk += bytes(frame_size - len(chunk))
            else:
                chunk = bytes(frame_size)

        frame = AudioFrame(format="s16", layout="mono", samples=480)
        frame.planes[0].update(chunk)
        frame.sample_rate = 24000
        frame.pts = self._pts
        frame.time_base = Fraction(1, 24000)
        self._pts += 480
        return frame

    async def close(self):
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

# WebRTC sesli kanalının güncel uygulaması: STT + LLM + TTS tek bir Deepgram
# Voice Agent bağlantısında çalışır. Yukarıdaki eski fonksiyon bu tanımla
# değiştirilir ve artık çağrılmaz.
async def handle_user_speech(track, pc, user_context, output_track):
    if not DEEPGRAM_API_KEY:
        print("Deepgram Agent Hatası: DEEPGRAM_API_KEY tanımlı değil.")
        return

    profile = user_context["app_profile"]
    profile_lines = []
    for label, key in (
        ("Sevdiği türler", "favorite_genres"),
        ("Sevdiği yönetmenler", "favorite_directors"),
        ("Sevdiği oyuncular", "favorite_actors"),
        ("Favori filmleri", "favorite_movies"),
    ):
        values = profile.get(key) or []
        if values:
            profile_lines.append(f"- {label}: {', '.join(values)}")

    prompt = VOICE_AGENT_PROMPT
    if profile_lines:
        prompt += (
            "\n\nKULLANICININ CİNEMATCH ZEVK PROFİLİ:\n"
            + "\n".join(profile_lines)
            + "\nFilm önerilerinde bu tercihleri doğal biçimde dikkate al."
        )

    agent_url = "wss://agent.deepgram.com/v1/agent/converse"
    settings_applied = asyncio.Event()
    input_resampler = AudioResampler(format="s16", layout="mono", rate=16000)
    drop_interrupted_audio = False
    new_response_started = False
    interrupted_user_committed = False
    pending_user_text = None
    turn_user_text = None
    pending_assistant_chunks = []
    persistence_tasks = set()

    def track_persistence_task(task):
        persistence_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error:
            print("WebRTC konuşma kayıt hatası:", repr(error))

    async def persist_turn(user_text, assistant_text):
        session_id = user_context.get("session_id")
        if not session_id:
            return
        logged = await asyncio.to_thread(
            log_chat,
            session_id,
            user_context["user_id"],
            user_context["username"],
            user_text,
            assistant_text,
            channel="voice_webrtc",
            input_type="streaming_audio",
        )
        await asyncio.to_thread(touch_session, session_id, logged)

    def flush_completed_turn():
        nonlocal turn_user_text, pending_assistant_chunks
        if turn_user_text and pending_assistant_chunks:
            assistant_text = " ".join(pending_assistant_chunks)
            task = asyncio.create_task(
                persist_turn(turn_user_text, assistant_text)
            )
            persistence_tasks.add(task)
            task.add_done_callback(track_persistence_task)
        turn_user_text = None
        pending_assistant_chunks = []

    def send_control(action):
        channel = user_context.get("control_channel")
        if channel and channel.readyState == "open":
            channel.send(json.dumps({"action": action}))

    settings = {
        "type": "Settings",
        "tags": ["cinematch", "webrtc"],
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": 16000},
            "output": {
                "encoding": "linear16",
                "sample_rate": 24000,
                "container": "none",
            },
        },
        "agent": {
            "language": "tr",
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "model": "nova-3",
                    "language": "tr",
                    "smart_format": True,
                }
            },
            "think": {
                "provider": {
                    "type": "google",
                    "model": "gemini-3.1-flash-lite",
                    "temperature": 0.2,
                },
                "prompt": prompt,
            },
            # Aura henüz Türkçe sunmadığı için Deepgram tarafından yönetilen
            # Cartesia kullanılır; istemci yalnızca Deepgram'a bağlanır.
            "speak": {
                "provider": {
                    "type": "cartesia",
                    "model_id": "sonic-3",
                    "voice": {
                        "mode": "id",
                        "id": "a167e0f3-df7e-4d52-a9c3-f949145efdab",
                    },
                    "language": "tr",
                    "speed": "normal",
                }
            },
        },
    }

    print("Kullanıcı dinleniyor (Deepgram Voice Agent)...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                agent_url,
                headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                heartbeat=20,
            ) as ws:

                async def send_microphone():
                    await settings_applied.wait()
                    try:
                        while pc.connectionState not in {"closed", "failed"}:
                            frame = await track.recv()
                            for converted in input_resampler.resample(frame):
                                await ws.send_bytes(bytes(converted.planes[0]))
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        print("Mikrofon akışı sona erdi:", repr(e))
                    finally:
                        if not ws.closed:
                            await ws.close()

                send_task = None
                async for msg in ws:
                    # Tarayıcıdaki yerel VAD, Deepgram transkriptini beklemeden
                    # kesme sinyali gönderebilir. Bir sonraki WebSocket olayından
                    # itibaren eski cevaba ait bütün ses paketlerini reddet.
                    if user_context.pop("client_interrupt_pending", False):
                        drop_interrupted_audio = True
                        new_response_started = False
                        interrupted_user_committed = False

                    if msg.type == aiohttp.WSMsgType.BINARY:
                        # Kullanıcı araya girdikten sonra eski cevaba ait, ağda
                        # kalmış ses paketlerini yeni cevap başlayana kadar at.
                        if not drop_interrupted_audio:
                            await output_track.write(msg.data)
                        continue

                    if msg.type == aiohttp.WSMsgType.ERROR:
                        print("Deepgram Agent WebSocket Hatası:", ws.exception())
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue

                    event = json.loads(msg.data)
                    event_type = event.get("type")

                    if event_type == "Welcome":
                        await ws.send_json(settings)
                    elif event_type == "SettingsApplied":
                        print("Deepgram Voice Agent ayarları uygulandı.")
                        settings_applied.set()
                        send_task = asyncio.create_task(send_microphone())
                    elif event_type == "ConversationText":
                        role = event.get("role", "unknown")
                        content = str(event.get("content") or "").strip()
                        # Ham transkript merkezi loga yazilmaz; yalnizca olay
                        # tipi ve metin uzunlugu operasyonel olarak kaydedilir.
                        print(
                            "Deepgram konuşma metni olayı:",
                            f"role={role}",
                            f"characters={len(content)}",
                        )
                        if role == "user" and content:
                            flush_completed_turn()
                            pending_user_text = content
                            # Araya giren kullanıcının yeni cümlesi Deepgram
                            # tarafından kesinleştirildi. Bundan sonra gelecek
                            # ilk assistant parçası yeni cevaba aittir.
                            if drop_interrupted_audio:
                                interrupted_user_committed = True
                        elif role == "assistant" and content:
                            if (
                                drop_interrupted_audio
                                and not interrupted_user_committed
                            ):
                                # Eski LLM cevabının kesme sinyalinden sonra
                                # ulaşan metin parçalarıdır; sesi yeniden açma.
                                print(
                                    "[Deepgram] Kesilen eski cevabın parçası "
                                    "yok sayıldı."
                                )
                            else:
                                # İlk normal cevap veya araya giren kullanıcının
                                # kesinleşmiş mesajına verilen yeni cevap.
                                if not new_response_started:
                                    await output_track.begin_utterance()
                                    new_response_started = True
                                    send_control("resume")
                                drop_interrupted_audio = False
                                interrupted_user_committed = False
                                if turn_user_text is None:
                                    turn_user_text = pending_user_text
                                pending_user_text = None
                                pending_assistant_chunks.append(content)
                    elif event_type == "UserStartedSpeaking":
                        # Barge-in: mevcut cevabı anında kes ve eski cevabın
                        # geç gelebilecek TTS paketlerini kabul etme. Mikrofon
                        # send_microphone içinde kesintisiz gönderilmeye devam eder.
                        drop_interrupted_audio = True
                        new_response_started = False
                        interrupted_user_committed = False
                        send_control("interrupt")
                        await output_track.clear()
                        print("[Deepgram] Kullanıcı araya girdi; önceki cevap kesildi.")
                    elif event_type == "AgentThinking":
                        print("[Deepgram] CineMatch düşünüyor...")
                    elif event_type == "AgentStartedSpeaking":
                        # Eski cevap hâlâ TTS'e geçmeye çalışıyorsa bu olay da
                        # sesi yeniden açmamalı. Yeni kullanıcı metni
                        # kesinleştiyse yeni cevabın başlamasına izin ver.
                        if not drop_interrupted_audio or interrupted_user_committed:
                            if not new_response_started:
                                await output_track.begin_utterance()
                                new_response_started = True
                                send_control("resume")
                            drop_interrupted_audio = False
                            interrupted_user_committed = False
                        print("[Deepgram] CineMatch konuşuyor...")
                    elif event_type == "AgentAudioDone":
                        await output_track.mark_utterance_done()
                        flush_completed_turn()
                        print("[Deepgram] Ajan ses yanıtı tamamlandı.")
                    elif event_type == "LatencyReport":
                        print("[Deepgram Metrik]:", event)
                    elif event_type in {"Warning", "Error"}:
                        print(f"Deepgram Agent {event_type}:", event)
                        if event_type == "Error":
                            break

                if send_task:
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass
    except Exception as e:
        print("Deepgram Voice Agent Bağlantı Hatası:", repr(e))
    finally:
        flush_completed_turn()
        if persistence_tasks:
            await asyncio.gather(*persistence_tasks, return_exceptions=True)


async def offer(request):
    cors_headers = build_cors_headers(request)
    if request.headers.get("Origin") and "Access-Control-Allow-Origin" not in cors_headers:
        return web.json_response(
            {"status": "error", "message": "Bu origin için erişim izni yok."},
            status=403,
            headers=cors_headers,
        )
    if not request_is_authorized(request):
        return web.json_response(
            {"status": "error", "message": "Yetkisiz erişim."},
            status=401,
            headers=cors_headers,
        )
    if not DEEPGRAM_API_KEY:
        return web.json_response(
            {"status": "error", "message": "Voice agent yapılandırılmamış."},
            status=503,
            headers=cors_headers,
        )

    try:
        params = await request.json()
        remote_sdp = params["sdp"]
        remote_type = params["type"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return web.json_response(
            {"status": "error", "message": "Geçerli sdp ve type alanları gerekli."},
            status=400,
            headers=cors_headers,
        )

    offer = RTCSessionDescription(sdp=remote_sdp, type=remote_type)
    pc = RTCPeerConnection(configuration=rtc_configuration())
    PEER_CONNECTIONS.add(pc)
    output_track = AgentAudioTrack()
    pc.addTrack(output_track)

    # İstemci bu alanları gönderirse normal CineMatch API'siyle aynı kullanıcı
    # hafızası ve zevk profili kullanılır. Eski istemciler için güvenli
    # varsayılanlar bırakıldı.
    user_context = {
        "user_id": str(params.get("user_id") or "webrtc_user"),
        "username": str(params.get("username") or "WebRTC_User"),
        "movie_name": params.get("movie_name") or params.get("movie_title"),
        "app_profile": {
            "favorite_genres": params.get("favorite_genres") or [],
            "favorite_directors": params.get("favorite_directors") or [],
            "favorite_actors": params.get("favorite_actors") or [],
            "favorite_movies": params.get("favorite_movies") or [],
        },
        "control_channel": None,
        "client_interrupt_pending": False,
        "session_id": None,
    }
    try:
        user_context["session_id"] = await asyncio.to_thread(
            get_or_create_session,
            user_context["user_id"],
            user_context["username"],
        )
    except Exception as session_error:
        # Eski WebRTC ses hattı veritabanı geçici olarak kullanılamadığında da
        # çalışabilsin; yalnızca bu görüşmenin kalıcı logu atlanır.
        print("WebRTC oturumu oluşturulamadı:", repr(session_error))

    @pc.on("datachannel")
    def on_datachannel(channel):
        if channel.label == "cinematch-control":
            user_context["control_channel"] = channel
            print("WebRTC kontrol kanalı hazır.")

            @channel.on("message")
            def on_control_message(message):
                try:
                    payload = json.loads(message)
                except (TypeError, json.JSONDecodeError):
                    return
                if payload.get("action") == "interrupt":
                    user_context["client_interrupt_pending"] = True
                    asyncio.create_task(output_track.clear())
                    print(
                        "[WebRTC] Yerel ses algılandı; Deepgram "
                        "transkripti beklenmeden cevap kesildi."
                    )
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("WebRTC Bağlantı Durumu:", pc.connectionState)
        if pc.connectionState in {"disconnected", "failed", "closed"}:
            await output_track.close()
            PEER_CONNECTIONS.discard(pc)
        if pc.connectionState == "failed":
            await pc.close()

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            print("Ses track'i alındı, dinleme başlatılıyor...")
            asyncio.create_task(
                handle_user_speech(track, pc, user_context, output_track)
            )

    try:
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except Exception:
        PEER_CONNECTIONS.discard(pc)
        await pc.close()
        raise

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        headers=cors_headers,
    )

async def options_handler(request):
    headers = build_cors_headers(request)
    if request.headers.get("Origin") and "Access-Control-Allow-Origin" not in headers:
        return web.Response(status=403, headers=headers)
    return web.Response(status=204, headers=headers)
