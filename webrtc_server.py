import asyncio
import json
import re
import time
import os
import aiohttp
from fractions import Fraction
from aiohttp import web
from aiohttp_wsgi import WSGIHandler
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from av import AudioFrame
from av.audio.resampler import AudioResampler
from ai_service import get_ai_response, text_to_speech
from app_guide import CINEMATCH_APP_GUIDE
from database import (
    get_or_create_session,
    get_session_transcript_recent,
    log_chat,
    log_performance_metric,
    touch_session,
)

# Ortam değişkenlerinden Deepgram anahtarını alıyoruz
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
VOICE_API_KEY = os.environ.get("VOICE_API_KEY", "")
VOICE_ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.environ.get("VOICE_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
}
PEER_CONNECTIONS = set()


def _cors_headers(request):
    origin = request.headers.get("Origin", "")
    if "*" in VOICE_ALLOWED_ORIGINS:
        allowed_origin = "*"
    elif origin in VOICE_ALLOWED_ORIGINS:
        allowed_origin = origin
    else:
        allowed_origin = ""

    headers = {
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, X-Voice-Api-Key",
        "Vary": "Origin",
    }
    if allowed_origin:
        headers["Access-Control-Allow-Origin"] = allowed_origin
    return headers


def _request_is_authorized(request):
    if not VOICE_API_KEY:
        # Yerel geliştirmeyi bozma; Render'da yanlışlıkla korumasız yayınlama.
        return not os.environ.get("RENDER")
    return request.headers.get("X-Voice-Api-Key", "") == VOICE_API_KEY


def _rtc_configuration():
    """İsteğe bağlı TURN sunucusunu aiortc bağlantısına ekler."""
    ice_servers = []
    stun_url = os.environ.get(
        "STUN_URL", "stun:stun.l.google.com:19302"
    ).strip()
    if stun_url:
        ice_servers.append(RTCIceServer(urls=stun_url))

    turn_url = os.environ.get("TURN_URL", "").strip()
    if turn_url:
        ice_servers.append(
            RTCIceServer(
                urls=turn_url,
                username=os.environ.get("TURN_USERNAME"),
                credential=os.environ.get("TURN_CREDENTIAL"),
            )
        )

    return RTCConfiguration(iceServers=ice_servers)

VOICE_AGENT_PROMPT = f"""
Sen CineMatch uygulamasının resmi yapay zeka asistanı, profesyonel bir sinema
asistanı ve film eleştirmenisin.

KURALLAR:
- Yalnızca filmler, diziler, yönetmenler, oyuncular, sinema sektörü ve CineMatch
  uygulaması hakkında cevap ver.
- İlgisiz bir soru gelirse kısa biçimde yalnızca sinema ve CineMatch hakkında
  yardımcı olabildiğini söyle ve konuşmayı sinemaya yönlendir.
- Her zaman Türkçe konuş. Cevapların doğal, kısa ve öz; normalde 2-4 cümle olsun.
- Ses üretiminin erken başlayabilmesi için ilk cümleyi mümkün olduğunca çabuk
  tamamla. Kısa ve tam cümleler kur; gereksiz virgül, üç nokta, parantez,
  ünlem tekrarı ve uzun duraklama oluşturacak ifadeler kullanma.
- Film önerirken kullanıcının belirttiği tür ve zevklere uy; uymayan bir filmi
  o türe aitmiş gibi gösterme.
- Bir filmden bahsederken kuru özet verme; oyunculuk, yönetmenlik veya
  sinematografi hakkında kısa bir eleştirmen yorumu da ekle.
- Doğrulayamadığın IMDb puanı, gişe hasılatı veya çıkış yılı gibi sayısal
  bilgileri uydurma.
- Robotik kapanışlar yapma; konuşmanın bağlamına uygun doğal bir soru sor.
- Kullanıcı bir tercih, duygu veya görüş belirttiğinde uygun olduğu zaman cevaba
  kısa ve doğal bir geri bildirimle başla: "Anladım", "Hı hı", "Haklısın"
  veya "Evet, seni anlıyorum" gibi. Bunu her cevapta tekrarlama ve kullanıcı
  konuşurken sözünü kesme; kullanıcı sözünü bitirdikten sonra söyle.
- Sesli yanıta uygun konuş; markdown, tablo, bağlantı, emoji ve
  [[FILMLER: ...]] gibi makine işaretleri kullanma.
- CineMatch hakkında sorulursa yalnızca aşağıdaki uygulama rehberindeki kesin
  bilgileri kullan; rehberde yoksa bilgi uydurma.

CINEMATCH UYGULAMA REHBERİ:
{CINEMATCH_APP_GUIDE}
""".strip()


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

async def handle_user_speech(track, pc, user_context):
    print("Kullanıcı dinleniyor (Native WebSocket Streaming)...")
    
    metrics = {
        "user_speech_end": None,
        "stt_latency_ms": None,
        "ttfb_ms": None,
        "tts_latency_ms": None,
        "ttfs_ms": None
    }
    
    final_transcript = ""
    last_interim_transcript = ""
    speech_detected = False
    is_speaking = True

    # Deepgram'ın doğrudan çekirdek (WebSocket) bağlantı adresi
    dg_url = "wss://api.deepgram.com/v1/listen?model=nova-2&language=tr&smart_format=true&encoding=linear16&sample_rate=48000&interim_results=true&utterance_end_ms=1000&vad_events=true"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                dg_url, 
                headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
            ) as ws:
                # Tarayıcıdan gelen WebRTC sesi genellikle float/planar olur.
                # Deepgram'a URL'de belirttiğimiz biçimde PCM linear16 gönder.
                resampler = AudioResampler(
                    format="s16",
                    layout="mono",
                    rate=48000,
                )
                
                # Görev 1: WebRTC'den sesi alıp Deepgram'a eşzamanlı akıt
                async def send_audio():
                    while is_speaking:
                        try:
                            frame = await track.recv()
                            converted_frames = resampler.resample(frame)
                            for converted_frame in converted_frames:
                                await ws.send_bytes(bytes(converted_frame.planes[0]))
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            print("Ses gönderme hatası:", repr(e))
                            break

                    # Dinleme bitince bağlantıyı temiz şekilde kapat
                    try:
                        await ws.send_json({"type": "CloseStream"})
                    except Exception:
                        pass

                send_task = asyncio.create_task(send_audio())

                # Görev 2: Deepgram'dan gelen anlık metni dinle
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        result = json.loads(msg.data)
                        
                        if result.get("type") == "Results":
                            channel = result.get("channel", {})
                            alts = channel.get("alternatives", [])
                            if alts:
                                sentence = alts[0].get("transcript", "")
                                is_final = result.get("is_final", False)
                                speech_final = result.get("speech_final", False)
                                
                                if sentence.strip():
                                    speech_detected = True
                                    last_interim_transcript = sentence.strip()

                                    if is_final:
                                        final_transcript += sentence.strip() + " "
                                        last_interim_transcript = ""
                                        print(f"[Anlık STT]: {final_transcript}")
                                    
                                # Boş bir VAD sonucu oturumu erken kapatmasın.
                                if speech_final and (final_transcript.strip() or sentence.strip()):
                                    metrics["user_speech_end"] = time.perf_counter()
                                    is_speaking = False
                                    break
                                    
                        # 1 saniyelik sessizlik (VAD) tespit edildiğinde
                        elif result.get("type") == "UtteranceEnd":
                            if speech_detected:
                                metrics["user_speech_end"] = time.perf_counter()
                                is_speaking = False
                                break
                            print("[Deepgram] Konuşma başlamadan gelen UtteranceEnd yok sayıldı.")
                        elif result.get("type") == "Error":
                            print("Deepgram STT Hatası:", result)
                            is_speaking = False
                            break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print("Deepgram WebSocket Hatası:", ws.exception())
                        break
                
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass
    except Exception as e:
        print("Deepgram Bağlantı Hatası:", e)
        return

    if metrics["user_speech_end"] is None:
        metrics["user_speech_end"] = time.perf_counter()

    # Bağlantı final sonuçtan önce kapanırsa son anlamlı ara sonucu kaybetme.
    if not final_transcript.strip() and last_interim_transcript:
        final_transcript = last_interim_transcript + " "
        print("[Deepgram] Son ara transkript kullanıldı.")
    
    if not final_transcript.strip():
        print("Sessizlik algılandı veya konuşma anlaşılamadı, işlem iptal.")
        return

    print(f"Kullanıcı (STT Tamamlandı): {final_transcript}")
    
    # STT gecikmesini hesapla
    metrics["stt_latency_ms"] = round((time.perf_counter() - metrics["user_speech_end"]) * 1000)
    print(f"[METRİK] STT Latency: {metrics['stt_latency_ms']} ms")

    # 3. CİNEMATCH AI ZİNCİRİ VE METRİK TAKİBİ
    #
    # Ham _call_openrouter_stream çağrısı, ai_service içindeki sistem promptunu,
    # kullanıcı hafızasını, uygulama rehberini ve OMDb araçlarını atlıyordu.
    # Burada normal Telegram/API akışıyla aynı get_ai_response fonksiyonunu
    # kullanarak bütün kanallarda aynı asistan davranışını koruyoruz.
    try:
        ai_start_time = time.perf_counter()
        ai_answer, recommended_movies, session_id, diagnostics = await asyncio.to_thread(
            get_ai_response,
            final_transcript.strip(),
            user_id=user_context["user_id"],
            username=user_context["username"],
            app_profile=user_context["app_profile"],
            movie_name=user_context["movie_name"],
            include_diagnostics=True,
            allow_stateless=True,
        )
        metrics["ttfb_ms"] = round((time.perf_counter() - ai_start_time) * 1000)
        print(f"[METRİK] AI Response Ready: {metrics['ttfb_ms']} ms")
        print(f"[CineMatch] Oturum: {session_id}, araç çağrısı: {diagnostics['tool_call_count']}")

        first_audio_generated = False

        # Film kartı işaretleri get_ai_response içinde ayrıştırıldığı için burada
        # yalnızca kullanıcıya okunacak temiz cevap bulunur.
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.?!])\s+", ai_answer.strip())
            if sentence.strip()
        ]
        if not sentences and ai_answer.strip():
            sentences = [ai_answer.strip()]

        for sentence_to_speak in sentences:
            print(f"Yapay Zeka: {sentence_to_speak}")

            tts_start_time = time.perf_counter()
            audio_chunk_bytes = await asyncio.to_thread(
                text_to_speech, sentence_to_speak
            )

            if not first_audio_generated:
                metrics["tts_latency_ms"] = round((time.perf_counter() - tts_start_time) * 1000)
                metrics["ttfs_ms"] = round((time.perf_counter() - metrics["user_speech_end"]) * 1000)

                print(f"[METRİK] TTS Latency: {metrics['tts_latency_ms']} ms")
                print(f"[METRİK] TTFS (Time to First Speech): {metrics['ttfs_ms']} ms")
                print("--> İlk ses verisi üretildi; istemciye aktarım henüz bağlı değil.")
                first_audio_generated = True
                    
    except Exception as e:
        print(f"İşlem sırasında hata: {e}")


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
                        content = event.get("content", "")
                        print(f"[{role}]: {content}")
                        if role == "user" and content.strip():
                            # Araya giren kullanıcının yeni cümlesi Deepgram
                            # tarafından kesinleştirildi. Bundan sonra gelecek
                            # ilk assistant parçası yeni cevaba aittir.
                            if drop_interrupted_audio:
                                interrupted_user_committed = True
                        elif role == "assistant" and content.strip():
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


async def offer(request):
    cors_headers = _cors_headers(request)
    if request.headers.get("Origin") and "Access-Control-Allow-Origin" not in cors_headers:
        return web.json_response(
            {"status": "error", "message": "Bu origin için erişim izni yok."},
            status=403,
            headers=cors_headers,
        )
    if not _request_is_authorized(request):
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
    pc = RTCPeerConnection(configuration=_rtc_configuration())
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
    }

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
    headers = _cors_headers(request)
    if request.headers.get("Origin") and "Access-Control-Allow-Origin" not in headers:
        return web.Response(status=403, headers=headers)
    return web.Response(status=204, headers=headers)


def _streaming_agent_settings(app_profile, history, input_sample_rate):
    history_messages = []
    for row in history:
        if row.get("user_message"):
            history_messages.append({
                "type": "History",
                "role": "user",
                "content": row["user_message"],
            })
        if row.get("bot_response"):
            history_messages.append({
                "type": "History",
                "role": "assistant",
                "content": row["bot_response"],
            })

    profile_text = json.dumps(app_profile, ensure_ascii=False)
    prompt = (
        f"{VOICE_AGENT_PROMPT}\n\n"
        f"KULLANICI ZEVK PROFİLİ: {profile_text}\n"
        "Bu tercihleri film önerilerinde doğal biçimde dikkate al."
    )
    return {
        "type": "Settings",
        "tags": ["cinematch", "websocket-streaming"],
        "audio": {
            "input": {
                "encoding": "linear16",
                "sample_rate": input_sample_rate,
            },
            "output": {
                "encoding": "linear16",
                "sample_rate": 24000,
                "container": "none",
            },
        },
        "agent": {
            "language": "tr",
            "context": {"messages": history_messages},
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "model": "nova-3",
                    "language": "tr",
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


async def voice_stream(request):
    """Tarayıcı ile Deepgram arasında düşük gecikmeli PCM WebSocket köprüsü."""
    cors_headers = _cors_headers(request)
    if request.headers.get("Origin") and "Access-Control-Allow-Origin" not in cors_headers:
        return web.json_response(
            {"status": "error", "message": "Bu origin için erişim izni yok."},
            status=403,
            headers=cors_headers,
        )

    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=15 * 1024 * 1024)
    await ws.prepare(request)

    try:
        auth_message = await asyncio.wait_for(ws.receive(), timeout=10)
        if auth_message.type != aiohttp.WSMsgType.TEXT:
            await ws.close(code=4001, message=b"Kimlik dogrulama gerekli")
            return ws

        try:
            context = json.loads(auth_message.data)
        except json.JSONDecodeError:
            context = {}

        if (
            context.get("type") != "auth"
            or not VOICE_API_KEY
            or context.get("api_key") != VOICE_API_KEY
        ):
            await ws.send_json({"type": "error", "message": "Yetkisiz erişim."})
            await ws.close(code=4001, message=b"Yetkisiz")
            return ws

        user_id = str(context.get("user_id") or "voice-user")
        username = str(context.get("username") or "Voice User")
        app_profile = {
            "favorite_genres": context.get("favorite_genres") or [],
            "favorite_directors": context.get("favorite_directors") or [],
            "favorite_actors": context.get("favorite_actors") or [],
            "favorite_movies": context.get("favorite_movies") or [],
        }
        session_id = await asyncio.to_thread(
            get_or_create_session, user_id, username
        )
        history = await asyncio.to_thread(
            get_session_transcript_recent, session_id, 6
        )
        try:
            input_sample_rate = int(context.get("sample_rate") or 48000)
        except (TypeError, ValueError):
            input_sample_rate = 48000
        if input_sample_rate < 8000 or input_sample_rate > 48000:
            input_sample_rate = 48000

        settings = _streaming_agent_settings(
            app_profile, history, input_sample_rate
        )
        await ws.send_json({"type": "connecting"})

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                "wss://agent.deepgram.com/v1/agent/converse",
                headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                heartbeat=20,
            ) as agent_ws:
                pending_user_text = None
                settings_applied = asyncio.Event()
                persistence_tasks = set()
                drop_interrupted_audio = False
                interrupted_user_committed = False
                deepgram_request_id = None
                turn_number = 0
                client_interrupt_started = None
                latency_logged_for_turn = False

                def track_persistence_task(task):
                    persistence_tasks.discard(task)
                    if task.cancelled():
                        return
                    error = task.exception()
                    if error:
                        print("Voice metrik/veritabanı kayıt hatası:", repr(error))

                async def persist_turn(user_text, assistant_text):
                    await asyncio.to_thread(
                        log_chat,
                        session_id,
                        user_id,
                        username,
                        user_text,
                        assistant_text,
                    )
                    await asyncio.to_thread(touch_session, session_id)

                async def persist_latency(event):
                    nonlocal turn_number
                    turn_number += 1

                    def milliseconds(field):
                        value = event.get(field)
                        if isinstance(value, (int, float)):
                            return round(value * 1000)
                        return None

                    ai_ms = milliseconds("ttt_text_latency")
                    if ai_ms is None:
                        ai_ms = milliseconds("ttt_token_latency")
                    if ai_ms is None:
                        ai_ms = milliseconds("ttt_latency")
                    tts_ms = milliseconds("tts_latency")
                    e2e_ms = milliseconds("total_latency")
                    if ai_ms is None and e2e_ms is not None:
                        # Bazı sağlayıcı/turlarda yalnızca total + TTS gelir.
                        # Belge admin başarı kohortundan düşmesin.
                        ai_ms = max(0, e2e_ms - (tts_ms or 0))
                    asr_ms = None
                    if e2e_ms is not None and ai_ms is not None and tts_ms is not None:
                        # Deepgram ayrı STT alanı vermiyor. Total = utterance
                        # sonu → ilk ses olduğundan kalan süre STT finalizasyonu
                        # ve Agent yönlendirme ek yükünün yaklaşık değeridir.
                        asr_ms = max(0, e2e_ms - ai_ms - tts_ms)

                    metric = {
                        "channel": "voice_websocket",
                        "input_type": "streaming_audio",
                        "user_id": user_id,
                        "username": username,
                        "session_id": session_id,
                        "deepgram_request_id": deepgram_request_id,
                        "turn_number": turn_number,
                        "asr_ms": asr_ms,
                        "asr_is_estimate": True,
                        "ai_ms": ai_ms,
                        "ai_ready_ms": (
                            e2e_ms - tts_ms
                            if e2e_ms is not None and tts_ms is not None
                            else None
                        ),
                        "ttfb_ms": ai_ms,
                        "tts_ms": tts_ms,
                        "tts_ready_ms": e2e_ms,
                        "ttfs_ms": e2e_ms,
                        "e2e_ms": e2e_ms,
                        "deepgram_ttt_token_ms": milliseconds(
                            "ttt_token_latency"
                        ),
                        "deepgram_ttt_text_ms": milliseconds(
                            "ttt_text_latency"
                        ),
                        "deepgram_ttt_tool_ms": milliseconds(
                            "ttt_tool_latency"
                        ),
                        "deepgram_ttt_thinking_ms": milliseconds(
                            "ttt_thinking_latency"
                        ),
                        "asr_model": "deepgram/nova-3",
                        "ai_model": "google/gemini-3.1-flash-lite",
                        "tts_model": "cartesia/sonic-3",
                        "tool_call_count": 0,
                        "status": "success",
                        "failed_stage": None,
                        "error_type": None,
                    }
                    await asyncio.to_thread(log_performance_metric, metric)

                async def browser_to_agent():
                    nonlocal drop_interrupted_audio
                    nonlocal interrupted_user_committed
                    nonlocal client_interrupt_started
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.BINARY:
                            if message.data:
                                await settings_applied.wait()
                                await agent_ws.send_bytes(message.data)
                        elif message.type == aiohttp.WSMsgType.TEXT:
                            try:
                                command = json.loads(message.data)
                            except json.JSONDecodeError:
                                continue
                            if command.get("type") == "keepalive":
                                await agent_ws.send_json({"type": "KeepAlive"})
                            elif command.get("type") == "interrupt":
                                drop_interrupted_audio = True
                                interrupted_user_committed = False
                                client_interrupt_started = time.perf_counter()
                                await ws.send_json({
                                    "type": "interrupt_acknowledged"
                                })
                        elif message.type in {
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        }:
                            break

                async def agent_to_browser():
                    nonlocal pending_user_text
                    nonlocal drop_interrupted_audio
                    nonlocal interrupted_user_committed
                    nonlocal deepgram_request_id
                    nonlocal client_interrupt_started
                    nonlocal latency_logged_for_turn
                    async for message in agent_ws:
                        if message.type == aiohttp.WSMsgType.BINARY:
                            if not drop_interrupted_audio:
                                await ws.send_bytes(message.data)
                            continue
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue

                        event = json.loads(message.data)
                        event_type = event.get("type")
                        if event_type == "Welcome":
                            deepgram_request_id = event.get("request_id")
                            await agent_ws.send_json(settings)
                        elif event_type == "SettingsApplied":
                            settings_applied.set()
                            await ws.send_json({
                                "type": "ready",
                                "session_id": session_id,
                                "input_sample_rate": input_sample_rate,
                                "output_sample_rate": 24000,
                            })
                        elif event_type == "ConversationText":
                            role = event.get("role")
                            content = str(event.get("content") or "").strip()
                            if not content:
                                continue
                            if (
                                role == "assistant"
                                and drop_interrupted_audio
                                and not interrupted_user_committed
                            ):
                                # Kesilmiş cevabın geç ulaşan metnini istemciye
                                # gönderme ve yeni turun parçası gibi kaydetme.
                                continue
                            if (
                                role == "assistant"
                                and drop_interrupted_audio
                                and interrupted_user_committed
                            ):
                                # Deepgram bazen AgentStartedSpeaking olayını
                                # ConversationText'ten önce yollar. Yeni kullanıcı
                                # turuna ait assistant metni kesinleştiğinde ses
                                # kapısını mutlaka yeniden aç.
                                drop_interrupted_audio = False
                                interrupted_user_committed = False
                                client_interrupt_started = None
                                await ws.send_json({"type": "speaking"})
                            await ws.send_json({
                                "type": "transcript",
                                "role": role,
                                "content": content,
                            })
                            if role == "user":
                                pending_user_text = content
                                if drop_interrupted_audio:
                                    interrupted_user_committed = True
                            elif role == "assistant" and pending_user_text:
                                user_text = pending_user_text
                                pending_user_text = None
                                persistence_task = asyncio.create_task(
                                    persist_turn(user_text, content)
                                )
                                persistence_tasks.add(persistence_task)
                                persistence_task.add_done_callback(
                                    track_persistence_task
                                )
                        elif event_type == "UserStartedSpeaking":
                            latency_logged_for_turn = False
                            drop_interrupted_audio = True
                            interrupted_user_committed = bool(pending_user_text)
                            await ws.send_json({"type": "interrupt"})
                        elif event_type == "AgentThinking":
                            await ws.send_json({"type": "processing"})
                        elif event_type == "AgentStartedSpeaking":
                            if (
                                not latency_logged_for_turn
                                and isinstance(event.get("total_latency"), (int, float))
                            ):
                                latency_task = asyncio.create_task(
                                    persist_latency(event)
                                )
                                persistence_tasks.add(latency_task)
                                latency_task.add_done_callback(
                                    track_persistence_task
                                )
                                latency_logged_for_turn = True
                            if drop_interrupted_audio and not interrupted_user_committed:
                                continue
                            client_interrupt_started = None
                            drop_interrupted_audio = False
                            interrupted_user_committed = False
                            await ws.send_json({"type": "speaking"})
                        elif event_type == "AgentAudioDone":
                            await ws.send_json({"type": "audio_done"})
                        elif event_type == "LatencyReport":
                            latency_task = asyncio.create_task(
                                persist_latency(event)
                            )
                            persistence_tasks.add(latency_task)
                            latency_task.add_done_callback(
                                track_persistence_task
                            )
                            latency_logged_for_turn = True
                        elif event_type in {"Warning", "Error"}:
                            await ws.send_json({
                                "type": event_type.lower(),
                                "message": event.get("description")
                                or event.get("message")
                                or "Deepgram Voice Agent hatası.",
                            })
                            if event_type == "Error":
                                break

                tasks = {
                    asyncio.create_task(browser_to_agent()),
                    asyncio.create_task(agent_to_browser()),
                }
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    task.result()
                if persistence_tasks:
                    await asyncio.gather(
                        *persistence_tasks, return_exceptions=True
                    )
    except asyncio.TimeoutError:
        await ws.close(code=4001, message=b"Kimlik dogrulama zaman asimi")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as exc:
        print("Voice streaming hatası:", repr(exc))
        if not ws.closed:
            await ws.send_json({
                "type": "error",
                "message": str(exc) or "Voice streaming hatası.",
            })

    return ws


async def close_peer_connections(app):
    await asyncio.gather(
        *(pc.close() for pc in tuple(PEER_CONNECTIONS)),
        return_exceptions=True,
    )
    PEER_CONNECTIONS.clear()


def create_app():
    # Flask REST/Telegram API'sini ve WebRTC sinyallemesini aynı portta yayınla.
    from main import app as flask_app

    app = web.Application()
    app.router.add_get("/api/voice/stream", voice_stream)
    app.router.add_post("/api/voice/offer", offer)
    app.router.add_options("/api/voice/offer", options_handler)
    app.router.add_route("*", "/{path_info:.*}", WSGIHandler(flask_app))
    app.on_shutdown.append(close_peer_connections)
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    print(f"CineMatch API ve WebRTC Voice Agent {port} portunda başlatılıyor...")
    web.run_app(create_app(), host="0.0.0.0", port=port)
