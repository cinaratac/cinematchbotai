import asyncio
import json
import time

import aiohttp
from aiohttp import web

from database import (
    get_or_create_session,
    get_session_transcript_recent,
    log_chat,
    log_performance_metric,
    touch_session,
)
from voice.config import (
    DEEPGRAM_API_KEY,
    VOICE_AGENT_PROMPT,
    VOICE_API_KEY,
    cors_headers,
)
from voice.recording import VoiceRecordingSession
from voice.metrics import build_voice_metric
from voice.barge_in import BargeInState


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
    cors_headers = cors_headers(request)
    if request.headers.get("Origin") and "Access-Control-Allow-Origin" not in cors_headers:
        return web.json_response(
            {"status": "error", "message": "Bu origin için erişim izni yok."},
            status=403,
            headers=cors_headers,
        )

    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=15 * 1024 * 1024)
    await ws.prepare(request)
    recording = None

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

        recording = VoiceRecordingSession.create_if_enabled(
            session_id,
            user_id,
            username,
            input_sample_rate,
        )

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
                barge_in = BargeInState()
                deepgram_request_id = None
                turn_number = 0
                latency_event = {}
                turn_metric_logged = False
                first_audio_at = None
                audio_done_at = None
                user_transcript_ready_at = None
                assistant_text_ready_at = None

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

                async def persist_latency(
                    measured_first_audio_at,
                    measured_audio_done_at,
                    measured_latency_event,
                ):
                    nonlocal turn_number
                    turn_number += 1
                    metric = build_voice_metric(
                        latency_event=measured_latency_event,
                        first_audio_at=measured_first_audio_at,
                        playback_done_at=measured_audio_done_at,
                        user_id=user_id,
                        username=username,
                        session_id=session_id,
                        deepgram_request_id=deepgram_request_id,
                        turn_number=turn_number,
                    )
                    await asyncio.to_thread(log_performance_metric, metric)

                def schedule_latency_if_complete():
                    nonlocal turn_metric_logged
                    if (
                        turn_metric_logged
                        or first_audio_at is None
                        or audio_done_at is None
                        or not isinstance(
                            latency_event.get("total_latency"), (int, float)
                        )
                    ):
                        return
                    turn_metric_logged = True
                    latency_task = asyncio.create_task(
                        persist_latency(
                            first_audio_at,
                            audio_done_at,
                            dict(latency_event),
                        )
                    )
                    persistence_tasks.add(latency_task)
                    latency_task.add_done_callback(track_persistence_task)

                async def browser_to_agent():
                    nonlocal audio_done_at
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.BINARY:
                            if message.data:
                                await settings_applied.wait()
                                if recording:
                                    recording.write("user", message.data)
                                await agent_ws.send_bytes(message.data)
                        elif message.type == aiohttp.WSMsgType.TEXT:
                            try:
                                command = json.loads(message.data)
                            except json.JSONDecodeError:
                                continue
                            if command.get("type") == "keepalive":
                                await agent_ws.send_json({"type": "KeepAlive"})
                            elif command.get("type") == "interrupt":
                                barge_in.drop_interrupted_audio = True
                                barge_in.interrupted_user_committed = False
                                barge_in.client_interrupt_started = time.perf_counter()
                                await ws.send_json({
                                    "type": "interrupt_acknowledged"
                                })
                            elif command.get("type") == "playback_done":
                                # Tarayıcıdaki son ses buffer'ı gerçekten
                                # oynatıldı. Tam kullanıcı E2E bitişi budur.
                                audio_done_at = time.perf_counter()
                                schedule_latency_if_complete()
                        elif message.type in {
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        }:
                            break

                async def agent_to_browser():
                    nonlocal pending_user_text
                    nonlocal deepgram_request_id
                    nonlocal latency_event
                    nonlocal turn_metric_logged
                    nonlocal first_audio_at
                    nonlocal audio_done_at
                    nonlocal user_transcript_ready_at
                    nonlocal assistant_text_ready_at
                    async for message in agent_ws:
                        if message.type == aiohttp.WSMsgType.BINARY:
                            if not barge_in.drop_interrupted_audio:
                                if first_audio_at is None:
                                    first_audio_at = time.perf_counter()
                                if recording:
                                    recording.write("agent", message.data)
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
                                and barge_in.drop_interrupted_audio
                                and not barge_in.interrupted_user_committed
                            ):
                                # Kesilmiş cevabın geç ulaşan metnini istemciye
                                # gönderme ve yeni turun parçası gibi kaydetme.
                                continue
                            if (
                                role == "assistant"
                                and barge_in.drop_interrupted_audio
                                and barge_in.interrupted_user_committed
                            ):
                                # Deepgram bazen AgentStartedSpeaking olayını
                                # ConversationText'ten önce yollar. Yeni kullanıcı
                                # turuna ait assistant metni kesinleştiğinde ses
                                # kapısını mutlaka yeniden aç.
                                barge_in.drop_interrupted_audio = False
                                barge_in.interrupted_user_committed = False
                                barge_in.client_interrupt_started = None
                                await ws.send_json({"type": "speaking"})
                            await ws.send_json({
                                "type": "transcript",
                                "role": role,
                                "content": content,
                            })
                            if role == "user":
                                pending_user_text = content
                                user_transcript_ready_at = time.perf_counter()
                                assistant_text_ready_at = None
                                turn_metric_logged = False
                                first_audio_at = None
                                audio_done_at = None
                                if barge_in.drop_interrupted_audio:
                                    barge_in.interrupted_user_committed = True
                            elif role == "assistant" and pending_user_text:
                                if assistant_text_ready_at is None:
                                    assistant_text_ready_at = time.perf_counter()
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
                            latency_event = {}
                            turn_metric_logged = False
                            first_audio_at = None
                            audio_done_at = None
                            user_transcript_ready_at = None
                            assistant_text_ready_at = None
                            barge_in.drop_interrupted_audio = True
                            barge_in.interrupted_user_committed = bool(pending_user_text)
                            await ws.send_json({"type": "interrupt"})
                        elif event_type == "AgentThinking":
                            await ws.send_json({"type": "processing"})
                        elif event_type == "AgentStartedSpeaking":
                            for field in (
                                "ttt_latency",
                                "tts_latency",
                                "total_latency",
                            ):
                                if isinstance(event.get(field), (int, float)):
                                    latency_event[field] = event[field]
                            if (
                                barge_in.drop_interrupted_audio
                                and not barge_in.interrupted_user_committed
                            ):
                                continue
                            if (
                                assistant_text_ready_at is None
                                and user_transcript_ready_at is not None
                            ):
                                assistant_text_ready_at = time.perf_counter()
                            barge_in.client_interrupt_started = None
                            barge_in.drop_interrupted_audio = False
                            barge_in.interrupted_user_committed = False
                            await ws.send_json({"type": "speaking"})
                        elif event_type == "AgentAudioDone":
                            await ws.send_json({"type": "audio_done"})
                        elif event_type == "LatencyReport":
                            latency_event.update({
                                key: value
                                for key, value in event.items()
                                if isinstance(value, (int, float))
                            })
                            schedule_latency_if_complete()
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
    finally:
        if recording:
            await recording.finalize()

    return ws
