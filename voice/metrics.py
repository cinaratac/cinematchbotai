def _milliseconds(latency_event, field):
    value = latency_event.get(field)
    if isinstance(value, (int, float)):
        return round(value * 1000)
    return None


def build_voice_metric(
    *,
    latency_event,
    first_audio_at,
    playback_done_at,
    user_id,
    username,
    session_id,
    deepgram_request_id,
    turn_number,
):
    """Deepgram raporu ve client playback zamanından Firestore metriği üretir."""
    ai_ms = _milliseconds(latency_event, "ttt_text_latency")
    ttfb_ms = _milliseconds(latency_event, "ttt_token_latency")
    if ttfb_ms is None:
        ttfb_ms = ai_ms
    tts_ms = _milliseconds(latency_event, "tts_latency")
    ttfs_ms = _milliseconds(latency_event, "total_latency")

    audio_stream_ms = round(
        max(0, playback_done_at - first_audio_at) * 1000
    )
    full_turn_ms = (
        ttfs_ms + audio_stream_ms if ttfs_ms is not None else None
    )
    asr_ms = None
    if ttfs_ms is not None and ai_ms is not None and tts_ms is not None:
        # Agent API ayrı ASR/EOT alanı vermediğinden kalan süre tahminidir.
        asr_ms = max(0, ttfs_ms - ai_ms - tts_ms)

    return {
        "channel": "voice_websocket",
        "input_type": "streaming_audio",
        "metric_version": 4,
        "measurement_definition": "deepgram_e2e_first_audio",
        "user_id": user_id,
        "username": username,
        "session_id": session_id,
        "deepgram_request_id": deepgram_request_id,
        "turn_number": turn_number,
        "asr_ms": asr_ms,
        "asr_is_estimate": True,
        "ai_ms": ai_ms,
        "ai_ready_ms": (
            ttfs_ms - tts_ms
            if ttfs_ms is not None and tts_ms is not None
            else None
        ),
        "ttfb_ms": ttfb_ms,
        "tts_ms": tts_ms,
        "tts_ready_ms": ttfs_ms,
        "voice_audio_stream_ms": audio_stream_ms,
        "ttfs_ms": ttfs_ms,
        "e2e_ms": ttfs_ms,
        "full_turn_ms": full_turn_ms,
        "deepgram_ttt_token_ms": _milliseconds(
            latency_event, "ttt_token_latency"
        ),
        "deepgram_ttt_text_ms": _milliseconds(
            latency_event, "ttt_text_latency"
        ),
        "deepgram_ttt_tool_ms": _milliseconds(
            latency_event, "ttt_tool_latency"
        ),
        "deepgram_ttt_thinking_ms": _milliseconds(
            latency_event, "ttt_thinking_latency"
        ),
        "asr_model": "deepgram/nova-3",
        "ai_model": "google/gemini-3.1-flash-lite",
        "tts_model": "cartesia/sonic-3",
        "tool_call_count": 0,
        "status": "success",
        "failed_stage": None,
        "error_type": None,
    }
