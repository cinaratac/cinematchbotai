"""Prometheus metrikleri ile Flask/aiohttp istek gozlemlenebilirligi."""

from __future__ import annotations

import os
import re
import secrets
import time
import uuid

from aiohttp import web
from flask import Response, g, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from logging_config import set_request_id
from otel_config import (
    record_active_voice_connections as record_otel_active_voice_connections,
    record_http_request as record_otel_http_request,
    record_performance_metric as record_otel_performance_metric,
    record_tool_call as record_otel_tool_call,
)


HTTP_REQUESTS = Counter(
    "cinebot_http_requests_total",
    "CineBot HTTP isteklerinin toplam sayisi.",
    ("method", "route", "status_code"),
)
HTTP_DURATION = Histogram(
    "cinebot_http_request_duration_seconds",
    "CineBot HTTP istek suresi.",
    ("method", "route"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 60),
)
PERFORMANCE_RECORDS = Counter(
    "cinebot_performance_records_total",
    "Uygulama pipeline sonuclarinin toplam sayisi.",
    ("channel", "status", "failed_stage", "outcome"),
)
PIPELINE_DURATION = Histogram(
    "cinebot_pipeline_stage_duration_seconds",
    "CineBot pipeline asama ve kilometre tasi sureleri.",
    ("channel", "stage", "status"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 60, 120),
)
TOOL_CALLS = Counter(
    "cinebot_tool_calls_total",
    "CineBot harici arac cagrilarinin toplam sayisi.",
    ("tool", "status"),
)
TOOL_DURATION = Histogram(
    "cinebot_tool_call_duration_seconds",
    "CineBot harici arac cagri suresi.",
    ("tool", "status"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40),
)
ACTIVE_VOICE_CONNECTIONS = Gauge(
    "cinebot_active_voice_connections",
    "Anlik aktif CineBot voice websocket baglantisi.",
)
BUILD_INFO = Info("cinebot_build", "CineBot calisan surum bilgisi.")
BUILD_INFO.info({
    "service": os.environ.get("OTEL_SERVICE_NAME", "cinebot-api"),
    "environment": os.environ.get("CINEBOT_ENVIRONMENT", "development"),
    "commit": os.environ.get("RENDER_GIT_COMMIT", "local")[:40],
})


_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,80}$")
_DURATION_FIELDS = (
    "telegram_download_ms", "asr_ms", "ai_ms", "ai_ready_ms",
    "telegram_text_send_ms", "ttfb_ms", "tts_ms", "tts_ready_ms",
    "telegram_voice_upload_ms", "voice_audio_stream_ms", "tool_total_ms",
    "ttfs_ms", "e2e_ms", "full_turn_ms", "barge_in_latency_ms",
)


def _label(value, fallback="unknown", limit=80):
    text = str(value or fallback).strip().lower()
    return text[:limit] or fallback


def _request_id_from_header(value):
    value = str(value or "")
    return value if _REQUEST_ID_RE.fullmatch(value) else uuid.uuid4().hex


def observe_http_request(method, route, status_code, duration_seconds):
    method = _label(method, "unknown", 12).upper()
    route = str(route or "unmatched")[:160]
    status_code = str(status_code or 0)
    HTTP_REQUESTS.labels(method, route, status_code).inc()
    HTTP_DURATION.labels(method, route).observe(max(0.0, duration_seconds))
    record_otel_http_request(
        method,
        route,
        status_code,
        max(0.0, duration_seconds),
    )


def observe_performance_metric(metric):
    channel = _label(metric.get("channel"))
    status = _label(metric.get("status"))
    failed_stage = _label(metric.get("failed_stage"), "none")
    outcome = _label(metric.get("outcome"), "unknown")
    PERFORMANCE_RECORDS.labels(channel, status, failed_stage, outcome).inc()

    otel_durations = {}
    for field in _DURATION_FIELDS:
        value = metric.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            duration_seconds = max(0.0, value / 1000)
            PIPELINE_DURATION.labels(channel, field, status).observe(duration_seconds)
            otel_durations[field] = duration_seconds
    record_otel_performance_metric(
        channel,
        status,
        failed_stage,
        outcome,
        otel_durations,
    )


def observe_tool_call(tool_name, duration_ms, success):
    tool = _label(tool_name, "unknown_tool")
    status = "success" if success else "error"
    TOOL_CALLS.labels(tool, status).inc()
    duration_seconds = None
    if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
        duration_seconds = max(0.0, duration_ms / 1000)
        TOOL_DURATION.labels(tool, status).observe(duration_seconds)
    record_otel_tool_call(tool, status, duration_seconds)


def set_active_voice_connections(count):
    count = max(0, int(count))
    ACTIVE_VOICE_CONNECTIONS.set(count)
    record_otel_active_voice_connections(count)


def _metrics_authorized():
    expected = os.environ.get("METRICS_BEARER_TOKEN", "")
    if not expected:
        return True
    provided = request.headers.get("Authorization", "")
    prefix = "Bearer "
    return provided.startswith(prefix) and secrets.compare_digest(
        provided[len(prefix):], expected
    )


def init_flask_observability(app):
    """Flask uygulamasina /metrics ve dusuk kardinaliteli istek metrikleri ekle."""

    @app.before_request
    def _before_request():
        request_id = _request_id_from_header(request.headers.get("X-Request-ID"))
        set_request_id(request_id)
        g.cinebot_request_id = request_id
        g.cinebot_request_started = time.perf_counter()

    @app.after_request
    def _after_request(response):
        response.headers["X-Request-ID"] = getattr(
            g, "cinebot_request_id", uuid.uuid4().hex
        )
        if request.path != "/metrics":
            started = getattr(g, "cinebot_request_started", time.perf_counter())
            route = request.url_rule.rule if request.url_rule else "unmatched"
            observe_http_request(
                request.method,
                route,
                response.status_code,
                time.perf_counter() - started,
            )
        return response

    @app.teardown_request
    def _teardown_request(_error):
        set_request_id(None)

    @app.get("/metrics")
    def prometheus_metrics():
        if not _metrics_authorized():
            return Response("Unauthorized\n", status=401, mimetype="text/plain")
        return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)


@web.middleware
async def aiohttp_voice_observability(request, handler):
    """Yalnizca aiohttp tarafinda sonlanan voice route'larini olcer."""
    if not request.path.startswith("/api/voice/"):
        return await handler(request)

    request_id = _request_id_from_header(request.headers.get("X-Request-ID"))
    set_request_id(request_id)
    started = time.perf_counter()
    status_code = 500
    try:
        response = await handler(request)
        status_code = response.status
        if not response.prepared:
            response.headers["X-Request-ID"] = request_id
        return response
    except web.HTTPException as exc:
        status_code = exc.status
        raise
    finally:
        route = getattr(getattr(request, "match_info", None), "route", None)
        resource = getattr(route, "resource", None)
        route_name = getattr(resource, "canonical", None) or request.path
        observe_http_request(
            request.method,
            route_name,
            status_code,
            time.perf_counter() - started,
        )
        set_request_id(None)
