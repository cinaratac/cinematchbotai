"""Grafana Cloud'a dogrudan OTLP/HTTP telemetri aktarimi.

Render Metrics Stream gerektirmez. OTEL ortam degiskenleri tanimli degilse
modul sessizce devre disi kalir ve mevcut Prometheus yapisi aynen devam eder.
"""

from __future__ import annotations

import logging
import os
import socket


_configured = False
_enabled = False
_instruments = {}


class _ExcludeExporterLogs(logging.Filter):
    """Exporter'in kendi loglarini tekrar export ederek dongu olusturmasini onler."""

    def filter(self, record):
        return not record.name.startswith(("opentelemetry", "urllib3"))


def configure_otel():
    """OTLP endpoint verilmisse metric ve log exporter'larini bir kez kurar."""
    global _configured, _enabled, _instruments
    if _configured:
        return _enabled
    _configured = True

    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip():
        return False

    try:
        from opentelemetry import metrics
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        logging.getLogger(__name__).exception(
            "OTLP endpoint tanimli ancak OpenTelemetry paketleri yuklu degil.",
            extra={"event": "otel_dependency_missing"},
        )
        return False

    try:
        service_name = os.environ.get("OTEL_SERVICE_NAME", "cinebot-api")
        environment = os.environ.get(
            "CINEBOT_ENVIRONMENT",
            os.environ.get("RENDER_SERVICE_NAME", "development"),
        )
        resource = Resource.create({
            "service.name": service_name,
            "service.version": os.environ.get("RENDER_GIT_COMMIT", "local")[:40],
            "service.instance.id": os.environ.get(
                "RENDER_INSTANCE_ID", socket.gethostname()
            ),
            "deployment.environment.name": environment,
        })

        interval_ms = int(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL", "15000"))
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(),
            export_interval_millis=max(5000, interval_ms),
        )
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )
        metrics.set_meter_provider(meter_provider)
        meter = metrics.get_meter("cinebot.observability")

        _instruments = {
            "http_requests": meter.create_counter(
                "cinebot.http.requests",
                unit="{request}",
                description="CineBot HTTP isteklerinin toplam sayisi.",
            ),
            "http_duration": meter.create_histogram(
                "cinebot.http.request.duration",
                unit="s",
                description="CineBot HTTP istek suresi.",
            ),
            "performance_records": meter.create_counter(
                "cinebot.performance.records",
                unit="{record}",
                description="CineBot pipeline sonuclarinin toplam sayisi.",
            ),
            "pipeline_duration": meter.create_histogram(
                "cinebot.pipeline.stage.duration",
                unit="s",
                description="CineBot pipeline asama suresi.",
            ),
            "tool_calls": meter.create_counter(
                "cinebot.tool.calls",
                unit="{call}",
                description="CineBot harici arac cagrilarinin toplam sayisi.",
            ),
            "tool_duration": meter.create_histogram(
                "cinebot.tool.call.duration",
                unit="s",
                description="CineBot harici arac cagri suresi.",
            ),
            "active_voice_connections": meter.create_gauge(
                "cinebot.active.voice.connections",
                unit="{connection}",
                description="Anlik aktif CineBot voice websocket baglantisi.",
            ),
        }

        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter())
        )
        otel_handler = LoggingHandler(
            level=logging.NOTSET,
            logger_provider=logger_provider,
        )
        otel_handler.addFilter(_ExcludeExporterLogs())
        logging.getLogger().addHandler(otel_handler)

        _enabled = True
        logging.getLogger(__name__).info(
            "Grafana Cloud OTLP metric ve log aktarimi etkin.",
            extra={"event": "otel_export_enabled", "status": "enabled"},
        )
        return True
    except Exception:
        logging.getLogger(__name__).exception(
            "OpenTelemetry exporter baslatilamadi; Prometheus devam ediyor.",
            extra={"event": "otel_export_init_error"},
        )
        _instruments = {}
        return False


def _emit(name, method, value, attributes):
    instrument = _instruments.get(name)
    if instrument is None:
        return
    try:
        getattr(instrument, method)(value, attributes=attributes)
    except Exception:
        # Telemetri arizasi kullanici istegini veya bot pipeline'ini bozmamali.
        logging.getLogger(__name__).debug(
            "OTLP metric kaydi basarisiz.",
            exc_info=True,
            extra={"event": "otel_metric_record_error"},
        )


def record_http_request(method, route, status_code, duration_seconds):
    attributes = {
        "http.request.method": method,
        "http.route": route,
        "http.response.status_code": int(status_code),
        # Mevcut Grafana sorgulariyla uyumluluk icin kisa adlar da korunur.
        "method": method,
        "route": route,
        "status_code": str(status_code),
    }
    _emit("http_requests", "add", 1, attributes)
    _emit("http_duration", "record", duration_seconds, attributes)


def record_performance_metric(channel, status, failed_stage, outcome, durations):
    attributes = {
        "channel": channel,
        "status": status,
        "failed_stage": failed_stage,
        "outcome": outcome,
    }
    _emit("performance_records", "add", 1, attributes)
    for stage, duration_seconds in durations.items():
        stage_attributes = {
            "channel": channel,
            "stage": stage,
            "status": status,
        }
        _emit("pipeline_duration", "record", duration_seconds, stage_attributes)


def record_tool_call(tool, status, duration_seconds=None):
    attributes = {"tool": tool, "status": status}
    _emit("tool_calls", "add", 1, attributes)
    if duration_seconds is not None:
        _emit("tool_duration", "record", duration_seconds, attributes)


def record_active_voice_connections(count):
    _emit("active_voice_connections", "set", count, {})
