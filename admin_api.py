"""
Admin Paneli API (Blueprint)
============================
CineMatch admin paneli (cinematch/public/admin-bot.html), veritabanındaki
konuşmaları, tool çağrılarını ve değerlendirme skorlarını görüntülemek için
bu uçları kullanır.

GÜVENLİK: Bu uçlar kullanıcıların özel sohbet verisini döndürdüğü için
herkese açık DEĞİLDİR. Her istekte `X-Admin-Key` header'ı, ortam
değişkeni ADMIN_API_KEY ile eşleşmelidir. Anahtar tanımlı değilse
(geliştirme ortamı hariç) tüm uçlar 503 döner -- yanlışlıkla korumasız
yayına çıkmayı engellemek için.

Anahtarı tarayıcıya (frontend) doğrudan yazmak yerine, CineMatch tarafında
Firebase Auth ile giriş yapmış + yetkili admin'lere Cloud Functions
üzerinden (getBotAdminAccess) dağıtılması önerilir. Bkz. functions/index.js
içindeki assertBotAdmin / getBotAdminAccess ve public/js/admin-bot.js.
"""

import csv
import io
import json
import os
import secrets
from datetime import datetime, timezone
from functools import wraps
from urllib.parse import urlparse

from flask import Blueprint, Response, jsonify, redirect, render_template, request

import database as db
from evaluation_service import EVALUATION_MODEL, schedule_voice_evaluation

admin_bp = Blueprint("admin_bp", __name__, url_prefix="/api/admin")

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
VOICE_RECORDINGS_API_KEY = os.environ.get("VOICE_RECORDINGS_API_KEY", "")


def require_admin_key(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        # Tarayıcı, X-Admin-Key taşıyan cross-origin POST isteklerinden önce
        # anahtarsız OPTIONS preflight gönderir. Bu istek yetki vermez; CORS
        # katmanının gerekli Access-Control-* başlıkları ekleyebilmesi için
        # başarıyla dönmesi gerekir.
        if request.method == "OPTIONS":
            return "", 204
        if not ADMIN_API_KEY:
            return jsonify({
                "status": "error",
                "message": "ADMIN_API_KEY tanımlı değil. Sunucu ortam değişkenlerine "
                           "ADMIN_API_KEY eklenmeden admin uçları kullanılamaz."
            }), 503

        provided = request.headers.get("X-Admin-Key", "")
        if provided != ADMIN_API_KEY:
            return jsonify({"status": "error", "message": "Yetkisiz erişim."}), 401

        return view_func(*args, **kwargs)
    return wrapper


def require_voice_recordings_key(view_func):
    """Koordinatöre yalnızca ses kayıtlarına erişim yetkisi verir."""
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not VOICE_RECORDINGS_API_KEY:
            return jsonify({
                "status": "error",
                "message": "VOICE_RECORDINGS_API_KEY tanımlı değil.",
            }), 503
        provided = request.headers.get("X-Voice-Recordings-Key", "")
        if not secrets.compare_digest(provided, VOICE_RECORDINGS_API_KEY):
            return jsonify({"status": "error", "message": "Yetkisiz erişim."}), 401
        return view_func(*args, **kwargs)
    return wrapper


def _pagination_params(default_limit=50, max_limit=200):
    try:
        limit = int(request.args.get("limit", default_limit))
    except (TypeError, ValueError):
        limit = default_limit
    limit = max(1, min(limit, max_limit))

    try:
        offset = int(request.args.get("offset", 0))
    except (TypeError, ValueError):
        offset = 0
    offset = max(0, offset)

    return limit, offset


def _bounded_int_arg(name, default, minimum, maximum):
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@"}:
        # Excel formulu enjeksiyonunu engelle; degeri gorunur tut.
        return "'" + value
    return value


def _csv_response(filename, headers, rows):
    output = io.StringIO(newline="")
    output.write("\ufeff")  # Excel'in Turkce karakterleri UTF-8 acmasi icin BOM.
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow([_csv_cell(value) for value in row])
    response = Response(
        output.getvalue(),
        status=200,
        content_type="text/csv; charset=utf-8",
    )
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    response.headers["Cache-Control"] = "no-store"
    return response


@admin_bp.route("/overview", methods=["GET"])
@require_admin_key
def overview():
    try:
        days = int(request.args.get("days", 14))
    except (TypeError, ValueError):
        days = 14
    days = max(1, min(days, 90))

    data = db.get_admin_overview(days=days)
    return jsonify({"status": "success", "data": data}), 200


@admin_bp.route("/outcomes", methods=["GET"])
@require_admin_key
def outcome_analytics():
    try:
        days = int(request.args.get("days", 30))
    except (TypeError, ValueError):
        days = 30
    days = max(1, min(days, 365))

    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 200))

    data = db.get_outcome_analytics_admin(
        days=days,
        limit=limit,
        intent=request.args.get("intent") or None,
        outcome=request.args.get("outcome") or None,
        channel=request.args.get("channel") or None,
    )
    return jsonify({"status": "success", "data": data}), 200


@admin_bp.route("/voice-qa/dashboard", methods=["GET"])
@require_admin_key
def voice_qa_dashboard():
    try:
        days = int(request.args.get("days", 14))
    except (TypeError, ValueError):
        days = 14
    days = max(1, min(days, 90))
    session_id = request.args.get("session_id") or None
    dashboard = db.get_voice_qa_trend(
        days=days,
        session_id=session_id,
    )
    return render_template(
        "voice_qa_dashboard.html",
        dashboard=dashboard,
    )


@admin_bp.route("/sessions", methods=["GET"])
@require_admin_key
def list_sessions():
    limit, offset = _pagination_params()
    search = request.args.get("search") or None

    sessions = db.get_sessions_admin(limit=limit, offset=offset, search=search)
    total = db.count_sessions_admin(search=search)

    return jsonify({
        "status": "success",
        "data": sessions,
        "pagination": {"limit": limit, "offset": offset, "total": total},
    }), 200


@admin_bp.route("/sessions/<session_id>", methods=["GET"])
@require_admin_key
def session_detail(session_id):
    detail = db.get_session_admin_detail(session_id)
    if not detail:
        return jsonify({"status": "error", "message": "Oturum bulunamadı."}), 404
    return jsonify({"status": "success", "data": detail}), 200


@admin_bp.route("/sessions/<session_id>/evaluate", methods=["POST"])
@require_admin_key
def evaluate_session(session_id):
    payload = request.get_json(silent=True) or {}

    try:
        rating = int(payload.get("rating"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "rating 1-5 arası bir tam sayı olmalı."}), 400

    if rating < 1 or rating > 5:
        return jsonify({"status": "error", "message": "rating 1-5 arası olmalı."}), 400

    note = str(payload.get("note", ""))[:1000]
    evaluator = str(payload.get("evaluator", ""))[:120]

    if not db.session_exists(session_id):
        return jsonify({"status": "error", "message": "Oturum bulunamadı."}), 404

    new_id = db.add_evaluation(session_id, rating, note=note, evaluator=evaluator)
    return jsonify({"status": "success", "data": {"id": new_id}}), 201


@admin_bp.route("/tool-calls", methods=["GET"])
@require_admin_key
def tool_calls():
    limit, offset = _pagination_params()
    calls = db.get_tool_calls_admin(limit=limit, offset=offset)
    total = db.count_tool_calls_admin()
    return jsonify({
        "status": "success",
        "data": calls,
        "pagination": {"limit": limit, "offset": offset, "total": total},
    }), 200
@admin_bp.route("/performance", methods=["GET"])
@require_admin_key
def performance_metrics():
    limit, offset = _pagination_params()
    session_id = request.args.get("session_id") or None

    bundle = db.get_performance_metrics_bundle(
        limit=limit,
        offset=offset,
        session_id=session_id,
    )

    return jsonify({
        "status": "success",
        "data": bundle["data"],
        "averages": bundle["averages"],
        "session_id": session_id,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": bundle["total"],
        },
    }), 200


@admin_bp.route("/monitoring", methods=["GET"])
@require_admin_key
def monitoring_config():
    """Admin paneline gizli bilgi icermeyen Grafana baglanti durumunu ver."""
    dashboard_url = os.environ.get("GRAFANA_DASHBOARD_URL", "").strip()
    parsed = urlparse(dashboard_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        dashboard_url = None
    return jsonify({
        "status": "success",
        "data": {
            "grafana_configured": bool(dashboard_url),
            "grafana_dashboard_url": dashboard_url,
            "prometheus_endpoint": "/metrics",
            "metrics_auth_enabled": bool(
                os.environ.get("METRICS_BEARER_TOKEN", "")
            ),
            "structured_logging": True,
            "service": os.environ.get("OTEL_SERVICE_NAME", "cinebot-api"),
            "environment": os.environ.get(
                "CINEBOT_ENVIRONMENT",
                os.environ.get("RENDER_SERVICE_NAME", "development"),
            ),
        },
    }), 200


@admin_bp.route("/reports/overview.csv", methods=["GET"])
@require_admin_key
def overview_csv_report():
    days = _bounded_int_arg("days", 30, 1, 90)
    data = db.get_admin_report_summary(days=days)
    headers = (
        "donem_baslangici_utc", "donem_bitisi_utc", "donem_gun",
        "toplam_oturum", "aktif_oturum", "toplam_mesaj",
        "benzersiz_kullanici", "tool_cagrisi", "basarili_tool_cagrisi",
        "basarisiz_tool_cagrisi", "tool_basari_orani_yuzde",
        "ortalama_puan", "degerlendirme_sayisi", "siniflandirilan_mesaj",
        "siniflandirilmayan_mesaj", "basari_orani_yuzde",
        "fallback_orani_yuzde", "teknik_hata_orani_yuzde",
        "en_yogun_niyet", "en_yogun_niyet_adedi", "en_sik_sonuc",
        "en_sik_sonuc_adedi", "en_cok_sorgulanan_film",
        "film_sorgu_adedi", "veri_sinirlandirildi",
    )
    values = (
        data.get("period_start"), data.get("period_end"), data.get("days"),
        data.get("total_sessions"), data.get("active_sessions"),
        data.get("total_messages"), data.get("total_users"),
        data.get("total_tool_calls"), data.get("successful_tool_calls"),
        data.get("failed_tool_calls"), data.get("tool_success_rate"),
        data.get("avg_rating"), data.get("total_evaluations"),
        data.get("classified_messages"), data.get("unclassified_messages"),
        data.get("success_rate"), data.get("fallback_rate"),
        data.get("technical_error_rate"), data.get("top_intent"),
        data.get("top_intent_count"), data.get("top_outcome"),
        data.get("top_outcome_count"), data.get("top_movie"),
        data.get("top_movie_count"), data.get("data_truncated"),
    )

    return _csv_response(
        f"cinebot-genel-rapor-{days}-gun.csv",
        headers,
        [values],
    )


@admin_bp.route("/reports/performance.csv", methods=["GET"])
@require_admin_key
def performance_csv_report():
    days = _bounded_int_arg("days", 30, 1, 365)
    limit = _bounded_int_arg("limit", 5000, 1, 5000)
    session_id = request.args.get("session_id") or None
    rows = db.get_performance_metrics_export_admin(
        days=days,
        limit=limit,
        session_id=session_id,
    )
    headers = (
        "id", "created_at", "session_id", "recording_id", "channel",
        "input_type", "status", "failed_stage", "error_type", "outcome",
        "measurement_valid", "measurement_errors", *db.PERFORMANCE_METRIC_FIELDS,
    )
    return _csv_response(
        f"cinebot-performans-{days}-gun.csv",
        headers,
        ([row.get(header) for header in headers] for row in rows),
    )


@admin_bp.route("/reports/outcomes.csv", methods=["GET"])
@require_admin_key
def outcomes_csv_report():
    days = _bounded_int_arg("days", 30, 1, 365)
    limit = _bounded_int_arg("limit", 5000, 1, 5000)
    rows = db.get_outcome_rows_export_admin(days=days, limit=limit)
    headers = (
        "id", "created_at", "session_id", "recording_id", "user_id",
        "username", "channel", "input_type", "user_message", "bot_response",
        "recommended_movies", "classification_status", "intent", "outcome",
        "intent_confidence", "outcome_confidence", "classification_version",
        "classification_reason", "error_stage", "error_type",
    )
    return _csv_response(
        f"cinebot-sonuclar-{days}-gun.csv",
        headers,
        ([row.get(header) for header in headers] for row in rows),
    )


@admin_bp.route("/voice-recordings", methods=["GET"])
@require_voice_recordings_key
def list_voice_recordings():
    limit, offset = _pagination_params(default_limit=50, max_limit=200)
    session_id = request.args.get("session_id") or None
    recordings = db.list_voice_recordings_api(
        limit=limit,
        offset=offset,
        session_id=session_id,
    )
    base_url = request.host_url.rstrip("/")
    for recording in recordings:
        recording_id = recording["id"]
        recording["user_download_url"] = (
            f"{base_url}/api/admin/voice-recordings/"
            f"{recording_id}/user"
        )
        recording["agent_download_url"] = (
            f"{base_url}/api/admin/voice-recordings/"
            f"{recording_id}/agent"
        )

    total = db.count_voice_recordings_api(session_id=session_id)
    return jsonify({
        "status": "success",
        "data": recordings,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total,
        },
    }), 200


@admin_bp.route(
    "/voice-recordings/<recording_id>/<track>",
    methods=["GET"],
)
@require_voice_recordings_key
def download_voice_recording(recording_id, track):
    if track not in {"user", "agent"}:
        return jsonify({
            "status": "error",
            "message": "track yalnızca user veya agent olabilir.",
        }), 400
    download_url = db.get_voice_recording_download_url(recording_id, track)
    if not download_url:
        return jsonify({
            "status": "error",
            "message": "Hazır ses kaydı bulunamadı.",
        }), 404
    return redirect(download_url, code=302)


@admin_bp.route(
    "/voice-recordings/<recording_id>/qa/re-evaluate",
    methods=["POST", "OPTIONS"],
)
@require_admin_key
def re_evaluate_voice_recording(recording_id):
    """Hazır bir WAV kaydı için QA hattını admin isteğiyle yeniden çalıştır."""
    recording = db.get_voice_recording_for_qa(recording_id)
    if not recording:
        return jsonify({
            "status": "error",
            "message": "Ses kaydı bulunamadı.",
        }), 404
    if recording.get("status") != "ready":
        return jsonify({
            "status": "error",
            "message": "Yalnızca hazır durumdaki ses kayıtları yeniden değerlendirilebilir.",
        }), 409
    if not recording.get("session_id") or not recording.get("user_storage_path"):
        return jsonify({
            "status": "error",
            "message": "Kayıt QA için gerekli session veya kullanıcı sesi bilgisini içermiyor.",
        }), 409

    current_status = db.get_voice_ai_evaluation_status(recording_id)
    if current_status in {"queued", "processing"}:
        return jsonify({
            "status": "error",
            "message": "Bu kayıt için zaten çalışan bir değerlendirme var.",
        }), 409

    model_chain = f"openrouter/{EVALUATION_MODEL}"
    db.queue_voice_ai_evaluation(
        recording["session_id"], recording_id, model_chain
    )
    try:
        schedule_voice_evaluation(recording["session_id"], recording_id)
    except Exception as exc:
        db.fail_voice_ai_evaluation(recording_id, exc)
        return jsonify({
            "status": "error",
            "message": "Değerlendirme kuyruğa alınamadı.",
        }), 503

    return jsonify({
        "status": "accepted",
        "data": {
            "recording_id": recording_id,
            "session_id": recording["session_id"],
            "evaluation_status": "queued",
        },
    }), 202
