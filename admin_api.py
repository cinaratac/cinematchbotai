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

import os
import secrets
from functools import wraps

from flask import Blueprint, jsonify, redirect, render_template, request

import database as db
from evaluation_service import EVALUATION_MODEL, schedule_voice_evaluation

admin_bp = Blueprint("admin_bp", __name__, url_prefix="/api/admin")

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
VOICE_RECORDINGS_API_KEY = os.environ.get("VOICE_RECORDINGS_API_KEY", "")


def require_admin_key(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
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

    if not db.get_session_admin_detail(session_id):
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

    metrics = db.get_performance_metrics_admin(
        limit=limit, offset=offset, session_id=session_id
    )
    total = db.count_performance_metrics_admin(session_id=session_id)
    averages = db.get_performance_metrics_averages(session_id=session_id)

    return jsonify({
        "status": "success",
        "data": metrics,
        "averages": averages,
        "session_id": session_id,
        "pagination": {"limit": limit, "offset": offset, "total": total},
    }), 200


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
    methods=["POST"],
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

    model_chain = f"openai/{EVALUATION_MODEL} -> openai/gpt-4o"
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
