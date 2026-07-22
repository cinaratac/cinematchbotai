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
from functools import wraps

from flask import Blueprint, jsonify, request

import database as db

admin_bp = Blueprint("admin_bp", __name__, url_prefix="/api/admin")

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")


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

    metrics = db.get_performance_metrics_admin(limit=limit, offset=offset)
    total = db.count_performance_metrics_admin()
    averages = db.get_performance_metrics_averages()

    return jsonify({
        "status": "success",
        "data": metrics,
        "averages": averages,
        "pagination": {"limit": limit, "offset": offset, "total": total},
    }), 200
