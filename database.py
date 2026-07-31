import os
import json
import logging
import time
import threading
import random
from datetime import datetime, timedelta, timezone

import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.api_core.exceptions import FailedPrecondition
from google.cloud.firestore_v1 import FieldFilter
from outcome_service import (
    OUTCOME_TECHNICAL_ERROR,
    VALID_OUTCOMES,
    categorize_interaction,
)
from observability import observe_performance_metric, observe_tool_call


logger = logging.getLogger(__name__)

# Oturum süresi
SESSION_TIMEOUT_MINUTES = 30
# kaç mesajda bir oturum özeti güncellenecek
SUMMARY_UPDATE_INTERVAL = 4
# Prompt'a gidecek son kaç mesajın ham metni alınacak
RECENT_TURNS_IN_PROMPT = 4

# --- Firebase Admin SDK kurulumu ---
# FIREBASE_SERVICE_ACCOUNT_JSON: Firebase Console > Project Settings >
# Service Accounts > Generate new private key ile indirilen JSON dosyasının
# TÜM içeriği, Render'ın Environment sekmesine tek bir env var olarak girilir.
_firebase_app = None


def _storage_bucket_name():
    return (
        os.environ.get("FIREBASE_STORAGE_BUCKET", "")
        .strip()
        .removeprefix("gs://")
        .rstrip("/")
    )


def _get_db():
    global _firebase_app
    if _firebase_app is None:
        raw_creds = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
        if not raw_creds:
            raise RuntimeError(
                "FIREBASE_SERVICE_ACCOUNT_JSON tanımlı değil. Render Environment "
                "sekmesine Firebase servis hesabı JSON'ını eklemen gerekiyor."
            )
        cred_dict = json.loads(raw_creds)
        cred = credentials.Certificate(cred_dict)
        options = {}
        storage_bucket = _storage_bucket_name()
        if storage_bucket:
            options["storageBucket"] = storage_bucket
        _firebase_app = firebase_admin.initialize_app(cred, options)
    return firestore.client()


# Koleksiyon isimleri (mevcut trivia/haber koleksiyonlarıyla çakışmasın diye
# hepsi "bot_" öneki taşıyor)
COL_SESSIONS = "bot_sessions"
COL_CHAT_LOGS = "bot_chat_logs"
COL_USER_PROFILE = "bot_user_profiles"
COL_API_LOGS = "bot_api_logs"
COL_EVALUATIONS = "bot_evaluations"
COL_PERFORMANCE_METRICS = "bot_performance_metrics"
COL_VOICE_RECORDINGS = "bot_voice_recordings"
COL_VOICE_AI_EVALUATIONS = "bot_voice_ai_evaluations"


_MISSING = object()
_history_cache_lock = threading.Lock()
_past_summary_cache = {}
_HISTORY_CACHE_TTL_SECONDS = max(
    0,
    int(os.environ.get("FIRESTORE_HISTORY_CACHE_TTL_SECONDS", "120")),
)
_HISTORY_CACHE_MAX_USERS = max(
    1,
    int(os.environ.get("FIRESTORE_HISTORY_CACHE_MAX_USERS", "1000")),
)


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    """Firestore'un datetime nesnelerini (DatetimeWithNanoseconds dahil)
    frontend'in güvenle parse edebileceği ISO 8601 string'e çevirir."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def setup_database():
    """Firestore şema-sız olduğu için tablo oluşturmaya gerek yok; sadece
    bağlantının kurulabildiğini doğruluyoruz."""
    try:
        _get_db()
        print("SİSTEM: Firestore bağlantısı hazır (bot_* koleksiyonları).")
        try:
            cleanup_stale_voice_ai_evaluations()
        except Exception as cleanup_error:
            print("Stale Voice QA temizleme uyarısı:", repr(cleanup_error))
    except Exception as e:
        print(f"SİSTEM HATASI: Firestore bağlantısı kurulamadı: {e}")


# ============================================================
# OTURUM YÖNETİMİ
# ============================================================

def _invalidate_past_summary_cache(user_id):
    with _history_cache_lock:
        _past_summary_cache.pop(str(user_id), None)


def _session_state(doc, data):
    message_count = data.get("message_count", 0)
    if not isinstance(message_count, int) or isinstance(message_count, bool):
        message_count = 0

    summary = data.get("summary", "") or ""
    summary_message_count = data.get("summary_message_count")
    if (
        not isinstance(summary_message_count, int)
        or isinstance(summary_message_count, bool)
        or summary_message_count < 0
        or summary_message_count > message_count
    ):
        # Eski oturumlarda bu alan bulunmaz. Son özet denemesinin başarısız
        # olmuş olma ihtimaline karşı bir önceki aralık sınırından başlarız.
        # Birkaç turu bir kez daha özetlemek, yeni turları atlamaktan güvenlidir.
        summary_message_count = (
            max(
                0,
                (
                    (message_count // SUMMARY_UPDATE_INTERVAL) - 1
                ) * SUMMARY_UPDATE_INTERVAL,
            )
            if summary
            else 0
        )

    return {
        "session_id": doc.id,
        "message_count": message_count,
        "summary": summary,
        "summary_message_count": summary_message_count,
    }


def get_or_create_session_state(user_id, username):
    """Aktif oturumu ve zaten okunan özet/sayaç durumunu birlikte döndürür."""
    db = _get_db()
    user_id = str(user_id)

    query = (
        db.collection(COL_SESSIONS)
        .where(filter=FieldFilter("user_id", "==", user_id))
        .where(filter=FieldFilter("is_active", "==", True))
        .order_by("last_active_at", direction=firestore.Query.DESCENDING)
        .limit(1)
    )
    docs = list(query.stream())

    if docs:
        doc = docs[0]
        data = doc.to_dict()
        last_active = data["last_active_at"]
        if isinstance(last_active, datetime):
            if _now() - last_active <= timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                return _session_state(doc, data)
        # Süresi dolmuş -> pasif işaretle
        doc.reference.update({"is_active": False})

    new_doc = db.collection(COL_SESSIONS).document()
    new_doc.set({
        "user_id": user_id,
        "username": username,
        "started_at": _now(),
        "last_active_at": _now(),
        "message_count": 0,
        "summary": "",
        "summary_message_count": 0,
        "is_active": True,
        "rating_sum": 0,
        "rating_count": 0,
        "intent_counts": {},
        "outcome_counts": {},
        "last_intent": None,
        "last_outcome": None,
    })
    _invalidate_past_summary_cache(user_id)
    logger.info(
        "Yeni kullanici oturumu acildi.",
        extra={"event": "session_created", "session_id": new_doc.id},
    )
    return {
        "session_id": new_doc.id,
        "message_count": 0,
        "summary": "",
        "summary_message_count": 0,
    }


def get_or_create_session(user_id, username):
    """Geriye uyumlu oturum-ID arayüzü."""
    return get_or_create_session_state(user_id, username)["session_id"]


def touch_session(
    session_id,
    classification=None,
    *,
    current_message_count=None,
    summary_text=None,
    summary_message_count=None,
):
    db = _get_db()
    ref = db.collection(COL_SESSIONS).document(session_id)
    updates = {
        "last_active_at": _now(),
        "message_count": firestore.Increment(1),
    }
    if classification:
        intent = classification.get("intent")
        outcome = classification.get("outcome")
        if intent:
            updates[f"intent_counts.{intent}"] = firestore.Increment(1)
            updates["last_intent"] = intent
        if outcome:
            updates[f"outcome_counts.{outcome}"] = firestore.Increment(1)
            updates["last_outcome"] = outcome
    if summary_text:
        updates["summary"] = summary_text
        if (
            isinstance(summary_message_count, int)
            and summary_message_count >= 0
        ):
            updates["summary_message_count"] = summary_message_count
    ref.update(updates)
    if isinstance(current_message_count, int) and not isinstance(
        current_message_count, bool
    ):
        return current_message_count + 1
    return None


def update_session_summary(session_id, summary_text, message_count=None):
    db = _get_db()
    updates = {"summary": summary_text}
    if isinstance(message_count, int) and message_count >= 0:
        updates["summary_message_count"] = message_count
    db.collection(COL_SESSIONS).document(session_id).update(updates)
    logger.info(
        "Oturum ozeti guncellendi.",
        extra={"event": "session_summary_updated", "session_id": session_id},
    )


# ============================================================
# TAM KONUŞMA DÖKÜMÜ (TRANSCRIPT)
# ============================================================

def log_chat(
    session_id,
    user_id,
    username,
    user_message,
    bot_response,
    recording_id=None,
    *,
    channel="unknown",
    input_type="text",
    classification=None,
    recommended_movies=None,
    outcome_hint=None,
    error_stage=None,
    error_type=None,
    tool_calls=None,
):
    classification = classification or categorize_interaction(
        user_message,
        bot_response,
        input_type=input_type,
        recommended_movies=recommended_movies,
        outcome_hint=outcome_hint,
        error_stage=error_stage,
        error_type=error_type,
        tool_calls=tool_calls,
    )
    db = _get_db()
    payload = {
        "session_id": session_id,
        "user_id": str(user_id),
        "username": username,
        "user_message": user_message,
        "bot_response": bot_response,
        "channel": channel,
        "input_type": input_type,
        **classification,
        "created_at": _now(),
    }
    if recording_id:
        payload["recording_id"] = recording_id
    if recommended_movies:
        payload["recommended_movies"] = list(recommended_movies)[:3]
    if error_stage:
        payload["error_stage"] = str(error_stage)[:120]
    if error_type:
        payload["error_type"] = str(error_type)[:200]
    doc_ref = db.collection(COL_CHAT_LOGS).document()
    doc_ref.set(payload)
    print("LOG BAŞARILI: Mesaj Firestore'a kaydedildi.")
    return {"id": doc_ref.id, **classification}


def log_failed_interaction(
    user_id,
    username,
    user_message,
    bot_response,
    *,
    channel,
    input_type,
    error_stage,
    error_type,
):
    """Ana cevap hattına giremeden biten bir isteği teknik hata olarak kaydeder.

    Veritabanının kendisi kullanılamıyorsa bu fonksiyon doğal olarak kayıt
    oluşturamaz; çağıranlar kullanıcı yanıtını bozmamak için hatayı yakalamalıdır.
    """
    session_id = get_or_create_session(user_id, username)
    logged = log_chat(
        session_id,
        user_id,
        username,
        user_message,
        bot_response,
        channel=channel,
        input_type=input_type,
        outcome_hint=OUTCOME_TECHNICAL_ERROR,
        error_stage=error_stage,
        error_type=error_type,
    )
    touch_session(session_id, logged)
    return {"session_id": session_id, **logged}


def update_chat_outcome(
    chat_log_id,
    outcome,
    *,
    error_stage=None,
    error_type=None,
    only_if_outcomes=None,
):
    """Kanal teslimi/TTS sonradan hata verirse kaydedilmiş sonucu düzeltir."""
    if not chat_log_id or outcome not in VALID_OUTCOMES:
        return False

    db = _get_db()
    ref = db.collection(COL_CHAT_LOGS).document(chat_log_id)
    snap = ref.get()
    if not snap.exists:
        return False
    data = snap.to_dict() or {}
    previous_outcome = data.get("outcome")
    if only_if_outcomes and previous_outcome not in set(only_if_outcomes):
        return False

    updates = {
        "outcome": outcome,
        "outcome_confidence": 1.0,
        "classification_reason.outcome": "explicit:channel_delivery_update",
    }
    if error_stage:
        updates["error_stage"] = str(error_stage)[:120]
    if error_type:
        updates["error_type"] = str(error_type)[:200]
    ref.update(updates)

    session_id = data.get("session_id")
    if session_id and previous_outcome and previous_outcome != outcome:
        db.collection(COL_SESSIONS).document(session_id).update({
            f"outcome_counts.{previous_outcome}": firestore.Increment(-1),
            f"outcome_counts.{outcome}": firestore.Increment(1),
            "last_outcome": outcome,
        })
    return True


def save_voice_recording(
    session_id,
    user_id,
    username,
    recording_id,
    user_audio_path,
    agent_audio_path,
    user_duration_ms,
    agent_duration_ms,
):
    """Bir voice bağlantısının iki WAV kanalını private Storage'a yükler."""
    _get_db()
    bucket_name = _storage_bucket_name()
    if not bucket_name:
        raise RuntimeError(
            "FIREBASE_STORAGE_BUCKET tanımlı değil; ses kaydı yüklenemedi."
        )

    base_path = f"bot_voice_recordings/{session_id}/{recording_id}"
    user_storage_path = f"{base_path}/user.wav"
    agent_storage_path = f"{base_path}/agent.wav"
    doc_ref = _get_db().collection(COL_VOICE_RECORDINGS).document(recording_id)
    payload = {
        "session_id": session_id,
        "user_id": str(user_id),
        "username": username,
        "user_storage_path": user_storage_path,
        "agent_storage_path": agent_storage_path,
        "user_duration_ms": user_duration_ms,
        "agent_duration_ms": agent_duration_ms,
        "status": "uploading",
        "error": None,
        "created_at": _now(),
    }
    try:
        bucket = storage.bucket(bucket_name)
        bucket.blob(user_storage_path).upload_from_filename(
            user_audio_path,
            content_type="audio/wav",
        )
        bucket.blob(agent_storage_path).upload_from_filename(
            agent_audio_path,
            content_type="audio/wav",
        )
        payload.update({
            "status": "ready",
            "uploaded_at": _now(),
        })
        # Başarılı kayıt için uploading + ready şeklinde iki ayrı Firestore
        # yazması yerine nihai metadata tek seferde oluşturulur.
        doc_ref.set(payload)
    except Exception as exc:
        payload.update({
            "status": "failed",
            "error": str(exc)[:1000],
        })
        # Hata görünürlüğü korunur; başarısız deneme de tek metadata yazmasıdır.
        doc_ref.set(payload)
        raise
    return recording_id


def _stream_ordered_with_fallback(
    ordered_query,
    fallback_query,
    *,
    sort_field,
    reverse,
    limit=None,
):
    """Birleşik index hazır değilse görünürlüğü koruyan geçici geri dönüş."""
    try:
        return list(ordered_query.stream())
    except FailedPrecondition:
        docs = list(fallback_query.stream())
        docs.sort(
            key=lambda doc: (
                (doc.to_dict() or {}).get(sort_field).timestamp()
                if isinstance(
                    (doc.to_dict() or {}).get(sort_field),
                    datetime,
                )
                else 0
            ),
            reverse=reverse,
        )
        return docs[:limit] if limit is not None else docs


def get_voice_recordings_admin(session_id):
    """Oturumdaki kayıtları, admin paneli için kısa ömürlü oynatma URL'leriyle döndürür."""
    bucket_name = _storage_bucket_name()
    if not bucket_name:
        return []

    query = (
        _get_db().collection(COL_VOICE_RECORDINGS)
        .where(filter=FieldFilter("session_id", "==", session_id))
    )
    docs = _stream_ordered_with_fallback(
        query.order_by(
            "created_at",
            direction=firestore.Query.DESCENDING,
        ),
        query,
        sort_field="created_at",
        reverse=True,
    )
    docs.reverse()
    bucket = storage.bucket(bucket_name)
    result = []
    for doc in docs:
        data = doc.to_dict()
        row = {
            "id": doc.id,
            "created_at": _iso(data.get("created_at")),
            "user_duration_ms": data.get("user_duration_ms"),
            "agent_duration_ms": data.get("agent_duration_ms"),
            "status": data.get("status"),
            "error": data.get("error"),
        }
        for track in ("user", "agent"):
            path = data.get(f"{track}_storage_path")
            row[f"{track}_audio_url"] = None
            if path and data.get("status") == "ready":
                try:
                    row[f"{track}_audio_url"] = bucket.blob(path).generate_signed_url(
                        version="v4",
                        expiration=timedelta(minutes=15),
                        method="GET",
                    )
                except Exception as exc:
                    row["status"] = "url_error"
                    row["error"] = str(exc)[:1000]
        result.append(row)
    return result


def list_voice_recordings_api(limit=50, offset=0, session_id=None):
    """Dış entegrasyon API'si için kayıt metadata listesini döndürür."""
    collection = _get_db().collection(COL_VOICE_RECORDINGS)
    if session_id:
        query = collection.where(
            filter=FieldFilter("session_id", "==", session_id)
        )
        docs = _stream_ordered_with_fallback(
            query.order_by(
                "created_at",
                direction=firestore.Query.DESCENDING,
            ).limit(limit + offset),
            query,
            sort_field="created_at",
            reverse=True,
            limit=limit + offset,
        )
    else:
        docs = list(
            collection
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit + offset)
            .stream()
        )

    rows = []
    for doc in docs[offset:offset + limit]:
        data = doc.to_dict()
        rows.append({
            "id": doc.id,
            "session_id": data.get("session_id"),
            "user_id": data.get("user_id"),
            "username": data.get("username"),
            "status": data.get("status"),
            "user_duration_ms": data.get("user_duration_ms"),
            "agent_duration_ms": data.get("agent_duration_ms"),
            "created_at": _iso(data.get("created_at")),
            "uploaded_at": _iso(data.get("uploaded_at")),
        })
    return rows


def count_voice_recordings_api(session_id=None):
    collection = _get_db().collection(COL_VOICE_RECORDINGS)
    query = collection
    if session_id:
        query = query.where(
            filter=FieldFilter("session_id", "==", session_id)
        )
    return query.count().get()[0][0].value


def get_voice_recording_download_url(recording_id, track):
    """Private WAV için 5 dakika geçerli indirme bağlantısı üretir."""
    if track not in {"user", "agent"}:
        return None
    snap = (
        _get_db().collection(COL_VOICE_RECORDINGS)
        .document(recording_id)
        .get()
    )
    if not snap.exists:
        return None
    data = snap.to_dict()
    if data.get("status") != "ready":
        return None
    storage_path = data.get(f"{track}_storage_path")
    bucket_name = _storage_bucket_name()
    if not storage_path or not bucket_name:
        return None

    return storage.bucket(bucket_name).blob(storage_path).generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=5),
        method="GET",
        response_disposition=(
            f'attachment; filename="{recording_id}-{track}.wav"'
        ),
    )


def download_voice_recording_audio(
    recording_id,
    track="user",
    *,
    recording=None,
):
    """QA işleyicisi için private Firebase Storage kaydını byte olarak döndürür."""
    if track not in {"user", "agent"}:
        raise ValueError("track user veya agent olmalıdır.")
    if recording is None:
        recording = get_voice_recording_for_qa(recording_id)
    if not recording:
        raise RuntimeError("Voice kaydı Firestore'da bulunamadı.")
    storage_path = recording.get(f"{track}_storage_path")
    bucket_name = _storage_bucket_name()
    if not storage_path or not bucket_name:
        raise RuntimeError("Voice kaydının Storage yolu bulunamadı.")
    return storage.bucket(bucket_name).blob(storage_path).download_as_bytes()


def get_voice_recording_for_qa(recording_id):
    """Admin kaynaklı tekrar QA isteği için kaydın güvenli metadata'sını döndür."""
    doc = (
        _get_db()
        .collection(COL_VOICE_RECORDINGS)
        .document(recording_id)
        .get()
    )
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    return {
        "recording_id": doc.id,
        "session_id": data.get("session_id"),
        "status": data.get("status"),
        "user_storage_path": data.get("user_storage_path"),
        "agent_storage_path": data.get("agent_storage_path"),
    }


def get_voice_ai_evaluation_status(recording_id):
    doc = (
        _get_db()
        .collection(COL_VOICE_AI_EVALUATIONS)
        .document(recording_id)
        .get()
    )
    if not doc.exists:
        return None
    return (doc.to_dict() or {}).get("status")


def queue_voice_ai_evaluation(session_id, recording_id, model_chain):
    """Admin'ın istediği yeniden QA işini görünür biçimde kuyruğa alır."""
    _get_db().collection(COL_VOICE_AI_EVALUATIONS).document(recording_id).set({
        "session_id": session_id,
        "recording_id": recording_id,
        "status": "queued",
        "model_chain": model_chain,
        "updated_at": _now(),
    }, merge=True)


def start_voice_ai_evaluation(session_id, recording_id, model_chain):
    ref = _get_db().collection(COL_VOICE_AI_EVALUATIONS).document(recording_id)
    ref.set({
        "session_id": session_id,
        "recording_id": recording_id,
        "status": "processing",
        "model_chain": model_chain,
        "created_at": _now(),
        "updated_at": _now(),
    })


def complete_voice_ai_evaluation(recording_id, result):
    payload = dict(result)
    payload.update({
        "status": "completed",
        "error": firestore.DELETE_FIELD,
        "skip_reason": firestore.DELETE_FIELD,
        "updated_at": _now(),
    })
    _get_db().collection(COL_VOICE_AI_EVALUATIONS).document(
        recording_id
    ).update(payload)


def fail_voice_ai_evaluation(recording_id, error):
    _get_db().collection(COL_VOICE_AI_EVALUATIONS).document(
        recording_id
    ).set({
        "recording_id": recording_id,
        "status": "failed",
        "error": str(error)[:1000],
        # Aynı belge daha önce tamamlandıysa eski rapor ile yeni hata aynı
        # admin kartında görünmesin.
        "overall_score": firestore.DELETE_FIELD,
        "summary": firestore.DELETE_FIELD,
        "criteria": firestore.DELETE_FIELD,
        "issues": firestore.DELETE_FIELD,
        "strengths": firestore.DELETE_FIELD,
        "prompt_recommendations": firestore.DELETE_FIELD,
        "turns": firestore.DELETE_FIELD,
        "transcript_comparison": firestore.DELETE_FIELD,
        "audio_transcript": firestore.DELETE_FIELD,
        "logged_user_transcript": firestore.DELETE_FIELD,
        "updated_at": _now(),
    }, merge=True)


def skip_voice_ai_evaluation(recording_id, reason):
    _get_db().collection(COL_VOICE_AI_EVALUATIONS).document(
        recording_id
    ).set({
        "recording_id": recording_id,
        "status": "skipped",
        "error": None,
        "skip_reason": str(reason)[:1000],
        "updated_at": _now(),
    }, merge=True)


def cleanup_stale_voice_ai_evaluations(max_age_minutes=30):
    """Restart sonrası kuyruğa alınmış veya processing kalmış QA işlerini kapatır."""
    cutoff = _now() - timedelta(minutes=max_age_minutes)
    docs = []
    for status in ("queued", "processing"):
        base_query = (
            _get_db().collection(COL_VOICE_AI_EVALUATIONS)
            .where(filter=FieldFilter("status", "==", status))
        )
        stale_query = (
            base_query
            .where(filter=FieldFilter("updated_at", "<", cutoff))
            .order_by("updated_at", direction=firestore.Query.ASCENDING)
            .limit(500)
        )
        try:
            docs.extend(stale_query.stream())
        except FailedPrecondition:
            # Index deploy edilene kadar eski sorguyla işlevi koru.
            docs.extend(base_query.stream())
    cleaned = 0
    for doc in docs:
        data = doc.to_dict() or {}
        updated_at = data.get("updated_at") or data.get("created_at")
        if isinstance(updated_at, datetime) and updated_at < cutoff:
            doc.reference.set({
                "status": "failed",
                "error": "Sunucu yeniden başladı veya QA işi zaman aşımına uğradı.",
                "updated_at": _now(),
            }, merge=True)
            cleaned += 1
    if cleaned:
        print("Stale Voice QA kayıtları kapatıldı:", cleaned)
    return cleaned


def get_voice_ai_evaluations_admin(session_id):
    query = (
        _get_db().collection(COL_VOICE_AI_EVALUATIONS)
        .where(filter=FieldFilter("session_id", "==", session_id))
    )
    docs = _stream_ordered_with_fallback(
        query.order_by(
            "created_at",
            direction=firestore.Query.DESCENDING,
        ),
        query,
        sort_field="created_at",
        reverse=True,
    )
    rows = []
    for doc in docs:
        data = doc.to_dict()
        data["created_at"] = _iso(data.get("created_at"))
        data["updated_at"] = _iso(data.get("updated_at"))
        rows.append({"id": doc.id, **data})
    return rows


def get_voice_qa_trend(days=14, session_id=None, limit=500):
    """Dashboard için QA skor, kriter, eşleşme ve barge-in trendini üretir."""
    cutoff = _now() - timedelta(days=days)
    base_query = _get_db().collection(COL_VOICE_AI_EVALUATIONS).where(
        filter=FieldFilter("status", "==", "completed"),
    )
    if session_id:
        base_query = base_query.where(
            filter=FieldFilter("session_id", "==", session_id)
        )
    query = (
        base_query
        .where(filter=FieldFilter("created_at", ">=", cutoff))
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    try:
        docs = list(query.stream())
    except FailedPrecondition:
        docs = list(base_query.limit(limit).stream())
    rows = []
    criterion_sums = {}
    criterion_counts = {}
    for doc in docs:
        data = doc.to_dict() or {}
        created_at = data.get("created_at")
        if not isinstance(created_at, datetime) or created_at < cutoff:
            continue
        criteria = data.get("criteria") or {}
        for key, item in criteria.items():
            score = item.get("score") if isinstance(item, dict) else None
            if isinstance(score, (int, float)):
                criterion_sums[key] = criterion_sums.get(key, 0) + score
                criterion_counts[key] = criterion_counts.get(key, 0) + 1
        comparison = data.get("transcript_comparison") or {}
        rows.append({
            "id": doc.id,
            "recording_id": data.get("recording_id") or doc.id,
            "session_id": data.get("session_id"),
            "created_at": _iso(created_at),
            "overall_score": data.get("overall_score"),
            "match_score": comparison.get("match_score"),
            "summary": data.get("summary", ""),
            "issues": data.get("issues") or [],
            "prompt_recommendations": (
                data.get("prompt_recommendations") or []
            ),
        })
    rows.sort(key=lambda row: row["created_at"] or "")
    criteria_averages = {
        key: round(criterion_sums[key] / criterion_counts[key], 1)
        for key in criterion_sums
        if criterion_counts.get(key)
    }
    metrics = _get_performance_metric_docs(
        session_id=session_id,
        scan_limit=2000,
        since=cutoff,
    )
    barge_in_latencies = []
    for metric_doc in metrics:
        data = metric_doc.to_dict() or {}
        created_at = data.get("created_at")
        latency = data.get("barge_in_latency_ms")
        if (
            isinstance(created_at, datetime)
            and created_at >= cutoff
            and isinstance(latency, (int, float))
        ):
            barge_in_latencies.append(latency)
    recommendations = []
    for row in reversed(rows):
        for item in row["prompt_recommendations"]:
            if not isinstance(item, dict):
                continue
            recommendations.append({
                **item,
                "recording_id": row["recording_id"],
            })
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(
        key=lambda item: priority_order.get(item.get("priority"), 3)
    )
    return {
        "days": days,
        "session_id": session_id,
        "evaluations": list(reversed(rows)),
        "series": rows,
        "criteria_averages": criteria_averages,
        "barge_in_latencies": barge_in_latencies,
        "prompt_recommendations": recommendations[:30],
        "average_overall_score": (
            round(sum(
                row["overall_score"] for row in rows
                if isinstance(row["overall_score"], (int, float))
            ) / len([
                row for row in rows
                if isinstance(row["overall_score"], (int, float))
            ]), 1)
            if any(
                isinstance(row["overall_score"], (int, float))
                for row in rows
            )
            else None
        ),
    }


def get_recording_transcript(recording_id, limit=100):
    """Yalnızca tek bir voice WAV kaydı sırasında oluşan konuşma turları."""
    query = (
        _get_db().collection(COL_CHAT_LOGS)
        .where(filter=FieldFilter("recording_id", "==", recording_id))
    )
    docs = _stream_ordered_with_fallback(
        query.order_by(
            "created_at",
            direction=firestore.Query.DESCENDING,
        ).limit(limit),
        query,
        sort_field="created_at",
        reverse=True,
        limit=limit,
    )
    docs.reverse()
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]


def get_session_transcript(session_id, limit=50):
    db = _get_db()
    query = (
        db.collection(COL_CHAT_LOGS)
        .where(filter=FieldFilter("session_id", "==", session_id))
        .order_by("created_at", direction=firestore.Query.ASCENDING)
        .limit(limit)
    )
    return [d.to_dict() for d in query.stream()]


def get_session_transcript_recent(session_id, n=RECENT_TURNS_IN_PROMPT):
    db = _get_db()
    query = (
        db.collection(COL_CHAT_LOGS)
        .where(filter=FieldFilter("session_id", "==", session_id))
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(n)
    )
    rows = [d.to_dict() for d in query.stream()]
    rows.reverse()
    return rows


def get_session_summary(session_id):
    db = _get_db()
    snap = db.collection(COL_SESSIONS).document(session_id).get()
    data = snap.to_dict() or {}
    return data.get("summary", "") or ""


# ============================================================
# KULLANICI GEÇMİŞİ
# ============================================================

def _get_cached_past_summaries(user_id):
    if _HISTORY_CACHE_TTL_SECONDS <= 0:
        return None
    cache_key = str(user_id)
    now_ts = time.monotonic()
    with _history_cache_lock:
        cached = _past_summary_cache.get(cache_key)
        if not cached:
            return None
        if cached["expires_at"] <= now_ts:
            _past_summary_cache.pop(cache_key, None)
            return None
        return [dict(row) for row in cached["rows"]]


def _cache_past_summaries(user_id, rows):
    if _HISTORY_CACHE_TTL_SECONDS <= 0:
        return
    cache_key = str(user_id)
    with _history_cache_lock:
        if (
            cache_key not in _past_summary_cache
            and len(_past_summary_cache) >= _HISTORY_CACHE_MAX_USERS
        ):
            oldest_key = next(iter(_past_summary_cache))
            _past_summary_cache.pop(oldest_key, None)
        _past_summary_cache[cache_key] = {
            "expires_at": time.monotonic() + _HISTORY_CACHE_TTL_SECONDS,
            "rows": [dict(row) for row in rows],
        }


def get_user_history(
    user_id,
    current_session_id,
    max_past_sessions=3,
    *,
    current_session_summary=_MISSING,
):
    db = _get_db()
    past_sessions = _get_cached_past_summaries(user_id)
    if past_sessions is None:
        query = (
            db.collection(COL_SESSIONS)
            .where(filter=FieldFilter("user_id", "==", str(user_id)))
            .order_by("started_at", direction=firestore.Query.DESCENDING)
            .limit(max_past_sessions + 1)  # mevcut oturumu eleyebilmek için
        )
        past_sessions = []
        for doc in query.stream():
            if doc.id == current_session_id:
                continue
            data = doc.to_dict()
            if not data.get("summary"):
                continue
            past_sessions.append({
                "session_id": doc.id,
                "started_at": data.get("started_at"),
                "summary": data.get("summary"),
            })
            if len(past_sessions) >= max_past_sessions:
                break
        _cache_past_summaries(user_id, past_sessions)
    else:
        past_sessions = past_sessions[:max_past_sessions]

    if current_session_summary is _MISSING:
        current_session_summary = get_session_summary(current_session_id)
    current_transcript = get_session_transcript_recent(current_session_id, RECENT_TURNS_IN_PROMPT)

    return {
        "past_summaries": past_sessions,
        "current_session_summary": current_session_summary,
        "current_transcript": current_transcript,
    }


# ============================================================
# TOOL (ARAÇ) ÇAĞRI LOGLARI
# ============================================================

def log_tool_call(
    session_id,
    user_id,
    movie_name,
    api_endpoint,
    api_response,
    username=None,
    tool_name="get_live_movie_data",
    query=None,
    duration_ms=None,
):
    response_text = str(api_response or "")
    success = (
        '"Response": "True"' in response_text
        or '"Response":"True"' in response_text
    )
    observe_tool_call(tool_name, duration_ms, success)
    db = _get_db()
    db.collection(COL_API_LOGS).document().set({
        "movie_name": movie_name,
        "api_endpoint": api_endpoint,
        "api_response": api_response,
        "session_id": session_id,
        "user_id": str(user_id) if user_id is not None else None,
        # YENİ: admin panelindeki "Tool Çağrıları" tablosu artık kullanıcı adını
        # göstermek için oturum başına ekstra bir sorgu atmıyor, doğrudan burada
        # denormalize edilmiş halde saklanıyor.
        "username": username,
        "tool_name": tool_name,
        "query": query if query is not None else movie_name,
        "duration_ms": duration_ms,
        "timestamp": _now(),
    })


def log_performance_metric(metric):
    """Bir pipeline ölçümünü ayrı Firestore koleksiyonuna kaydeder.

    Çağıran kodun hazırladığı sözlük kopyalanır; zaman damgası sunucu tarafında
    eklenir. Mesaj/transkript/ses içeriği bu koleksiyona yazılmamalıdır.
    """
    payload = dict(metric)
    # Firestore basarili kayitlari orneklese bile Prometheus tum olaylari gorur.
    observe_performance_metric(payload)
    try:
        configured_sample_rate = float(os.environ.get(
            "PERFORMANCE_METRIC_SUCCESS_SAMPLE_RATE",
            "0.25",
        ))
    except (TypeError, ValueError):
        configured_sample_rate = 0.25
    sample_rate = min(
        1.0,
        max(0.0, configured_sample_rate),
    )
    input_type = str(payload.get("input_type") or "")
    preserve_every_sample = (
        payload.get("channel") == "voice_websocket"
        or bool(payload.get("recording_id"))
        or input_type in {"voice", "audio", "streaming_audio"}
    )
    if (
        payload.get("status") == "success"
        and not preserve_every_sample
        and sample_rate < 1.0
        and random.random() >= sample_rate
    ):
        return None
    payload["success_sample_rate"] = (
        1.0 if preserve_every_sample else sample_rate
    )
    ai_ms = payload.get("ai_ms")
    ttfb_ms = payload.get("ttfb_ms")
    ttfs_ms = payload.get("ttfs_ms")
    e2e_ms = payload.get("e2e_ms")
    numeric = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)

    invariant_errors = []
    if numeric(ai_ms) and numeric(e2e_ms) and e2e_ms < ai_ms:
        invariant_errors.append("e2e_ms_lt_ai_ms")
    if numeric(ttfb_ms) and numeric(e2e_ms) and e2e_ms < ttfb_ms:
        invariant_errors.append("e2e_ms_lt_ttfb_ms")
    if numeric(ttfs_ms) and numeric(e2e_ms) and e2e_ms < ttfs_ms:
        invariant_errors.append("e2e_ms_lt_ttfs_ms")

    payload["measurement_valid"] = not invariant_errors
    payload["measurement_errors"] = invariant_errors
    payload["created_at"] = _now()
    return _get_db().collection(COL_PERFORMANCE_METRICS).document().set(payload)


# ============================================================
# KULLANICI PROFİLİ (uzun vadeli hafıza)
# ============================================================

def get_user_facts(user_id):
    db = _get_db()
    snap = db.collection(COL_USER_PROFILE).document(str(user_id)).get()
    if not snap.exists:
        return {}
    data = snap.to_dict() or {}
    return data.get("facts", {}) or {}


def update_user_facts(user_id, username, new_facts, *, existing_facts=None):
    if not new_facts:
        return dict(existing_facts or {})
    existing = (
        dict(existing_facts)
        if existing_facts is not None
        else get_user_facts(user_id)
    )
    existing.update(new_facts)

    db = _get_db()
    db.collection(COL_USER_PROFILE).document(str(user_id)).set({
        "username": username,
        "facts": existing,
        "updated_at": _now(),
    }, merge=True)
    logger.info(
        "Kullanici profili guncellendi.",
        extra={"event": "user_profile_updated"},
    )
    return existing


# ============================================================
# ADMIN PANELİ SORGULARI
# ============================================================

# YENİ: get_admin_overview çok pahalı (tüm sessions/evaluations/son 14 günün
# tüm chat_logs'unu tarıyor). Panel her açıldığında/yenilendiğinde tekrar
# tekrar çalışmasın diye kısa süreli bellek-içi cache kullanılıyor.
_overview_cache = {"data": None, "expires_at": 0, "days": None}
_OVERVIEW_CACHE_TTL_SECONDS = max(
    0,
    int(os.environ.get("FIRESTORE_OVERVIEW_CACHE_TTL_SECONDS", "60")),
)
_outcome_cache_lock = threading.Lock()
_outcome_cache = {}
_OUTCOME_CACHE_TTL_SECONDS = max(
    0,
    int(os.environ.get("FIRESTORE_OUTCOME_CACHE_TTL_SECONDS", "120")),
)
_SESSION_SEARCH_CACHE_TTL_SECONDS = max(
    0,
    int(os.environ.get("FIRESTORE_SESSION_SEARCH_CACHE_TTL_SECONDS", "60")),
)
_session_search_cache_lock = threading.Lock()
_session_search_cache = {"rows": None, "expires_at": 0}
_unique_user_cache_lock = threading.Lock()
_unique_user_cache = {"count": None, "expires_at": 0}
_UNIQUE_USER_CACHE_TTL_SECONDS = max(
    0,
    int(os.environ.get("FIRESTORE_UNIQUE_USER_CACHE_TTL_SECONDS", "300")),
)
_pagination_cache_lock = threading.Lock()
_pagination_cursor_cache = {}
_PAGINATION_CURSOR_TTL_SECONDS = max(
    0,
    int(os.environ.get("FIRESTORE_CURSOR_CACHE_TTL_SECONDS", "300")),
)
_performance_cache_lock = threading.Lock()
_performance_docs_cache = {}
_PERFORMANCE_CACHE_TTL_SECONDS = max(
    0,
    int(os.environ.get("FIRESTORE_PERFORMANCE_CACHE_TTL_SECONDS", "60")),
)


def _query_page_with_cursor(query, *, cache_key, limit, offset):
    """Sıralı admin sorgularında ardışık sayfaları start_after ile getirir."""
    now_ts = time.monotonic()
    cursor = None
    if offset > 0 and _PAGINATION_CURSOR_TTL_SECONDS > 0:
        with _pagination_cache_lock:
            cached = _pagination_cursor_cache.get(
                (cache_key, limit, offset)
            )
            if cached and cached["expires_at"] > now_ts:
                cursor = cached["snapshot"]

    if cursor is not None:
        docs = list(query.start_after(cursor).limit(limit).stream())
    else:
        fetched = list(query.limit(limit + offset).stream())
        docs = fetched[offset:offset + limit]
        # Doğrudan ileri bir sayfaya gelinmişse aradaki cursor'ları da sakla.
        if _PAGINATION_CURSOR_TTL_SECONDS > 0:
            with _pagination_cache_lock:
                for boundary in range(limit, len(fetched) + 1, limit):
                    _pagination_cursor_cache[(
                        cache_key,
                        limit,
                        boundary,
                    )] = {
                        "snapshot": fetched[boundary - 1],
                        "expires_at": (
                            now_ts + _PAGINATION_CURSOR_TTL_SECONDS
                        ),
                    }

    if docs and _PAGINATION_CURSOR_TTL_SECONDS > 0:
        next_offset = offset + len(docs)
        with _pagination_cache_lock:
            if len(_pagination_cursor_cache) >= 256:
                oldest_key = next(iter(_pagination_cursor_cache))
                _pagination_cursor_cache.pop(oldest_key, None)
            _pagination_cursor_cache[(cache_key, limit, next_offset)] = {
                "snapshot": docs[-1],
                "expires_at": now_ts + _PAGINATION_CURSOR_TTL_SECONDS,
            }
    return docs


def _count_unique_users(db):
    now_ts = time.monotonic()
    if _UNIQUE_USER_CACHE_TTL_SECONDS > 0:
        with _unique_user_cache_lock:
            if (
                _unique_user_cache["count"] is not None
                and _unique_user_cache["expires_at"] > now_ts
            ):
                return _unique_user_cache["count"]

    user_ids = set()
    for doc in db.collection(COL_SESSIONS).select(["user_id"]).stream():
        user_id = (doc.to_dict() or {}).get("user_id")
        if user_id is not None:
            user_ids.add(user_id)
    count = len(user_ids)
    if _UNIQUE_USER_CACHE_TTL_SECONDS > 0:
        with _unique_user_cache_lock:
            _unique_user_cache["count"] = count
            _unique_user_cache["expires_at"] = (
                now_ts + _UNIQUE_USER_CACHE_TTL_SECONDS
            )
    return count


def get_admin_overview(days=14):
    now_ts = time.time()
    if (
        _overview_cache["data"] is not None
        and _overview_cache["days"] == days
        and now_ts < _overview_cache["expires_at"]
    ):
        return _overview_cache["data"]

    db = _get_db()

    total_sessions = db.collection(COL_SESSIONS).count().get()[0][0].value
    active_sessions = (
        db.collection(COL_SESSIONS)
        .where(filter=FieldFilter("is_active", "==", True))
        .count().get()[0][0].value
    )
    total_messages = db.collection(COL_CHAT_LOGS).count().get()[0][0].value
    total_tool_calls = db.collection(COL_API_LOGS).count().get()[0][0].value

    # Firestore distinct-count sunmadığı için ilk hesaplamada session belgeleri
    # okunur; sonuç ayrıca cache'lenerek her overview yenilemesinde tarama önlenir.
    total_users = _count_unique_users(db)

    # Tool başarı oranı + en çok sorulan filmler: son 500 tool çağrısı üzerinden
    tool_docs = list(
        db.collection(COL_API_LOGS)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(500)
        .stream()
    )
    sampled_successful_tool_calls = 0
    movie_counts = {}
    movie_duration_sums = {}
    movie_duration_counts = {}
    for d in tool_docs:
        data = d.to_dict()
        resp = data.get("api_response", "") or ""
        if '"Response": "True"' in resp or '"Response":"True"' in resp:
            sampled_successful_tool_calls += 1
        name = data.get("movie_name")
        if name:
            movie_counts[name] = movie_counts.get(name, 0) + 1
            duration = data.get("duration_ms")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                movie_duration_sums[name] = movie_duration_sums.get(name, 0) + duration
                movie_duration_counts[name] = movie_duration_counts.get(name, 0) + 1
    top_movies = [
        {
            "movie_name": k,
            "c": v,
            "avg_duration_ms": (
                round(movie_duration_sums[k] / movie_duration_counts[k])
                if movie_duration_counts.get(k)
                else None
            ),
        }
        for k, v in sorted(movie_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    if tool_docs:
        sampled_success_rate = sampled_successful_tool_calls / len(tool_docs)
        successful_tool_calls = round(total_tool_calls * sampled_success_rate)
    else:
        successful_tool_calls = 0

    # Son N günün günlük mesaj hacmi
    since = _now() - timedelta(days=days)
    daily_counts = {}
    intent_counts = {}
    outcome_counts = {}
    classified_messages = 0
    for doc in (
        db.collection(COL_CHAT_LOGS)
        .where(filter=FieldFilter("created_at", ">=", since))
        .select(["created_at", "intent", "outcome"])
        .stream()
    ):
        row = doc.to_dict()
        created = row.get("created_at")
        if isinstance(created, datetime):
            day = created.strftime("%Y-%m-%d")
            daily_counts[day] = daily_counts.get(day, 0) + 1
        intent = row.get("intent")
        outcome = row.get("outcome")
        if intent and outcome:
            classified_messages += 1
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    daily_messages = [{"day": k, "c": v} for k, v in sorted(daily_counts.items())]

    # Tam değerlendirme belgelerini indirmek yerine beş küçük index
    # aggregation sorgusuyla aynı dağılımı hesapla.
    rating_dist = {}
    for rating in range(1, 6):
        count = (
            db.collection(COL_EVALUATIONS)
            .where(filter=FieldFilter("rating", "==", rating))
            .count()
            .get()[0][0]
            .value
        )
        if count:
            rating_dist[rating] = count
    total_evaluations = sum(rating_dist.values())
    rating_sum = sum(rating * count for rating, count in rating_dist.items())
    rating_distribution = [{"rating": k, "c": v} for k, v in sorted(rating_dist.items())]
    avg_rating = (
        round(rating_sum / total_evaluations, 2)
        if total_evaluations
        else None
    )

    result = {
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "total_messages": total_messages,
        "total_users": total_users,
        "total_tool_calls": total_tool_calls,
        "successful_tool_calls": successful_tool_calls,
        "failed_tool_calls": max(0, total_tool_calls - successful_tool_calls),
        "tool_call_sample_size": len(tool_docs),
        "sampled_successful_tool_calls": sampled_successful_tool_calls,
        "daily_messages": daily_messages,
        "rating_distribution": rating_distribution,
        "avg_rating": avg_rating,
        "total_evaluations": total_evaluations,
        "top_movies": top_movies,
        "classified_messages": classified_messages,
        "intent_distribution": [
            {"intent": code, "count": count}
            for code, count in sorted(
                intent_counts.items(), key=lambda item: item[1], reverse=True
            )
        ],
        "outcome_distribution": [
            {"outcome": code, "count": count}
            for code, count in sorted(
                outcome_counts.items(), key=lambda item: item[1], reverse=True
            )
        ],
        "success_rate": (
            round(
                outcome_counts.get("islem_basarili", 0)
                * 100
                / classified_messages,
                2,
            )
            if classified_messages
            else None
        ),
        "fallback_rate": (
            round(
                outcome_counts.get("anlasilamadi_fallback", 0)
                * 100
                / classified_messages,
                2,
            )
            if classified_messages
            else None
        ),
        "technical_error_rate": (
            round(
                outcome_counts.get("teknik_hata", 0)
                * 100
                / classified_messages,
                2,
            )
            if classified_messages
            else None
        ),
    }

    _overview_cache["data"] = result
    _overview_cache["days"] = days
    _overview_cache["expires_at"] = now_ts + _OVERVIEW_CACHE_TTL_SECONDS
    return result


def get_outcome_analytics_admin(
    days=30,
    limit=50,
    *,
    intent=None,
    outcome=None,
    channel=None,
    scan_limit=5000,
):
    """Etiket dağılımları, çapraz tablo ve incelenebilir son turları döndürür."""
    cache_key = (
        days,
        limit,
        intent,
        outcome,
        channel,
        scan_limit,
    )
    now_ts = time.monotonic()
    if _OUTCOME_CACHE_TTL_SECONDS > 0:
        with _outcome_cache_lock:
            cached = _outcome_cache.get(cache_key)
            if cached and cached["expires_at"] > now_ts:
                return cached["data"]

    since = _now() - timedelta(days=days)
    docs = list(
        _get_db().collection(COL_CHAT_LOGS)
        .where(filter=FieldFilter("created_at", ">=", since))
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(scan_limit)
        .stream()
    )

    rows = []
    unclassified_count = 0
    for doc in docs:
        data = doc.to_dict() or {}
        row_intent = data.get("intent")
        row_outcome = data.get("outcome")
        if not row_intent or not row_outcome:
            unclassified_count += 1
            continue
        if intent and row_intent != intent:
            continue
        if outcome and row_outcome != outcome:
            continue
        if channel and data.get("channel") != channel:
            continue
        rows.append((doc.id, data))

    intent_counts = {}
    outcome_counts = {}
    matrix = {}
    daily = {}
    for _doc_id, data in rows:
        row_intent = data["intent"]
        row_outcome = data["outcome"]
        intent_counts[row_intent] = intent_counts.get(row_intent, 0) + 1
        outcome_counts[row_outcome] = outcome_counts.get(row_outcome, 0) + 1
        matrix_key = (row_intent, row_outcome)
        matrix[matrix_key] = matrix.get(matrix_key, 0) + 1
        created_at = data.get("created_at")
        if isinstance(created_at, datetime):
            day = created_at.strftime("%Y-%m-%d")
            daily.setdefault(day, {})
            daily[day][row_outcome] = daily[day].get(row_outcome, 0) + 1

    total = len(rows)

    def distribution(counts, key):
        return [
            {
                key: code,
                "count": count,
                "percentage": round(count * 100 / total, 2) if total else 0,
            }
            for code, count in sorted(
                counts.items(), key=lambda item: item[1], reverse=True
            )
        ]

    recent_interactions = []
    for doc_id, data in rows[:limit]:
        recent_interactions.append({
            "id": doc_id,
            "session_id": data.get("session_id"),
            "user_id": data.get("user_id"),
            "username": data.get("username"),
            "channel": data.get("channel"),
            "input_type": data.get("input_type"),
            "intent": data.get("intent"),
            "outcome": data.get("outcome"),
            "intent_confidence": data.get("intent_confidence"),
            "outcome_confidence": data.get("outcome_confidence"),
            "classification_version": data.get("classification_version"),
            "classification_reason": data.get("classification_reason"),
            "error_stage": data.get("error_stage"),
            "error_type": data.get("error_type"),
            "user_message": str(data.get("user_message") or "")[:500],
            "bot_response": str(data.get("bot_response") or "")[:500],
            "created_at": _iso(data.get("created_at")),
        })

    result = {
        "days": days,
        "filters": {
            "intent": intent,
            "outcome": outcome,
            "channel": channel,
        },
        "scan_limit": scan_limit,
        "scanned_count": len(docs),
        "scan_truncated": len(docs) == scan_limit,
        "classified_count": total,
        "unclassified_count": unclassified_count,
        "intent_distribution": distribution(intent_counts, "intent"),
        "outcome_distribution": distribution(outcome_counts, "outcome"),
        "intent_outcome_matrix": [
            {"intent": key[0], "outcome": key[1], "count": count}
            for key, count in sorted(
                matrix.items(), key=lambda item: item[1], reverse=True
            )
        ],
        "daily_outcomes": [
            {"day": day, "outcomes": outcomes}
            for day, outcomes in sorted(daily.items())
        ],
        "rates": {
            "success": (
                round(outcome_counts.get("islem_basarili", 0) * 100 / total, 2)
                if total
                else None
            ),
            "fallback": (
                round(
                    outcome_counts.get("anlasilamadi_fallback", 0)
                    * 100
                    / total,
                    2,
                )
                if total
                else None
            ),
            "technical_error": (
                round(
                    outcome_counts.get("teknik_hata", 0) * 100 / total,
                    2,
                )
                if total
                else None
            ),
        },
        "recent_interactions": recent_interactions,
    }
    if _OUTCOME_CACHE_TTL_SECONDS > 0:
        with _outcome_cache_lock:
            if len(_outcome_cache) >= 32 and cache_key not in _outcome_cache:
                oldest_key = next(iter(_outcome_cache))
                _outcome_cache.pop(oldest_key, None)
            _outcome_cache[cache_key] = {
                "expires_at": now_ts + _OUTCOME_CACHE_TTL_SECONDS,
                "data": result,
            }
    return result


def _format_session_admin_row(doc_id, data):
    # Ortalama puan session belgesinde denormalize tutulduğu için burada
    # oturum başına evaluations sorgusu gerekmez.
    rating_sum = data.get("rating_sum", 0) or 0
    rating_count = data.get("rating_count", 0) or 0
    return {
        "session_id": doc_id,
        "user_id": data.get("user_id"),
        "username": data.get("username"),
        "started_at": _iso(data.get("started_at")),
        "last_active_at": _iso(data.get("last_active_at")),
        "message_count": data.get("message_count", 0),
        "is_active": data.get("is_active", False),
        "summary": data.get("summary", ""),
        "avg_rating": round(rating_sum / rating_count, 2) if rating_count else None,
        "evaluation_count": rating_count,
        "intent_counts": data.get("intent_counts", {}),
        "outcome_counts": data.get("outcome_counts", {}),
        "last_intent": data.get("last_intent"),
        "last_outcome": data.get("last_outcome"),
    }


def _get_session_search_rows():
    now_ts = time.monotonic()
    if _SESSION_SEARCH_CACHE_TTL_SECONDS > 0:
        with _session_search_cache_lock:
            if (
                _session_search_cache["rows"] is not None
                and _session_search_cache["expires_at"] > now_ts
            ):
                return _session_search_cache["rows"]

    docs = (
        _get_db().collection(COL_SESSIONS)
        .order_by("last_active_at", direction=firestore.Query.DESCENDING)
        .stream()
    )
    rows = [(doc.id, doc.to_dict() or {}) for doc in docs]
    if _SESSION_SEARCH_CACHE_TTL_SECONDS > 0:
        with _session_search_cache_lock:
            _session_search_cache["rows"] = rows
            _session_search_cache["expires_at"] = (
                now_ts + _SESSION_SEARCH_CACHE_TTL_SECONDS
            )
    return rows


def _filter_session_search_rows(search):
    needle = str(search or "").casefold()
    return [
        (doc_id, data)
        for doc_id, data in _get_session_search_rows()
        if needle in (
            f"{data.get('username', '')} {data.get('user_id', '')}".casefold()
        )
    ]


def get_sessions_admin(limit=50, offset=0, search=None):
    if search:
        rows = _filter_session_search_rows(search)[offset:offset + limit]
        return [_format_session_admin_row(doc_id, data) for doc_id, data in rows]

    query = _get_db().collection(COL_SESSIONS).order_by(
        "last_active_at", direction=firestore.Query.DESCENDING
    )
    docs = _query_page_with_cursor(
        query,
        cache_key="sessions",
        limit=limit,
        offset=offset,
    )
    return [
        _format_session_admin_row(doc.id, doc.to_dict() or {})
        for doc in docs
    ]


def count_sessions_admin(search=None):
    if not search:
        return _get_db().collection(COL_SESSIONS).count().get()[0][0].value
    # Liste ile aynı kısa süreli cache kullanılır; her tuş vuruşunda koleksiyon
    # yeniden taranmaz ve sayım ikinci bir Firestore sorgusu oluşturmaz.
    return len(_filter_session_search_rows(search))


def session_exists(session_id):
    """Puanlama gibi yalnızca varlık kontrolü isteyen yollar için tek okuma."""
    return (
        _get_db().collection(COL_SESSIONS).document(session_id).get().exists
    )


def get_session_admin_detail(session_id):
    db = _get_db()
    snap = db.collection(COL_SESSIONS).document(session_id).get()
    if not snap.exists:
        return None

    raw = snap.to_dict()
    raw["started_at"] = _iso(raw.get("started_at"))
    raw["last_active_at"] = _iso(raw.get("last_active_at"))
    session = {"session_id": snap.id, **raw}

    transcript = []
    for d in (
        db.collection(COL_CHAT_LOGS)
        .where(filter=FieldFilter("session_id", "==", session_id))
        .order_by("created_at", direction=firestore.Query.ASCENDING)
        .stream()
    ):
        row = d.to_dict()
        row["created_at"] = _iso(row.get("created_at"))
        transcript.append({"id": d.id, **row})

    tool_calls = []
    for d in (
        db.collection(COL_API_LOGS)
        .where(filter=FieldFilter("session_id", "==", session_id))
        .order_by("timestamp", direction=firestore.Query.ASCENDING)
        .stream()
    ):
        row = d.to_dict()
        row["timestamp"] = _iso(row.get("timestamp"))
        tool_calls.append({"id": d.id, **row})

    evaluations = []
    for d in (
        db.collection(COL_EVALUATIONS)
        .where(filter=FieldFilter("session_id", "==", session_id))
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .stream()
    ):
        row = d.to_dict()
        row["created_at"] = _iso(row.get("created_at"))
        evaluations.append({"id": d.id, **row})

    user_facts = get_user_facts(session.get("user_id"))
    performance_bundle = get_performance_metrics_bundle(
        limit=100,
        offset=0,
        session_id=session_id,
        include_total=False,
    )
    voice_recordings = get_voice_recordings_admin(session_id)
    voice_ai_evaluations = get_voice_ai_evaluations_admin(session_id)

    return {
        "session": session,
        "transcript": transcript,
        "tool_calls": tool_calls,
        "evaluations": evaluations,
        "user_facts": user_facts,
        "performance_metrics": performance_bundle["data"],
        "performance_averages": performance_bundle["averages"],
        "voice_recordings": voice_recordings,
        "voice_ai_evaluations": voice_ai_evaluations,
    }


def get_tool_calls_admin(limit=100, offset=0):
    db = _get_db()
    query = db.collection(COL_API_LOGS).order_by(
        "timestamp", direction=firestore.Query.DESCENDING
    )
    docs = _query_page_with_cursor(
        query,
        cache_key="tool_calls",
        limit=limit,
        offset=offset,
    )

    # YENİ: kullanıcı adı artık log_tool_call() içinde doğrudan kaydediliyor,
    # bu yüzden burada oturum başına ekstra bir sorgu atmıyoruz (eski N+1
    # problemi tamamen kaldırıldı).
    results = []
    for d in docs:
        data = d.to_dict()
        data["timestamp"] = _iso(data.get("timestamp"))
        results.append({"id": d.id, **data})
    return results


def count_tool_calls_admin():
    db = _get_db()
    return db.collection(COL_API_LOGS).count().get()[0][0].value


def add_evaluation(session_id, rating, note='', evaluator=''):
    db = _get_db()
    doc_ref = db.collection(COL_EVALUATIONS).document()
    doc_ref.set({
        "session_id": session_id,
        "rating": rating,
        "note": note,
        "evaluator": evaluator,
        "created_at": _now(),
    })
    # YENİ: sidebar'daki ortalama puanı ayrı bir sorgu atmadan gösterebilmek
    # için toplam/adet doğrudan session dokümanına da yazılıyor.
    db.collection(COL_SESSIONS).document(session_id).update({
        "rating_sum": firestore.Increment(rating),
        "rating_count": firestore.Increment(1),
    })
    return doc_ref.id
# ============================================================
# PERFORMANS METRİKLERİ (ADMIN)
# ============================================================

PERFORMANCE_METRIC_FIELDS = [
    "telegram_download_ms", "asr_ms", "ai_ms", "ai_ready_ms",
    "telegram_text_send_ms", "ttfb_ms", "tts_ms", "tts_ready_ms",
    "telegram_voice_upload_ms", "voice_audio_stream_ms", "tool_total_ms",
    "ttfs_ms", "e2e_ms", "full_turn_ms", "barge_in_latency_ms",
    "interrupt_count",
]


def _get_performance_metric_docs(
    session_id=None,
    recording_id=None,
    scan_limit=None,
    since=None,
):
    """Performans belgelerini Firestore'da sıralayıp sınırlandırarak getirir."""
    db = _get_db()
    collection = db.collection(COL_PERFORMANCE_METRICS)
    query = collection
    if recording_id:
        query = query.where(
            filter=FieldFilter("recording_id", "==", recording_id)
        )
    elif session_id:
        query = query.where(
            filter=FieldFilter("session_id", "==", session_id)
        )
    if since is not None:
        query = query.where(filter=FieldFilter("created_at", ">=", since))

    ordered_query = query.order_by(
        "created_at", direction=firestore.Query.DESCENDING
    )
    if scan_limit is not None:
        ordered_query = ordered_query.limit(scan_limit)
    try:
        return list(ordered_query.stream())
    except FailedPrecondition:
        # Kod ile index deploy'u kısa süre farklı sürümlerde kalırsa admin/QA
        # ekranı bozulmasın. Index hazır olana kadar eski, pahalı yol kullanılır.
        if not (recording_id or session_id):
            raise
        fallback_query = collection
        if recording_id:
            fallback_query = fallback_query.where(
                filter=FieldFilter("recording_id", "==", recording_id)
            )
        else:
            fallback_query = fallback_query.where(
                filter=FieldFilter("session_id", "==", session_id)
            )
        docs = list(fallback_query.stream())
        if since is not None:
            docs = [
                doc for doc in docs
                if isinstance((doc.to_dict() or {}).get("created_at"), datetime)
                and (doc.to_dict() or {}).get("created_at") >= since
            ]
        docs.sort(
            key=lambda d: (
                (d.to_dict() or {}).get("created_at").timestamp()
                if isinstance((d.to_dict() or {}).get("created_at"), datetime)
                else 0
            ),
            reverse=True,
        )
        return docs[:scan_limit] if scan_limit is not None else docs


def _format_performance_metric_docs(docs):
    results = []
    for d in docs:
        data = d.to_dict()
        row = {
            "id": d.id,
            "created_at": _iso(data.get("created_at")),
            "session_id": data.get("session_id"),
            "recording_id": data.get("recording_id"),
            "channel": data.get("channel"),
            "input_type": data.get("input_type"),
            "status": data.get("status"),
            "failed_stage": data.get("failed_stage"),
            "error_type": data.get("error_type"),
            "outcome": data.get("outcome"),
            "measurement_valid": data.get("measurement_valid", True),
            "measurement_errors": data.get("measurement_errors", []),
        }
        for f in PERFORMANCE_METRIC_FIELDS:
            row[f] = data.get(f)
        results.append(row)
    return results


def get_performance_metrics_admin(limit=25, offset=0, session_id=None):
    docs = _get_performance_metric_docs(
        session_id=session_id,
        scan_limit=limit + offset,
    )[offset:offset + limit]
    return _format_performance_metric_docs(docs)


def get_performance_metrics_export_admin(days=30, limit=5000, session_id=None):
    """Yonetim CSV raporu icin zaman araligina gore performans satirlari."""
    since = _now() - timedelta(days=max(1, min(int(days), 365)))
    docs = _get_performance_metric_docs(
        session_id=session_id,
        scan_limit=max(1, min(int(limit), 5000)),
        since=since,
    )
    return _format_performance_metric_docs(docs)


def count_performance_metrics_admin(session_id=None):
    db = _get_db()
    query = db.collection(COL_PERFORMANCE_METRICS)
    if session_id:
        query = query.where(filter=FieldFilter("session_id", "==", session_id))
    return query.count().get()[0][0].value


def get_performance_metrics_averages(
    sample_size=200,
    session_id=None,
    recording_id=None,
    *,
    docs=None,
):
    """Aynı geçerli başarı kohortu üzerinden karşılaştırılabilir ortalamalar.

    AI, TTFB ve E2E alanları farklı belge kümelerinden hesaplanırsa E2E
    ortalaması AI ortalamasından kısa çıkabilir. Bu nedenle ortalamaya yalnızca
    üç temel alanı da dolu, başarılı ve süre değişmezlerini sağlayan belgeler
    alınır. Eski/bozuk belgeler ortalamayı etkilemez.
    """
    if docs is None:
        docs = _get_performance_metric_docs(
            session_id=session_id,
            recording_id=recording_id,
            # Hatalı/eksik kayıtları eledikten sonra da yeterli örnek kalabilsin.
            scan_limit=max(sample_size * 5, sample_size),
        )

    def is_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    eligible = []
    excluded_count = 0
    for d in docs:
        data = d.to_dict()
        ai_ms = data.get("ai_ms")
        ttfb_ms = data.get("ttfb_ms")
        e2e_ms = data.get("e2e_ms")
        voice_schema_valid = (
            data.get("channel") != "voice_websocket"
            or data.get("metric_version") in {4, 5}
        )
        valid = (
            data.get("status") == "success"
            and data.get("measurement_valid", True) is not False
            and voice_schema_valid
            and is_number(ai_ms)
            and is_number(ttfb_ms)
            and is_number(e2e_ms)
            and e2e_ms >= ai_ms
            and e2e_ms >= ttfb_ms
        )
        if valid:
            eligible.append(data)
            if len(eligible) >= sample_size:
                break
        else:
            excluded_count += 1

    sums = {f: 0 for f in PERFORMANCE_METRIC_FIELDS}
    counts = {f: 0 for f in PERFORMANCE_METRIC_FIELDS}
    for data in eligible:
        for f in PERFORMANCE_METRIC_FIELDS:
            v = data.get(f)
            if is_number(v):
                sums[f] += v
                counts[f] += 1

    averages = {
        f: round(sums[f] / counts[f]) if counts[f] else None
        for f in PERFORMANCE_METRIC_FIELDS
    }
    averages["_sample_count"] = len(eligible)
    averages["_excluded_count"] = excluded_count
    # Aynı Firestore okumasından gelen noktalar frontend'de dakika/saat/gün
    # bazında gruplanır. Bunun için ek bir Firestore sorgusu yapılmaz.
    averages["_e2e_points"] = [
        {
            "created_at": _iso(data.get("created_at")),
            "e2e_ms": data.get("e2e_ms"),
        }
        for data in eligible
        if data.get("created_at") is not None and is_number(data.get("e2e_ms"))
    ]
    return averages


def get_performance_metrics_bundle(
    limit=25,
    offset=0,
    session_id=None,
    *,
    sample_size=200,
    include_total=True,
):
    """Tablo ve ortalamayı aynı Firestore belge kümesinden üretir."""
    scan_limit = max(limit + offset, sample_size * 5, sample_size)
    cache_key = str(session_id or "__global__")
    now_ts = time.monotonic()
    cached = None
    if _PERFORMANCE_CACHE_TTL_SECONDS > 0:
        with _performance_cache_lock:
            candidate = _performance_docs_cache.get(cache_key)
            if (
                candidate
                and candidate["expires_at"] > now_ts
                and (
                    len(candidate["docs"]) >= scan_limit
                    or len(candidate["docs"]) < candidate["scan_limit"]
                )
            ):
                cached = candidate

    if cached is None:
        docs = _get_performance_metric_docs(
            session_id=session_id,
            scan_limit=scan_limit,
        )
        cached = {
            "docs": docs,
            "total": None,
            "scan_limit": scan_limit,
            "expires_at": now_ts + _PERFORMANCE_CACHE_TTL_SECONDS,
        }
    else:
        docs = cached["docs"]

    if include_total and cached["total"] is None:
        cached["total"] = count_performance_metrics_admin(
            session_id=session_id
        )

    if _PERFORMANCE_CACHE_TTL_SECONDS > 0:
        with _performance_cache_lock:
            if (
                cache_key not in _performance_docs_cache
                and len(_performance_docs_cache) >= 16
            ):
                oldest_key = next(iter(_performance_docs_cache))
                _performance_docs_cache.pop(oldest_key, None)
            _performance_docs_cache[cache_key] = cached

    return {
        "data": _format_performance_metric_docs(
            docs[offset:offset + limit]
        ),
        "averages": get_performance_metrics_averages(
            sample_size=sample_size,
            session_id=session_id,
            docs=docs,
        ),
        "total": (
            cached["total"]
            if include_total
            else None
        ),
    }
