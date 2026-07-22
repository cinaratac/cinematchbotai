import os
import json
import time
from datetime import datetime, timedelta, timezone

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter

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
        _firebase_app = firebase_admin.initialize_app(cred)
    return firestore.client()


# Koleksiyon isimleri (mevcut trivia/haber koleksiyonlarıyla çakışmasın diye
# hepsi "bot_" öneki taşıyor)
COL_SESSIONS = "bot_sessions"
COL_CHAT_LOGS = "bot_chat_logs"
COL_USER_PROFILE = "bot_user_profiles"
COL_API_LOGS = "bot_api_logs"
COL_EVALUATIONS = "bot_evaluations"


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
    except Exception as e:
        print(f"SİSTEM HATASI: Firestore bağlantısı kurulamadı: {e}")


# ============================================================
# OTURUM YÖNETİMİ
# ============================================================

def get_or_create_session(user_id, username):
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
                return doc.id
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
        "is_active": True,
        "rating_sum": 0,
        "rating_count": 0,
    })
    print(f"SİSTEM: Kullanıcı {user_id} için yeni oturum açıldı -> session_id={new_doc.id}")
    return new_doc.id


def touch_session(session_id):
    db = _get_db()
    ref = db.collection(COL_SESSIONS).document(session_id)
    ref.update({
        "last_active_at": _now(),
        "message_count": firestore.Increment(1),
    })
    snap = ref.get()
    data = snap.to_dict() or {}
    return data.get("message_count", 0)


def update_session_summary(session_id, summary_text):
    db = _get_db()
    db.collection(COL_SESSIONS).document(session_id).update({"summary": summary_text})
    print(f"SİSTEM: Oturum #{session_id} özeti güncellendi.")


# ============================================================
# TAM KONUŞMA DÖKÜMÜ (TRANSCRIPT)
# ============================================================

def log_chat(session_id, user_id, username, user_message, bot_response):
    db = _get_db()
    db.collection(COL_CHAT_LOGS).document().set({
        "session_id": session_id,
        "user_id": str(user_id),
        "username": username,
        "user_message": user_message,
        "bot_response": bot_response,
        "created_at": _now(),
    })
    print("LOG BAŞARILI: Mesaj Firestore'a kaydedildi.")


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

def get_user_history(user_id, current_session_id, max_past_sessions=3):
    db = _get_db()
    query = (
        db.collection(COL_SESSIONS)
        .where(filter=FieldFilter("user_id", "==", str(user_id)))
        .order_by("started_at", direction=firestore.Query.DESCENDING)
        .limit(max_past_sessions + 1)  # +1: mevcut oturumu eleyebilmek için
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

def log_tool_call(session_id, user_id, movie_name, api_endpoint, api_response, username=None):
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
        "timestamp": _now(),
    })


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


def update_user_facts(user_id, username, new_facts):
    if not new_facts:
        return
    existing = get_user_facts(user_id)
    existing.update(new_facts)

    db = _get_db()
    db.collection(COL_USER_PROFILE).document(str(user_id)).set({
        "username": username,
        "facts": existing,
        "updated_at": _now(),
    }, merge=True)
    print(f"SİSTEM: Kullanıcı {user_id} profili güncellendi -> {existing}")


# ============================================================
# ADMIN PANELİ SORGULARI
# ============================================================

# YENİ: get_admin_overview çok pahalı (tüm sessions/evaluations/son 14 günün
# tüm chat_logs'unu tarıyor). Panel her açıldığında/yenilendiğinde tekrar
# tekrar çalışmasın diye kısa süreli bellek-içi cache kullanılıyor.
_overview_cache = {"data": None, "expires_at": 0, "days": None}
_OVERVIEW_CACHE_TTL_SECONDS = 60


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

    # Benzersiz kullanıcı sayısı (küçük ölçek için Python tarafında hesaplanıyor)
    user_ids = set()
    for doc in db.collection(COL_SESSIONS).select(["user_id"]).stream():
        user_ids.add(doc.to_dict().get("user_id"))
    total_users = len(user_ids)

    # Tool başarı oranı + en çok sorulan filmler: son 500 tool çağrısı üzerinden
    tool_docs = list(
        db.collection(COL_API_LOGS)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(500)
        .stream()
    )
    successful_tool_calls = 0
    movie_counts = {}
    for d in tool_docs:
        data = d.to_dict()
        resp = data.get("api_response", "") or ""
        if '"Response": "True"' in resp or '"Response":"True"' in resp:
            successful_tool_calls += 1
        name = data.get("movie_name")
        if name:
            movie_counts[name] = movie_counts.get(name, 0) + 1
    top_movies = [
        {"movie_name": k, "c": v}
        for k, v in sorted(movie_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # Son N günün günlük mesaj hacmi
    since = _now() - timedelta(days=days)
    daily_counts = {}
    for doc in (
        db.collection(COL_CHAT_LOGS)
        .where(filter=FieldFilter("created_at", ">=", since))
        .select(["created_at"])
        .stream()
    ):
        created = doc.to_dict().get("created_at")
        if isinstance(created, datetime):
            day = created.strftime("%Y-%m-%d")
            daily_counts[day] = daily_counts.get(day, 0) + 1
    daily_messages = [{"day": k, "c": v} for k, v in sorted(daily_counts.items())]

    # Değerlendirme dağılımı + ortalama
    eval_docs = [d.to_dict() for d in db.collection(COL_EVALUATIONS).stream()]
    ratings = [e.get("rating") for e in eval_docs if e.get("rating") is not None]
    rating_dist = {}
    for r in ratings:
        rating_dist[r] = rating_dist.get(r, 0) + 1
    rating_distribution = [{"rating": k, "c": v} for k, v in sorted(rating_dist.items())]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None

    result = {
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "total_messages": total_messages,
        "total_users": total_users,
        "total_tool_calls": total_tool_calls,
        "successful_tool_calls": successful_tool_calls,
        "failed_tool_calls": total_tool_calls - successful_tool_calls,
        "daily_messages": daily_messages,
        "rating_distribution": rating_distribution,
        "avg_rating": avg_rating,
        "total_evaluations": len(eval_docs),
        "top_movies": top_movies,
    }

    _overview_cache["data"] = result
    _overview_cache["days"] = days
    _overview_cache["expires_at"] = now_ts + _OVERVIEW_CACHE_TTL_SECONDS
    return result


def get_sessions_admin(limit=50, offset=0, search=None):
    db = _get_db()
    query = db.collection(COL_SESSIONS).order_by(
        "last_active_at", direction=firestore.Query.DESCENDING
    )
    # Not: Firestore'da OFFSET büyük veri setlerinde verimsizdir; küçük/orta
    # ölçekli bir bot için sorun teşkil etmez.
    docs = list(query.limit(limit + offset).stream())[offset:offset + limit]

    results = []
    for doc in docs:
        data = doc.to_dict()
        if search:
            haystack = f"{data.get('username','')} {data.get('user_id','')}".lower()
            if search.lower() not in haystack:
                continue

        # YENİ: her oturum için ayrı bir evaluations sorgusu atmak yerine
        # (N+1 problemi), toplam/adet doğrudan session dokümanında tutuluyor
        # (bkz. add_evaluation), ortalama buradan anlık hesaplanıyor.
        rating_sum = data.get("rating_sum", 0) or 0
        rating_count = data.get("rating_count", 0) or 0

        results.append({
            "session_id": doc.id,
            "user_id": data.get("user_id"),
            "username": data.get("username"),
            "started_at": _iso(data.get("started_at")),
            "last_active_at": _iso(data.get("last_active_at")),
            "message_count": data.get("message_count", 0),
            "is_active": data.get("is_active", False),
            "summary": data.get("summary", ""),
            "avg_rating": round(rating_sum / rating_count, 2) if rating_count else None,
            "evaluation_count": rating_count,
        })
    return results


def count_sessions_admin(search=None):
    db = _get_db()
    if not search:
        return db.collection(COL_SESSIONS).count().get()[0][0].value
    # search verilmişse tam sayım için tüm dokümanları taramak gerekiyor
    count = 0
    for doc in db.collection(COL_SESSIONS).select(["username", "user_id"]).stream():
        data = doc.to_dict()
        haystack = f"{data.get('username','')} {data.get('user_id','')}".lower()
        if search.lower() in haystack:
            count += 1
    return count


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

    return {
        "session": session,
        "transcript": transcript,
        "tool_calls": tool_calls,
        "evaluations": evaluations,
        "user_facts": user_facts,
    }


def get_tool_calls_admin(limit=100, offset=0):
    db = _get_db()
    query = db.collection(COL_API_LOGS).order_by(
        "timestamp", direction=firestore.Query.DESCENDING
    )
    docs = list(query.limit(limit + offset).stream())[offset:offset + limit]

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
