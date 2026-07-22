import base64
import os
import re
import time
import requests
import json
from urllib.parse import urlencode

from database import (
    get_or_create_session,
    touch_session,
    log_chat,
    get_user_history,
    update_session_summary,
    get_session_transcript,
    get_user_facts,
    update_user_facts,
    log_tool_call,
    SUMMARY_UPDATE_INTERVAL,
)
from app_guide import CINEMATCH_APP_GUIDE

# --- GİZLİ ANAHTARLAR: ortam değişkenlerinden okunuyor. Render'da
# "Environment" sekmesinden şunları tanımlaman gerekiyor:
#   OPENROUTER_API_KEY, OMDB_API_KEY
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

if not OMDB_API_KEY:
    print("UYARI: OMDB_API_KEY tanımlı değil; film verisi çekme aracı çalışmayacak.")
if not OPENROUTER_API_KEY:
    print("UYARI: OPENROUTER_API_KEY tanımlı değil; AI yanıtları çalışmayacak.")

MODEL_NAME = "google/gemma-4-26b-a4b-it"
#model geçmişi:
# - qwen/qwen-2.5-7b-instruct: çalışıyordu ama halüsinasyon oranı çok yüksekti.
# - openai/gpt-oss-120b:free ve google/gemma-4-26b-a4b-it:free: ikisi de
#   OpenRouter'ın PAYLAŞIMLI ücretsiz kotasına takıldı 
# - google/gemma-4-26b-a4b-it  platform seviyesinde istek
#   limiti yok, sadece token başına ücretlendiriliyor . Şu an bu kullanılıyor.
# Alternatif: google/gemini-2.0-flash-001 de benzer şekilde ucuz ve kararlı.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_TRANSCRIPTIONS_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
OPENROUTER_SPEECH_URL = "https://openrouter.ai/api/v1/audio/speech"
ASR_MODEL_NAME = os.environ.get("ASR_MODEL_NAME", "openai/whisper-large-v3")
TTS_MODEL_NAME = os.environ.get("TTS_MODEL_NAME", "openai/gpt-4o-mini-tts-2025-12-15")
TTS_VOICE = os.environ.get("TTS_VOICE", "nova")


VISION_MODEL_NAME = MODEL_NAME


def transcribe_audio(audio_bytes, audio_format="ogg", language=None):
    """Ses dosyasını OpenRouter STT API üzerinden metne çevirir.

    Telegram voice mesajları OGG/Opus biçiminde geldiği için varsayılan format
    ``ogg``'dir. ``language`` verilmezse konuşulan dil model tarafından otomatik
    algılanır; böylece bot Türkçe dışındaki sesli mesajları da işleyebilir.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY tanımlı değil; ses yazıya çevrilemiyor.")
    if not audio_bytes:
        raise ValueError("Ses dosyası boş.")

    payload = {
        "model": ASR_MODEL_NAME,
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
            "format": audio_format,
        },
    }
    if language:
        payload["language"] = language

    response = requests.post(
        OPENROUTER_TRANSCRIPTIONS_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Ses servisi geçersiz bir cevap döndürdü.") from exc

    if not response.ok:
        error = data.get("error", {})
        message = error.get("message") if isinstance(error, dict) else None
        raise RuntimeError(message or f"Ses servisi HTTP {response.status_code} hatası döndürdü.")

    transcript = str(data.get("text", "")).strip()
    if not transcript:
        raise RuntimeError("Ses anlaşılamadı veya boş bir transkript döndü.")
    return transcript


def text_to_speech(text, voice=None):
    """Metni OpenRouter TTS API ile MP3 ses verisine çevirir."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY tanımlı değil; ses üretilemiyor.")

    clean_text = str(text or "").strip()
    if not clean_text:
        raise ValueError("Seslendirilecek metin boş.")

    response = requests.post(
        OPENROUTER_SPEECH_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": TTS_MODEL_NAME,
            "input": clean_text,
            "voice": voice or TTS_VOICE,
            "response_format": "mp3",
            "speed": 1.0,
        },
        timeout=90,
    )

    if not response.ok:
        try:
            error_data = response.json()
            error = error_data.get("error", {})
            message = error.get("message") if isinstance(error, dict) else None
        except ValueError:
            message = None
        raise RuntimeError(message or f"TTS servisi HTTP {response.status_code} hatası döndürdü.")

    if not response.content:
        raise RuntimeError("TTS servisi boş ses verisi döndürdü.")
    return response.content


def get_live_movie_data(movie_name, session_id=None, user_id=None, username=None):
    """Bu fonksiyon OMDb API'ye bağlanıp anlık film verilerini çeker ve loglar.
    session_id/user_id verilirse (normal akışta her zaman verilir), admin
    panelinde bu tool çağrısı ilgili konuşmayla ilişkilendirilerek gösterilir.
    username verilirse admin panelindeki "Tool Çağrıları" tablosunda ekstra
    sorguya gerek kalmadan doğrudan gösterilir."""
    url = "https://www.omdbapi.com/"
    # Admin logunda API anahtarı görünmesin; gerçek istekte params içine eklenir.
    logged_url = f"{url}?{urlencode({'t': movie_name})}"
    try:
        response = requests.get(
            url,
            params={"t": movie_name, "apikey": OMDB_API_KEY},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        logged_response = json.dumps(data, ensure_ascii=False)
    except Exception as e:
        # Başarısız çağrılar da admin panelindeki Tool Çağrıları bölümünde
        # görülsün; aksi halde çağrı hiç yapılmamış gibi görünüyordu.
        safe_error_message = str(e)
        if OMDB_API_KEY:
            safe_error_message = safe_error_message.replace(OMDB_API_KEY, "[REDACTED]")
        logged_response = json.dumps({
            "Response": "False",
            "Error": type(e).__name__,
            "message": safe_error_message[:500],
        }, ensure_ascii=False)
        log_tool_call(
            session_id, user_id, movie_name, logged_url,
            logged_response, username=username,
        )
        raise

    log_tool_call(
        session_id, user_id, movie_name, logged_url,
        logged_response, username=username,
    )

    if data.get("Response") == "True":
        return (f"Film: {data.get('Title')}, "
                f"Çıkış Yılı: {data.get('Year')}, "
                f"Yönetmen: {data.get('Director')}, "
                f"Tür: {data.get('Genre')}, "
                f"Oyuncular: {data.get('Actors')}, "
                f"IMDb Puanı: {data.get('imdbRating')}, "
                f"Gişe Hasılatı: {data.get('BoxOffice')}.")
    else:
        return "Film veritabanında bulunamadı."


def _call_openrouter(messages, tools=None, tool_choice=None, model=None, _retry=True):
    """OpenRouter'a istek atan ortak yardımcı fonksiyon (kod tekrarını önlemek için).
    tool_choice="auto" (varsayılan, model kendi karar verir) ya da modeli belirli
    bir aracı ÇAĞIRMAYA ZORLAMAK için {"type": "function", "function": {"name": "..."}}
    olabilir. Bu, modelin "aracı çağırmayı unutması/atlaması" (halüsinasyon) riskini
    tamamen ortadan kaldırır -- prompt kuralına güvenmek yerine kod garantisi sağlar.

    model=None ise varsayılan MODEL_NAME kullanılır; görüntü analizinde olduğu gibi
    farklı bir model (VISION_MODEL_NAME) ile çağırmak için doldurulabilir.

    429 (rate-limit) hatası alınırsa, kısa bir bekleme sonrası BİR KEZ otomatik
    tekrar dener; kullanıcı çoğu zaman geçici yoğunluğu hiç fark etmez."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model or MODEL_NAME,
        "temperature": 0.3,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    data = response.json()

    if _retry and 'choices' not in data and data.get('error', {}).get('code') == 429:
        print("SİSTEM: 429 rate-limit alındı, 2 saniye sonra bir kez daha deneniyor...")
        time.sleep(2)
        return _call_openrouter(messages, tools=tools, tool_choice=tool_choice, model=model, _retry=False)

    return data



MOVIE_FACT_KEYWORDS = [
    "imdb", "ımdb", "puan", "puanı", "puani",
    "gişe", "gise", "hasılat", "hasilat", "box office",
    "çıkış yılı", "cikis yili", "hangi yıl", "hangi yil",
    "ne zaman çıktı", "ne zaman cikti", "vizyon tarihi",
]


def wants_movie_facts(user_message):
    """Kullanıcı mesajında IMDb puanı, gişe hasılatı veya çıkış yılı gibi
    somut bir film verisi istendiğine dair bir işaret var mı?"""
    lowered = user_message.lower()
    return any(keyword in lowered for keyword in MOVIE_FACT_KEYWORDS)


_EXPLICIT_MOVIE_PATTERNS = [
    # Tırnak içine alınmış başlık: "Inception" hakkında ne düşünüyorsun?
    re.compile(
        r"^[^\"“”']*[\"“”']([^\"“”']{1,120})[\"“”']\s*"
        r"(?:filmi\s+)?(?:hakkında|nasıl|sence|imdb|puan|gişe|çıkış)",
        re.IGNORECASE,
    ),
    # Inception filmi/dizisi hakkında... / Inception filmi nasıl?
    re.compile(
        r"^\s*(.{1,120}?)\s+(?:filmi|dizisi)\s+"
        r"(?:hakkında|nasıl(?:dır)?\b|sence\b|iyi\s+mi\b|kötü\s+mü\b)",
        re.IGNORECASE,
    ),
    # Inception nasıl? / Inception hakkında ne düşünüyorsun?
    re.compile(
        r"^\s*(.{1,120}?)\s+"
        r"(?:hakkında\s+(?:ne\s+düşünüyorsun|bilgi)|nasıl(?:dır)?\??\s*$|sence\s+nasıl)",
        re.IGNORECASE,
    ),
]


def extract_explicit_movie_title(user_message):
    """Açıkça tek bir filme yönelen basit mesajlardan başlığı çıkarır.

    Serbest doğal dilde kusursuz film-adı tespiti mümkün olmadığı için yalnızca
    yanlış pozitif üretme ihtimali düşük kalıplar kabul edilir. Uygulama film
    kartından soru gönderiyorsa ``movie_name`` alanını ayrıca göndermesi daha
    güvenilirdir.
    """
    text = str(user_message or "").strip()
    for pattern in _EXPLICIT_MOVIE_PATTERNS:
        match = pattern.search(text)
        if match:
            title = match.group(1).strip(" \t\n.,!?;:\"'“”")
            if title:
                return title
    return None




NAME_PATTERNS = [
    r"(?:benim\s+)?ad[ıi]m\s+([A-ZÇĞİÖŞÜ][a-zçğıöşüA-ZÇĞİÖŞÜ]+)",
    r"(?:benim\s+)?ismim\s+([A-ZÇĞİÖŞÜ][a-zçğıöşüA-ZÇĞİÖŞÜ]+)",
]


def extract_simple_facts(user_message):
    """Mesajdan güvenilir, basit desenlerle (isim gibi) bilgi çıkarır."""
    facts = {}
    for pattern in NAME_PATTERNS:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            facts["isim"] = match.group(1).strip().capitalize()
            break
    return facts


def build_app_profile_context(app_profile):
    """Cinematch uygulamasından gelen zevk verisini (favori tür/oyuncu/
    yönetmen/film -- kullanıcının uygulamada kendi seçtiği KESİN veri)
    okunabilir bir metne çevirir. Veri yoksa None döner."""
    if not app_profile:
        return None

    genres = [g for g in (app_profile.get('favorite_genres') or []) if g]
    directors = [d for d in (app_profile.get('favorite_directors') or []) if d]
    actors = [a for a in (app_profile.get('favorite_actors') or []) if a]
    movies = [m for m in (app_profile.get('favorite_movies') or []) if m]

    if not (genres or directors or actors or movies):
        return None

    lines = []
    if genres:
        lines.append(f"- Sevdiği türler: {', '.join(genres)}")
    if directors:
        lines.append(f"- Sevdiği yönetmenler: {', '.join(directors)}")
    if actors:
        lines.append(f"- Sevdiği oyuncular: {', '.join(actors)}")
    if movies:
        lines.append(f"- Beğendiği / favorilere eklediği filmler: {', '.join(movies)}")
    return "\n".join(lines)


_MOVIE_MARKER_RE = re.compile(r"\[\[FILMLER:\s*(.*?)\]\]", re.IGNORECASE | re.DOTALL)


def extract_movie_recommendations(text):
    """Modelin cevabının sonuna eklediği [[FILMLER: A | B]] bloğunu bulur,
    görünür metinden temizler ve önerilen film adlarını bir listeye çevirir.
    Marker yoksa metni olduğu gibi, film listesini boş olarak döner."""
    if not text:
        return text, []

    match = _MOVIE_MARKER_RE.search(text)
    if not match:
        return text, []

    movies = [m.strip() for m in match.group(1).split('|') if m.strip()]
    cleaned = _MOVIE_MARKER_RE.sub('', text).strip()
    return cleaned, movies


def build_profile_context(user_facts):
    """Kalıcı kullanıcı profilini (varsa) net, tartışmasız bir metne çevirir."""
    if not user_facts:
        return "Bu kullanıcı hakkında henüz kayıtlı kesin bir bilgi yok."
    lines = [f"- {key}: {value}" for key, value in user_facts.items()]
    return "\n".join(lines)



# GEÇMİŞİ OKUNABİLİR METNE ÇEVİRME


def build_background_context(history):
    """
    database.get_user_history() çıktısından SADECE arka plan bilgisini
    (geçmiş oturum özetleri + mevcut oturumun eski kısmının özeti) metne çevirir.

    ÖNEMLİ: Mevcut oturumun SON mesajları burada YOK — onlar artık sistem
    promptuna gömülü metin olarak değil, gerçek user/assistant rollü mesajlar
    olarak messages dizisine ekleniyor (bkz. get_ai_response). Bunun sebebi:
    küçük bir modelin, sistem promptu içine gömülü "Kullanıcı: ... Asistan: ..."
    metin bloğunu yorumlamaktansa, gerçek çok turlu (multi-turn) bir sohbeti
    doğal şekilde sürdürmesi çok daha güvenilir oluyor (tekrar/kopyalama gibi
    bozulma davranışlarını belirgin şekilde azaltıyor).
    """
    parts = []

    past_summaries = history.get("past_summaries", [])
    if past_summaries:
        parts.append("### GEÇMİŞ OTURUM ÖZETLERİ (bu kullanıcıyla önceki konuşmalar):")
        for s in reversed(past_summaries):  # en eskiden en yeniye doğru sırala
            parts.append(f"- ({s['started_at']}): {s['summary']}")

    current_session_summary = history.get("current_session_summary", "")
    if current_session_summary:
        parts.append(
            "\n### BU OTURUMUN DAHA ÖNCEKİ KISMININ ÖZETİ "
            "(aşağıda mesaj dizisinde gördüğün son mesajlardan ÖNCE konuşulanlar):"
        )
        parts.append(f"- {current_session_summary}")

    if not parts:
        return "Bu kullanıcıyla ilgili kayıtlı bir geçmiş yok, bu ilk konuşma."

    return "\n".join(parts)



# PERİYODİK OLARAK ÇAĞRILAN ÖZETLEME ARACI

def summarize_session(session_id):
    transcript = get_session_transcript(session_id)
    if not transcript:
        return

    transcript_text = "\n".join(
        f"Kullanıcı: {t['user_message']}\nAsistan: {t['bot_response']}"
        for t in transcript
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Aşağıda bir sinema asistanı ile kullanıcı arasındaki konuşma dökümü var. "
                "Bunu; kullanıcının hangi filmleri/türleri/yönetmenleri sorduğunu, hangi "
                "önerilerin yapıldığını ve kullanıcının belli olan zevklerini içeren "
                "3-5 cümlelik KISA bir özete çevir. Sadece özeti yaz, başka açıklama ekleme."
            ),
        },
        {"role": "user", "content": transcript_text},
    ]

    try:
        response_data = _call_openrouter(messages)
        summary = response_data["choices"][0]["message"]["content"].strip()
        update_session_summary(session_id, summary)
    except Exception as e:
        print(f"SİSTEM UYARISI: Oturum özeti çıkarılamadı: {e}")



# ANA FONKSİYON


def get_ai_response(user_message, user_id="anon", username="anon", app_profile=None, movie_name=None):
    # 1. OTURUMU BUL / OLUŞTUR (session bazlı yapı)
    session_id = get_or_create_session(user_id, username)

    # Film adı istemci tarafından açıkça gönderildiyse veya mesajdaki güvenli
    # bir kalıptan çıkarılabildiyse, model cevap üretmeden ÖNCE OMDb'yi çağır.
    # Böylece çağrı modelin tool kullanma tercihine bağlı kalmaz ve admin
    # panelindeki Tool Çağrıları bölümüne her zaman loglanır.
    explicit_movie_name = str(movie_name or "").strip() or extract_explicit_movie_title(user_message)
    prefetched_movie_data = None
    if explicit_movie_name:
        try:
            prefetched_movie_data = get_live_movie_data(
                explicit_movie_name,
                session_id=session_id,
                user_id=user_id,
                username=username,
            )
        except Exception as e:
            print(f"SİSTEM UYARISI: OMDb ön sorgusu başarısız: {e}")

    # 2. BASİT BİLGİ ÇIKARIMI: mesajda isim gibi kesin bir bilgi varsa kalıcı profile hemen kaydet 
    new_facts = extract_simple_facts(user_message)
    if new_facts:
        update_user_facts(user_id, username, new_facts)

    # 3. BOT CEVAP VERMEDEN ÖNCE KULLANICI GEÇMİŞİNİ ÇEK
    
    user_facts = get_user_facts(user_id)

    # YENİ: Eğer bu kullanıcı için kayıtlı bir isim yoksa ve Cinematch
    # uygulamasından (Firebase Auth displayName) güvenilir bir ad geldiyse,
    # onu kalıcı profile otomatik kaydet. Böylece kullanıcı adını sohbette
    # tekrar yazmasına gerek kalmadan daha ilk mesajdan itibaren ismiyle
    # hitap edilebilir.
    _GENERIC_USERNAMES = {"anon", "api_user", "kullanıcı", "kullanici", "unknown", "flutter_user", ""}
    if "isim" not in user_facts and username and username.strip().lower() not in _GENERIC_USERNAMES:
        user_facts["isim"] = username.strip()
        update_user_facts(user_id, username, {"isim": username.strip()})

    profile_context = build_profile_context(user_facts)

    # YENİ: Cinematch uygulamasından gelen zevk profili (favori tür/oyuncu/
    # yönetmen/film). Telegram tarafından çağrıldığında app_profile None
    # olur, o zaman normal şekilde yoksayılır.
    app_profile_context = build_app_profile_context(app_profile)

    history = get_user_history(user_id, session_id)
    background_context = build_background_context(history)

    system_prompt = f"""Sen profesyonel bir sinema asistanı ve film eleştirmenisin. Aynı zamanda CineMatch uygulamasının içindeki resmi yapay zeka asistanısın ve uygulamanın nasıl kullanılacağı hakkında da doğru bilgi verebilirsin.

Sadece filmler, diziler, yönetmenler, oyuncular, sinema sektörü VE CineMatch uygulamasının kendisi (özellikleri, ayarları, nasıl kullanılacağı) hakkındaki sorulara cevap verebilirsin. Sinema hakkında her türlü soruya cevap verebilirisin.

Vizyondaki filmler, klasikler, ödüllü filmler ve popüler diziler hakkında önerilerde bulunabilirsin. Yönetmen ve oyuncuların filmografileri veya hayatları hakkında bilgi verebilirsin.

ÜSLUP KURALI (ÇOK ÖNEMLİ): Cevapların KISA VE ÖZ olsun. Normal bir soruya/sohbete 2-4 cümleyle cevap ver; bir film önerirken bile sinopsis + eleştirmen yorumunu birlikte en fazla 4-5 cümlede toparla, gereksiz uzatma ve tekrar yapma. Kullanıcı açıkça "detaylı anlat", "uzun uzun anlat" gibi bir şey istemedikçe uzun paragraflar yazma.

KURALLAR:

1. Eğer kullanıcı belirli bir ülkenin (örn: Portekiz, Fransa), belirli bir konunun (örn: uzay, aşk) veya belirli bir türün filmlerini sorarsa, bu sinema ile İLGİLİDİR. Hemen o kategoriye uygun harika film önerileri yap.

2. SADECE kullanıcı hem sinemayla hem de CineMatch uygulamasıyla HİÇ ALAKASI OLMAYAN şeyler sorarsa (Örneğin: "Portekiz'in başkenti neresi?", "Nasıl kod yazılır?", "Kek tarifi ver") KESİNLİKLE bilgi verme ve sadece şu cümleyi söyle:

'Üzgünüm, ben bir sinema asistanıyım ve sadece filmler/diziler ile CineMatch uygulaması hakkında konuşabilirim. Size güzel bir film önermemi ister misiniz?'

Kullanıcı CineMatch uygulamasının kendisi hakkında bir şey sorarsa (örn. "kullanıcı adımı nasıl değiştiririm", "eşleşme nasıl çalışıyor", "bildirimleri nasıl kapatırım") bu KESİNLİKLE reddedilmez; aşağıdaki 16. kuralı ve CINEMATCH UYGULAMA REHBERİ'ni kullanarak cevapla.

3. DİL UYUMU: Kullanıcı hangi dilde yazıyorsa (Türkçe, İngilizce, İspanyolca vb.), SADECE o dilde cevap ver.
4. DİKKAT KESİN KURAL: Asla ve asla kendi hafızandaki IMDb puanlarını, gişe hasılatlarını veya çıkış yıllarını kullanıcıya söyleme! Bu bilgiler doğrudan istenirse VEYA bir film hakkında (öneri, tanıtım, karşılaştırma fark etmez) detaylı bahsediyorsan, sayısal veriler için İSTİSNASIZ OLARAK önce "get_live_movie_data" aracını çağıracaksın, sonra aracın döndürdüğü GERÇEK veriyi kullanacaksın. Araç sonuç getirmezse rakam söylemeden devam et; asla tahmini/hafızandan bir sayı yazma. Bu kuralı çiğnemek KABUL EDİLEMEZ.
5. Kullanıcı kişisel bir konudan bahsettiğinde (örn: "Okul kötü geçti, iş yerinde mobbinge uğruyorum, nasılsın, bir ai olmak nasıl bir his) gibi kişisel şeyler sorduğunda konuyu hemen sinemaya getir. Unutma sen bir sinema robotusun ancak kullanıcıların isteklerini de çok fazla geri çevirme.
6. HAFIZA KURALI (ÇOK ÖNEMLİ): Aşağıda "KULLANICI PROFİLİ" ve "KULLANICI GEÇMİŞİ" başlıkları altında bu kullanıcıyla ilgili kesin bilgiler, geçmiş oturum özetleri ve mevcut oturumdaki son mesajlar verilmiştir. Bu bilgiyi SADECE İLGİLİ OLDUĞU YERDE kullan:
   - KULLANICI PROFİLİ'nde bir bilgi (örn. isim) varsa, bunu KESİN DOĞRU kabul et. Kullanıcı "ismim ne?" gibi doğrudan sorarsa, profildeki bilgiyi birebir söyle.
   - ASLA "[isim]" gibi köşeli parantezli bir yer tutucu (placeholder) yazma. Eğer bir bilgi profilde/geçmişte YOKSA, o bilgiyi bilmediğini söyle veya normal şekilde sor; asla köşeli parantez kullanma.
   - Kullanıcının daha önce sorduğu ya da söylediği bir bilgiyi (örn. sevdiği tür, daha önce sorduğu film) tekrar sorma.
   - Daha önce önerdiğin filmleri, kullanıcı açıkça "başka" ya da "farklı bir şey" istemediği sürece tekrar önerme.
   - Kullanıcının geçmişte belirttiği zevklerini hatırlıyormuş gibi doğal şekilde devam ettir; "hatırlıyor musun" diye sorma, doğrudan bilgiyi kullan.
   - Aynı soruyu ya da aynı bilgiyi tekrar tekrar isteme; geçmişte cevaplanmış bir şeyi tekrar sorma.
   - KULLANICI PROFİLİ'nde bir "isim" bilgisi varsa, kullanıcıya doğal aralıklarla (özellikle sohbetin başında/selamlaşmada ya da bir öneri sunarken) ismiyle hitap et. Ama her tek mesajda ismi tekrarlayıp yapmacık/robotik durma; doğal bir insan gibi ara sıra kullan.
7. SADECE İLGİLİ OLAN YERDE HAFIZA KULLAN (ÇOK ÖNEMLİ): Kullanıcı sadece kısa bir selamlaşma yazdıysa ("merhaba", "selam", "naber", "günaydın" gibi), KULLANICI PROFİLİ'ndeki ya da GEÇMİŞ'teki bilgileri cevaba ZORLA doldurma. Sadece doğal, kısa bir şekilde selamlaş (istersen ismiyle hitap et), film önerisi veya geçmiş tercihlerden bahsetmeyi kullanıcı bir şey sormadan/istemeden başlatma. Geçmiş bilgiler ancak kullanıcının isteği bunu gerektirdiğinde (film önerisi istemesi, tercihinin sorulması vb.) devreye girer.
8. TEKRAR YASAĞI (ÇOK ÖNEMLİ): Bir önceki kendi mesajını birebir ya da neredeyse birebir tekrar ETME. Kullanıcı kısa bir onay/istek mesajı yazdıysa ("hadi yap", "tamam", "evet", "devam et", "yap" gibi), bunu SON mesajında verdiğin teklifin gerçekleştirilmesi isteği olarak yorumla ve GERÇEKTEN SOMUT BİR AKSİYON AL (örn. gerçek bir film adı ve kısa açıklamasıyla öneri yap) — sadece aynı teklif/selamlama cümlesini tekrar yazma.
9. HASSAS KONU KURALI: Kullanıcı zorbalık, kötü muamele, yalnızlık gibi üzücü/hassas bir durumdan bahsettiyse, doğrudan kalıp cümleyle ("ben sadece sinema asistanıyım...") geçme. Önce 1 cümlelik samimi bir empati göster, sonra nazikçe sinemaya bağla. Kullanıcı bu konuyla ilgili film isterse, konuyu doğrudan sahneleyen ağır/rahatsız edici filmler yerine; dayanıklılık, dostluk, iyileşme ve umut temalı, insanı güçlendiren filmler öner (örn. Wonder, Sing Street, The Karate Kid, Good Will Hunting gibi).
10. TÜR DÜRÜSTLÜĞÜ KURALI (ÇOK ÖNEMLİ): Kullanıcının istediği tür(ler)e gerçekten uyan bir film seç; uymayan bir filmi seçip sonra onu o türmüş GİBİ göstermeye çalışma (örn. romantik-dramatik bir filmi "aksiyonlu ilerliyor" diye tanımlama). Filmin türünü her zaman DOĞRU ve DÜRÜST tanımla. Kullanıcı birden fazla tür istiyorsa (örn. "hem aksiyon hem duygusal"), gerçekten o türlerin ikisini de taşıyan bir film bulmaya çalış (örn. Gladiator, Warrior (2011), Logan). Tam eşleşen bir film bulamıyorsan, bunu açıkça söyle ("tam ikisini birden karşılayan bir film yerine, [X] öneriyorum çünkü...") ve filmi olduğu gibi tanıt, gerçek türünü saklamaya veya çarpıtmaya çalışma.
11. YORUM KATMA GÖREVİ: Bir film sorulduğunda ya da önerildiğinde SADECE özet (sinopsis) verme! Sen bir sinema ELEŞTİRMENİSİN; sinopsisin yanında mutlaka kendi (yapay ama profesyonel) görüşünü de kat: sinematografi, oyunculuklar (örneğin belirli bir oyuncunun performansı), yönetmenin tarzı ya da filmin sinema tarihindeki/türündeki yeri hakkında kısa bir yorum ekle. Kuru bir Wikipedia özeti gibi yazma; bir eleştirmenin sesiyle konuş.
12. DOĞAL DİYALOG (ÇOK ÖNEMLİ): Mesajlarının sonuna asla "Başka bir öneri mi arıyorsunuz?", "Başka bir konuda yardımcı olabilir miyim?" gibi basmakalıp, çağrı-merkezi tarzı robotik kapanış cümleleri ekleme. Bunun yerine, konuşmayı doğal şekilde ileri götürecek, kullanıcının FİKRİNİ soran açık uçlu bir soru sor (örn. bahsettiğin filmin belirli bir sahnesi, karakteri veya oyuncusu hakkında ne düşündüğünü sor). Her mesajın kapanışı birbirinin aynısı olmasın, konuşmanın akışına özgü olsun.
13. CİNEMATCH ZEVK PROFİLİ KULLANIMI (ÇOK ÖNEMLİ): Aşağıda "CİNEMATCH ZEVK PROFİLİ" başlığı altında, kullanıcının Cinematch uygulamasında kendi seçtiği favori tür/oyuncu/yönetmen/film bilgisi varsa, bunu KESİN VE GÜVENİLİR kabul et (SQLite'daki KULLANICI PROFİLİ ile aynı güvenilirlikte). Kullanıcı film önerisi istediğinde ya da "ne izlesem", "bana bir şey öner" gibi genel bir istek yaptığında -başka bir tercih belirtmediği sürece- bu zevk profilini dikkate alarak öneri yap (örn. sevdiği türe/yönetmenine/oyuncusuna yakın ama daha önce izlemediği bir şey seç). Kullanıcı açıkça farklı bir tür/tarz isterse, o isteğe öncelik ver. Zevk profilini her mesajda ZORLA tekrar söyleme ("senin sevdiğin türler şunlar" diye listeleme); sadece öneri yaparken sessizce kullan, istisnai olarak kullanıcı doğrudan "ben neyi severim" gibi sorarsa bahset.
14. YETENEKLERİN HAKKINDA SORULARSA (ÇOK ÖNEMLİ): Kullanıcı "neler yapabilirsin", "ne işe yararsın", "nasıl kullanılırsın" gibi bir şey sorarsa, KISACA ve doğal bir dille şunları anlat (madde madde değil, akıcı cümlelerle):
   - Film/dizi önerisi yapabildiğini ve sinema hakkında sohbet edebildiğini,
   - Cinematch'teki favori tür/oyuncu/yönetmen/filmlerine bakarak ona ÖZEL öneriler sunabildiğini,
   - Kendisinin de sohbete bir film gönderip o film hakkında (yorum, benzerleri, karşılaştırma vb.) konuşabileceğini,
   - Önerdiğin filmleri, uygulama içinde doğrudan tıklanıp detayına gidilebilen film kartları olarak gönderdiğini,
   - CineMatch uygulamasının kendisiyle ilgili (profil, ayarlar, eşleşme, kulüpler vb.) sorularını da cevaplayabildiğini.
15. FİLM KARTI KURALI (ÇOK ÖNEMLİ, KESİNLİKLE UYULMALI): Kullanıcıya spesifik bir veya birkaç film ÖNERDİĞİNDE (izlemesini tavsiye ettiğinde), cevabının EN SONUNA, ayrı bir satırda ve başka HİÇBİR ŞEY eklemeden şu formatta bir blok koy:
[[FILMLER: Film Adı 1 | Film Adı 2]]
   - En fazla 3 film adı, "|" ile ayrılmış olarak.
   - Film adını TMDB'de aratıldığında bulunabilecek şekilde doğru, net ve tercihen orijinal/uluslararası adıyla yaz (örn. "Yaşamın Kıyısında" değil "Manchester by the Sea").
   - SADECE gerçekten ÖNERDİĞİN filmleri bu bloğa yaz; sohbette bahsi geçen ama önermediğin (örn. karşılaştırma için adı geçen) filmleri EKLEME.
   - Öneri yapmıyorsan (genel sohbet, soru cevaplama vb.) bu bloğu HİÇ ekleme.
   - Bu blok kullanıcıya gösterilmeyecek, uygulama tarafından ayrıştırılıp tıklanabilir film kartına çevrilecek; bu yüzden formatı BİREBİR koru ve blok İÇİNDE/ÖNCESİNDE/SONRASINDA başka açıklama yazma.
16. CİNEMATCH UYGULAMA REHBERİ KULLANIMI (ÇOK ÖNEMLİ): Aşağıda "CİNEMATCH UYGULAMA REHBERİ" başlığı altında, uygulamanın gerçek ekran/menü/buton isimleriyle yazılmış KESİN VE GÜNCEL bir rehber var. Kullanıcı "kullanıcı adımı nasıl değiştiririm", "bildirimleri nasıl kapatırım", "eşleşme nasıl çalışıyor", "kulüp nasıl kurarım" gibi uygulamanın kendisiyle ilgili bir şey sorduğunda:
   - SADECE bu rehberdeki bilgiyi kullan, uydurma menü/buton ismi söyleme.
   - Cevabını rehberdeki gerçek isimlerle, adım adım ve KISA ver (örn. "Profil sekmesi > Profili Düzenle > Temel Bilgiler bölümünden kullanıcı adını değiştirip Değişiklikleri Kaydet'e basman yeterli.").
   - Rehberde cevabı olmayan bir soru sorulursa uydurma; "Bu konuda kesin bilgim yok, uygulama içindeki Ayarlar > Destek Al bölümünden ekiple iletişime geçebilirsin" de.
   - Bu rehberi sadece uygulama hakkında bir soru geldiğinde kullan; film sohbetlerine zorla karıştırma.

KULLANICI PROFİLİ (kesin bilgiler):
{profile_context}

CİNEMATCH ZEVK PROFİLİ (uygulamadan gelen kesin veri):
{app_profile_context if app_profile_context else "Uygulamadan gelen kayıtlı bir zevk profili yok."}

CİNEMATCH UYGULAMA REHBERİ (uygulamanın kendisi hakkında sorular için kesin kaynak):
{CINEMATCH_APP_GUIDE}

KULLANICI GEÇMİŞİ (arka plan - eski oturumlar / bu oturumun eski kısmı):
{background_context}

ÖNCEDEN ÇEKİLMİŞ OMDb VERİSİ:
{prefetched_movie_data if prefetched_movie_data else "Bu mesaj için önceden çekilmiş OMDb verisi yok."}

Not: Bu oturumun SON mesajları, bu sistem talimatından sonra gerçek konuşma
turları (user/assistant) olarak sana ayrıca gösteriliyor; onlara normal bir
sohbetin doğal devamıymış gibi bak, aynı cümleleri tekrar etme.
"""

    # 4. MEVCUT OTURUMUN SON MESAJLARINI GERÇEK user/assistant TURLARI OLARAK EKLE.
    
    recent_turns = history.get("current_transcript", [])

    messages = [{"role": "system", "content": system_prompt}]
    for turn in recent_turns:
        messages.append({"role": "user", "content": turn["user_message"]})
        messages.append({"role": "assistant", "content": turn["bot_response"]})
    messages.append({"role": "user", "content": user_message})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_live_movie_data",
                "description": "Belirli bir filmin güncel IMDb puanını, çıkış yılını ve gişe hasılatını internetten çeker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "movie_name": {
                            "type": "string",
                            "description": "Hakkında bilgi istenen filmin adı (örneğin: The Matrix, Inception)"
                        }
                    },
                    "required": ["movie_name"]
                }
            }
        }
    ]

    final_answer = ""
    try:
        # Kullanıcı puan/gişe/yıl gibi somut bir veri istiyorsa, modele "istersen çağır" demek yerine aracı DOĞRUDAN ZORUNLU kılıyoruz. 
        forced_tool_choice = None
        if not prefetched_movie_data and wants_movie_facts(user_message):
            forced_tool_choice = {"type": "function", "function": {"name": "get_live_movie_data"}}
            print("SİSTEM: Kullanıcı somut film verisi istiyor -> get_live_movie_data ZORUNLU kılındı.")

        # OMDb verisi önceden çekildiyse aynı filmi modelin tekrar tool ile
        # sorgulamasını engelle; mevcut veri doğrudan system prompt'tadır.
        available_tools = None if prefetched_movie_data else tools
        response_data = _call_openrouter(messages, tools=available_tools, tool_choice=forced_tool_choice)

        if 'choices' not in response_data:
            error_info = response_data.get('error', {})
            error_code = error_info.get('code')
            print("OPENROUTER'DAN GELEN HATA:", response_data)

            if error_code == 429:
                # Rate-limit
                
                final_answer = ("Şu anda yoğunluk var, model geçici olarak yanıt veremiyor. "
                                 "Birkaç saniye sonra tekrar dener misin?")
            else:
                final_answer = "Üzgünüm, şu anda bir bağlantı sorunu yaşıyorum. Birazdan tekrar dener misin?"
        else:
            assistant_message = response_data['choices'][0]['message']

            if assistant_message.get("tool_calls"):
                tool_call = assistant_message["tool_calls"][0]
                function_name = tool_call["function"]["name"]

                if function_name == "get_live_movie_data":
                    arguments = json.loads(tool_call["function"]["arguments"])
                    movie_name = arguments.get("movie_name")

                    if not movie_name:
                        # Zorunlu tool_choice modeli aracı çağırmaya itti ama hangi film olduğunu çıkaramadı 
                        final_answer = "Hangi filmden bahsediyorsun, film adını yazabilir misin?"
                    else:
                        print(f"SİSTEM: Yapay zeka OMDb'den şu filmi çekiyor: {movie_name}")
                        function_result = get_live_movie_data(
                            movie_name, session_id=session_id, user_id=user_id, username=username
                        )

                        messages.append(assistant_message)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": function_name,
                            "content": function_result
                        })

                        second_response_data = _call_openrouter(messages, tools=tools)
                        final_answer = second_response_data['choices'][0]['message']['content']
                else:
                    final_answer = assistant_message.get('content') or ""
            else:
                final_answer = assistant_message['content']

    except Exception as e:
        final_answer = f"Üzgünüm, bir sistem hatası oluştu: {str(e)}"

    if not final_answer or final_answer.strip() == "":
        final_answer = "⚠️ Üzgünüm, yapay zeka modeli boş bir cevap döndürdü."

    # YENİ: Model film önerdiyse cevabın sonuna eklediği [[FILMLER: ...]]
    # bloğunu ayıklayıp görünür metni ve tıklanabilir film listesini ayır.
    final_answer, recommended_movies = extract_movie_recommendations(final_answer)

    # 5. TAM KONUŞMA DÖKÜMÜNE KAYDET 
    log_chat(session_id, user_id, username, user_message, final_answer)

    # 6. OTURUM SAYACINI GÜNCELLE
    message_count = touch_session(session_id)

    # 7. BELİRLİ ARALIKLARLA ÖZETİ GÜNCELLEYEN ARACI TETİKLE
    if message_count % SUMMARY_UPDATE_INTERVAL == 0:
        summarize_session(session_id)

    return final_answer, recommended_movies, session_id



# GÖREV 4: GÖRÜNTÜ (FOTOĞRAF) ANALİZİ



def analyze_image(image_base64, caption, user_id="anon", username="anon"):
    """image_base64: Telegram'dan indirilen fotoğrafın base64 (data URL'siz) hali.
    caption: kullanıcının fotoğrafla birlikte yazdığı yazı (yoksa None/boş olabilir)."""


    session_id = get_or_create_session(user_id, username)

    if caption:
        new_facts = extract_simple_facts(caption)
        if new_facts:
            update_user_facts(user_id, username, new_facts)

    user_facts = get_user_facts(user_id)
    profile_context = build_profile_context(user_facts)

    history = get_user_history(user_id, session_id)
    background_context = build_background_context(history)

    system_prompt = f"""Sen profesyonel bir sinema asistanı ve film eleştirmenisin. Sana bir FOTOĞRAF gönderildi.

GÖREVİN:
1. Fotoğrafın sinema ile İLGİLİ olup olmadığını değerlendir. Film afişi, film karesi/sahnesi,
   oyuncu/yönetmen fotoğrafı, sinema salonu, film seti, kamera/çekim ekipmanı, sinema bileti,
   DVD/Blu-ray kutusu gibi şeyler İLGİLİ sayılır.
2. İLGİLİYSE: Bir sinema eleştirmeni gibi yorum yap. Ne gördüğünü kısaca tarif et, mümkünse
   hangi film/sahne/oyuncu olabileceğini tahmin et (KESİN EMİN DEĞİLSEN iddialı konuşma,
   "...olabilir", "...anımsatıyor" gibi ifadeler kullan; uydurma bir film adı söyleme).
   Görsel/sinematografik bir yorum da ekle (kadraj, ışık, renk paleti, atmosfer gibi).
3. İLGİLİ DEĞİLSE (ör. kişisel bir selfie, yemek, manzara, hayvan fotoğrafı vb.): Kibarca bunun
   sinemayla ilgili görünmediğini belirt ve kullanıcıyı sinemayla ilgili bir şey sormaya/paylaşmaya
   yönlendir. Fotoğrafın içeriğini detaylıca tarif ETME, sadece nazikçe konu dışı olduğunu söyle.
4. DİL UYUMU: Kullanıcı fotoğrafla birlikte bir yazı (caption) gönderdiyse o dilde cevap ver;
   yazı yoksa Türkçe cevap ver.
5. TEKRAR YASAĞI ve DOĞAL DİYALOG kuralları burada da geçerli: kalıp cümle tekrar etme, robotik
   kapanış ("başka bir şey var mı?" gibi) kullanma.

KULLANICI PROFİLİ (kesin bilgiler):
{profile_context}

KULLANICI GEÇMİŞİ (arka plan):
{background_context}
"""

    user_text = caption if caption else "Bu fotoğrafı yorumlar mısın?"
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }
    ]

    final_answer = ""
    try:
        response_data = _call_openrouter(messages, model=VISION_MODEL_NAME)

        if 'choices' not in response_data:
            error_info = response_data.get('error', {})
            print("OPENROUTER'DAN GELEN HATA (görüntü):", response_data)
            if error_info.get('code') == 429:
                final_answer = ("Şu anda yoğunluk var, fotoğrafı analiz edemedim. "
                                 "Birkaç saniye sonra tekrar gönderir misin?")
            else:
                final_answer = "Üzgünüm, fotoğrafı analiz ederken bir sorun oluştu, tekrar dener misin?"
        else:
            final_answer = response_data['choices'][0]['message']['content']

    except Exception as e:
        final_answer = f"Üzgünüm, fotoğrafı analiz ederken bir sistem hatası oluştu: {str(e)}"

    if not final_answer or final_answer.strip() == "":
        final_answer = "⚠️ Üzgünüm, fotoğraf için boş bir cevap döndü, tekrar dener misin?"

    log_message = "[FOTOĞRAF]" + (f" - {caption}" if caption else "")
    log_chat(session_id, user_id, username, log_message, final_answer)

    message_count = touch_session(session_id)
    if message_count % SUMMARY_UPDATE_INTERVAL == 0:
        summarize_session(session_id)

    return final_answer
