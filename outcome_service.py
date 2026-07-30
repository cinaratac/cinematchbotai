"""CineBot konuşmaları için açıklanabilir intent/outcome sınıflandırması.

Bu modül dış servise veya veritabanına bağlı değildir. Böylece sınıflandırma
ana sohbet hattına ek gecikme/maliyet eklemez ve bir sağlayıcı hatasında da
çalışmaya devam eder.
"""

import re
import unicodedata


CLASSIFICATION_VERSION = "rules-v1"

INTENT_MOVIE_RECOMMENDATION = "film_onerisi_istendi"
INTENT_CATEGORY_MOVIES = "kategoriye_gore_film_arama"
INTENT_ACTOR_MOVIES = "oyuncuya_gore_film_arama"
INTENT_ACTOR_INFO = "oyuncu_bilgisi_soruldu"
INTENT_DIRECTOR_MOVIES = "yonetmene_gore_film_arama"
INTENT_DIRECTOR_INFO = "yonetmen_bilgisi_soruldu"
INTENT_MOVIE_INFO = "film_bilgisi_soruldu"
INTENT_CINEMATCH_HELP = "cinematch_destegi_istendi"
INTENT_IMAGE_ANALYSIS = "gorsel_analizi_istendi"
INTENT_CINEMA_CHAT = "sinema_sohbeti"
INTENT_GREETING = "selamlasma"
INTENT_UNRELATED = "alakasiz_sohbet"
INTENT_UNCLEAR = "anlasilamayan_istek"

OUTCOME_SUCCESS = "islem_basarili"
OUTCOME_FALLBACK = "anlasilamadi_fallback"
OUTCOME_TECHNICAL_ERROR = "teknik_hata"
OUTCOME_PARTIAL_SUCCESS = "kismi_basarili"
OUTCOME_NOT_FOUND = "veri_bulunamadi"
OUTCOME_OUT_OF_SCOPE = "kapsam_disi_yonlendirildi"

VALID_INTENTS = {
    INTENT_MOVIE_RECOMMENDATION,
    INTENT_CATEGORY_MOVIES,
    INTENT_ACTOR_MOVIES,
    INTENT_ACTOR_INFO,
    INTENT_DIRECTOR_MOVIES,
    INTENT_DIRECTOR_INFO,
    INTENT_MOVIE_INFO,
    INTENT_CINEMATCH_HELP,
    INTENT_IMAGE_ANALYSIS,
    INTENT_CINEMA_CHAT,
    INTENT_GREETING,
    INTENT_UNRELATED,
    INTENT_UNCLEAR,
}

VALID_OUTCOMES = {
    OUTCOME_SUCCESS,
    OUTCOME_FALLBACK,
    OUTCOME_TECHNICAL_ERROR,
    OUTCOME_PARTIAL_SUCCESS,
    OUTCOME_NOT_FOUND,
    OUTCOME_OUT_OF_SCOPE,
}


def _normalize(value):
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.translate(str.maketrans({"ı": "i", "æ": "ae", "ø": "o"}))
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def _contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


def _looks_like_named_person_filmography(original_text):
    """``Tom Hanks filmleri`` gibi açık kişi filmografisi kalıplarını bulur."""
    return bool(re.search(
        r"\b[A-ZÇĞİÖŞÜ][\wçğıöşüÇĞİÖŞÜ.-]+"
        r"(?:\s+[A-ZÇĞİÖŞÜ][\wçğıöşüÇĞİÖŞÜ.-]+)+"
        r"(?:['’](?:in|ın|un|ün|nin|nın|nun|nün))?"
        r"\s+(?:filmi|filmler(?:i|ini)?|filmografisi)\b",
        str(original_text or ""),
    ))


def _contains_named_person(original_text):
    return bool(re.search(
        r"\b[A-ZÇĞİÖŞÜ][\wçğıöşüÇĞİÖŞÜ.-]+"
        r"(?:\s+[A-ZÇĞİÖŞÜ][\wçğıöşüÇĞİÖŞÜ.-]+)+\b",
        str(original_text or ""),
    ))


def _classify_intent(user_message, bot_response, input_type, recommended_movies):
    text = _normalize(user_message)
    response = _normalize(bot_response)
    original = str(user_message or "").strip()
    input_kind = _normalize(input_type)

    if input_kind in {"photo", "image", "gorsel"} or original.startswith("[FOTOĞRAF]"):
        return INTENT_IMAGE_ANALYSIS, 1.0, "input_type:image"

    app_terms = {
        "cinematch", "uygulama", "profil", "kullanici adi", "bildirim",
        "ayarlar", "eslesme", "sinefiller", "kulup", "letterboxd",
        "izleme listesi", "favoriler", "rozet", "liderlik", "trivia",
        "hesabi sil", "sifre", "engellenen kullanici",
        "username", "profile", "notification", "settings", "watchlist",
        "delete account", "password", "blocked user",
    }
    if _contains_any(text, app_terms):
        return INTENT_CINEMATCH_HELP, 0.96, "keyword:cinematch_help"

    actor_terms = {
        "oyuncu", "aktor", "aktris", "cast", "kim oynuyor", "kimler oynuyor",
        "hangi filmlerde oynadi", "oynadigi filmler", "filmografisi",
        "actor", "actress", "starring", "who stars", "quien actua",
        "actriz",
    }
    director_terms = {
        "yonetmen", "kim yonetti", "yonettigi filmler", "yonetmenligini",
        "directed by", "director", "quien dirigio", "director de",
    }
    film_search_terms = {
        "filmler", "filmografi", "film oner", "dizi oner",
        "film tavsiye", "dizi tavsiye",
        "ne izlesem", "ne izleyeyim", "bana bir film", "bana film",
        "benzer film", "recommend", "what should i watch", "movies",
        "films", "filmography", "peliculas", "filmografia",
    }

    if _contains_any(text, actor_terms):
        if _contains_any(text, film_search_terms):
            return INTENT_ACTOR_MOVIES, 0.94, "keyword:actor_film_search"
        return INTENT_ACTOR_INFO, 0.91, "keyword:actor"

    if _contains_any(text, director_terms):
        if _contains_any(text, film_search_terms):
            return INTENT_DIRECTOR_MOVIES, 0.94, "keyword:director_film_search"
        return INTENT_DIRECTOR_INFO, 0.91, "keyword:director"

    if _looks_like_named_person_filmography(original):
        return INTENT_ACTOR_MOVIES, 0.72, "pattern:named_person_filmography"

    if (
        _contains_named_person(original)
        and _contains_any(text, {"kimdir", "hakkinda bilgi", "hayati"})
    ):
        if _contains_any(response, {"yonetmen", "director"}):
            return INTENT_DIRECTOR_INFO, 0.74, "response:named_director_info"
        return INTENT_ACTOR_INFO, 0.68, "pattern:named_person_info"

    movie_fact_terms = {
        "imdb", "puan", "gise", "hasilat", "box office", "cikis yili",
        "hangi yil", "ne zaman cikti", "vizyon tarihi", "konusu ne",
        "hakkinda", "ozeti", "incele", "yorumla", "elestir", "review",
        "rating", "release year", "release date", "plot", "synopsis",
        "resena", "argumento", "fecha de estreno",
    }
    if _contains_any(text, movie_fact_terms):
        return INTENT_MOVIE_INFO, 0.88, "keyword:movie_info"

    recommendation_terms = {
        "oner", "tavsiye", "ne izlesem", "ne izleyeyim", "izlemelik",
        "film ariyorum", "dizi ariyorum", "benzer film", "recommend",
        "what should i watch", "suggest a movie", "recomienda",
        "que pelicula veo", "que puedo ver",
    }
    category_terms = {
        "aksiyon", "dram", "komedi", "korku", "gerilim", "bilim kurgu",
        "fantastik", "romantik", "animasyon", "belgesel", "suc", "gizem",
        "macera", "aile", "savas", "western", "muzikal", "uzay", "ask",
        "polisiye", "biyografi", "spor", "tarihi",
        "action", "drama", "comedy", "horror", "thriller", "science fiction",
        "fantasy", "romance", "documentary", "mystery", "adventure",
        "accion", "comedia", "terror", "ciencia ficcion", "romance",
    }
    category_film_pattern = bool(re.search(
        r"\b.+\s+(?:filmi|filmleri|dizisi|dizileri)\s+"
        r"(?:oner|tavsiye)",
        text,
    ))
    if (
        _contains_any(text, recommendation_terms)
        and (
            _contains_any(text, category_terms)
            or category_film_pattern
        )
    ):
        return INTENT_CATEGORY_MOVIES, 0.91, "signal:category_film_search"

    if recommended_movies or _contains_any(text, recommendation_terms):
        return INTENT_MOVIE_RECOMMENDATION, 0.93, "signal:movie_recommendation"

    greeting_terms = {
        "merhaba", "selam", "selamlar", "gunaydin", "iyi aksamlar",
        "iyi geceler", "hello", "hi", "hey", "start",
    }
    if text.strip(" !?.,") in greeting_terms:
        return INTENT_GREETING, 0.98, "exact:greeting"

    out_of_scope_signals = {
        "ben bir sinema asistaniyim",
        "sadece filmler diziler ile cinematch",
        "yalnizca sinema ve cinematch",
        "only help with movies",
        "solo puedo ayudar con peliculas",
    }
    if _contains_any(response, out_of_scope_signals):
        return INTENT_UNRELATED, 0.97, "response:out_of_scope"

    cinema_terms = {
        "film", "dizi", "sinema", "sahne", "karakter", "senaryo",
        "fragman", "oscar", "festival", "vizyon", "belgesel", "anime",
        "movie", "series", "cinema", "scene", "character", "screenplay",
        "trailer", "pelicula", "serie", "cine", "escena", "personaje",
    }
    if _contains_any(text, cinema_terms):
        return INTENT_CINEMA_CHAT, 0.78, "keyword:cinema"

    if not text or len(text.strip()) <= 2:
        return INTENT_UNCLEAR, 0.9, "input:empty_or_too_short"

    return INTENT_UNRELATED, 0.58, "default:outside_known_domains"


def _classify_outcome(
    bot_response,
    outcome_hint,
    error_stage,
    error_type,
    tool_calls,
):
    if outcome_hint in VALID_OUTCOMES:
        return outcome_hint, 1.0, "explicit:outcome_hint"

    if error_stage or error_type:
        return OUTCOME_TECHNICAL_ERROR, 1.0, "signal:technical_error"

    response = _normalize(bot_response)
    if not response.strip():
        return OUTCOME_FALLBACK, 1.0, "response:empty"

    tool_statuses = {
        item.get("status")
        for item in (tool_calls or [])
        if isinstance(item, dict)
    }
    if "error" in tool_statuses:
        return OUTCOME_PARTIAL_SUCCESS, 0.98, "tool:error_with_response"
    if (
        "not_found" in tool_statuses
        and _contains_any(response, {"bulunamadi", "bulamadim", "not found"})
    ):
        return OUTCOME_NOT_FOUND, 0.98, "tool:not_found"

    technical_signals = {
        "bir sistem hatasi olustu", "baglanti sorunu yasiyorum",
        "model gecici olarak yanit veremiyor", "servisi http",
        "fotografi analiz ederken bir sorun olustu",
        "yapay zeka modeli bos bir cevap dondurdu",
    }
    if _contains_any(response, technical_signals):
        return OUTCOME_TECHNICAL_ERROR, 0.94, "response:technical_fallback"

    clarification_signals = {
        "tekrar eder misin", "tekrar dener misin", "film adini yazabilir misin",
        "hangi filmden bahsediyorsun", "anlayamadim", "anlasilmadi",
        "could you clarify", "which movie",
        "puedes aclarar", "que pelicula",
    }
    if _contains_any(response, clarification_signals):
        return OUTCOME_FALLBACK, 0.88, "response:clarification"

    out_of_scope_signals = {
        "ben bir sinema asistaniyim",
        "sadece filmler diziler ile cinematch",
        "yalnizca sinema ve cinematch",
        "only help with movies",
        "solo puedo ayudar con peliculas",
    }
    if _contains_any(response, out_of_scope_signals):
        return OUTCOME_OUT_OF_SCOPE, 0.97, "response:out_of_scope"

    return OUTCOME_SUCCESS, 0.9, "default:completed_response"


def categorize_interaction(
    user_message,
    bot_response,
    *,
    input_type="text",
    recommended_movies=None,
    outcome_hint=None,
    error_stage=None,
    error_type=None,
    tool_calls=None,
):
    """Tek bir kullanıcı-bot turunun sürümlenmiş sınıflandırmasını döndürür."""
    intent, intent_confidence, intent_reason = _classify_intent(
        user_message,
        bot_response,
        input_type,
        recommended_movies or [],
    )
    outcome, outcome_confidence, outcome_reason = _classify_outcome(
        bot_response,
        outcome_hint,
        error_stage,
        error_type,
        tool_calls or [],
    )

    if outcome == OUTCOME_FALLBACK and intent == INTENT_UNRELATED:
        intent = INTENT_UNCLEAR
        intent_confidence = max(intent_confidence, 0.82)
        intent_reason = "outcome:unrecognized_fallback"

    return {
        "intent": intent,
        "outcome": outcome,
        "intent_confidence": round(intent_confidence, 2),
        "outcome_confidence": round(outcome_confidence, 2),
        "classification_version": CLASSIFICATION_VERSION,
        "classification_method": "deterministic_rules",
        "classification_reason": {
            "intent": intent_reason,
            "outcome": outcome_reason,
        },
    }
