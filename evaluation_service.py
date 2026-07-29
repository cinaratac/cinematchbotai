import asyncio
import copy
import hashlib
import json
import os
import re
import unicodedata

import aiohttp

from database import (
    complete_voice_ai_evaluation,
    download_voice_recording_audio,
    fail_voice_ai_evaluation,
    get_performance_metrics_averages,
    get_recording_transcript,
    get_session_transcript,
    skip_voice_ai_evaluation,
    start_voice_ai_evaluation,
)
from voice.activity import wait_for_voice_idle



DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EVALUATION_MODEL = os.environ.get(
    "VOICE_EVALUATION_MODEL", "gpt-4o-mini"
)
if not EVALUATION_MODEL.startswith("gpt-"):
    print(
        "VOICE_EVALUATION_MODEL open_ai sağlayıcısıyla uyumsuz; "
        "gpt-4o-mini kullanılacak:",
        EVALUATION_MODEL,
    )
    EVALUATION_MODEL = "gpt-4o-mini"
EVALUATION_TASKS = set()
QA_EVALUATOR_VERSION = "openai-responses-json-schema-v1"
STT_REFERENCE_VERSION = "deepgram-nova-3-tr"
TRANSCRIPT_METRIC_VERSION = "wer-token-v1"

CRITERIA = {
    "speech_recognition_accuracy": (
        "Gerçek kullanıcı sesinin agent transkriptine doğru aktarılması"
    ),
    "intent_understanding": "Kullanıcının niyetini doğru anlama",
    "answer_relevance": "Cevabın soruyla doğrudan ilgili olması",
    "accuracy": "Olgusal doğruluk ve halüsinasyondan kaçınma",
    "naturalness": "Türkçe konuşmanın doğallığı",
    "conciseness": "Gereksiz tekrar ve uzunluktan kaçınma",
    "task_completion": "Kullanıcının hedefini tamamlama",
    "context_retention": "Önceki konuşma bağlamını doğru kullanma",
    "barge_in_handling": "Kesilme sonrası yeni isteğe odaklanma",
    "tool_usage": "Araç kullanımının gerekliliği ve doğruluğu",
    "safety": "Güvenli, şeffaf ve uygun yanıt verme",
}

EVALUATOR_PROMPT = """
Sen CineMatch Voice Agent görüşmelerini denetleyen bağımsız QA agentısın.
Kullanıcı ses kaydı bağımsız olarak yeniden yazıya çevrilmiştir. Sana bu
audio_transcript ile canlı voice agentın ürettiği logged_transcript ve
performans özeti verilir. Önce iki transkripti karşılaştır; yanlış anlaşılan,
atlanmış veya anlamı değiştirilmiş ifadeleri belirle. Sonra agent cevaplarının
gerçekte söylenenlere uygunluğunu değerlendir. Her kriteri 0-10 arasında
puanla. Gözlenemeyen kriter için score=null ve observed=false kullan.
Veri uydurma. Belirgin konuşma tanıma hataları varsa bunları issue olarak
raporla ve overall_score değerini gereksiz şekilde yüksek verme.

İki transkript arasındaki yalnızca yazım, noktalama, ek, çekim veya aynı anlamı
taşıyan eş anlamlı ifade farklarını speech_recognition_error sayma. Örneğin
"iletişime geçmek" ve "iletişim kurmak" tek başına anlam kaybı değildir.
Hata ancak kullanıcının niyeti, varlık adı, sayı, olumsuzluk, tercih veya
cevabın yönü değişiyorsa raporlanmalıdır. Kanıt alanında iki taraftaki gerçek
ifadeyi aynen göster; transkriptte bulunmayan örnek uydurma.
logged_transcript içindeki "user" alanı canlı STT'nin kullanıcıdan anladığı
metindir; "assistant" alanı agent cevabıdır. speech_recognition_error için
assistant cevabını, agentın duyduğu kullanıcı metni gibi kullanma. Bu kriterde
yalnızca audio_utterances ile "user" alanlarını karşılaştır.
deterministic_match_score kodla hesaplanmıştır; match_score alanına bu değeri
aynen yaz ve kendi tahmininle değiştirme.

PUAN KALİBRASYONU:
- 9-10: Tam veya tama yakın başarı.
- 7-8: Genel olarak başarılı, sınırlı ve düşük etkili sorunlar var.
- 5-6: Birden fazla belirgin sorun var ama görüşme kısmen işe yarıyor.
- 0-4: Ciddi/tekrarlayan başarısızlık veya hedefin gerçekleştirilememesi.
Tek bir hatayı ilgililik, niyet, doğruluk, bağlam ve görev tamamlama altında
kanıtsız biçimde beş kez cezalandırma. Her observed=true kriterinin reason
alanında ilgili turn_index ve somut ifade bulunmalıdır; somut kanıt yoksa
observed=false kullan. "accuracy" yalnızca yanlış olgusal iddia veya
halüsinasyon içindir; STT/niyet hatasını accuracy kriterine tekrar yazma.
Doğal ve akıcı gerçek assistant metni varsa naturalness puanını genel bir
"doğal değildi" cümlesiyle düşürme. overall_score gözlenen kriterlerle ve
sorunların gerçek kullanıcı etkisiyle tutarlı olmalıdır.

Sonucu serbest metin olarak yazma. Structured Output JSON'unu aşağıdaki
şemaya göre doldur:
{
  "overall_score": 0-100,
  "summary": "kısa yönetici özeti",
  "criteria": {
    "<criterion>": {
      "score": 0-10 veya null,
      "observed": true veya false,
      "reason": "kanıta dayalı kısa gerekçe"
    }
  },
  "issues": [
    {
      "type": "snake_case",
      "severity": "low|medium|high|critical",
      "evidence": "transkriptten kısa kanıt",
      "recommendation": "somut düzeltme"
    }
  ],
  "strengths": ["kanıta dayalı güçlü yön"],
  "prompt_recommendations": [
    {
      "priority": "low|medium|high",
      "problem": "tespit edilen tekrar eden veya önemli sorun",
      "suggested_instruction": "sistem promptuna eklenebilecek net talimat",
      "expected_effect": "beklenen sonuç",
      "evidence": "bu görüşmedeki dayanak"
    }
  ],
  "transcript_comparison": {
    "match_score": 0-100,
    "summary": "ses temelli ve agent transkriptinin karşılaştırması",
    "mismatches": [
      {
        "audio_says": "bağımsız STT sonucu",
        "agent_understood": "canlı agent transkripti",
        "impact": "yanlış anlamanın cevaba etkisi"
      }
    ]
  },
  "turns": [
    {
      "turn_index": 1,
      "notable": true,
      "issues": ["bu tura özgü kanıta dayalı sorun"]
    }
  ]
}

Prompt önerisi yalnızca transkriptte gerçek bir problem varsa üret. Sistem
promptunu otomatik değiştirme. Barge-in ve tool kullanımı transkriptten
gözlenemiyorsa bunları puanlama. "Daha fazla eğitim verisi kullan", "modeli
eğit" veya "STT sistemini güncelle" gibi bu uygulamada doğrudan yapılamayan
genel öneriler verme. Öneriler yalnızca sistem promptuna eklenecek somut bir
talimat veya değiştirilebilecek voice/STT konfigürasyonu olmalıdır. Sorun
prompt ile düzeltilemezse prompt_recommendations alanına ekleme; issues içinde
teknik konfigürasyon önerisi olarak belirt.
Her logged_transcript turu için aynı turn_index ile turns dizisinde bir kayıt
oluştur. Sorunsuz turda issues boş liste ve notable=false olmalıdır.
Barge-in metriği varsa barge_in_handling kriterini bu ölçüme dayanarak puanla;
yoksa gözlenemedi olarak bırak.
""".strip()


def _criterion_function_schema():
    return {
        "type": "object",
        "properties": {
            "score": {
                "anyOf": [
                    {"type": "number"},
                    {"type": "null"},
                ],
                "description": "Gözlenebiliyorsa 0-10 puan.",
            },
            "observed": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["score", "observed", "reason"],
    }


EVALUATION_FUNCTION = {
    "name": "submit_evaluation",
    "description": (
        "CineMatch görüşmesinin yapılandırılmış kalite raporunu teslim eder. "
        "Her değerlendirmede bu fonksiyon tam bir kez çağrılmalıdır."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "overall_score": {"type": "number"},
            "summary": {"type": "string"},
            "criteria": {
                "type": "object",
                "properties": {
                    key: _criterion_function_schema() for key in CRITERIA
                },
                "required": list(CRITERIA),
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                        },
                        "evidence": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                    "required": [
                        "type", "severity", "evidence", "recommendation"
                    ],
                },
            },
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
            },
            "prompt_recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                        "problem": {"type": "string"},
                        "suggested_instruction": {"type": "string"},
                        "expected_effect": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": [
                        "priority",
                        "problem",
                        "suggested_instruction",
                        "expected_effect",
                        "evidence",
                    ],
                },
            },
            "transcript_comparison": {
                "type": "object",
                "properties": {
                    "match_score": {"type": "number"},
                    "summary": {"type": "string"},
                    "mismatches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "audio_says": {"type": "string"},
                                "agent_understood": {"type": "string"},
                                "impact": {"type": "string"},
                            },
                            "required": [
                                "audio_says",
                                "agent_understood",
                                "impact",
                            ],
                        },
                    },
                },
                "required": ["match_score", "summary", "mismatches"],
            },
            "turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "turn_index": {"type": "number"},
                        "notable": {"type": "boolean"},
                        "issues": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["turn_index", "notable", "issues"],
                },
            },
        },
        "required": [
            "overall_score",
            "summary",
            "criteria",
            "issues",
            "strengths",
            "prompt_recommendations",
            "transcript_comparison",
            "turns",
        ],
    },
}


def _strict_json_schema(schema):
    """Responses Structured Outputs için şemayı strict uyumlu hale getir."""
    value = copy.deepcopy(schema)
    if value.get("type") == "object":
        properties = value.get("properties", {})
        value["additionalProperties"] = False
        value["required"] = list(properties)
        value["properties"] = {
            key: _strict_json_schema(item)
            for key, item in properties.items()
        }
    if value.get("type") == "array" and isinstance(value.get("items"), dict):
        value["items"] = _strict_json_schema(value["items"])
    if isinstance(value.get("anyOf"), list):
        value["anyOf"] = [
            _strict_json_schema(item) for item in value["anyOf"]
        ]
    return value


EVALUATION_JSON_SCHEMA = _strict_json_schema(
    EVALUATION_FUNCTION["parameters"]
)
QA_PROMPT_VERSION = hashlib.sha256(
    EVALUATOR_PROMPT.encode("utf-8")
).hexdigest()[:12]


def _extract_json(text):
    clean = str(text or "").strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    start = clean.find("{")
    end = clean.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Değerlendirme agentı geçerli JSON döndürmedi.")
    return json.loads(clean[start:end + 1])


def _normalize_transcript(text):
    normalized = unicodedata.normalize("NFKC", str(text or "").casefold())
    normalized = re.sub(r"[^a-zçğıöşü0-9 ]+", " ", normalized)
    return " ".join(normalized.split())


def _transcript_wer(audio_text, logged_user_texts):
    """İki STT çıktısındaki token düzeyi edit mesafesini ölç.

    Bağımsız STT insan doğrulamalı ground truth değildir; bu değer iki motor
    arasındaki anlaşmazlığı ölçer. WER = (S + D + I) / referans_token_sayısı.
    """
    audio_words = _normalize_transcript(audio_text).split()
    logged_words = _normalize_transcript(
        " ".join(str(text or "") for text in logged_user_texts)
    ).split()
    if not audio_words:
        return {
            "reference_word_count": 0,
            "hypothesis_word_count": len(logged_words),
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(logged_words),
            "wer": None,
            "match_score": 0,
        }

    # Hücre: (toplam hata, substitution, deletion, insertion). Eşitlikte
    # substitution yerine deletion/insertion tercih edilmez; hata türü izlenir.
    matrix = [[(0, 0, 0, 0)] * (len(logged_words) + 1)
              for _ in range(len(audio_words) + 1)]
    for index in range(1, len(audio_words) + 1):
        matrix[index][0] = (index, 0, index, 0)
    for index in range(1, len(logged_words) + 1):
        matrix[0][index] = (index, 0, 0, index)
    for audio_index, audio_word in enumerate(audio_words, start=1):
        for logged_index, logged_word in enumerate(logged_words, start=1):
            if audio_word == logged_word:
                matrix[audio_index][logged_index] = matrix[
                    audio_index - 1
                ][logged_index - 1]
                continue
            previous = matrix[audio_index - 1][logged_index - 1]
            substitute = (previous[0] + 1, previous[1] + 1, previous[2], previous[3])
            previous = matrix[audio_index - 1][logged_index]
            delete = (previous[0] + 1, previous[1], previous[2] + 1, previous[3])
            previous = matrix[audio_index][logged_index - 1]
            insert = (previous[0] + 1, previous[1], previous[2], previous[3] + 1)
            matrix[audio_index][logged_index] = min(
                substitute, delete, insert, key=lambda item: item[0]
            )
    _, substitutions, deletions, insertions = matrix[-1][-1]
    wer = (substitutions + deletions + insertions) / len(audio_words)
    return {
        "reference_word_count": len(audio_words),
        "hypothesis_word_count": len(logged_words),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "wer": round(wer, 4),
        "match_score": round(max(0, 1 - wer) * 100),
    }


def _contains_quoted_evidence(source_text, claimed_text):
    source = _normalize_transcript(source_text)
    claimed = _normalize_transcript(claimed_text)
    return bool(claimed and len(claimed.split()) >= 2 and claimed in source)


def _has_evidence_overlap(source_text, evidence):
    """Serbest kanıtta kaynak dökümden en az bir gerçek iki kelimelik parça ara."""
    source_words = _normalize_transcript(source_text).split()
    evidence_words = _normalize_transcript(evidence).split()
    if len(evidence_words) < 2:
        return False
    source_pairs = set(zip(source_words, source_words[1:]))
    return any(
        pair in source_pairs
        for pair in zip(evidence_words, evidence_words[1:])
    )


def _validate_qa_evidence(result, audio_text, turns):
    """LLM'nin transkriptte bulunmayan kanıtlarla rapor üretmesini reddet."""
    if not isinstance(result, dict):
        raise ValueError("QA sonucu nesne değil.")
    logged_users = " ".join(
        str(turn.get("user") or "") for turn in turns
    )
    all_transcript = " ".join([
        audio_text,
        logged_users,
        *(
            str(turn.get("assistant") or "")
            for turn in turns
        ),
    ])
    comparison = result.get("transcript_comparison") or {}
    mismatches = comparison.get("mismatches") or []
    valid_mismatch_count = 0
    for mismatch in mismatches:
        if not isinstance(mismatch, dict):
            raise ValueError("QA uyuşmazlık kanıtı nesne değil.")
        audio_says = mismatch.get("audio_says")
        agent_understood = mismatch.get("agent_understood")
        if not (
            _contains_quoted_evidence(audio_text, audio_says)
            and _contains_quoted_evidence(logged_users, agent_understood)
        ):
            raise ValueError(
                "QA transkriptte bulunmayan STT uyuşmazlığı üretti: "
                f"audio={audio_says!r}, logged={agent_understood!r}"
            )
        valid_mismatch_count += 1
    for issue in result.get("issues") or []:
        if not isinstance(issue, dict):
            raise ValueError("QA issue kaydı nesne değil.")
        if not _has_evidence_overlap(all_transcript, issue.get("evidence")):
            raise ValueError(
                "QA issue kanıtı konuşma dökümünde bulunamadı: "
                f"{issue.get('evidence')!r}"
            )
        if (
            issue.get("type") == "speech_recognition_error"
            and valid_mismatch_count == 0
        ):
            raise ValueError(
                "QA doğrulanmış uyuşmazlık olmadan STT hatası üretti."
            )
    return result


def _validate_result(result):
    if not isinstance(result, dict):
        raise ValueError("Değerlendirme sonucu nesne değil.")
    overall = result.get("overall_score")
    if not isinstance(overall, (int, float)) or not 0 <= overall <= 100:
        raise ValueError("overall_score 0-100 aralığında değil.")
    criteria = result.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("criteria alanı eksik.")
    comparison = result.get("transcript_comparison") or {}
    match_score = comparison.get("match_score")
    if not isinstance(match_score, (int, float)):
        raise ValueError("transcript_comparison.match_score eksik.")
    comparison["match_score"] = max(0, min(100, match_score))
    comparison["summary"] = str(comparison.get("summary") or "")[:2000]
    comparison["mismatches"] = list(
        comparison.get("mismatches") or []
    )[:30]
    result["transcript_comparison"] = comparison
    normalized = {}
    for key in CRITERIA:
        item = criteria.get(key) or {}
        score = item.get("score")
        observed = item.get("observed") is True
        if key == "speech_recognition_accuracy":
            # Bu kriterin nesnel kaynağı iki transkriptin eşleşme yüzdesidir.
            # LLM score alanını atlasa bile tüm raporu kaybetme.
            score = round(comparison["match_score"] / 10, 1)
            observed = True
        if not observed:
            score = None
        elif not isinstance(score, (int, float)) or not 0 <= score <= 10:
            # Tek bir eksik opsiyonel kriter tüm değerlendirmeyi bozmasın.
            score = None
            observed = False
        normalized[key] = {
            "label": CRITERIA[key],
            "score": score,
            "observed": observed,
            "reason": str(item.get("reason") or "")[:600],
        }
    result["criteria"] = normalized
    result["summary"] = str(result.get("summary") or "")[:2000]
    result["issues"] = list(result.get("issues") or [])[:20]
    result["strengths"] = list(result.get("strengths") or [])[:20]
    result["prompt_recommendations"] = list(
        result.get("prompt_recommendations") or []
    )[:15]
    result["turns"] = list(result.get("turns") or [])[:50]
    # Model, diğer kriterler iyi diye ciddi STT hatalarını örtemesin.
    # Ses-transkript eşleşmesi düşükse toplam kalite puanına deterministik tavan koy.
    if comparison["match_score"] < 60:
        result["overall_score"] = min(result["overall_score"], 60)
    elif comparison["match_score"] < 80:
        result["overall_score"] = min(result["overall_score"], 75)
    return result


async def _transcribe_recording(recording_id):
    """Firebase'deki gerçek kullanıcı WAV kaydını bağımsız olarak yazıya çevir."""
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY tanımlı değil.")
    audio_bytes = await asyncio.to_thread(
        download_voice_recording_audio,
        recording_id,
        "user",
    )
    if not audio_bytes:
        raise RuntimeError("Kullanıcı ses kaydı boş.")
    timeout = aiohttp.ClientTimeout(total=180)
    params = {
        "model": "nova-3",
        "language": "tr",
        "smart_format": "true",
        "utterances": "true",
        "punctuate": "true",
    }
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            },
            data=audio_bytes,
        ) as response:
            body = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError(
                    f"Ses kaydı STT hatası ({response.status}): {body}"
                )
    alternatives = (
        body.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [])
    )
    transcript = (
        str(alternatives[0].get("transcript") or "").strip()
        if alternatives
        else ""
    )
    if not transcript:
        raise RuntimeError("Ses kaydından bağımsız transkript çıkarılamadı.")
    print(
        "Voice QA ses kaydı yeniden transkript edildi:",
        recording_id,
        f"chars={len(transcript)}",
    )
    utterances = [
        str(item.get("transcript") or "").strip()
        for item in body.get("results", {}).get("utterances", [])
        if str(item.get("transcript") or "").strip()
    ]
    return {
        "transcript": transcript,
        "utterances": utterances or [transcript],
    }


def _response_output_json(response):
    """Responses API'nin çıktı bloklarından Structured Output JSON'unu al."""
    if response.get("status") != "completed":
        detail = response.get("error") or response.get("incomplete_details")
        raise RuntimeError(f"OpenAI QA isteği tamamlanmadı: {detail}")
    for output in response.get("output") or []:
        for content in output.get("content") or []:
            if content.get("type") == "refusal":
                raise RuntimeError(
                    f"OpenAI QA isteği reddedildi: {content.get('refusal')}"
                )
            if content.get("type") == "output_text":
                return json.loads(content.get("text") or "")
    raise RuntimeError("OpenAI QA isteği Structured Output döndürmedi.")


async def _call_openai_evaluator_once(payload, model):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY tanımlı değil.")
    serialized_payload = json.dumps(payload, ensure_ascii=False)
    request_body = {
        "model": model,
        "instructions": EVALUATOR_PROMPT,
        "input": serialized_payload,
        "temperature": 0.1,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "voice_qa_evaluation",
                "strict": True,
                "schema": EVALUATION_JSON_SCHEMA,
            },
        },
    }
    print(
        "Voice QA OpenAI isteği:",
        f"model={model}",
        f"prompt_version={QA_PROMPT_VERSION}",
        f"payload_chars={len(serialized_payload)}",
    )
    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=request_body,
        ) as response:
            body = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError(
                    f"OpenAI QA hatası ({response.status}): {body}"
                )
    return _response_output_json(body)


async def _call_openai_evaluator(payload, model, attempts=3):
    """Geçici REST/provider hatalarında üstel geri çekilmeli yeniden deneme."""
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            return await _call_openai_evaluator_once(payload, model)
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
            errors.append(str(exc))
            detail = str(exc).lower()
            transient = (
                isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))
                or any(marker in detail for marker in (
                    "timeout",
                    "timed out",
                    "internal_server_error",
                    "failed_to_think",
                    "connection",
                    "disconnect",
                    "502",
                    "503",
                    "504",
                ))
            )
            if not transient:
                raise
            if attempt >= attempts:
                break
            delay = 2 ** (attempt - 1)
            print(
                "Voice QA geçici hata; yeniden denenecek:",
                f"attempt={attempt}/{attempts}",
                f"delay={delay}s",
                repr(exc),
            )
            await asyncio.sleep(delay)
    raise RuntimeError(
        f"QA {attempts} denemede tamamlanamadı: " + " | ".join(errors)
    )


async def evaluate_voice_session(session_id, recording_id):
    model_chain = list(dict.fromkeys([EVALUATION_MODEL, "gpt-4o"]))
    try:
        print(
            "Voice AI değerlendirmesi canlı görüşmelerin bitmesini bekliyor:",
            recording_id,
        )
        await wait_for_voice_idle(grace_seconds=15)
        await asyncio.to_thread(
            start_voice_ai_evaluation,
            session_id,
            recording_id,
            " -> ".join(f"openai/{model}" for model in model_chain),
        )
        transcript = await asyncio.to_thread(
            get_recording_transcript, recording_id, 100
        )
        # Eski kayıtlar recording_id alanı eklenmeden oluşturulmuş olabilir.
        # Onlar için geriye dönük olarak oturumun son turlarını kullan.
        if not transcript:
            transcript = await asyncio.to_thread(
                get_session_transcript, session_id, 20
            )
        if not transcript:
            await asyncio.to_thread(
                skip_voice_ai_evaluation,
                recording_id,
                "Görüşmede değerlendirilecek transkript oluşmadı.",
            )
            print(
                "Voice AI değerlendirmesi atlandı; transkript yok:",
                recording_id,
            )
            return
        audio_result = await _transcribe_recording(recording_id)
        audio_transcript = audio_result["transcript"]
        performance = await asyncio.to_thread(
            get_performance_metrics_averages,
            200,
            session_id,
            recording_id,
        )
        # QA agentına yalnızca son 20 turu gönder. Tüm performans trend
        # noktalarını ve yüzlerce eski turu göndermek managed LLM bağlamını
        # gereksiz büyütüp think çağrısının reddedilmesine yol açabiliyor.
        turns = [
            {
                "turn_index": index,
                "user": row.get("user_message"),
                "assistant": row.get("bot_response"),
            }
            for index, row in enumerate(transcript[-20:], start=1)
        ]
        performance_summary = {
            key: value
            for key, value in performance.items()
            if not key.startswith("_")
        }
        logged_user_texts = [
            turn["user"] for turn in turns if turn.get("user")
        ]
        transcript_wer = _transcript_wer(
            audio_transcript,
            logged_user_texts,
        )
        deterministic_match_score = transcript_wer["match_score"]
        evaluation_payload = {
            "criteria": CRITERIA,
            "audio_transcript": audio_transcript,
            "audio_utterances": audio_result["utterances"],
            "logged_transcript": turns,
            "deterministic_match_score": deterministic_match_score,
            "transcript_disagreement": transcript_wer,
            "performance_metrics_ms": performance_summary,
            "barge_in_metrics": {
                "average_latency_ms": performance_summary.get(
                    "barge_in_latency_ms"
                ),
                "average_interrupt_count": performance_summary.get(
                    "interrupt_count"
                ),
            },
        }
        provider_errors = []
        raw_result = None
        used_provider = None
        used_model = None
        for model in model_chain:
            try:
                candidate_result = await _call_openai_evaluator(
                    evaluation_payload, model
                )
                _validate_qa_evidence(
                    candidate_result,
                    audio_transcript,
                    turns,
                )
                raw_result = candidate_result
                used_provider = "openai"
                used_model = model
                break
            except Exception as provider_error:
                provider_errors.append(
                    f"openai/{model}: {provider_error}"
                )
                print(
                    "Voice QA model denemesi başarısız:",
                    provider_errors[-1],
                )
        if raw_result is None:
            raise RuntimeError(
                "Tüm OpenAI QA modelleri başarısız: "
                + " | ".join(provider_errors)
            )
        comparison = raw_result.setdefault("transcript_comparison", {})
        comparison["match_score"] = deterministic_match_score
        if deterministic_match_score >= 92:
            raw_result["issues"] = [
                issue
                for issue in raw_result.get("issues") or []
                if (
                    not isinstance(issue, dict)
                    or issue.get("type") != "speech_recognition_error"
                )
            ]
        result = _validate_result(raw_result)
        result["evaluator_provider"] = used_provider
        result["evaluator_model"] = used_model
        result["qa_evaluator_version"] = QA_EVALUATOR_VERSION
        result["qa_prompt_version"] = QA_PROMPT_VERSION
        result["stt_reference_version"] = STT_REFERENCE_VERSION
        result["transcript_metric_version"] = TRANSCRIPT_METRIC_VERSION
        result["transcript_disagreement"] = transcript_wer
        result["audio_transcript"] = audio_transcript
        result["logged_user_transcript"] = logged_user_texts
        await asyncio.to_thread(
            complete_voice_ai_evaluation, recording_id, result
        )
        print("Voice AI değerlendirmesi tamamlandı:", recording_id)
    except Exception as exc:
        print("Voice AI değerlendirme hatası:", repr(exc))
        try:
            await asyncio.to_thread(
                fail_voice_ai_evaluation, recording_id, exc
            )
        except Exception as persist_exc:
            print("Değerlendirme hata kaydı yazılamadı:", repr(persist_exc))


def schedule_voice_evaluation(session_id, recording_id):
    """WebSocket requestinden bağımsız bir QA taskı başlatır."""
    print("Voice AI değerlendirmesi arka plana alındı:", recording_id)
    task = asyncio.create_task(
        evaluate_voice_session(session_id, recording_id),
        name=f"voice-evaluation-{recording_id}",
    )
    EVALUATION_TASKS.add(task)

    def evaluation_done(completed_task):
        EVALUATION_TASKS.discard(completed_task)
        if completed_task.cancelled():
            print("Voice AI değerlendirme taskı iptal edildi:", recording_id)
            return
        error = completed_task.exception()
        if error:
            print("Voice AI background task hatası:", repr(error))

    task.add_done_callback(evaluation_done)
    return task
