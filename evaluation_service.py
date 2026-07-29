import asyncio
from difflib import SequenceMatcher
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
EVALUATION_MODEL = os.environ.get(
    "VOICE_EVALUATION_MODEL", "gpt-4o-mini"
)
if not EVALUATION_MODEL.startswith("gpt-"):
    EVALUATION_MODEL = "gpt-4o-mini"
EVALUATION_TASKS = set()
VOICE_EVALUATION_LOOP = None

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
taşıyan eş anlamlı ifade farklarını speech_recognition_error sayma.
Hata ancak kullanıcının niyeti, varlık adı, sayı, olumsuzluk, tercih veya
cevabın yönü değişiyorsa raporlanmalıdır. Kanıt alanında iki taraftaki gerçek
ifadeyi aynen göster; transkriptte bulunmayan ifade uydurma.
audio_says ve agent_understood aynı ifadeyse veya yalnızca biçimsel fark
taşıyorsa transcript_comparison.mismatches dizisini boş bırak.
logged_transcript içindeki "user" alanı canlı STT'nin kullanıcıdan anladığı
metindir; "assistant" alanı agent cevabıdır. speech_recognition_error için
assistant cevabını, agentın duyduğu kullanıcı metni gibi kullanma. Bu kriterde
yalnızca audio_utterances ile "user" alanlarını karşılaştır.
deterministic_match_score kodla hesaplanmıştır; match_score alanına bu değeri
aynen yaz ve kendi tahmininle değiştirme.

agent_audio_transcript, agent'ın gerçekte söylediği sesin (agent WAV kaydının)
bağımsız STT çıktısıdır — logged_transcript'teki "assistant" alanından
BAĞIMSIZ, ayrı bir kaynaktır. Eğer agent_audio_transcript mevcutsa, her tur
için logged_transcript.assistant ile agent_audio_transcript'i karşılaştır;
agent'ın sesle gerçekte söylediği ile logaki metin belirgin şekilde
uyuşmuyorsa (ör. log eksik/kesik, log'da olmayan bir cümle sesle söylenmiş)
bunu transcript_comparison.mismatches içine "audio_says" alanına ses
kaydından alıntı, "agent_understood" alanına logdaki metinden alıntı olacak
şekilde ekle ve issues içinde "log_fidelity_error" tipiyle raporla. Bu durumda
raporun agent tarafındaki puanlamayı agent_audio_transcript'e göre yap; loga
değil sese güven, çünkü log eksik olabilir. agent_audio_transcript
gönderilmemişse (agent_audio_available=false) bu karşılaştırmayı atla.

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

Sonucu serbest metin olarak yazma. Daima submit_evaluation fonksiyonunu çağır.
Fonksiyon argümanlarını aşağıdaki şemaya göre doldur:
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

ZORUNLU KAYNAK KURALI: Bu QA için tek gerçek kaynak payload içindeki
audio_transcript, agent_audio_transcript ve logged_transcript alanlarıdır.
Dünya bilgini, varsayımını, örnek konuşmaları veya başka oturumları kullanma.
Bu alanlarda açıkça geçmeyen hiçbir konu, kişi, film, yer, sayı veya olay
rapora yazılamaz.

Her observed=true kriterinin reason alanı `Turn <numara>: "tam alıntı" —`
ile başlamalıdır; alıntı payload'daki gerçek kaynak metinden harfiyen
alınmalıdır. Böyle bir alıntı yoksa observed=false ve score=null kullan.
Her issue.evidence alanı da gerçek bir turn alıntısı olmalıdır. Önce turn'leri
ve alıntıları içinden belirle, sonra yalnızca bunlara dayanarak raporla.
İki transkriptin farklı olması QA'nın inceleme konusudur; onları eşit kabul
etme veya fark gördüğün için raporu reddetme.
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


def _transcript_match_score(audio_text, logged_user_texts):
    """İki STT çıktısının deterministik kelime dizisi benzerliği."""
    audio_words = _normalize_transcript(audio_text).split()
    logged_words = _normalize_transcript(
        " ".join(str(text or "") for text in logged_user_texts)
    ).split()
    if not audio_words or not logged_words:
        return 0
    return round(
        SequenceMatcher(None, audio_words, logged_words).ratio() * 100
    )


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


async def _transcribe_recording(recording_id, track="user"):
    """Firebase'deki gerçek WAV kaydını (kullanıcı veya agent) bağımsız
    olarak yazıya çevir. track: 'user' veya 'agent'."""
    if track not in {"user", "agent"}:
        raise ValueError("track yalnızca 'user' veya 'agent' olabilir.")
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY tanımlı değil.")
    audio_bytes = await asyncio.to_thread(
        download_voice_recording_audio,
        recording_id,
        track,
    )
    if not audio_bytes:
        raise RuntimeError(f"{track} ses kaydı boş.")
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
        raise RuntimeError(
            f"{track} ses kaydından bağımsız transkript çıkarılamadı."
        )
    print(
        "Voice QA ses kaydı yeniden transkript edildi:",
        recording_id,
        f"track={track}",
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


async def _call_deepgram_evaluator_once(payload, provider_type, model):
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY tanımlı değil.")
    serialized_payload = json.dumps(payload, ensure_ascii=False)
    settings = {
        "type": "Settings",
        "tags": ["cinematch", "qa-evaluator"],
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": 16000},
            "output": {"encoding": "linear16", "sample_rate": 24000, "container": "none"},
        },
        "agent": {
            "language": "tr",
            "listen": {"provider": {"type": "deepgram", "model": "nova-3", "language": "tr"}},
            "think": {
                "provider": {"type": provider_type, "model": model, "temperature": 0.1},
                "prompt": EVALUATOR_PROMPT,
                "functions": [EVALUATION_FUNCTION],
            },
            "speak": {"provider": {
                "type": "cartesia", "model_id": "sonic-3",
                "voice": {"mode": "id", "id": "a167e0f3-df7e-4d52-a9c3-f949145efdab"},
                "language": "tr", "speed": "normal",
            }},
        },
    }
    print(
        "Voice QA Deepgram isteği:",
        f"provider={provider_type}",
        f"model={model}",
        f"system_prompt_chars={len(EVALUATOR_PROMPT)}",
        f"payload_chars={len(serialized_payload)}",
    )
    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(
            "wss://agent.deepgram.com/v1/agent/converse",
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            heartbeat=20,
        ) as ws:
            async for message in ws:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(message.data)
                event_type = event.get("type")
                if event_type == "Welcome":
                    await ws.send_json(settings)
                elif event_type == "SettingsApplied":
                    await ws.send_json({"type": "InjectUserMessage", "content": serialized_payload})
                elif event_type == "FunctionCallRequest":
                    for function_call in event.get("functions") or []:
                        if function_call.get("name") != "submit_evaluation":
                            continue
                        arguments = function_call.get("arguments")
                        if isinstance(arguments, dict):
                            return arguments
                        if isinstance(arguments, str):
                            return json.loads(arguments)
                    raise RuntimeError("QA agentı beklenmeyen bir function çağırdı.")
                elif event_type == "ConversationText" and event.get("role") == "assistant":
                    # Model function-call yapmak yerine serbest metin
                    # döndürdü. Bunu JSON'a zorlayıp kabul etmek, kanıtsız/
                    # yapılandırılmamış çıktının rapora sızmasına yol açar.
                    # Bunun yerine hata say; çağıran taraf yeniden dener
                    # veya bir sonraki modele düşer.
                    preview = str(event.get("content") or "")[:200]
                    raise RuntimeError(
                        "QA agentı submit_evaluation çağırmadı, serbest "
                        f"metin döndürdü: {preview!r}"
                    )
                elif event_type == "Error":
                    detail = event.get("description") or event.get("message") or "Deepgram evaluator hatası."
                    raise RuntimeError(detail)
    raise RuntimeError("Değerlendirme agentından cevap alınamadı.")


async def _call_deepgram_evaluator(payload, provider_type, model, attempts=3):
    """Geçici WS/provider hatalarında üstel geri çekilmeli yeniden deneme."""
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            return await _call_deepgram_evaluator_once(payload, provider_type, model)
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


async def evaluate_voice_session(
    session_id,
    recording_id,
    wait_for_idle=True,
    allow_session_fallback=True,
    idle_grace_seconds=15,
):
    provider_chain = list(dict.fromkeys([
        ("open_ai", EVALUATION_MODEL),
        ("open_ai", "gpt-4o"),
    ]))
    try:
        if wait_for_idle:
            print(
                "Voice AI değerlendirmesi canlı görüşmelerin bitmesini bekliyor:",
                recording_id,
            )
            await wait_for_voice_idle(grace_seconds=idle_grace_seconds)
        else:
            print("Voice AI admin yeniden değerlendirmesi başlıyor:", recording_id)
        await asyncio.to_thread(
            start_voice_ai_evaluation,
            session_id,
            recording_id,
            " -> ".join(
                f"{provider}/{model}" for provider, model in provider_chain
            ),
        )
        transcript = await asyncio.to_thread(
            get_recording_transcript, recording_id, 100
        )
        # Otomatik QA eski kayıt uyumluluğu için oturum fallback'i kullanır.
        # Admin tekrar değerlendirmesinde bu fallback seçili WAV ile ilgisiz
        # son konuşma turlarını eşleştirebildiği için bilinçli olarak kapatılır.
        if not transcript and allow_session_fallback:
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
        audio_result = await _transcribe_recording(recording_id, "user")
        audio_transcript = audio_result["transcript"]

        agent_audio_transcript = ""
        agent_audio_utterances = []
        agent_audio_available = False
        try:
            agent_audio_result = await _transcribe_recording(
                recording_id, "agent"
            )
            agent_audio_transcript = agent_audio_result["transcript"]
            agent_audio_utterances = agent_audio_result["utterances"]
            agent_audio_available = True
        except Exception as agent_audio_error:
            # Agent WAV'ı olmayan eski kayıtlar için değerlendirmeyi
            # tamamen durdurmuyoruz; kullanıcı sesi üzerinden değerlendirme
            # devam eder, sadece agent tarafı bağımsız doğrulanamaz.
            print(
                "Voice QA agent ses kaydı transkript edilemedi:",
                recording_id,
                repr(agent_audio_error),
            )

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
        logged_assistant_texts = [
            turn["assistant"] for turn in turns if turn.get("assistant")
        ]
        deterministic_match_score = _transcript_match_score(
            audio_transcript,
            logged_user_texts,
        )
        agent_deterministic_match_score = (
            _transcript_match_score(
                agent_audio_transcript, logged_assistant_texts
            )
            if agent_audio_available
            else None
        )
        evaluation_payload = {
            "criteria": CRITERIA,
            "audio_transcript": audio_transcript,
            "audio_utterances": audio_result["utterances"],
            "agent_audio_available": agent_audio_available,
            "agent_audio_transcript": agent_audio_transcript,
            "agent_audio_utterances": agent_audio_utterances,
            "logged_transcript": turns,
            "deterministic_match_score": deterministic_match_score,
            "agent_deterministic_match_score": agent_deterministic_match_score,
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
        for provider_type, model in provider_chain:
            try:
                raw_result = await _call_deepgram_evaluator(
                    evaluation_payload,
                    provider_type,
                    model,
                )
                used_provider = provider_type
                used_model = model
                break
            except Exception as provider_error:
                provider_errors.append(
                    f"{provider_type}/{model}: {provider_error}"
                )
                print(
                    "Voice QA model denemesi başarısız:",
                    provider_errors[-1],
                )
        if raw_result is None:
            raise RuntimeError( 
                "Tüm Deepgram managed QA modelleri başarısız: "
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
        result["audio_transcript"] = audio_transcript
        result["logged_user_transcript"] = logged_user_texts
        result["agent_audio_available"] = agent_audio_available
        result["agent_audio_transcript"] = agent_audio_transcript
        result["logged_assistant_transcript"] = logged_assistant_texts
        result["agent_deterministic_match_score"] = (
            agent_deterministic_match_score
        )
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
def set_voice_evaluation_loop(loop):
    """WSGI admin isteklerinin QA işini ana aiohttp loop'una iletmesini sağlar."""
    global VOICE_EVALUATION_LOOP
    VOICE_EVALUATION_LOOP = loop


def _create_voice_evaluation_task(
    session_id,
    recording_id,
    wait_for_idle=True,
    allow_session_fallback=True,
    idle_grace_seconds=15,
):
    """Bu fonksiyon mutlaka çalışan bir asyncio loop'u içinde çağrılmalıdır."""
    print("Voice AI değerlendirmesi arka plana alındı:", recording_id)
    task = asyncio.create_task(
        evaluate_voice_session(
            session_id,
            recording_id,
            wait_for_idle,
            allow_session_fallback,
            idle_grace_seconds,
        ),
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


def schedule_voice_evaluation(
    session_id,
    recording_id,
    wait_for_idle=True,
    allow_session_fallback=True,
    idle_grace_seconds=15,
):
    """QA işini aktif loop'ta veya WSGI'den ana aiohttp loop'unda başlatır."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = VOICE_EVALUATION_LOOP
        if loop is None or loop.is_closed():
            raise RuntimeError(
                "Voice QA scheduler henüz hazır değil; sunucu başlatılıyor olabilir."
            )
        return asyncio.run_coroutine_threadsafe(
            _schedule_voice_evaluation_on_loop(
                session_id,
                recording_id,
                wait_for_idle,
                allow_session_fallback,
                idle_grace_seconds,
            ), loop
        )
    return _create_voice_evaluation_task(
        session_id,
        recording_id,
        wait_for_idle,
        allow_session_fallback,
        idle_grace_seconds,
    )


async def _schedule_voice_evaluation_on_loop(
    session_id,
    recording_id,
    wait_for_idle,
    allow_session_fallback,
    idle_grace_seconds,
):
    return _create_voice_evaluation_task(
        session_id,
        recording_id,
        wait_for_idle,
        allow_session_fallback,
        idle_grace_seconds,
    )
