import asyncio
import json
import os
import re

import aiohttp

from database import (
    complete_voice_ai_evaluation,
    fail_voice_ai_evaluation,
    get_performance_metrics_averages,
    get_session_transcript,
    skip_voice_ai_evaluation,
    start_voice_ai_evaluation,
)



DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
EVALUATION_MODEL = os.environ.get(
    "VOICE_EVALUATION_MODEL",
    # QA gibi uzun ve yapılandırılmış çıktı işlerinde kararlı managed model.
    "gemini-2.5-flash",
)
EVALUATION_TASKS = set()

CRITERIA = {
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
Sana verilen transkripti ve performans özetini yalnızca kanıta dayanarak
değerlendir. Her kriteri 0-10 arasında puanla. Transkriptte gözlenemeyen bir
kriter için score=null ve observed=false kullan; veri uydurma.

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
  ]
}

Prompt önerisi yalnızca transkriptte gerçek bir problem varsa üret. Sistem
promptunu otomatik değiştirme. Barge-in ve tool kullanımı transkriptten
gözlenemiyorsa bunları puanlama.
""".strip()


def _criterion_function_schema():
    return {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "description": "Gözlenebiliyorsa 0-10 puan.",
            },
            "observed": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["observed", "reason"],
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
        },
        "required": [
            "overall_score",
            "summary",
            "criteria",
            "issues",
            "strengths",
            "prompt_recommendations",
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


def _validate_result(result):
    if not isinstance(result, dict):
        raise ValueError("Değerlendirme sonucu nesne değil.")
    overall = result.get("overall_score")
    if not isinstance(overall, (int, float)) or not 0 <= overall <= 100:
        raise ValueError("overall_score 0-100 aralığında değil.")
    criteria = result.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("criteria alanı eksik.")
    normalized = {}
    for key in CRITERIA:
        item = criteria.get(key) or {}
        score = item.get("score")
        observed = item.get("observed") is True
        if not observed:
            score = None
        elif not isinstance(score, (int, float)) or not 0 <= score <= 10:
            raise ValueError(f"{key} puanı geçersiz.")
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
    return result


async def _call_deepgram_evaluator(payload, provider_type, model):
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY tanımlı değil.")

    settings = {
        "type": "Settings",
        "tags": ["cinematch", "qa-evaluator"],
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": 16000},
            "output": {
                "encoding": "linear16",
                "sample_rate": 24000,
                "container": "none",
            },
        },
        "agent": {
            "language": "tr",
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "model": "nova-3",
                    "language": "tr",
                }
            },
            "think": {
                "provider": {
                    "type": provider_type,
                    "model": model,
                    "temperature": 0.1,
                },
                "prompt": EVALUATOR_PROMPT,
                "functions": [EVALUATION_FUNCTION],
            },
            "speak": {
                "provider": {
                    # QA sonucu metin olarak alınır ve ses paketleri kullanılmaz.
                    # Agent API yine de bir speak sağlayıcısı istediği için ana
                    # voice agentta çalışan Türkçe Cartesia ayarı kullanılır.
                    "type": "cartesia",
                    "model_id": "sonic-3",
                    "voice": {
                        "mode": "id",
                        "id": "a167e0f3-df7e-4d52-a9c3-f949145efdab",
                    },
                    "language": "tr",
                    "speed": "normal",
                }
            },
        },
    }

    serialized_payload = json.dumps(payload, ensure_ascii=False)
    print(
        "Voice QA isteği:",
        f"provider={provider_type}",
        f"model={model}",
        f"system_prompt_chars={len(EVALUATOR_PROMPT)}",
        f"payload_chars={len(serialized_payload)}",
    )

    timeout = aiohttp.ClientTimeout(total=90)
    provider_warnings = []
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
                    await ws.send_json({
                        "type": "InjectUserMessage",
                        "content": serialized_payload,
                    })
                elif (
                    event_type == "FunctionCallRequest"
                ):
                    for function_call in event.get("functions") or []:
                        if function_call.get("name") != "submit_evaluation":
                            continue
                        arguments = function_call.get("arguments")
                        if isinstance(arguments, dict):
                            return arguments
                        if isinstance(arguments, str):
                            return json.loads(arguments)
                    raise RuntimeError(
                        "QA agentı beklenmeyen bir function çağırdı."
                    )
                elif (
                    event_type == "ConversationText"
                    and event.get("role") == "assistant"
                ):
                    return _extract_json(event.get("content"))
                elif event_type == "Warning":
                    warning = (
                        f"{event.get('code') or 'WARNING'}: "
                        f"{event.get('description') or event.get('message') or ''}"
                    )
                    provider_warnings.append(warning)
                    print("Voice QA Deepgram uyarısı:", warning)
                elif event_type == "Error":
                    detail = (
                        event.get("description")
                        or event.get("message")
                        or "Deepgram evaluator hatası."
                    )
                    code = event.get("code")
                    warning_detail = (
                        " | warnings=" + " || ".join(provider_warnings)
                        if provider_warnings
                        else ""
                    )
                    raise RuntimeError(
                        f"{detail}{f' (code={code})' if code else ''}"
                        f"{warning_detail}"
                    )
    raise RuntimeError("Değerlendirme agentından cevap alınamadı.")


async def evaluate_voice_session(session_id, recording_id):
    provider_chain = [
        ("google", EVALUATION_MODEL),
        # Bu OpenRouter değildir; Deepgram'ın yönettiği standart fallback'tir.
        ("open_ai", "gpt-4o-mini"),
    ]
    try:
        await asyncio.to_thread(
            start_voice_ai_evaluation,
            session_id,
            recording_id,
            " -> ".join(
                f"{provider}/{model}" for provider, model in provider_chain
            ),
        )
        transcript = await asyncio.to_thread(
            get_session_transcript, session_id, 100
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
        performance = await asyncio.to_thread(
            get_performance_metrics_averages,
            200,
            session_id,
        )
        # QA agentına yalnızca son 20 turu gönder. Tüm performans trend
        # noktalarını ve yüzlerce eski turu göndermek managed LLM bağlamını
        # gereksiz büyütüp think çağrısının reddedilmesine yol açabiliyor.
        turns = [
            {
                "user": row.get("user_message"),
                "assistant": row.get("bot_response"),
            }
            for row in transcript[-20:]
        ]
        performance_summary = {
            key: value
            for key, value in performance.items()
            if not key.startswith("_")
        }
        evaluation_payload = {
            "criteria": CRITERIA,
            "transcript": turns,
            "performance_metrics_ms": performance_summary,
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
                "Tüm managed QA modelleri başarısız: "
                + " | ".join(provider_errors)
            )
        result = _validate_result(raw_result)
        result["evaluator_provider"] = used_provider
        result["evaluator_model"] = used_model
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
