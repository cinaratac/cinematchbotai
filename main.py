import base64
import os
import time
from io import BytesIO
import telebot
from flask import Flask, request, jsonify
from flask_cors import CORS
from ai_service import (
    ASR_MODEL_NAME,
    MODEL_NAME,
    TTS_MODEL_NAME,
    get_ai_response,
    analyze_image,
    transcribe_audio,
    text_to_speech,
)
from admin_api import admin_bp
from database import (
    setup_database,
    get_session_admin_detail,
    add_evaluation,
    get_or_create_session,
    log_chat,
    log_failed_interaction,
    log_performance_metric,
    touch_session,
    update_chat_outcome,
)
from outcome_service import (
    OUTCOME_OUT_OF_SCOPE,
    OUTCOME_PARTIAL_SUCCESS,
    OUTCOME_SUCCESS,
    OUTCOME_TECHNICAL_ERROR,
)


# --- GİZLİ ANAHTARLAR (Görev 11): artık koddan değil ortam değişkenlerinden
# okunuyor. Render'da "Environment" sekmesinden şunları tanımlaman gerekiyor:
#   TELEGRAM_BOT_TOKEN  -> BotFather'dan aldığın token (ESKİ TOKEN'I İPTAL EDİP
#                          YENİSİNİ KULLAN, eskisi GitHub geçmişinde public kaldı)
#   PUBLIC_BASE_URL     -> (opsiyonel) Render dışında bir yere deploy edersen
#                          servisin dışarıdan erişilebilir adresi
# Render'daysan PUBLIC_BASE_URL'i elle girmene bile gerek yok: Render her
# servise otomatik olarak RENDER_EXTERNAL_URL ortam değişkenini enjekte eder,
# aşağıdaki kod onu otomatik kullanır.
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Sadece saf yerel geliştirme + ngrok ile test ederken kullanılan yedek adres.
_LOCAL_NGROK_FALLBACK = "https://unlit-wildfire-problem.ngrok-free.dev"
PUBLIC_BASE_URL = (
    os.environ.get("RENDER_EXTERNAL_URL")
    or os.environ.get("PUBLIC_BASE_URL")
    or _LOCAL_NGROK_FALLBACK
)

bot = telebot.TeleBot(TOKEN, threaded=False) if TOKEN else None
app = Flask(__name__)
app.json.ensure_ascii = False

# CineMatch web sitesindeki (Firebase Hosting) sohbet widget'ı bu API'yi
# tarayıcıdan farklı bir origin'den (site domaini) çağıracağı için CORS'u
# sadece /api/ altındaki uçlar için açıyoruz.
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Admin paneli uçları (/api/admin/...): konuşma/tool/değerlendirme verisi.
# Kendi X-Admin-Key header kontrolüyle korunur (bkz. admin_api.py).
app.register_blueprint(admin_bp)


def _register_webhook():
    """Telegram'a, gelen mesajları PUBLIC_BASE_URL/{TOKEN} adresine göndermesini
    söyler."""
    if not bot:
        print("UYARI: TELEGRAM_BOT_TOKEN tanımlı değil; Telegram botu devre dışı, sadece /api/chat aktif.")
        return
    try:
        bot.remove_webhook()
        bot.set_webhook(url=f"{PUBLIC_BASE_URL}/{TOKEN}")
        # Bot tokenı webhook yolunun bir parçasıdır; deploy loguna yazmayalım.
        print("Telegram webhook ayarlandı.")
    except Exception as e:
        print("WEBHOOK AYARLAMA HATASI:", e)


def initialize_services():
    """Firestore ve Telegram gibi dış servisleri başlatır."""
    setup_database()
    _register_webhook()


# Gunicorn ile yalnızca main:app çalıştırıldığında eski davranışı koru.
# Birleşik aiohttp sunucusu bu değeri importtan önce ayarlar ve kurulumu port
# açıldıktan sonra arka planda başlatır; böylece Render port taraması beklemez.
if os.environ.get("CINEMATCH_DEFER_SERVICE_INITIALIZATION") != "1":
    initialize_services()


if bot:

    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        answer = "Merhaba! Ben staj yapan bir adamın ürettiği demoyum"
        bot.reply_to(message, answer)
        username = (
            message.from_user.username or message.from_user.first_name
        )
        try:
            session_id = get_or_create_session(message.from_user.id, username)
            logged = log_chat(
                session_id,
                message.from_user.id,
                username,
                "/start",
                answer,
                channel="telegram",
                input_type="command",
            )
            touch_session(session_id, logged)
        except Exception as log_error:
            print("START MESAJI LOG HATASI:", log_error)

    @bot.message_handler(content_types=['photo'])
    def handle_photo(message):
        """GÖREV 4: Bota gönderilen fotoğrafları analiz eder.
        content_types=['photo'] sayesinde bu handler SADECE fotoğraf mesajlarını
        yakalar; aşağıdaki metin handler'ı (varsayılan content_types=['text'])
        fotoğraflarla çakışmaz."""
        print("Fotoğraf geldi.")
        msg = bot.reply_to(message, "Fotoğrafı inceliyorum...")

        username = message.from_user.username if message.from_user.username else message.from_user.first_name

        diagnostics = None
        try:
            file_id = message.photo[-1].file_id
            file_info = bot.get_file(file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            image_base64 = base64.b64encode(downloaded_file).decode('utf-8')

            caption = message.caption

            answer, _session_id, diagnostics = analyze_image(
                image_base64,
                caption,
                user_id=message.from_user.id,
                username=username,
                include_diagnostics=True,
                channel="telegram",
            )
        except Exception as e:
            print("FOTOĞRAF İŞLEME HATASI:", e)
            answer = "Üzgünüm, fotoğrafı işlerken bir sorun oluştu, tekrar gönderir misin?"
            try:
                failed = log_failed_interaction(
                    message.from_user.id,
                    username,
                    "[FOTOĞRAF]" + (f" - {message.caption}" if message.caption else ""),
                    answer,
                    channel="telegram",
                    input_type="photo",
                    error_stage="telegram_photo_processing",
                    error_type=type(e).__name__,
                )
                diagnostics = {
                    "chat_log_id": failed["id"],
                    "outcome": failed["outcome"],
                }
            except Exception as log_error:
                print("FOTOĞRAF HATA LOGU KAYDEDİLEMEDİ:", log_error)

        try:
            bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=msg.message_id,
                text=answer,
            )
        except Exception as send_error:
            print("FOTOĞRAF CEVABI GÖNDERİLEMEDİ:", send_error)
            if diagnostics and diagnostics.get("chat_log_id"):
                try:
                    update_chat_outcome(
                        diagnostics["chat_log_id"],
                        OUTCOME_TECHNICAL_ERROR,
                        error_stage="telegram_photo_send",
                        error_type=type(send_error).__name__,
                    )
                except Exception as update_error:
                    print("FOTOĞRAF OUTCOME GÜNCELLENEMEDİ:", update_error)

    @bot.message_handler(content_types=['voice', 'audio'])
    def handle_audio(message):
        """Telegram voice/ses dosyasını yazıya çevirip normal sohbet akışına verir."""
        pipeline_started = time.perf_counter()
        current_stage = "telegram_ack"
        metrics = {
            "channel": "telegram",
            "input_type": message.content_type,
            "user_id": str(message.from_user.id),
            "username": message.from_user.username or message.from_user.first_name,
            "session_id": None,
            "telegram_download_ms": None,
            "asr_ms": None,
            "ai_ms": None,
            "ai_ready_ms": None,
            "telegram_text_send_ms": None,
            "tts_ms": None,
            "tts_ready_ms": None,
            "telegram_voice_upload_ms": None,
            "tool_call_count": 0,
            "tool_total_ms": None,
            "tool_calls": [],
            "ttfb_ms": None,
            "ttfs_ms": None,
            "e2e_ms": None,
            "audio_duration_seconds": None,
            "audio_size_bytes": None,
            "transcript_length": None,
            "answer_length": None,
            "asr_model": ASR_MODEL_NAME,
            "ai_model": MODEL_NAME,
            "tts_model": TTS_MODEL_NAME,
            "status": "started",
            "failed_stage": None,
            "error_type": None,
        }

        def save_metrics():
            """Ölçüm kaydı hatası kullanıcı cevabını bozmasın."""
            try:
                log_performance_metric(metrics)
            except Exception as metric_error:
                print("PERFORMANS LOG HATASI:", metric_error)

        msg = bot.reply_to(message, "Sesini dinliyorum...")
        username = message.from_user.username or message.from_user.first_name
        transcript = None
        diagnostics = None

        try:
            media = message.voice if message.content_type == 'voice' else message.audio
            metrics["audio_duration_seconds"] = getattr(media, "duration", None)

            current_stage = "telegram_download"
            download_started = time.perf_counter()
            file_info = bot.get_file(media.file_id)
            audio_bytes = bot.download_file(file_info.file_path)
            metrics["telegram_download_ms"] = round(
                (time.perf_counter() - download_started) * 1000
            )
            metrics["audio_size_bytes"] = len(audio_bytes)

            # Telegram voice mesajları OGG/Opus'tur. Normal audio mesajlarında
            # dosya uzantısını STT API'nin beklediği format bilgisi olarak kullan.
            if message.content_type == 'voice':
                audio_format = 'ogg'
            else:
                filename = getattr(media, 'file_name', '') or ''
                extension = os.path.splitext(filename)[1].lower().lstrip('.')
                mime_format = {
                    'audio/mpeg': 'mp3',
                    'audio/mp4': 'm4a',
                    'audio/x-m4a': 'm4a',
                    'audio/ogg': 'ogg',
                    'audio/wav': 'wav',
                    'audio/x-wav': 'wav',
                    'audio/flac': 'flac',
                    'audio/aac': 'aac',
                }.get(getattr(media, 'mime_type', None))
                audio_format = extension or mime_format or 'mp3'

            current_stage = "asr"
            asr_started = time.perf_counter()
            transcript = transcribe_audio(audio_bytes, audio_format=audio_format)
            metrics["asr_ms"] = round((time.perf_counter() - asr_started) * 1000)
            metrics["transcript_length"] = len(transcript)

            current_stage = "ai"
            ai_started = time.perf_counter()
            answer, _recommended_movies, _session_id, diagnostics = get_ai_response(
                transcript,
                user_id=message.from_user.id,
                username=username,
                include_diagnostics=True,
                channel="telegram",
                input_type=message.content_type,
            )
            metrics["ai_ms"] = round((time.perf_counter() - ai_started) * 1000)
            metrics["ai_ready_ms"] = round((time.perf_counter() - pipeline_started) * 1000)
            metrics["answer_length"] = len(answer)
            metrics["session_id"] = _session_id
            metrics["tool_call_count"] = diagnostics["tool_call_count"]
            metrics["tool_total_ms"] = diagnostics["tool_total_ms"]
            metrics["tool_calls"] = diagnostics["tool_calls"]
            metrics["intent"] = diagnostics["intent"]
            metrics["outcome"] = diagnostics["outcome"]

            current_stage = "telegram_text_send"
            text_send_started = time.perf_counter()
            bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=msg.message_id,
                text=f"🎙️ {transcript}\n\n{answer}",
            )
            metrics["telegram_text_send_ms"] = round(
                (time.perf_counter() - text_send_started) * 1000
            )
            # Telegram akışında TTFB: işlem başlangıcından ilk gerçek yazılı
            # cevabın Telegram API tarafından kabul edilmesine kadar geçen süre.
            metrics["ttfb_ms"] = round((time.perf_counter() - pipeline_started) * 1000)

            # Yazılı cevap kullanıcıda kalır; TTS ayrıca MP3 olarak gönderilir.
            # TTS tek başına hata verirse başarılı ASR/AI cevabını bozma.
            try:
                current_stage = "tts"
                tts_started = time.perf_counter()
                speech_bytes = text_to_speech(answer)
                metrics["tts_ms"] = round((time.perf_counter() - tts_started) * 1000)
                metrics["tts_ready_ms"] = round(
                    (time.perf_counter() - pipeline_started) * 1000
                )
                speech_file = BytesIO(speech_bytes)
                speech_file.name = "cinebot-cevap.mp3"

                current_stage = "telegram_voice_upload"
                voice_upload_started = time.perf_counter()
                bot.send_voice(
                    chat_id=message.chat.id,
                    voice=speech_file,
                    reply_to_message_id=message.message_id,
                )
                metrics["telegram_voice_upload_ms"] = round(
                    (time.perf_counter() - voice_upload_started) * 1000
                )
                # Streaming kullanılmadığı için ilk ses, MP3 tamamen üretilip
                # Telegram'a gönderildikten sonra kullanılabilir hale gelir.
                metrics["ttfs_ms"] = round(
                    (time.perf_counter() - pipeline_started) * 1000
                )
                metrics["e2e_ms"] = metrics["ttfs_ms"]
                metrics["status"] = "success"
                save_metrics()
            except Exception as tts_error:
                print("TTS HATASI:", tts_error)
                metrics["status"] = "partial_success"
                metrics["failed_stage"] = current_stage
                metrics["error_type"] = type(tts_error).__name__
                metrics["e2e_ms"] = round(
                    (time.perf_counter() - pipeline_started) * 1000
                )
                if diagnostics and diagnostics.get("chat_log_id"):
                    try:
                        changed = update_chat_outcome(
                            diagnostics["chat_log_id"],
                            OUTCOME_PARTIAL_SUCCESS,
                            error_stage=current_stage,
                            error_type=type(tts_error).__name__,
                            only_if_outcomes={
                                OUTCOME_SUCCESS,
                                OUTCOME_OUT_OF_SCOPE,
                            },
                        )
                        if changed:
                            metrics["outcome"] = OUTCOME_PARTIAL_SUCCESS
                    except Exception as update_error:
                        print("SES OUTCOME GÜNCELLENEMEDİ:", update_error)
                save_metrics()
        except Exception as e:
            print("SES İŞLEME HATASI:", e)
            metrics["status"] = "error"
            metrics["failed_stage"] = current_stage
            metrics["error_type"] = type(e).__name__
            metrics["outcome"] = OUTCOME_TECHNICAL_ERROR
            metrics["e2e_ms"] = round((time.perf_counter() - pipeline_started) * 1000)
            if diagnostics and diagnostics.get("chat_log_id"):
                try:
                    update_chat_outcome(
                        diagnostics["chat_log_id"],
                        OUTCOME_TECHNICAL_ERROR,
                        error_stage=current_stage,
                        error_type=type(e).__name__,
                    )
                except Exception as update_error:
                    print("SES OUTCOME GÜNCELLENEMEDİ:", update_error)
            else:
                try:
                    failed = log_failed_interaction(
                        message.from_user.id,
                        username,
                        transcript or "[SES MESAJI - transkript alınamadı]",
                        (
                            "Üzgünüm, sesini anlayamadım. Biraz daha net "
                            "konuşup tekrar gönderir misin?"
                        ),
                        channel="telegram",
                        input_type=message.content_type,
                        error_stage=current_stage,
                        error_type=type(e).__name__,
                    )
                    metrics["session_id"] = failed["session_id"]
                    metrics["intent"] = failed["intent"]
                except Exception as log_error:
                    print("SES HATA LOGU KAYDEDİLEMEDİ:", log_error)
            save_metrics()
            bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=msg.message_id,
                text="Üzgünüm, sesini anlayamadım. Biraz daha net konuşup tekrar gönderir misin?",
            )

    @bot.message_handler(func=lambda message: True)
    def handle_all_messages(message):
        pipeline_started = time.perf_counter()
        print("Soru geldi:", message.text)
        msg = bot.reply_to(message, "Düşünmekteyim...")

        username = message.from_user.username if message.from_user.username else message.from_user.first_name
        metrics = {
            "channel": "telegram",
            "input_type": "text",
            "user_id": str(message.from_user.id),
            "username": username,
            "session_id": None,
            "ai_ms": None,
            "ai_ready_ms": None,
            "telegram_text_send_ms": None,
            "tool_call_count": 0,
            "tool_total_ms": None,
            "tool_calls": [],
            "ttfb_ms": None,
            "ttfs_ms": None,
            "e2e_ms": None,
            "message_length": len(message.text or ""),
            "answer_length": None,
            "ai_model": MODEL_NAME,
            "status": "started",
            "failed_stage": None,
            "error_type": None,
        }
        diagnostics = None

        try:
            ai_started = time.perf_counter()
            answer, _recommended_movies, _session_id, diagnostics = get_ai_response(
                message.text,
                user_id=message.from_user.id,
                username=username,
                include_diagnostics=True,
                channel="telegram",
                input_type="text",
            )
            metrics["ai_ms"] = round((time.perf_counter() - ai_started) * 1000)
            metrics["ai_ready_ms"] = round(
                (time.perf_counter() - pipeline_started) * 1000
            )
            metrics["session_id"] = _session_id
            metrics["tool_call_count"] = diagnostics["tool_call_count"]
            metrics["tool_total_ms"] = diagnostics["tool_total_ms"]
            metrics["tool_calls"] = diagnostics["tool_calls"]
            metrics["intent"] = diagnostics["intent"]
            metrics["outcome"] = diagnostics["outcome"]
            metrics["answer_length"] = len(answer)

            send_started = time.perf_counter()
            bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=msg.message_id,
                text=answer,
            )
            metrics["telegram_text_send_ms"] = round(
                (time.perf_counter() - send_started) * 1000
            )
            metrics["ttfb_ms"] = round(
                (time.perf_counter() - pipeline_started) * 1000
            )
            metrics["e2e_ms"] = metrics["ttfb_ms"]
            metrics["status"] = "success"
        except Exception as e:
            metrics["status"] = "error"
            metrics["failed_stage"] = (
                "telegram_text_send" if metrics["ai_ready_ms"] is not None else "ai"
            )
            metrics["error_type"] = type(e).__name__
            metrics["outcome"] = OUTCOME_TECHNICAL_ERROR
            metrics["e2e_ms"] = round(
                (time.perf_counter() - pipeline_started) * 1000
            )
            print("TELEGRAM METİN İŞLEME HATASI:", e)
            if diagnostics and diagnostics.get("chat_log_id"):
                try:
                    update_chat_outcome(
                        diagnostics["chat_log_id"],
                        OUTCOME_TECHNICAL_ERROR,
                        error_stage=metrics["failed_stage"],
                        error_type=type(e).__name__,
                    )
                except Exception as update_error:
                    print("METİN OUTCOME GÜNCELLENEMEDİ:", update_error)
            else:
                try:
                    failed = log_failed_interaction(
                        message.from_user.id,
                        username,
                        message.text or "",
                        (
                            "Üzgünüm, mesajını işlerken bir sorun oluştu. "
                            "Tekrar dener misin?"
                        ),
                        channel="telegram",
                        input_type="text",
                        error_stage=metrics["failed_stage"],
                        error_type=type(e).__name__,
                    )
                    metrics["session_id"] = failed["session_id"]
                    metrics["intent"] = failed["intent"]
                except Exception as log_error:
                    print("METİN HATA LOGU KAYDEDİLEMEDİ:", log_error)
            try:
                bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=msg.message_id,
                    text="Üzgünüm, mesajını işlerken bir sorun oluştu. Tekrar dener misin?",
                )
            except Exception as send_error:
                print("TELEGRAM HATA MESAJI GÖNDERİLEMEDİ:", send_error)
        finally:
            try:
                log_performance_metric(metrics)
            except Exception as metric_error:
                print("PERFORMANS LOG HATASI:", metric_error)

    @app.route('/' + TOKEN, methods=['POST'])
    def getMessage():
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return "!", 200


@app.route('/api/chat', methods=['POST'])
def api_chat():
    pipeline_started = time.perf_counter()
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({"error": "Lütfen JSON formatında bir 'message' parametresi gönderin."}), 400

    user_message = data['message']

    user_id = data.get('user_id', 'api_user')
    username = data.get('username', 'API_User')
    metrics = {
        "channel": "api",
        "input_type": "text",
        "user_id": str(user_id),
        "username": username,
        "session_id": None,
        "ai_ms": None,
        "ai_ready_ms": None,
        "tool_call_count": 0,
        "tool_total_ms": None,
        "tool_calls": [],
        "ttfb_ms": None,
        "ttfs_ms": None,
        "e2e_ms": None,
        "message_length": len(str(user_message)),
        "answer_length": None,
        "ai_model": MODEL_NAME,
        "status": "started",
        "failed_stage": None,
        "error_type": None,
    }
    diagnostics = None

    # YENİ: Cinematch uygulamasından gelen zevk profili (opsiyonel).
    # Flutter tarafı bunu her istekte gönderiyor; hiçbiri yoksa boş liste
    # olarak kabul edilip normal şekilde devam edilir.
    app_profile = {
        'favorite_genres': data.get('favorite_genres') or [],
        'favorite_directors': data.get('favorite_directors') or [],
        'favorite_actors': data.get('favorite_actors') or [],
        'favorite_movies': data.get('favorite_movies') or [],
    }

    try:
        ai_started = time.perf_counter()
        ai_response, recommended_movies, session_id, diagnostics = get_ai_response(
            user_message,
            user_id=user_id,
            username=username,
            app_profile=app_profile,
            movie_name=data.get('movie_name') or data.get('movie_title'),
            include_diagnostics=True,
            channel="api",
            input_type="text",
        )
        metrics["ai_ms"] = round((time.perf_counter() - ai_started) * 1000)
        metrics["ai_ready_ms"] = round(
            (time.perf_counter() - pipeline_started) * 1000
        )
        metrics["session_id"] = session_id
        metrics["tool_call_count"] = diagnostics["tool_call_count"]
        metrics["tool_total_ms"] = diagnostics["tool_total_ms"]
        metrics["tool_calls"] = diagnostics["tool_calls"]
        metrics["intent"] = diagnostics["intent"]
        metrics["outcome"] = diagnostics["outcome"]
        metrics["answer_length"] = len(ai_response)
        # Flask cevabı streaming değil; backend tarafında TTFB, JSON cevabının
        # dönmeye hazır olduğu andır. Ağ ve istemcide render süresi dahil değildir.
        metrics["ttfb_ms"] = round(
            (time.perf_counter() - pipeline_started) * 1000
        )
        metrics["e2e_ms"] = metrics["ttfb_ms"]
        metrics["status"] = "success"
        try:
            log_performance_metric(metrics)
        except Exception as metric_error:
            print("PERFORMANS LOG HATASI:", metric_error)
        print("LOG BAŞARILI: API isteği Firestore veritabanına kaydedildi.")
        return jsonify({
            "status": "success",
            "bot_response": ai_response,
            "recommended_movies": recommended_movies,
            "session_id": session_id,
            "intent": diagnostics["intent"],
            "outcome": diagnostics["outcome"],
        }), 200
    except Exception as e:
        metrics["status"] = "error"
        metrics["failed_stage"] = "ai"
        metrics["error_type"] = type(e).__name__
        metrics["outcome"] = OUTCOME_TECHNICAL_ERROR
        metrics["e2e_ms"] = round(
            (time.perf_counter() - pipeline_started) * 1000
        )
        if diagnostics and diagnostics.get("chat_log_id"):
            try:
                update_chat_outcome(
                    diagnostics["chat_log_id"],
                    OUTCOME_TECHNICAL_ERROR,
                    error_stage="api_response",
                    error_type=type(e).__name__,
                )
            except Exception as update_error:
                print("API OUTCOME GÜNCELLENEMEDİ:", update_error)
        else:
            try:
                failed = log_failed_interaction(
                    user_id,
                    username,
                    user_message,
                    "İstek teknik bir hata nedeniyle tamamlanamadı.",
                    channel="api",
                    input_type="text",
                    error_stage="ai",
                    error_type=type(e).__name__,
                )
                metrics["session_id"] = failed["session_id"]
                metrics["intent"] = failed["intent"]
            except Exception as log_error:
                print("API HATA LOGU KAYDEDİLEMEDİ:", log_error)
        try:
            log_performance_metric(metrics)
        except Exception as metric_error:
            print("PERFORMANS LOG HATASI:", metric_error)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/sessions/<session_id>/rate', methods=['POST'])
def rate_session(session_id):
    """Kullanıcının kendi oturumunu değerlendirmesi için (admin key GEREKMEZ).
    Web widget'ı sohbeti bitirirken/kapatırken bunu çağırır."""
    data = request.get_json(silent=True) or {}
    try:
        rating = int(data.get('rating'))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "rating 1-5 arası bir tam sayı olmalı."}), 400
    if rating < 1 or rating > 5:
        return jsonify({"status": "error", "message": "rating 1-5 arası olmalı."}), 400

    note = str(data.get('note', ''))[:500]

    if not get_session_admin_detail(session_id):
        return jsonify({"status": "error", "message": "Oturum bulunamadı."}), 404

    new_id = add_evaluation(session_id, rating, note=note, evaluator="user")
    return jsonify({"status": "success", "data": {"id": new_id}}), 201


@app.route("/")
def health_check():
    """Render'ın (ve uptime izleyicilerin) servisin ayakta olduğunu kontrol
    etmesi için basit bir health-check ucu. ARTIK BURADA webhook ayarlamıyoruz
    -- eskiden bu route ziyaret edildiğinde webhook'u NGROK_URL'e (yerel
    makineye) yeniden yönlendiriyordu, bu da botu "lokal bilgisayar açık
    olmalı" durumuna sokan asıl sebepti."""
    return {"status": "ok", "telegram_enabled": bot is not None}, 200


if __name__ == "__main__":
    # initialize_services() yukarıda çağrıldı; burada tekrar çağırmaya gerek yok.
    app.run(host="0.0.0.0", port=5001)
