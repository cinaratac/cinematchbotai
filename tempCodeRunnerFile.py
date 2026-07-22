import telebot
from flask import Flask, request, jsonify
from ai_service import get_ai_response
from database import setup_database, log_chat
from database import update_user_session

TOKEN = "8790113808:AAF7eyGmwkW-Q1f1O3FH4Zqobh7tBvbaI-g"
NGROK_URL = "https://unlit-wildfire-problem.ngrok-free.dev"

bot = telebot.TeleBot(TOKEN)        
app = Flask(__name__) 
app.json.ensure_ascii = False


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Merhaba! Ben staj yapan bir adamın ürettiği demoyum")

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    print("Soru geldi:", message.text)
    msg = bot.reply_to(message, "Düşünmekteyim...") 
    
    answer = get_ai_response(message.from_user.id, message.text)
    if not answer or answer.strip() == "":
        answer = "⚠️ Üzgünüm, yapay zeka modeli boş bir cevap döndürdü."

    bot.edit_message_text(chat_id=message.chat.id, message_id=msg.message_id, text=answer)

    username = message.from_user.username if message.from_user.username else message.from_user.first_name
    log_chat(message.from_user.id, username, message.text, answer)

    from ai_service import maintain_user_profile
    maintain_user_profile(message.from_user.id, message.text, answer)
   
    from database import update_user_session
    update_user_session(message.from_user.id, f"Son konuştuğu konu: {message.text}")

@app.route('/' + TOKEN, methods=['POST'])
def getMessage():
    json_string = request.get_data().decode('utf-8')
    update = telebot.types.Update.de_json(json_string)
    bot.process_new_updates([update])
    return "!", 200

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    user_id = data.get('user_id', 'API_User') 
    user_message = data.get('message', '')
    if not data or 'message' not in data:
        return jsonify({"error": "Lütfen JSON formatında bir 'message' parametresi gönderin."}), 400
        
    user_message = data['message']
    
    try:
        # 1. Yapay zekadan cevabı al (OMDb Tool çalışacaktır)
        ai_response = get_ai_response(user_id, user_message)
        
        # 2. VERİTABANINA KAYDET
        # API istekleri için sabit ID'yi 0, kullanıcı adını "API_User" olarak logluyoruz
        log_chat(user_id, "API_User", user_message, ai_response)
        
        # 🌟 YENİ: Akıllı kümülatif session hafızasını tetikle
        from ai_service import maintain_user_profile
        maintain_user_profile(user_id, user_message, ai_response)
        
        return jsonify({
            "status": "success",
            "bot_response": ai_response
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route("/")
def webhook_ayarla():
    bot.remove_webhook()
    bot.set_webhook(url=NGROK_URL + '/' + TOKEN)
    return "Sistem Hazır", 200

if __name__ == "__main__":
    setup_database()
    print("Webhook ayarını tetiklemek için http://127.0.0.1:5001 adresine gidin.")

    app.run(host="0.0.0.0", port=5001)