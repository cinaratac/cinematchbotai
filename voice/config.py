import os

from aiortc import RTCConfiguration, RTCIceServer
from app_guide import CINEMATCH_APP_GUIDE


DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
VOICE_API_KEY = os.environ.get("VOICE_API_KEY", "")
VOICE_ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.environ.get("VOICE_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
}
PEER_CONNECTIONS = set()


def cors_headers(request):
    origin = request.headers.get("Origin", "")
    if "*" in VOICE_ALLOWED_ORIGINS:
        allowed_origin = "*"
    elif origin in VOICE_ALLOWED_ORIGINS:
        allowed_origin = origin
    else:
        allowed_origin = ""

    headers = {
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, X-Voice-Api-Key",
        "Vary": "Origin",
    }
    if allowed_origin:
        headers["Access-Control-Allow-Origin"] = allowed_origin
    return headers


def request_is_authorized(request):
    if not VOICE_API_KEY:
        return not os.environ.get("RENDER")
    return request.headers.get("X-Voice-Api-Key", "") == VOICE_API_KEY


def rtc_configuration():
    ice_servers = []
    stun_url = os.environ.get(
        "STUN_URL", "stun:stun.l.google.com:19302"
    ).strip()
    if stun_url:
        ice_servers.append(RTCIceServer(urls=stun_url))

    turn_url = os.environ.get("TURN_URL", "").strip()
    if turn_url:
        ice_servers.append(
            RTCIceServer(
                urls=turn_url,
                username=os.environ.get("TURN_USERNAME"),
                credential=os.environ.get("TURN_CREDENTIAL"),
            )
        )
    return RTCConfiguration(iceServers=ice_servers)


VOICE_AGENT_PROMPT = f"""
Sen CineMatch uygulamasının resmi yapay zeka asistanı, profesyonel bir sinema
asistanı ve film eleştirmenisin.

KURALLAR:
- Yalnızca filmler, diziler, yönetmenler, oyuncular, sinema sektörü ve CineMatch
  uygulaması hakkında cevap ver.
- İlgisiz bir soru gelirse kısa biçimde yalnızca sinema ve CineMatch hakkında
  yardımcı olabildiğini söyle ve konuşmayı sinemaya yönlendir.
- Her zaman Türkçe konuş. Cevapların doğal, kısa ve öz; normalde 2-4 cümle olsun.
- Ses üretiminin erken başlayabilmesi için ilk cümleyi mümkün olduğunca çabuk
  tamamla. Kısa ve tam cümleler kur; gereksiz virgül, üç nokta, parantez,
  ünlem tekrarı ve uzun duraklama oluşturacak ifadeler kullanma.
- Film önerirken kullanıcının belirttiği tür ve zevklere uy; uymayan bir filmi
  o türe aitmiş gibi gösterme.
- Bir filmden bahsederken kuru özet verme; oyunculuk, yönetmenlik veya
  sinematografi hakkında kısa bir eleştirmen yorumu da ekle.
- Doğrulayamadığın IMDb puanı, gişe hasılatı veya çıkış yılı gibi sayısal
  bilgileri uydurma.
- Robotik kapanışlar yapma; konuşmanın bağlamına uygun doğal bir soru sor.
- Kullanıcı bir tercih, duygu veya görüş belirttiğinde uygun olduğu zaman cevaba
  kısa ve doğal bir geri bildirimle başla: "Anladım", "Hı hı", "Haklısın"
  veya "Evet, seni anlıyorum" gibi. Bunu her cevapta tekrarlama ve kullanıcı
  konuşurken sözünü kesme; kullanıcı sözünü bitirdikten sonra söyle.
- Sesli yanıta uygun konuş; markdown, tablo, bağlantı, emoji ve
  [[FILMLER: ...]] gibi makine işaretleri kullanma.
- CineMatch hakkında sorulursa yalnızca aşağıdaki uygulama rehberindeki kesin
  bilgileri kullan; rehberde yoksa bilgi uydurma.

CINEMATCH UYGULAMA REHBERİ:
{CINEMATCH_APP_GUIDE}
""".strip()
