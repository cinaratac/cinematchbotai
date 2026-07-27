# CineMatch Voice API

Voice agent, metin API'siyle aynı dış adres üzerinden WebRTC kullanır:

```text
POST /api/voice/offer
```

## Sunucu ortam değişkenleri

Zorunlu:

```text
DEEPGRAM_API_KEY=...
VOICE_API_KEY=uzun-rastgele-bir-deger
```

Önerilen:

```text
VOICE_ALLOWED_ORIGINS=https://uygulaman.example,https://admin.example
STUN_URL=stun:stun.l.google.com:19302
TURN_URL=turn:turn.example:3478
TURN_USERNAME=...
TURN_CREDENTIAL=...
```

`TURN_*` değişkenleri, sunucu ile istemci arasında doğrudan WebRTC medya yolu
kurulamayan hosting ortamlarında gereklidir. `VOICE_ALLOWED_ORIGINS` virgülle
ayrılmış origin listesidir. Render ortamında `VOICE_API_KEY` tanımlı değilse
voice endpoint güvenlik amacıyla bütün istekleri reddeder.

Deploy başlangıç komutu:

```text
python webrtc_server.py
```

Repo kökündeki `Procfile` bu komutu zaten tanımlar.

## Tarayıcı istemcisi

```js
const pc = new RTCPeerConnection({
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
});

const remoteAudio = new Audio();
remoteAudio.autoplay = true;
pc.ontrack = ({ streams: [stream] }) => {
  remoteAudio.srcObject = stream;
};

const control = pc.createDataChannel("cinematch-control");
const microphone = await navigator.mediaDevices.getUserMedia({ audio: true });
for (const track of microphone.getTracks()) {
  pc.addTrack(track, microphone);
}

await pc.setLocalDescription(await pc.createOffer());
await new Promise((resolve) => {
  if (pc.iceGatheringState === "complete") return resolve();
  pc.addEventListener("icegatheringstatechange", () => {
    if (pc.iceGatheringState === "complete") resolve();
  });
});

const response = await fetch("https://API_ADRESI/api/voice/offer", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-Voice-Api-Key": "VOICE_API_KEY",
  },
  body: JSON.stringify({
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
    user_id: "kullanici-id",
    username: "Kullanıcı",
    favorite_genres: ["Bilim Kurgu"],
  }),
});

if (!response.ok) {
  throw new Error(`Voice API hatası: ${response.status}`);
}

await pc.setRemoteDescription(await response.json());
```

Bir web uygulamasına gömülen API anahtarı kullanıcı tarafından görülebilir.
Üretimde web istemcileri için kısa ömürlü, backend tarafından verilen bir
oturum tokenına geçilmesi; sabit `VOICE_API_KEY` kullanımının güvenilir sunucu
veya kontrollü mobil istemcilerle sınırlandırılması gerekir.
