# CineBot Intent / Outcome Sistemi

Her tamamlanan kullanıcı turu Firestore'daki `bot_chat_logs` koleksiyonuna
mesaj metinleriyle birlikte otomatik olarak sınıflandırılmış biçimde yazılır.
Sınıflandırma dış bir LLM çağrısı yapmaz; gecikme ve maliyet oluşturmayan,
sürümlenmiş kurallarla çalışır.

## Kaydedilen alanlar

```json
{
  "intent": "kategoriye_gore_film_arama",
  "outcome": "islem_basarili",
  "intent_confidence": 0.91,
  "outcome_confidence": 0.9,
  "classification_version": "rules-v1",
  "classification_method": "deterministic_rules",
  "classification_reason": {
    "intent": "signal:category_film_search",
    "outcome": "default:completed_response"
  },
  "channel": "telegram",
  "input_type": "voice"
}
```

Teknik bir sorun varsa ayrıca yalnızca güvenli teşhis alanları tutulur:
`error_stage` ve `error_type`. API anahtarı veya tam sağlayıcı hata cevabı
sohbet kaydına yazılmaz.

## Intent kodları

- `film_onerisi_istendi`
- `kategoriye_gore_film_arama`
- `oyuncuya_gore_film_arama`
- `oyuncu_bilgisi_soruldu`
- `yonetmene_gore_film_arama`
- `yonetmen_bilgisi_soruldu`
- `film_bilgisi_soruldu`
- `cinematch_destegi_istendi`
- `gorsel_analizi_istendi`
- `sinema_sohbeti`
- `selamlasma`
- `alakasiz_sohbet`
- `anlasilamayan_istek`

## Outcome kodları

- `islem_basarili`: İstek normal bir cevapla tamamlandı.
- `anlasilamadi_fallback`: Bot açıklama/tekrar istedi veya boş cevap aldı.
- `teknik_hata`: Model, ASR, ağ, kanal gönderimi gibi bir aşama başarısız oldu.
- `kismi_basarili`: Örneğin yazılı cevap ulaştı ama TTS ya da yardımcı araç
  başarısız oldu.
- `veri_bulunamadi`: İstenen dış veri kaynağında bulunamadı.
- `kapsam_disi_yonlendirildi`: Sinema/CineMatch dışı istek doğru biçimde
  kapsam içine yönlendirildi.

Oturum belgelerinde de `intent_counts`, `outcome_counts`, `last_intent` ve
`last_outcome` alanları güncellenir. Böylece oturum listesinde ek sorgu
yapmadan kısa bir özet gösterilebilir.

## Admin analitiği

Admin anahtarıyla:

```text
GET /api/admin/outcomes?days=30&limit=50
X-Admin-Key: ADMIN_API_KEY
```

Opsiyonel filtreler:

```text
intent=oyuncuya_gore_film_arama
outcome=anlasilamadi_fallback
channel=telegram
```

Yanıt; intent ve outcome dağılımlarını, başarı/fallback/teknik hata oranlarını,
intent-outcome çapraz tablosunu, günlük sonucu ve incelenebilir son konuşma
turlarını içerir. Eski, henüz etiketlenmemiş kayıtlar `unclassified_count`
alanında ayrı gösterilir. İncelenen zaman aralığı 5.000 kaydı aşarsa
`scan_truncated=true` döner; böylece oranların örneklem üzerinden hesaplandığı
gizlenmez.

Genel `/api/admin/overview` yanıtına da son seçilen gün aralığı için
`intent_distribution`, `outcome_distribution`, `success_rate`,
`fallback_rate` ve `technical_error_rate` alanları eklenmiştir.

## Test

```bash
python3 -m unittest -v test_outcome_service.py
```

Kural listesi değiştiğinde `CLASSIFICATION_VERSION` artırılmalıdır. Böylece
eski ve yeni etiketlerin hangi mantıkla üretildiği analiz sırasında ayrılabilir.
