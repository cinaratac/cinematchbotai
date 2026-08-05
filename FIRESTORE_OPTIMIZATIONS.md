# Firestore yük optimizasyonları

Bu projede aşağıdaki davranışlar varsayılan olarak etkindir:

- Geçmiş oturum özetleri 120 saniye önbelleğe alınır.
- Outcome analizi 120 saniye önbelleğe alınır.
- Performans ekranının belge kümesi 60 saniye önbelleğe alınır.
- Admin sayfalama cursor'ları 300 saniye saklanır.
- Başarılı metin/görsel performans kayıtlarının %25'i örneklenir.
- Hatalar ile ses/audio performans kayıtlarının tamamı saklanır.

Ortam değişkenleriyle varsayılanlar değiştirilebilir:

```text
FIRESTORE_HISTORY_CACHE_TTL_SECONDS=120
FIRESTORE_HISTORY_CACHE_MAX_USERS=1000
FIRESTORE_OVERVIEW_CACHE_TTL_SECONDS=60
FIRESTORE_UNIQUE_USER_CACHE_TTL_SECONDS=300
FIRESTORE_OUTCOME_CACHE_TTL_SECONDS=120
FIRESTORE_SESSION_SEARCH_CACHE_TTL_SECONDS=60
FIRESTORE_CURSOR_CACHE_TTL_SECONDS=300
FIRESTORE_PERFORMANCE_CACHE_TTL_SECONDS=60
PERFORMANCE_METRIC_SUCCESS_SAMPLE_RATE=0.25
```

`PERFORMANCE_METRIC_SUCCESS_SAMPLE_RATE=1` bütün başarılı performans
ölçümlerini yeniden kaydeder. Bu ayar hata ve ses kayıtlarını etkilemez.

## Birleşik indeksleri devreye alma

Kod, indeksler hazır değilken eski sorguya güvenli biçimde geri döner. İndeksler
30 Temmuz 2026 tarihinde `movie-matching-8a836` projesine deploy edilmiştir.
Gelecekte tanımlar değişirse yeniden deploy etmek için:

```bash
firebase deploy --only firestore:indexes
```

İndeks tanımları `firestore.indexes.json`, Firebase CLI bağlantısı ise
`firebase.json` içindedir. Aynı dosya sorgulanmayan büyük metin/array
alanlarının otomatik indekslerini de kapatarak indeks depolamasını ve write
fanout'unu azaltır.

Firebase projesinde CineMatch uygulamasına ait başka birleşik indeksler varsa,
deploy öncesinde mevcut indeksleri dışa aktararak bu dosyayla birleştirin.
CLI'nin başka koleksiyonlara ait mevcut indeksleri silme teklifini onaylamayın.
Bu dosya 30 Temmuz 2026 itibarıyla mevcut `chats`, `custom_lists`, `matches`,
`notifications`, `posts` ve `scores` indekslerini de içerir. İndeks deploy'ları
boş/eski bir yapılandırma yerine bu backend klasöründen yapılmalıdır.
