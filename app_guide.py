# CineMatch uygulaması hakkında CineBot AI'nın kullanacağı rehber metni.
#
# Bu metin, gerçek uygulama koduna (Flutter tarafındaki ekran ve ayar
# isimlerine) bakılarak yazılmıştır; kullanıcı "kullanıcı adımı nasıl
# değiştiririm" gibi uygulama hakkında bir soru sorduğunda CineBot bu
# rehberi kullanarak doğru, uygulamadaki gerçek menü/buton isimleriyle
# birebir örtüşen bir yanıt verir.
#
# NOT: Uygulamaya yeni bir özellik eklendiğinde ya da bir ekran/menü ismi
# değiştiğinde, botun yanlış yönlendirme yapmaması için bu metni de
# güncellemek gerekir.

CINEMATCH_APP_GUIDE = """
CineMatch uygulaması alt menüde 4 ana sekmeden oluşur: Feed, Sinefiller (eşleşme), Mesajlar, Profil.

PROFİLİ DÜZENLEME (Profil sekmesi > "Profili Düzenle"):
- Kullanıcı adını değiştirme: "Profili Düzenle" ekranındaki "Temel Bilgiler" bölümünde "Kullanıcı Adı" alanına yeni adı yaz, sonra "Değişiklikleri Kaydet"e bas. Kullanıcı adı 3-20 karakter olmalı, sadece harf/rakam/alt çizgi/nokta/tire içerebilir ve başka biri tarafından kullanılmıyor olmalı.
- Profil fotoğrafını değiştirme: "Profili Düzenle" ekranında üstteki yuvarlak profil fotoğrafına dokunup galeriden yeni bir fotoğraf seçebilirsin.
- Biyografi (bio) ve yaş bilgisi de "Temel Bilgiler" bölümünde düzenlenir.
- Sevdiğin türler, favori yönetmenler ve favori oyuncular "Profili Düzenle" ekranında ayrı bölümler halinde eklenip çıkarılabilir. Bu bilgiler benim (CineBot) sana film önerirken kullandığım kişisel zevk profilini oluşturur.
- Letterboxd hesabını bağlama: "Letterboxd Bağlantısı" bölümüne Letterboxd kullanıcı adını girip verilerini içeri aktarabilirsin.

AYARLAR (Profil sekmesi > sağ üstteki dişli/ayarlar ikonu):
- TERCİHLER: Bildirimleri açma/kapatma, Görünüm (Aydınlık / Karanlık / Sistem teması).
- HESAP: Şifre Sıfırla (kayıtlı e-postana sıfırlama bağlantısı gönderilir), Letterboxd Bağlantısını Kes, Engellenen Kullanıcılar (engellediğin kişileri görüp engeli kaldırabilirsin).
- UYGULAMA: Kullanım Koşulları, Gizlilik Politikası, Destek Al (uygulama ekibine e-posta atmanı sağlar), Hakkında.
- En altta Çıkış Yap ve Hesabı Sil seçenekleri var. Hesabı silmek GERİ ALINAMAZ bir işlemdir, bunu belirtmeyi unutma.

FİLM/DİZİ TAKİBİ VE RAFLAR:
- Bir filmi "5 Yıldız", "Favoriler", "İzleme Listesi" ya da "Beğenmedim" olarak işaretleyebilirsin; bu, filmin detay sayfasındaki ilgili butonlardan yapılır.
- Kendi özel film listelerini oluşturabilir, "Keşfet: Tüm Listeler" bölümünden başka kullanıcıların listelerine göz atabilirsin; listelerini istersen gizli de yapabilirsin.
- Film, oyuncu ve yönetmen arama özelliğiyle uygulama içinden arama yapabilirsin.

SİNEFİLLER / EŞLEŞME:
- "Sinefiller" sekmesinde sana ortak film zevkine göre başka kullanıcılar önerilir; profilleri beğenip geçebilirsin. Karşılıklı beğeni olursa eşleşirsiniz ve birbirinizle mesajlaşabilirsiniz. Ne kadar çok film puanlarsan, eşleşme önerileri o kadar isabetli olur.

MESAJLAR:
- Bireysel sohbetlerde film paylaşabilir (tıklanabilir film kartı olarak gönderilir), etkinlik ya da anket oluşturabilir, "watchlist çarkı" ile birlikte ne izleyeceğinize rastgele karar verebilirsiniz.
- "Kulüplerim" sekmesinde grup sohbetleri (kulüpler) bulunur; bir kulüpte "haftanın filmi" öne çıkarılabilir.
- Ben (CineBot AI) de Mesajlar ekranındaki sohbet listesinde diğer sohbetlerle birlikte görünürüm; benimle de film gönderip o film hakkında konuşabilirsin.

KULÜPLER:
- Yeni bir kulüp (grup sohbeti) oluşturabilir, arkadaşlarını davet edebilir, birlikte film önerileri paylaşabilirsiniz.

OYUNLAŞTIRMA:
- Rozetler: Uygulamadaki çeşitli aktiviteleri (film puanlama, eşleşme, sohbet vb.) tamamladıkça rozet kazanılır; Profil sekmesinden rozet ilerlemesi görülebilir.
- Liderlik Tablosu: "En Popüler", "En Sinefiller" kategorileri ve haftalık trivia yarışması liderlerini gösterir.
- Haftalık Trivia: Sinema bilgi yarışmasına katılıp puan toplanabilir, haftalık liderlik tablosunda sıralamanı görebilirsin.

FEED VE HABERLER:
- Feed sekmesinde diğer kullanıcıların paylaşımlarını görebilir, kendi gönderini oluşturabilirsin.
- "Sinema Gündemi" bölümünde güncel sinema haberleri okunabilir.

GİZLİLİK / GÜVENLİK:
- İstenmeyen bir kullanıcıyı profilinden engelleyebilir, Ayarlar > Engellenen Kullanıcılar'dan istersen sonradan engeli kaldırabilirsin.
""".strip()
