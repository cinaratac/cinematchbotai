# CineBot Grafana Monitoring

Bu kurulum mevcut admin panelini degistirmez. Admin paneli Firestore tabanli
is/kullanici analitigini; Grafana ise servis sagligi, gecikme, hata, process
kaynaklari, merkezi log ve alarmlari gosterir.

## Eklenen telemetri

- `GET /metrics`: Prometheus formati. Python process/GC metrikleri de dahildir.
- Normalize HTTP route sayaclari ve gecikme histogramlari.
- AI, ASR, TTS, TTFB, TTFS, E2E ve tool asama histogramlari.
- Pipeline status/outcome sayaclari ve aktif voice baglanti gauge'i.
- `X-Request-ID` ile iliskilendirilebilen JSON stdout loglari.
- Kullanici mesaji, kullanici adi ve kullanici ID'si Prometheus label'i olmaz.
- Genel, outcome ve performans icin Excel uyumlu CSV admin endpoint'leri.

## Yerelde calistirma

Gereksinimler: Python ortami, Docker Desktop ve Docker Compose.

1. Backend'i JSON dosya logu acik sekilde baslatin:

   ```bash
   CINEBOT_LOG_FILE=var/log/cinebot.json.log .venv/bin/python webrtc_server.py
   ```

2. Ayri terminalde gozlemleme servislerini baslatin:

   ```bash
   docker compose -f observability/docker-compose.yml up -d
   ```

3. `http://localhost:3000` adresini acin. Varsayilan yerel kullanici/sifre
   `admin/admin` degeridir. Ilk calistirmadan once guclu sifre vermek icin:

   ```bash
   GRAFANA_ADMIN_PASSWORD='guclu-bir-sifre' docker compose -f observability/docker-compose.yml up -d
   ```

Hazir `CineBot / CineBot Operasyon Izleme` dashboard'u otomatik yuklenir.
Prometheus `http://localhost:9090`, Loki `http://localhost:3100`, Alloy ise
`http://localhost:12345` uzerindedir ve yalnizca localhost'a bind edilir.

## Metrik endpoint guvenligi

`METRICS_BEARER_TOKEN` tanimli degilse `/metrics` ag seviyesinde korunmalidir.
Token tanimlanirsa Prometheus job'una su blok eklenir:

```yaml
authorization:
  type: Bearer
  credentials: GUCLU_TOKEN
```

Token'i repoya veya dashboard JSON'una yazmayin. Render secret/environment
degiskeni ve Prometheus secret dosyasi kullanin.

## Render ve Grafana Cloud

Onerilen canli ortam mimarisi:

1. Render Pro veya ustunde `Observability > Metrics Stream` ile CPU, RAM,
   network, HTTP ve disk metriklerini Grafana Cloud endpoint'ine gonderin.
2. Uygulama metrikleri icin Render private network'te Prometheus/Alloy calistirip
   `https://<cinebot-private-host>/metrics` hedefini scrape edin ve Grafana
   Cloud Prometheus remote-write endpoint'ine aktarim yapin.
3. JSON stdout loglarini Render Log Stream ile TLS syslog kabul eden bir Alloy
   gateway'e, oradan Grafana Cloud Loki'ye gonderin. Promtail kullanmayin;
   yeni kurulumlarda Grafana Alloy kullanilir.
4. `METRICS_BEARER_TOKEN`, Grafana Cloud URL/kullanici/token degerlerini yalnizca
   Render secret'larinda tutun.
5. Bu repodaki dashboard JSON'unu Grafana Cloud'a import edin. Data source
   UID'lerini Cloud stack'in Prometheus ve Loki UID'leriyle eslestirin.
6. Render backend environment'ina admin panelindeki Monitoring sekmesinin
   acacagi tam adresi ekleyin:

   ```text
   GRAFANA_DASHBOARD_URL=https://<grafana-host>/d/cinebot-operations
   ```

Render plana veya Grafana Cloud kimlik bilgilerine koddan karar verilemedigi
icin bu adimlar deploy yoneticisi tarafindan tamamlanmalidir.

## Alarmlar

`observability/prometheus/rules/cinebot-alerts.yml` su kurallari icerir:

- Servis iki dakika erisilemezse kritik alarm.
- HTTP 5xx orani bes dakika boyunca yuzde 5'i asarsa uyari.
- P95 yanit suresi on dakika boyunca 10 saniyeyi asarsa uyari.
- Pipeline hata orani yuzde 5'i asarsa uyari.
- Process RAM 512 MiB uzerinde on dakika kalirsa uyari.

E-posta/Slack/Teams alicisi kuruma ozel oldugu icin repoya eklenmez. Grafana
Alerting contact point veya Alertmanager receiver canli ortamda tanimlanmalidir.

## CSV raporlari

Tum endpoint'ler `X-Admin-Key` gerektirir:

```text
GET /api/admin/reports/overview.csv?days=30
GET /api/admin/reports/outcomes.csv?days=30&limit=5000
GET /api/admin/reports/performance.csv?days=30&limit=5000
```

Dosyalar UTF-8 BOM icerir, Excel'de Turkce karakterlerle acilir ve kullanici
kaynakli hucrelerde CSV/Excel formul enjeksiyonuna karsi koruma uygulanir.
Grafana panelleri ayrica `Inspect > Data > Download CSV` ile indirilebilir.

## Saklama ve maliyet

Yerel stack Prometheus ve Loki verisini 30 gun tutar. Canli ortam retention,
log hacmi, Firestore rapor araligi ve Grafana Cloud kotasi birlikte izlenmelidir.
Raw sohbet metni Loki'ye label olarak veya Prometheus'a kesinlikle gonderilmez.
