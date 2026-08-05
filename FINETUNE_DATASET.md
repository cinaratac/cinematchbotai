# CineMatch Fine-tune Dataset

Bu proje su anda modeli egitmiyor; `bot_chat_logs` ve voice QA kayitlarini
egitim verisine donusturmek icin once kaliteli turlari secmek gerekir.

## Kaynak alanlar

`bot_chat_logs` koleksiyonundan egitim girdisi:

- `user_message`: kullanici mesaji
- `bot_response`: modelin cevabi
- `intent`, `outcome`, `outcome_confidence`: kalite filtresi
- `channel`, `input_type`: chat/voice ayrimi
- `recording_id`: voice QA ile eslestirme

`bot_voice_ai_evaluations` koleksiyonu dogrudan egitim cevabi degil, kalite
kapisi olarak kullanilir. Ornegin `overall_score >= 80` ve
`transcript_comparison.match_score >= 85` olan ses kayitlarindaki turlar
egitime alinabilir.

## Hedef format

Her satir bir JSON objesidir:

```json
{"messages":[{"role":"system","content":"Sen CineMatch uygulamasinin resmi Turkce sinema asistanisin."},{"role":"user","content":"Bana psikolojik gerilim onerir misin?"},{"role":"assistant","content":"Tabii. Karanlik atmosfer seviyorsan Prisoners iyi bir secim olur..."}]}
```

Bu format LoRA/QLoRA ile supervised fine-tuning icin uygundur. Ek metadata
gerekiyorsa export komutuna `--include-metadata` eklenebilir; bazi provider
uploadlari ekstra alanlari kabul etmeyebilir.

## Export

```bash
python scripts/export_finetune_dataset.py --days 90 --limit 5000
```

Sadece voice agent icin, QA skoru iyi kayitlari al:

```bash
python scripts/export_finetune_dataset.py \
  --input-type voice \
  --require-voice-qa \
  --min-voice-overall-score 80 \
  --min-voice-match-score 85 \
  --system-prompt voice
```

Varsayilan ciktilar:

- `var/finetune/cinematch_sft.jsonl`
- `var/finetune/cinematch_sft.train.jsonl`
- `var/finetune/cinematch_sft.valid.jsonl`
- `var/finetune/cinematch_sft.jsonl.manifest.json`

## Onemli filtreleme kurali

`teknik_hata`, dusuk QA skoru, bos cevap, fallback ve transkript uyusmazligi
olan kayitlar modeli egitmek icin dogrudan kullanilmamalidir. Bunlar egitim
datasina degil, hata analizi veya daha sonra hazirlanacak preference/DPO
datasina ayrilmalidir.
