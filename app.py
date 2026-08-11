import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

# ============================================================
# AYARLAR
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY bulunamadı.")
    exit()

client = Groq(api_key=api_key)

VIDEO_FILE = "test.mp4"
OUTPUT_FILE = "subtitle_words_tr.json"

# Her seferinde kaç subtitle çevrilecek
BATCH_SIZE = 20
CONTEXT_SIZE = 3

# ============================================================
# JSON CEVABINI TEMİZLE
# ============================================================

def parse_translation_json(text):
    text = text.strip()

    # LLM ```json ... ``` döndürürse temizle
    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    # Sadece JSON array bölümünü al
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError("Çeviri cevabında JSON bulunamadı.")

    return json.loads(text[start:end + 1])


# ============================================================
# WHISPER
# ============================================================

print("\n🎧 Whisper Large-v3 çalışıyor...")

with open(VIDEO_FILE, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
        temperature=0.0
    )

data = transcription.model_dump()

segments = data.get("segments", [])
all_words = data.get("words", [])

print(f"✅ {len(segments)} segment bulundu.")
print(f"✅ {len(all_words)} kelime timestamp'i bulundu.")


# ============================================================
# TÜRKÇE ÇEVİRİ
# ============================================================

translations = {}

print("\n🇹🇷 Türkçe çeviri başlıyor...\n")

for batch_start in range(0, len(segments), BATCH_SIZE):

    batch = segments[
        batch_start:batch_start + BATCH_SIZE
    ]

    # Çevrilecek batch'in öncesinden ve sonrasından
    # 3'er segmenti context olarak ekliyoruz
    context_start = max(
        0,
        batch_start - CONTEXT_SIZE
    )

    context_end = min(
        len(segments),
        batch_start + len(batch) + CONTEXT_SIZE
    )

    context_items = []

    for i in range(context_start, context_end):

        context_items.append({
            "id": i,
            "text": segments[i]["text"].strip(),

            # Sadece mevcut batch çevrilecek
            "translate": (
                batch_start
                <= i
                < batch_start + len(batch)
            )
        })

    prompt = f"""
Sen profesyonel bir dizi, film ve video altyazı çevirmenisin.

Görevin, aşağıdaki konuşmaları Türkçeye çevirmek.

ÖNEMLİ:
Bu cümleler birbirinden bağımsız değildir.
Aynı sahnenin ve konuşmanın devamıdır.
Bu nedenle HER CÜMLEYİ çevirirken önceki ve sonraki cümlelerin bağlamını dikkate al.

Ama yalnızca verilen id'lere ait çevirileri döndür.

============================================================
ÇEVİRİ KURALLARI
============================================================

1. Çeviriyi doğal Türkçe yap.
Kelime kelime mekanik çeviri yapma.

2. Diyaloğun anlamını, karakterin niyetini, tonunu ve sahnenin bağlamını koru.

3. Zamirleri bağlama göre çöz.
Örneğin Almancadaki "sie", "er", "es" gibi ifadelerin neye veya kime
gönderme yaptığını önceki cümlelerden belirle.

4. Almanca deyimleri, günlük ifadeleri ve argoyu kelime kelime çevirme.
Türkçedeki doğal karşılığını kullan.

Örneğin:
"Piepen" bağlama göre para anlamına geliyorsa "ses" diye çevirme.
"heißer Ofen" bir araç için kullanılıyorsa "sıcak fırın" diye çevirme.
"Schiefe Bahn" deyim olarak kullanılıyorsa "eğri yol" diye çevirme.

5. Mizahı ve kelime oyunlarını mümkün olduğunca Türkçede doğal hale getir.
Çocuk çizgi filmi, komedi veya günlük konuşmaysa aşırı resmi Türkçe kullanma.

6. Özel isimleri koru.
SpongeBob, Patrick, Mr. Krabs, Plankton gibi isimleri değiştirme.

7. Karakter birine hitap ediyorsa Türkçe hitabı doğal biçimde koru.

8. Cümlenin kelime kelime yapısını değil, GERÇEK ANLAMINI çevir.

Örneğin:

"Ich bin auf den Kopf gefallen."
YANLIŞ:
"Başıma çarptım."

DOĞRU:
"Kafamın üstüne düştüm."

9. Kaynak transkript bazen Whisper tarafından yanlış yazılmış olabilir.

Eğer bir kelime veya cümle:
- anlamsız görünüyorsa,
- yarım kalmışsa,
- konuşmanın bağlamına uymuyorsa,
- bariz bir speech-to-text hatası içeriyorsa,

önceki ve sonraki cümleleri kullanarak ne söylendiğini anlamaya çalış.

Ancak bağlamdan güvenilir biçimde çıkarılamıyorsa yeni anlam UYDURMA.
Bu durumda eldeki metne göre en makul çeviriyi yap.

10. Bir segment yarım cümleyse, önceki veya sonraki segmentle bağlantısını
anla ve Türkçeyi buna göre doğal oluştur.

11. Türkçe altyazılar mümkün olduğunca kısa, akıcı ve okunabilir olsun.

12. Kaynakta olmayan açıklamalar ekleme.

13. Hiçbir segmenti atlama.

14. id değerlerini ASLA değiştirme.

15. Her id için tam olarak bir translation döndür.

"translate": true olan segmentleri Türkçeye çevir.

"translate": false olan segmentler yalnızca konuşmanın bağlamını
anlaman içindir. Bu segmentler için çıktı üretme.

Çıktıda yalnızca "translate": true olan id'ler bulunmalıdır.

============================================================
BAĞLAM KULLANIMI
============================================================

Aşağıdaki segmentler kronolojik sıradadır.

Bir cümleyi çevirirken sadece kendi metnine değil:
- önceki konuşmalara,
- sonraki konuşmalara,
- karakterlerin ne hakkında konuştuğuna,
- aynı sahnedeki tekrar eden kelime ve kavramlara

bak.

Örneğin bir önceki cümlede "Wunschbrunnen"dan bahsediliyorsa,
sonraki cümlelerde "Brunnen" geçtiğinde bunun "dilek kuyusu"
olduğunu hatırla.

Aynı nesne veya kavram için çeviri boyunca mümkün olduğunca
tutarlı Türkçe terimler kullan.

============================================================
ÇIKTI FORMATI
============================================================

SADECE geçerli JSON array döndür.

Markdown kullanma.
``` kullanma.
Açıklama yazma.
JSON'dan önce veya sonra hiçbir metin yazma.

Format:

[
  {{"id": 0, "translation": "Türkçe çeviri"}},
  {{"id": 1, "translation": "Türkçe çeviri"}}
]

============================================================
ÇEVRİLECEK DİYALOGLAR
============================================================

{json.dumps(context_items, ensure_ascii=False)}
"""

    success = False

    for attempt in range(3):
        try:

            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            result_text = (
                response
                .choices[0]
                .message
                .content
            )

            result = parse_translation_json(
                result_text
            )

            for item in result:
                translations[int(item["id"])] = (
                    item["translation"].strip()
                )

            success = True
            break

        except Exception as e:

            print(
                f"⚠️ Batch hatası "
                f"(deneme {attempt + 1}/3): {e}"
            )

            time.sleep(2)

    if not success:
        print(
            f"❌ {batch_start} batch'i çevrilemedi."
        )

    done = min(
        batch_start + BATCH_SIZE,
        len(segments)
    )

    print(
        f"   {done} / {len(segments)} cümle çevrildi"
    )

# ============================================================
# TRANSLATION QA / EDITOR
# ============================================================

print("\n🧠 Türkçe çeviri kalite kontrolü başlıyor...\n")

qa_translations = translations.copy()

for batch_start in range(0, len(segments), BATCH_SIZE):

    batch_end = min(
        batch_start + BATCH_SIZE,
        len(segments)
    )

    # QA için de önceki ve sonraki context'i göster
    context_start = max(
        0,
        batch_start - CONTEXT_SIZE
    )

    context_end = min(
        len(segments),
        batch_end + CONTEXT_SIZE
    )

    qa_context_items = []

    for i in range(context_start, context_end):

        qa_context_items.append({
            "id": i,
            "source": segments[i]["text"].strip(),
            "translation": translations.get(i, ""),
            "edit": batch_start <= i < batch_end
        })

    qa_prompt = f"""
Sen profesyonel bir Türkçe altyazı editörüsün.

Aşağıda video altyazılarının:

- kaynak metni
- mevcut Türkçe çevirisi
- önceki ve sonraki konuşma bağlamı

verilmiştir.

Görevin mevcut Türkçe çevirileri KONTROL ETMEK ve gerekiyorsa düzeltmektir.

============================================================
ÖNEMLİ
============================================================

"edit": true olan segmentleri kontrol et ve çıktı üret.

"edit": false olan segmentler SADECE bağlam içindir.
Bunlar için çıktı üretme.

============================================================
KONTROL ETMEN GEREKENLER
============================================================

1. Kaynak metnin gerçek anlamı Türkçeye doğru aktarılmış mı?

2. Türkçe doğal mı?
Kelime kelime veya robotik çeviri varsa düzelt.

3. Almanca deyim, argo veya günlük ifade yanlış çevrilmiş mi?

Örnek:

"Piepen"
bağlama göre para anlamındaysa
"ses" şeklinde çevrilmemeli.

"heißer Ofen"
bir araçtan bahsediliyorsa
"sıcak fırın" şeklinde çevrilmemeli.

"Schiefe Bahn"
deyim olarak kullanılıyorsa
"eğri yol" şeklinde çevrilmemeli.

4. Zamirleri bağlamdan çöz:

er / sie / es / ihnen / ihr vb.

5. Türkçe dilbilgisini düzelt.

Örneğin:

YANLIŞ:
"Bana gurur duyacaksınız."

DOĞRU:
"Benimle gurur duyacaksınız."

6. Gereksiz tekrarları veya yapay ifadeleri düzelt.

YANLIŞ:
"Bu buna değdi."

DOĞRU:
"Buna değdi."

7. Kaynakta hareket veya fiziksel eylem varsa anlamını değiştirme.

Örneğin:

"Ich bin auf den Kopf gefallen."

YANLIŞ:
"Başıma çarptım."

DOĞRU:
"Kafamın üstüne düştüm."

8. Çocuk dizisi, çizgi film, komedi veya günlük konuşma ise
doğal ve konuşma diline uygun Türkçe kullan.

9. Whisper kaynak metni bozuk görünüyorsa önceki ve sonraki
segmentlerden anlam çıkarmaya çalış.

Ancak emin değilsen yeni bir anlam UYDURMA.

10. translation alanı boşsa mutlaka Türkçe çeviri üret.

11. Çeviri zaten doğru ve doğalsa değiştirmek zorunda değilsin.

12. Özel isimleri koru.

13. id değerlerini değiştirme.

14. Her "edit": true id için TAM OLARAK bir çıktı üret.

============================================================
ÇIKTI
============================================================

SADECE geçerli JSON array döndür.

Markdown kullanma.
Açıklama yazma.

Format:

[
  {{"id": 0, "translation": "Düzeltilmiş Türkçe altyazı"}},
  {{"id": 1, "translation": "Düzeltilmiş Türkçe altyazı"}}
]

============================================================
ALTYAZILAR
============================================================

{json.dumps(qa_context_items, ensure_ascii=False)}
"""

    success = False

    for attempt in range(3):

        try:

            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": qa_prompt
                    }
                ],
                temperature=0
            )

            result_text = (
                response
                .choices[0]
                .message
                .content
            )

            result = parse_translation_json(
                result_text
            )

            for item in result:

                item_id = int(item["id"])

                # Sadece bu batch'teki segmentleri kabul et
                if batch_start <= item_id < batch_end:

                    translation = (
                        item
                        .get("translation", "")
                        .strip()
                    )

                    if translation:
                        qa_translations[item_id] = translation

            success = True
            break

        except Exception as e:

            print(
                f"⚠️ QA batch hatası "
                f"(deneme {attempt + 1}/3): {e}"
            )

            time.sleep(2)

    if not success:

        print(
            f"❌ QA batch "
            f"{batch_start}-{batch_end} kontrol edilemedi."
        )

    print(
        f"   QA: {batch_end} / {len(segments)}"
    )


# QA sonucunu ana translations'a aktar
translations = qa_translations


# ============================================================
# BOŞ ÇEVİRİ KONTROLÜ
# ============================================================

missing_ids = []

for i in range(len(segments)):

    if not translations.get(i, "").strip():
        missing_ids.append(i)


if missing_ids:

    print(
        f"\n⚠️ {len(missing_ids)} boş çeviri bulundu."
    )

    print(
        "🔧 Boş çeviriler tekrar deneniyor..."
    )

    for missing_id in missing_ids:

        context_start = max(
            0,
            missing_id - CONTEXT_SIZE
        )

        context_end = min(
            len(segments),
            missing_id + CONTEXT_SIZE + 1
        )

        missing_context = []

        for i in range(
            context_start,
            context_end
        ):

            missing_context.append({
                "id": i,
                "text": segments[i]["text"].strip(),
                "target": i == missing_id
            })

        repair_prompt = f"""
Aşağıdaki video altyazılarında yalnızca
"target": true olan segmenti doğal Türkçeye çevir.

Önceki ve sonraki segmentleri bağlam olarak kullan.

Kurallar:

- Doğal Türkçe kullan.
- Deyim ve argoyu kelime kelime çevirme.
- Kaynak metin Whisper hatası içeriyorsa bağlamdan anlamaya çalış.
- Emin olmadığın anlamı uydurma.
- Özel isimleri koru.
- SADECE JSON döndür.

Format:

[
  {{"id": {missing_id}, "translation": "Türkçe çeviri"}}
]

Konuşma:

{json.dumps(missing_context, ensure_ascii=False)}
"""

        try:

            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": repair_prompt
                    }
                ],
                temperature=0
            )

            result = parse_translation_json(
                response
                .choices[0]
                .message
                .content
            )

            if result:

                repaired = (
                    result[0]
                    .get("translation", "")
                    .strip()
                )

                if repaired:
                    translations[missing_id] = repaired

        except Exception as e:

            print(
                f"❌ ID {missing_id} "
                f"yeniden çevrilemedi: {e}"
            )


# Son kontrol

remaining_missing = [
    i
    for i in range(len(segments))
    if not translations.get(i, "").strip()
]

if remaining_missing:

    print(
        f"⚠️ Hâlâ boş kalan ID'ler: "
        f"{remaining_missing}"
    )

else:

    print(
        "\n✅ Tüm segmentlerin Türkçe çevirisi mevcut."
    )
    
# ============================================================
# CORC JSON OLUŞTUR
# ============================================================

print("\n🧩 CORC JSON hazırlanıyor...")

word_data = []

for segment_index, segment in enumerate(segments):

    seg_start = float(segment["start"])
    seg_end = float(segment["end"])

    # Bu segmente ait kelimeleri bul
    segment_words = []

    for word in all_words:

        word_start = float(word["start"])
        word_end = float(word["end"])

        # Kelimenin orta noktası
        center = (
            word_start + word_end
        ) / 2

        if seg_start <= center <= seg_end:

            word_text = (
                word["word"].strip()
            )

            if not word_text:
                continue

            segment_words.append({
                "word": word_text,
                "start": round(word_start, 3),
                "end": round(word_end, 3)
            })

    segment_data = {
        "start": round(seg_start, 3),
        "end": round(seg_end, 3),

        "text":
            segment["text"].strip(),

        "translation":
            translations.get(
                segment_index,
                ""
            ),

        "words":
            segment_words
    }

    word_data.append(segment_data)


# ============================================================
# DOSYAYA KAYDET
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        word_data,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SONUÇ
# ============================================================

total_words = sum(
    len(segment["words"])
    for segment in word_data
)

translated_count = sum(
    1
    for segment in word_data
    if segment["translation"]
)

print("\n===================================")
print("🎉 CORC DOSYASI HAZIR")
print("===================================")

print(
    f"💬 Segment: {len(word_data)}"
)

print(
    f"🇹🇷 Çeviri: {translated_count}"
)

print(
    f"🔤 Kelime: {total_words}"
)

print(
    f"📄 Dosya: {OUTPUT_FILE}"
)