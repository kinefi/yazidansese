---
title: Yazıdan Sese
emoji: 🎤
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.18.0"
python_version: "3.13"
app_file: main.py
pinned: false
---
# Yazıdan Sese

[![Yazıdan Sese](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/tekrei/yazidansese)

Türkçe metinleri yüksek kaliteli ses kayıtlarına dönüştüren web uygulaması.

Facebook'un MMS-TTS (VITS) modellerini kullanarak metinden konuşmaya (TTS) dönüşümü sağlar.

## Özellikler

- **Türkçe Desteği:** `facebook/mms-tts-tur` VITS modeli ile optimize edilmiş doğal ses sentezi.
- **Hız Kontrolü:** Konuşma hızını 0.5x ile 2.0x arasında ayarlayabilme.
- **Uzun Metin İşleme:** Metni anlamlı parçalara (chunk) bölerek prosodiyi korur ve bellek hatalarını önler.
- **Deterministik Çıktı:** Seed (tohum) değeri ile aynı metinden her zaman aynı sesin üretilmesini sağlar.
- **Hızlı Kurulum:** `uv` ile saniyeler içinde çalışma ortamı hazırlama.
- **Web Arayüzü:** Gradio tabanlı kullanıcı dostu arayüz.

## Kurulum

Uygulamayı çalıştırmanın en kolay yolu **uv** kullanmaktır. Bu araç, gerekli olan Python sürümünü ve kütüphaneleri sizin yerinize otomatik olarak ayarlar.

### 1. Adım: uv Paket Yöneticisini Kurun
- **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. Adım: Uygulamayı İndirin
Bu projeyi ZIP olarak indirin ve bir klasöre çıkartın. Terminali (veya Komut İstemi'ni) bu klasörün içinde açın.

---

## Kullanım

Terminalde şu tek komutu yazmanız yeterlidir:
```bash
uv run python main.py
```

### Seçenekler

| Parametre | Açıklama | Varsayılan |
| --- | --- | --- |
| `--port` | Web arayüzü portu | `7860` |
| `--share` | Gradio paylaşım linki oluşturur | `False` |

### Hot reload (geliştirme)

```bash
uv run gradio main.py
```

## Proje Yapısı

```text
├── main.py               # Giriş noktası
├── pyproject.toml
└── README.md
```

## Bağımlılıklar

| Paket | Görev |
| --- | --- |
| `transformers` | Model yükleme ve çıkarım |
| `torch` | Derin öğrenme arka ucu |
| `nltk` | Metin işleme ve cümle bölme |
| `gradio` | Web arayüzü |

## HuggingFace Space Deployment

Öncelikle space deposu ek remote olarak eklenmeli:

    git remote add space https://huggingface.co/spaces/tekrei/yazidansese

Daha sonra yeni sürümler de oraya itilebilir:

    git push --force space main

İtmeden önce bağımlılıklar yenilenmelidir:

    ./generate_requirements.sh

Yapılandırma bilgileri için: <https://huggingface.co/docs/hub/spaces-config-reference>

`sdk_version`, [requirements.txt](./requirements.txt) içindeki Gradio sürümüne eşit olmalıdır.

İşlem başarılı olursa uygulamaya şuradan erişilebilir: <https://huggingface.co/spaces/tekrei/yazidansese>