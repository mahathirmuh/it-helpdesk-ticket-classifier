# Klasifikasi Tiket IT Helpdesk — Multi-Skema Komparasi

Proyek ini membandingkan sembilan skema klasifikasi teks untuk memprediksi **category** dan **priority** tiket IT helpdesk dari kolom `description`.

## Skema yang Dibandingkan

| # | Skema | Pendekatan |
|---|-------|------------|
| 1 | SVM | TF-IDF + LinearSVC |
| 2 | Random Forest | TF-IDF + RandomForestClassifier |
| 3 | Logistic Regression | TF-IDF + LogisticRegression |
| 4 | BERT | Fine-tuned DistilBERT multilingual |
| 5 | GenAI-only | OpenAI API, zero-shot |
| 6 | Hybrid SVM | SVM + GenAI koreksi mismatch SVM |
| 7 | Hybrid RF | RF + GenAI koreksi mismatch RF |
| 8 | Hybrid LR | LR + GenAI koreksi mismatch LR |
| 9 | Hybrid BERT | BERT + GenAI koreksi mismatch BERT |

---

## Struktur File

```
rpl-svm1/
├── cobacek.xlsx                  # Dataset input
├── bert_classifier.py            # Modul BERT (sklearn-compatible wrapper)
│
├── train_svm.py                  # Standalone trainer: SVM
├── train_svm.ipynb
├── train_rf.py                   # Standalone trainer: Random Forest
├── train_rf.ipynb
├── train_logres.py               # Standalone trainer: Logistic Regression
├── train_logres.ipynb
├── train_bert.py                 # Standalone trainer: BERT
├── train_bert.ipynb
│
├── compare_svm_genai.py          # Komparasi lengkap semua 9 skema
├── compare_svm_genai.ipynb
│
├── plan.md                       # Rencana proyek
└── phase.md                      # Tahapan pengerjaan
```

---

## Instalasi

```bash
pip install pandas scikit-learn openpyxl openai torch transformers
```

---

## Penggunaan

### 1. Standalone Trainer (tanpa GenAI)

```bash
# SVM
python train_svm.py

# Random Forest
python train_rf.py

# Logistic Regression
python train_logres.py --input cobacek.xlsx --output cobacek_lr_pred.xlsx

# BERT
python train_bert.py --input cobacek.xlsx --output cobacek_bert_pred.xlsx \
    --bert-model distilbert-base-multilingual-cased \
    --bert-epochs 3
```

### 2. Komparasi Lengkap (semua skema)

```bash
# Dengan satu model GenAI
python compare_svm_genai.py \
    --input cobacek.xlsx \
    --output cobacek_compare.xlsx \
    --models gpt-4.1-mini

# Dengan beberapa model GenAI sekaligus
python compare_svm_genai.py \
    --models gpt-4.1-mini,gpt-4o-mini

# Skip BERT (lebih cepat, tanpa GPU)
python compare_svm_genai.py \
    --models gpt-4.1-mini \
    --skip-bert

# Skip Logistic Regression
python compare_svm_genai.py \
    --models gpt-4.1-mini \
    --skip-lr
```

> **Catatan:** OpenAI API key harus tersedia di environment variable `OPENAI_API_KEY` atau file `.env`.

### 3. Notebook Interaktif

Buka file `.ipynb` di Jupyter Notebook atau VS Code. Ubah parameter di sel **Konfigurasi** sesuai kebutuhan, lalu jalankan semua sel secara berurutan.

---

## Output

### Standalone Trainer
File Excel dengan tiga sheet:
- `Predictions` — seluruh data + kolom prediksi
- `Category Accuracy` — metrik per kelas untuk category
- `Priority Accuracy` — metrik per kelas untuk priority

### Komparasi Lengkap (`compare_svm_genai`)

File Excel dengan tiga sheet:

- `Predictions_Compare` — seluruh data + prediksi semua skema
- `Metrics` — tabel metrik semua skema, diurutkan descending by accuracy
- `Summary` — ringkasan run (model, waktu, konfigurasi)

Setiap header kolom dilengkapi **comment/note** (tooltip segitiga merah) yang menjelaskan arti kolom — termasuk kolom hybrid dinamis.

### Visualisasi (Notebook)

`compare_svm_genai.ipynb` menyertakan visualisasi di setiap fase training:

- **Confusion Matrix Heatmap** (2-panel: Category + Priority) setelah tiap model dilatih
  — Blues=SVM, Greens=RF, Oranges=LR, Purples=BERT
- **Grouped Bar Chart** (Section 16) membandingkan Accuracy + F1 semua skema,
  disimpan ke `comparison_chart.png`

---

## Konfigurasi Utama

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `--input` | `cobacek.xlsx` | File dataset (.xlsx) |
| `--output` | `cobacek_compare.xlsx` | File hasil |
| `--models` | `gpt-4.1-mini` | ID model OpenAI (pisah koma) |
| `--bert-model` | `distilbert-base-multilingual-cased` | HuggingFace model ID |
| `--bert-epochs` | `3` | Epoch fine-tuning BERT |
| `--skip-bert` | — | Lewati training BERT |
| `--skip-lr` | — | Lewati Logistic Regression |

---

## Dataset

File `cobacek.xlsx` dengan kolom:

| Kolom | Keterangan |
|-------|------------|
| `subject` | Judul tiket |
| `description` | Detail masalah — **input utama model** |
| `answer` | Solusi yang diberikan |
| `type` | Tipe tiket dari sumber data |
| `category` | Label category (ground truth) |
| `priority` | Label priority (ground truth) |

---

## Catatan

- **BERT di CPU** membutuhkan waktu lebih lama. Gunakan GPU (CUDA) untuk hasil lebih cepat, atau gunakan `--skip-bert` untuk melewati BERT.
- Model BERT default `distilbert-base-multilingual-cased` 40% lebih kecil dan lebih cepat dari `bert-base-multilingual-cased`, cocok untuk teks multibahasa (Indonesia + Inggris).
- Jika file output terkunci (sedang dibuka Excel), hasil otomatis disimpan ke `*_new.xlsx`.
