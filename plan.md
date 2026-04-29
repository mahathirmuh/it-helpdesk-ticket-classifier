# Plan — Klasifikasi Tiket IT Helpdesk: Multi-Skema Komparasi

## Tujuan

Membandingkan performa berbagai pendekatan klasifikasi teks pada dataset tiket IT helpdesk (`cobacek.xlsx`) untuk dua label target: **category** dan **priority**.

---

## Skema Komparasi

| # | Skema | Metode |
|---|-------|--------|
| 1 | **SVM** | TF-IDF (unigram+bigram) + LinearSVC |
| 2 | **Random Forest** | TF-IDF (unigram+bigram) + RandomForestClassifier(n_estimators=200) |
| 3 | **Logistic Regression** | TF-IDF (unigram+bigram) + LogisticRegression(max_iter=1000) |
| 4 | **BERT** | Fine-tuned `distilbert-base-multilingual-cased` (HuggingFace) |
| 5 | **GenAI-only** | OpenAI API, zero-shot per model |
| 6 | **Hybrid SVM** | SVM prediksi dasar; GenAI koreksi baris yang SVM salah |
| 7 | **Hybrid RF** | RF prediksi dasar; GenAI koreksi baris yang RF salah |
| 8 | **Hybrid LR** | LR prediksi dasar; GenAI koreksi baris yang LR salah |
| 9 | **Hybrid BERT** | BERT prediksi dasar; GenAI koreksi baris yang BERT salah |

---

## Dataset

- **File:** `cobacek.xlsx`
- **Kolom input:** `description`
- **Kolom target:** `category`, `priority`
- **Split:** 80% train / 20% test (stratified, random_state=42)

---

## Struktur File

```
rpl-svm1/
├── cobacek.xlsx                  # Dataset utama
│
├── bert_classifier.py            # Modul BERT (sklearn-compatible wrapper)
│
├── train_svm.py / .ipynb         # Standalone trainer: TF-IDF + SVM
├── train_rf.py  / .ipynb         # Standalone trainer: TF-IDF + Random Forest
├── train_logres.py / .ipynb      # Standalone trainer: TF-IDF + Logistic Regression
├── train_bert.py / .ipynb        # Standalone trainer: fine-tuned BERT
│
└── compare_svm_genai.py / .ipynb # Script komparasi lengkap (semua 9 skema)
```

---

## Metrik Evaluasi

Setiap skema dievaluasi dengan:
- **Accuracy**
- **Macro Precision**
- **Macro Recall**
- **Macro F1**
- **Weighted F1**
- **Jumlah sampel** (support)

Output tersimpan di satu file Excel dengan sheet: `Predictions_Compare`, `Metrics`, `Summary`.

---

## Visualisasi (Notebook)

Setiap model training phase dilengkapi dengan:

- **Confusion Matrix Heatmap** (2-panel: Category + Priority) — warna berbeda per model (Blues=SVM, Greens=RF, Oranges=LR, Purples=BERT)
- **Markdown Penjelasan** — cara kerja model, kelebihan/kekurangan, dan cara baca confusion matrix

Di akhir notebook (Section 16) terdapat **Grouped Bar Chart** untuk membandingkan Accuracy dan F1 semua skema sekaligus, beserta tabel karakteristik per skema.

---

## Catatan Excel Output

Setiap header kolom pada file Excel output dilengkapi **comment/note** (tooltip merah) yang menjelaskan arti kolom tersebut — mencakup kolom prediksi, ground truth, metrik, dan kolom hybrid dinamis.

---

## Konfigurasi Utama (`compare_svm_genai`)

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `--input` | `cobacek.xlsx` | File dataset |
| `--output` | `cobacek_compare.xlsx` | File hasil |
| `--models` | `gpt-4.1-mini` | OpenAI model ID (pisah koma untuk multi-model) |
| `--bert-model` | `distilbert-base-multilingual-cased` | HuggingFace model ID |
| `--bert-epochs` | `3` | Jumlah epoch fine-tuning BERT |
| `--skip-bert` | `False` | Lewati BERT (untuk run cepat tanpa GPU) |
| `--skip-lr` | `False` | Lewati Logistic Regression |

---

## Dependensi

```
pandas
openpyxl
scikit-learn
openai
torch
transformers
```
