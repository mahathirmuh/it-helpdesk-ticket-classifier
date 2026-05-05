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
| 5 | **Hybrid SVM** | SVM prediksi dasar; GenAI koreksi baris yang SVM salah (mismatch category atau priority) |

---

## Dataset

- **File:** `data/cobacek.xlsx`
- **Kolom input:** `description`
- **Kolom target:** `category` (81 kelas), `priority` (3 kelas: low, medium, high)
- **Total baris:** 16.338 tiket
- **Split:** Stratified K-Fold Cross-Validation (SVM/RF/LR: 5 fold; BERT: 3 fold)

---

## Metrik Evaluasi

Setiap skema dievaluasi dengan:

- **Accuracy**
- **Macro Precision**
- **Macro Recall**
- **Macro F1**
- **Weighted F1**
- **Jumlah sampel** (support)

Output tersimpan di satu file Excel (`hasil_final.xlsx`) dengan sheet:
`Predictions_Compare`, `Metrics`, `Summary`, `Category_Analysis`.

---

## Visualisasi

- **Confusion Matrix** per model (Category + Priority side-by-side) — disimpan ke `paper/figures/confusion_matrix_{model}.png`
- **Per-Category Accuracy Table** — akurasi tiap model per kategori, highlight model terbaik
- **Bar Chart Perbandingan** — grouped bar chart Accuracy & F1 semua skema (`paper/figures/fig5_accuracy_compare.png`, `fig6_macro_f1_compare.png`)

---

## Konfigurasi Utama (`compare_svm_genai`)

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `INPUT_FILE` | `../data/cobacek.xlsx` | File dataset |
| `OUTPUT_FILE` | `../results/hasil_final.xlsx` | File hasil |
| `N_SPLITS` | `5` | Jumlah fold K-Fold (SVM/RF/LR) |
| `BERT_N_SPLITS` | `3` | Jumlah fold K-Fold khusus BERT |
| `MULTI_MODELS` | dari `.env` | OpenAI model ID (pisah koma) |
| `SKIP_BERT` | `False` | Lewati BERT (untuk run cepat tanpa GPU) |
| `SKIP_LR` | `False` | Lewati Logistic Regression |

---

## Dependensi

```
pandas
openpyxl
scikit-learn
openai
torch
transformers
matplotlib
```
