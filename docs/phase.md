# Phase — Tahapan Pengerjaan

## Phase 1 — Baseline ML (SVM & Random Forest)

**Status:** Selesai

### Deliverables

- [x] `train_svm.py` — TF-IDF + LinearSVC, output `cobacek_pred.xlsx`
- [x] `train_svm.ipynb` — versi notebook interaktif
- [x] `train_rf.py` — TF-IDF + RandomForestClassifier, output `cobacek_rf_pred.xlsx`
- [x] `train_rf.ipynb` — versi notebook interaktif

### Hasil

- Evaluasi terpisah: category model & priority model
- Metrik: accuracy + classification report per kelas
- Retrain pada full data → prediksi seluruh dataset
- Output Excel: sheet `Predictions`, `Category Accuracy`, `Priority Accuracy`

---

## Phase 2 — Logistic Regression

**Status:** Selesai

### Deliverables

- [x] `train_logres.py` — TF-IDF + LogisticRegression, output `cobacek_lr_pred.xlsx`
- [x] `train_logres.ipynb` — versi notebook interaktif

### Detail Implementasi

- `LogisticRegression(max_iter=1000, random_state=42)` dengan TF-IDF (unigram+bigram)
- Struktur identik dengan SVM dan RF: train/eval split, retrain full data, export Excel

---

## Phase 3 — BERT Fine-Tuning

**Status:** Selesai

### Deliverables

- [x] `bert_classifier.py` — modul inti; sklearn-compatible wrapper untuk HuggingFace BERT
- [x] `train_bert.py` — standalone trainer, output `cobacek_bert_pred.xlsx`
- [x] `train_bert.ipynb` — versi notebook interaktif

### Detail Implementasi

- Model default: `distilbert-base-multilingual-cased` (multilingual, 40% lebih kecil dari BERT penuh)
- Alternatif: `bert-base-multilingual-cased` untuk akurasi lebih tinggi
- PyTorch DataLoader dengan custom `_TextDataset`
- `LabelEncoder` untuk konversi label string → integer dan sebaliknya
- Otomatis deteksi GPU (CUDA) / fallback ke CPU

---

## Phase 4 — Komparasi Multi-Skema dengan GenAI

**Status:** Selesai

### Deliverables

- [x] `compare_svm_genai.py` — script komparasi lengkap (9 skema)
- [x] `compare_svm_genai.ipynb` — versi notebook interaktif (17 seksi)

### Alur Kerja

1. Load & validasi dataset
2. Split 80/20 (stratified)
3. Train SVM (category + priority)
4. Train Random Forest (category + priority)
5. Train Logistic Regression (category + priority) — opsional via `--skip-lr`
6. Train BERT (category + priority) — opsional via `--skip-bert`
7. Deteksi baris mismatch per model (SVM, RF, LR, BERT)
8. Loop per GenAI model:
   - Klasifikasi GenAI-only pada test set
   - Hybrid SVM: SVM prediksi, GenAI koreksi mismatch SVM
   - Hybrid RF: RF prediksi, GenAI koreksi mismatch RF
   - Hybrid LR: LR prediksi, GenAI koreksi mismatch LR
   - Hybrid BERT: BERT prediksi, GenAI koreksi mismatch BERT
9. Kumpulkan semua metrik → sort descending by accuracy
10. Export Excel: `Predictions_Compare`, `Metrics`, `Summary`

### Visualisasi Per-Phase (Notebook)

Setiap model training section dilengkapi dua sel tambahan:

- **Confusion Matrix Heatmap** (2-panel: Category + Priority)
  — Blues=SVM, Greens=RF, Oranges=LR, Purples=BERT
- **Markdown Penjelasan** — cara kerja model, kelebihan/kekurangan, cara baca confusion matrix

Sel LR dan BERT dibungkus `if lr_available:` / `if bert_available:` agar tidak error saat di-skip.

### Section 16 — Visualisasi Perbandingan

Grouped bar chart (Accuracy + F1) membandingkan semua skema sekaligus,
disimpan ke `comparison_chart.png`. Dilengkapi tabel karakteristik (cara kerja,
kelebihan, kekurangan) dan panduan membaca grafik.

### Excel Column Comments

Setiap header kolom pada ketiga sheet output (`Predictions_Compare`, `Metrics`, `Summary`)
dilengkapi **comment/note** (tooltip segitiga merah) yang menjelaskan arti kolom —
termasuk kolom hybrid dinamis yang namanya bergantung pada model GenAI yang dipakai.

### Skema yang Dibandingkan

| Skema | Label di Output |
| --- | --- |
| SVM | `svm` |
| Random Forest | `random_forest` |
| Logistic Regression | `logistic_regression` |
| BERT | `bert` |
| GenAI-only | `genai_{model_key}` |
| Hybrid SVM | `hybrid_svm_{model_key}` |
| Hybrid RF | `hybrid_rf_{model_key}` |
| Hybrid LR | `hybrid_lr_{model_key}` |
| Hybrid BERT | `hybrid_bert_{model_key}` |

---

## Phase 5 — Analisis & Laporan (Opsional / Selanjutnya)

**Status:** Belum dimulai

### Rencana

- [ ] Visualisasi perbandingan akurasi antar skema (bar chart / heatmap)
- [ ] Analisis cost-efficiency Hybrid vs GenAI-only (berapa % baris yang di-callout ke GenAI)
- [ ] Laporan ringkas: tabel perbandingan 9 skema untuk category & priority
- [ ] Rekomendasi skema terbaik berdasarkan trade-off akurasi vs biaya API

---

## Catatan Teknis

| Topik | Keputusan |
| --- | --- |
| BERT default | `distilbert-base-multilingual-cased` — lebih cepat, cocok CPU |
| Skip BERT | Flag `--skip-bert` / variabel `SKIP_BERT` di notebook |
| Skip LR | Flag `--skip-lr` / variabel `SKIP_LR` di notebook |
| Multi-model GenAI | Satu run bisa tes beberapa model sekaligus via `--models a,b,c` |
| Output fallback | Jika file terkunci, simpan ke `*_new.xlsx` otomatis |
| Hybrid parameter | `classify_with_genai()` pakai `ml_category`/`ml_priority` (generik) |
| Confusion matrix import | `confusion_matrix`, `ConfusionMatrixDisplay` ditambah di sel SVM (pertama jalan) |
| Visualisasi chart | Disimpan ke `comparison_chart.png` via `plt.savefig()` |
| Excel comments | Kolom hybrid ditangani via prefix matching, bukan exact dict key |
