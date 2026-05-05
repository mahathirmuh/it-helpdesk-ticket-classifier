# Phase — Tahapan Pengerjaan

## Phase 1 — Baseline ML (SVM & Random Forest)

**Status:** Selesai

### Deliverables

- [x] `train_svm.py` — TF-IDF + LinearSVC, output `cobacek_pred.xlsx`
- [x] `train_rf.py` — TF-IDF + RandomForestClassifier, output `cobacek_rf_pred.xlsx`

### Hasil

- Evaluasi terpisah: category model & priority model
- Metrik: accuracy + classification report per kelas
- Output Excel: sheet `Predictions`, `Category Accuracy`, `Priority Accuracy`

---

## Phase 2 — Logistic Regression

**Status:** Selesai

### Deliverables

- [x] `train_logres.py` — TF-IDF + LogisticRegression, output `cobacek_lr_pred.xlsx`

### Detail Implementasi

- `LogisticRegression(max_iter=1000, random_state=42)` dengan TF-IDF (unigram+bigram)
- Struktur identik dengan SVM dan RF

---

## Phase 3 — BERT Fine-Tuning

**Status:** Selesai

### Deliverables

- [x] `src/bert_classifier.py` — modul inti; sklearn-compatible wrapper untuk HuggingFace BERT
- [x] `train_bert.py` — standalone trainer

### Detail Implementasi

- Model default: `distilbert-base-multilingual-cased` (multilingual, lebih cepat dari BERT penuh)
- PyTorch DataLoader dengan custom `_TextDataset`
- `LabelEncoder` untuk konversi label string → integer dan sebaliknya
- Otomatis deteksi GPU (CUDA) / fallback ke CPU
- Di CPU: ~6,3 jam untuk 16K baris dengan 3-fold, 2 epoch, batch 32

---

## Phase 4 — Komparasi Multi-Skema dengan GenAI

**Status:** Selesai

### Deliverables

- [x] `compare_svm_genai.ipynb` — notebook komparasi lengkap (17+ seksi)

### Alur Kerja

1. Load & validasi dataset (`cobacek.xlsx`, 16.338 baris)
2. Stratified K-Fold (5 fold) → train & prediksi OOF untuk SVM, RF, LR
3. Stratified K-Fold (3 fold) → train & prediksi OOF untuk BERT
4. Tentukan baris mismatch SVM (salah di category **atau** priority) → 6.607 baris
5. Loop per GenAI model: Hybrid SVM — SVM prediksi dasar, GenAI koreksi mismatch
6. Kumpulkan semua metrik → sort descending by accuracy
7. Export Excel `hasil_final.xlsx`: `Predictions_Compare`, `Metrics`, `Summary`, `Category_Analysis`

### Model GenAI yang Digunakan

- `gpt-5.4-mini`
- `gpt-4.1-mini`
- `gpt-4o-mini`

### Hasil Utama (OOF, 16.338 baris)

| Model | Acc Category | Acc Priority | Waktu |
|---|---|---|---|
| SVM | **0.8122** | 0.7250 | 70 s |
| BERT | 0.7868 | 0.4940 | 22.607 s |
| LR | 0.7734 | 0.6422 | 583 s |
| RF | 0.7648 | 0.7175 | 317 s |
| Hybrid (gpt-4o-mini) | 0.7301 | **0.7503** | 9.241 s |
| Hybrid (gpt-5.4-mini) | 0.6919 | 0.7341 | 7.144 s |
| Hybrid (gpt-4.1-mini) | 0.6697 | 0.7438 | 8.935 s |

### Excel Column Comments

Setiap header kolom pada keempat sheet output dilengkapi **comment/note** (tooltip segitiga merah)
yang menjelaskan arti kolom — termasuk kolom hybrid dinamis dan kolom `acc_*` per kategori.

---

## Phase 5 — Analisis Lanjutan & Visualisasi

**Status:** Sebagian selesai

### Deliverables

- [x] Confusion Matrix per model (Category + Priority) — `paper/figures/confusion_matrix_{model}.png`
- [x] Per-category accuracy analysis — sheet `Category_Analysis` di `hasil_final.xlsx`
- [ ] Perbarui bar chart di `paper/figures/` dengan data terbaru (fig5–fig7)
- [ ] Analisis lebih lanjut: mengapa Hybrid menurunkan akurasi kategori (+16K data)
- [ ] Bandingkan efek prompt GenAI lama vs baru pada dataset yang sama

---

## Catatan Teknis

| Topik | Keputusan |
| --- | --- |
| Metode evaluasi | Stratified K-Fold OOF (tidak ada data leakage) |
| BERT default | `distilbert-base-multilingual-cased` — lebih cepat, cocok CPU |
| Skip BERT/LR | Variabel `SKIP_BERT` / `SKIP_LR` di notebook |
| Multi-model GenAI | Satu run bisa tes beberapa model sekaligus via `MULTI_MODELS` |
| Hybrid target | Hanya Hybrid SVM yang diimplementasi (6.607 baris mismatch dikoreksi GenAI) |
| Output file | `hasil_final.xlsx` — 4 sheet dengan header tooltip |
| Confusion matrix | Disimpan ke `paper/figures/` DPI 150, tanpa anotasi jika >30 kelas |
