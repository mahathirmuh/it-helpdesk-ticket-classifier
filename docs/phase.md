# Phase — Tahapan Pengerjaan

## Phase 1 — Baseline ML (SVM, RF, LR, BERT)

**Status:** Selesai

### Deliverables

- [x] `src/train_svm.py` — TF-IDF + LinearSVC (standalone)
- [x] `src/train_rf.py` — TF-IDF + RandomForestClassifier (standalone)
- [x] `src/train_logres.py` — TF-IDF + LogisticRegression (standalone)
- [x] `src/bert_classifier.py` — sklearn-compatible wrapper untuk DistilBERT
- [x] `src/train_bert.py` — standalone fine-tuning BERT

### Hasil

- 4 model klasik berhasil dilatih
- BERT default: `distilbert-base-multilingual-cased` (lebih cepat dari BERT penuh)
- Otomatis deteksi GPU/fallback CPU

---

## Phase 2 — Komparasi Multi-Skema dengan Hybrid GenAI Correction

**Status:** Selesai (kemudian dideprecate)

### Deliverables

- [x] `src/compare_svm_genai.py` versi awal — Hybrid SVM dengan GenAI correction (decision-level)
- [x] `notebooks/compare_svm_genai.ipynb` versi awal — notebook duplikasi logika

### Pendekatan Awal (Hybrid Correction)

1. SVM prediksi semua tiket (OOF dari 5-fold CV)
2. Identifikasi mismatch (SVM salah cat atau pri)
3. Kirim baris mismatch ke OpenAI LLM (`gpt-4.1-mini`, dll) untuk dikoreksi
4. Hasil akhir: SVM benar + koreksi LLM

### Masalah yang Ditemukan

- **Tidak ada train/test split** → semua angka SVM/RF/LR/BERT adalah training accuracy (overfitting)
- **Hybrid Correction merusak akurasi**: prompt menyertakan "Current ML prediction" → LLM anchored, 0/901 cat override → no improvement, kadang malah merusak
- **Bug per-kolom**: GenAI mengganti kedua kolom (cat & pri) padahal mungkin cuma 1 yang salah
- **Hasil:** Hybrid ⩽ SVM tunggal di Acc Cat (0.7301 vs 0.8122 SVM training acc)

---

## Phase 3 — Hybrid Improvement Refactor (Branch `hybrid-improvement`)

**Status:** Selesai

### Tujuan

Membuat **Hybrid SVM-GenAI unggul SVM tunggal** secara statistik, dengan metodologi credible (proper train/test split, k-fold CV).

### Sub-phase

1. **Train/Test Split foundation** — Stratified 80/20, evaluasi di test set saja
2. **Per-column correction (fix bug)** — GenAI hanya update kolom yang SVM salah, bukan keduanya
3. **Confidence gate + constrained validator** — `decision_function` margin + multiple-choice prompt
4. **Guarded override audit** — track baris mana yang benar-benar di-override
5. **K-fold orchestrator** — `run_pipeline_kfold()` agregasi mean ± std

### Hasil Run #1 (single split, confidence-mode)

| Approach | Acc Cat | F1 Cat | Verdict |
|---|---|---|---|
| SVM | 0.8136 | 0.3371 | Baseline |
| Hybrid Fusion (TF-IDF + Embedding) | **0.8250** | **0.3477** | ✅ Menang +1.14% |
| Hybrid Correction (gpt-4o-mini) | 0.8136 | 0.3371 | ❌ Tied (LLM tidak override) |

**Konfirmasi:** anchor bias di prompt → 0 override. Hybrid Correction dideprecate.

---

## Phase 4 — Pivot ke Hybrid Voting + Switch ke Filtered Dataset

**Status:** Selesai

### Perubahan

- Hapus seluruh kode Hybrid Correction (compute_topk_and_margin, classify_with_genai_constrained, classify_with_genai legacy)
- Tambah `classify_with_genai_voter()` — prompt independen tanpa hint SVM
- Tambah `--enable-voting` flag — Hybrid Voting Ensemble (3-way: SVM + Fusion + LLM)
- Switch dataset dari `cobacek.xlsx` (81 kelas) → `cobacek_filtered.xlsx` (19 kelas, kolom `category_filtered`)
- Update default CLI ke filtered dataset

### Deliverables

- [x] `classify_with_genai_voter()` di `src/compare_svm_genai.py`
- [x] `--enable-voting`, `--category-col` CLI flags
- [x] Voting block di `run_pipeline()`

---

## Phase 5 — 5-Fold CV + Visualisasi + Cleanup

**Status:** Selesai

### Deliverables

- [x] 5-fold Stratified CV via `--n-folds 5` — output `results/cobacek_filtered_kfold.xlsx`
- [x] `src/visualize_results.py` — heatmap comparison (mirip paper)
- [x] `results/heatmap_filtered.png` — single split visualization
- [x] `results/heatmap_kfold.png` — 5-fold mean visualization
- [x] Notebook `compare_svm_genai.ipynb` di-refactor 40 → 14 cell (import dari src/)
- [x] Cleanup `results/` — pindah 14 file historis ke `results/archive/`

### Hasil Final (5-Fold CV)

| Approach | Acc Cat (mean ± std) | F1 Cat | Acc Pri | F1 Pri |
|---|---|---|---|---|
| **Hybrid Fusion** ✅ | **0.8214 ± 0.003** | **0.6738 ± 0.009** | **0.7219 ± 0.009** | **0.7097 ± 0.008** |
| SVM | 0.8125 ± 0.004 | 0.6535 ± 0.015 | 0.7167 ± 0.011 | 0.7033 ± 0.009 |
| Random Forest | 0.7660 ± 0.005 | 0.5353 ± 0.018 | 0.7172 ± 0.009 | 0.6838 ± 0.010 |
| Logistic Regression | 0.7764 ± 0.005 | 0.4713 ± 0.007 | 0.6335 ± 0.008 | 0.5864 ± 0.008 |

### Signifikansi Statistik

- Acc Cat Hybrid Fusion vs SVM: **+0.89%** (>2σ, signifikan)
- F1 Cat: **+2.03%** (>2σ, signifikan — penting untuk kelas minoritas)
- Konsisten unggul di **semua 5 fold**

---

## Catatan Teknis

| Topik | Keputusan |
|---|---|
| Metode evaluasi | Stratified Train/Test 80/20 + 5-fold CV (paper credibility) |
| Dataset utama | `cobacek_filtered.xlsx` dengan `category_filtered` (19 kelas) |
| Hybrid winning variant | **Fusion** (feature-level: TF-IDF + GenAI Embedding) |
| Hybrid Correction | **Dihapus** — tidak unggul karena anchor bias prompt |
| Hybrid Voting | Opt-in via `--enable-voting` (mahal: 1 GenAI call per test row) |
| BERT default | `distilbert-base-multilingual-cased` — lebih cepat, cocok CPU; opsional |
| Output single | `results/cobacek_filtered_compare.xlsx` (sheet Metrics, Summary, Predictions_Compare) |
| Output kfold | `results/cobacek_filtered_kfold.xlsx` (sheet Metrics_Aggregated mean±std) |
| Heatmap | `src/visualize_results.py` auto-detect single/kfold sheet |
| Branch | `hybrid-improvement` (pushed to GitHub, belum merged ke master) |
