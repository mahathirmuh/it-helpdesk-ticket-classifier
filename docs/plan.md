# Plan — Klasifikasi Tiket IT Helpdesk: Multi-Skema Komparasi

## Tujuan

Membandingkan performa berbagai pendekatan klasifikasi teks pada dataset tiket IT helpdesk (`cobacek_filtered.xlsx`) untuk dua label target: **category** (19 kelas) dan **priority** (3 kelas), dengan fokus utama membuktikan **Hybrid SVM-GenAI** unggul vs single SVM.

---

## Skema Komparasi

| # | Skema | Metode |
|---|-------|--------|
| 1 | **SVM** | TF-IDF (unigram+bigram) + LinearSVC |
| 2 | **Random Forest** | TF-IDF (unigram+bigram) + RandomForestClassifier(n_estimators=200) |
| 3 | **Logistic Regression** | TF-IDF (unigram+bigram) + LogisticRegression(max_iter=1000) |
| 4 | **BERT** | Fine-tuned `distilbert-base-multilingual-cased` (HuggingFace) — opsional, lambat di CPU |
| 5 | **Hybrid SVM-GenAI (Fusion)** ✅ | TF-IDF + OpenAI Embedding (`text-embedding-3-small`) → LinearSVC. Feature-level fusion. **Winning variant** |
| 6 | **Hybrid SVM-GenAI (Voting)** | Majority vote: SVM + Fusion + LLM (`gpt-4.1-mini` prediksi independen). Decision-level. Opsional, mahal |

> **Hybrid Correction (decision-level via LLM correction)** sebelumnya dicoba namun **tidak unggul** karena prompt menyertakan "Current ML prediction" yang membuat GenAI anchored ke jawaban SVM (0/901 cat override). Variant ini telah **dihapus** dari pipeline.

---

## Dataset

- **File:** `data/cobacek_filtered.xlsx`
- **Kolom input:** `description`
- **Kolom target:**
  - `category_filtered` (19 kelas digabung dari 81 asli) — default
  - `priority` (3 kelas: low, medium, high)
- **Total baris:** 16.338 tiket
- **Split:** Stratified Train/Test 80/20 (`random_state=42`)
  - Train: 13.070 baris
  - Test: 3.268 baris
- **Cross-validation:** 5-fold Stratified CV via `--n-folds 5` untuk paper credibility

---

## Metrik Evaluasi

Setiap skema dievaluasi pada **test set** (tidak overfit) dengan:

- **Accuracy**
- **Macro Precision** — rata-rata antar kelas (fairness terhadap kelas minoritas)
- **Macro Recall**
- **Macro F1** — metrik utama untuk kelas imbalanced
- **Weighted F1**
- **Jumlah sampel** (support)

Output single split: `results/cobacek_filtered_compare.xlsx` dengan sheet `Predictions_Compare`, `Metrics`, `Summary`.

Output 5-fold CV: `results/cobacek_filtered_kfold.xlsx` dengan sheet `Metrics_Aggregated` (mean ± std), `Metrics_Per_Fold`, `Summary`.

---

## Visualisasi

- **Heatmap comparison** — tabel mirip paper, color-coded per kolom (`results/heatmap_filtered.png`, `results/heatmap_kfold.png`)
- **Per-Fold breakdown** — sheet `Metrics_Per_Fold` di output k-fold

---

## Konfigurasi Utama (CLI)

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `--input` | `data/cobacek_filtered.xlsx` | File dataset |
| `--output` | `results/cobacek_filtered_compare.xlsx` | File hasil |
| `--category-col` | `category_filtered` | Kolom target (`category` untuk 81 kelas asli) |
| `--n-folds` | `1` | 1 = single 80/20 split, ≥2 = StratifiedKFold |
| `--base-seed` | `42` | Random state untuk split |
| `--enable-voting` | `false` | Aktifkan Hybrid Voting Ensemble (mahal) |
| `--skip-bert` | `false` | Skip BERT (lambat di CPU) |
| `--skip-lr` | `false` | Skip Logistic Regression |
| `--skip-fusion` | `false` | Skip Fusion (hemat OpenAI Embedding API) |
| `--model` / `--models` | (auto) | Pilih model GenAI untuk voting (single/multi) |

---

## Hasil Final (5-Fold CV, filtered dataset)

| Approach | Acc Cat (mean ± std) | F1 Cat | Acc Pri | F1 Pri |
|---|---|---|---|---|
| **Hybrid Fusion** ✅ | **0.8214 ± 0.003** | **0.6738 ± 0.009** | **0.7219 ± 0.009** | **0.7097 ± 0.008** |
| SVM | 0.8125 ± 0.004 | 0.6535 ± 0.015 | 0.7167 ± 0.011 | 0.7033 ± 0.009 |
| Random Forest | 0.7660 ± 0.005 | 0.5353 ± 0.018 | 0.7172 ± 0.009 | 0.6838 ± 0.010 |
| Logistic Regression | 0.7764 ± 0.005 | 0.4713 ± 0.007 | 0.6335 ± 0.008 | 0.5864 ± 0.008 |

**Selisih Hybrid Fusion vs SVM:**

- Acc Cat: +0.89% (>2σ — signifikan secara statistik)
- F1 Cat: +2.03% (>2σ — penting untuk kelas minoritas)
- Acc Pri: +0.52% (marginal, ~0.5σ)
- Konsisten unggul di **semua 5 fold**

---

## Dependensi

```
pandas
openpyxl
scikit-learn
openai
python-dotenv
torch
transformers
matplotlib
numpy
scipy
```
