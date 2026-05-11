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

**Paired t-test (Hybrid Fusion vs SVM, per-fold):**

- Acc Cat: **p = 0.007** (signifikan p<0.01)
- F1 Cat: **p = 0.004** (signifikan p<0.01)
- Konsisten unggul di **semua 5 fold**

## Analysis Tambahan untuk Paper Q2

### Tahap 2 — Hybrid Voting Ensemble (Decision-Level)

| Approach | Acc Cat | F1 Cat | Cost |
|---|---|---|---|
| SVM | 0.8146 | 0.6698 | Free |
| Hybrid Fusion | 0.8250 | 0.6881 | $0.03 + 5min |
| GenAI Voter alone | 0.5346 | 0.4274 | $3 + 3h |
| Hybrid Voting (3-way) | 0.8265 | 0.6888 | $3 + 3h |

**Finding:** Voting Ensemble (3-way: SVM + Fusion + GenAI Voter) marginal vs Fusion (+0.15% Acc). Voting agreement dengan Fusion 99.2% → adding GenAI Voter doesn't shift voting outcomes much. Cost-efficiency tidak worth it.

### Tahap 3 — Anchor Bias Ablation (5 prompt variants, N=300 per variant)

| Variant | Override | Correction | LLM Correct |
|---|---|---|---|
| V1 NO_ML | 52.3% | 8.0% | 45.3% |
| V2 NEUTRAL_ML | 48.7% | 7.0% | 44.7% |
| V3 DEFER_ML (original Hybrid Correction prompt) | 54.0% | 8.0% | 43.3% |
| V4 CHALLENGE_ML | 51.7% | 6.7% | 43.7% |
| **V5 TOP3_CHOICES** ⭐ | **37.0%** | **12.7%** | **56.7%** |

**Finding:** Anchor framing (mentioning ML prediction in prompt) tidak signifikan affect LLM correction quality. Sebelumnya "0% override" di Hybrid Correction asli akibat **JSON parsing bug** (markdown code fence), bukan anchor bias. Constraining label space (V5 top-3) menghasilkan +13.3% LLM correct rate vs unconstrained variants.

### Tahap 3b — Label Space Size Ablation (K=3,5,7,10,19, N=300)

| K | Override | Correction | LLM Correct |
|---|---|---|---|
| **3** | 35.3% | **12.0%** | **56.3%** |
| 5 | 39.0% | 9.3% | 52.7% |
| 7 | 41.3% | 9.3% | 52.3% |
| 10 | 45.7% | 9.3% | 49.7% |
| 19 (all) | 51.7% | 7.0% | 45.3% |

**Finding:** **Monotonic relationship** — makin besar K, makin buruk LLM correction quality. From K=3 to K=19, LLM correct rate turun 56.3% → 45.3% (relative drop 24%). **Strong evidence:** label space size adalah real lever, bukan anchor framing. Practical guideline: gunakan top-3 ML candidates untuk LLM hybrid correction.

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
