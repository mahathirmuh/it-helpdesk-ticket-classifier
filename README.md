# Rekaya Perangkat Lunak
## Nama Peserta Kelompok & NRP
<table>
  <tr>
    <td>
      <div>Nama Peserta:</div>
      <ul>
        <li>Ananta Dwi Prayoga Alwy - 6025252007</li>
        <li>Dhayu Intan Nareswari - 6025252005</li>
        <li>Mahathir Muhammad - 6025252008</li>
      </ul>
    </td>
    <td align="right" width="40%">
      <img src="https://media.giphy.com/media/L8K62iTDkzGX6/giphy.gif" alt="Tim" />
    </td>
  </tr>
</table>


# Klasifikasi Tiket IT Helpdesk — Multi-Skema Komparasi

Proyek ini membandingkan beberapa skema klasifikasi teks untuk memprediksi **category** dan **priority** tiket IT helpdesk dari kolom `description`.

## Skema yang Dibandingkan

| # | Skema | Pendekatan |
|---|-------|------------|
| 1 | SVM | TF-IDF + LinearSVC |
| 2 | Random Forest | TF-IDF + RandomForestClassifier |
| 3 | Logistic Regression | TF-IDF + LogisticRegression |
| 4 | BERT | Fine-tuned DistilBERT multilingual |
| 5 | **Hybrid SVM-GenAI (Fusion)** | TF-IDF + OpenAI Embedding (semantic) → LinearSVC |
| 6 | Hybrid SVM-GenAI (Voting, opsional) | Majority vote dari SVM + Fusion + LLM voter (gpt-4.1-mini) |

### Arsitektur Hybrid SVM-GenAI (Fusion)

Menggabungkan dua representasi teks:

```
                ┌─→ TF-IDF (lexical, ~50.000 dim, sparse) ──┐
Teks tiket ─────┤                                           ├─→ Concat → LinearSVC → Prediksi
                └─→ OpenAI Embedding (semantic, 1536 dim) ──┘
                    text-embedding-3-small via API
```

- **TF-IDF** menangkap kata-kata spesifik (bagus untuk istilah teknis).
- **OpenAI Embedding** menangkap arti/sinonim/parafrase (bagus untuk variasi bahasa).
- Penggabungan keduanya memberi SVM dua sumber informasi → akurasi lebih tinggi.

---

## Hasil Utama

### 5-Fold Stratified CV (filtered dataset, 16.338 tiket, 19 kategori)

| Model | Acc Cat (mean ± std) | F1 Cat | Acc Pri | F1 Pri |
|---|---|---|---|---|
| **Hybrid SVM-GenAI (Fusion)** ✅ | **0.8214 ± 0.003** | **0.6738 ± 0.009** | **0.7219 ± 0.009** | **0.7097 ± 0.008** |
| SVM | 0.8125 ± 0.004 | 0.6535 ± 0.015 | 0.7167 ± 0.011 | 0.7033 ± 0.009 |
| Random Forest | 0.7660 ± 0.005 | 0.5353 ± 0.018 | 0.7172 ± 0.009 | 0.6838 ± 0.010 |
| Logistic Regression | 0.7764 ± 0.005 | 0.4713 ± 0.007 | 0.6335 ± 0.008 | 0.5864 ± 0.008 |

**Paired t-test (per-fold, Hybrid Fusion vs SVM):** Acc Cat **p=0.007** (signifikan), F1 Cat **p=0.004** (signifikan). Hybrid Fusion unggul SVM di **semua 5 fold**.

> Run: `python src/compare_svm_genai.py --skip-bert --n-folds 5`
> Heatmap: `results/heatmap_kfold.png`

### Findings Tambahan untuk Paper

| # | Finding | File Bukti |
|---|---|---|
| 1 | **Hybrid Fusion > SVM signifikan** (p<0.01) di 5-fold CV | `results/analysis_phase1.xlsx` |
| 2 | **Hybrid Voting marginal** vs Fusion (+0.15% Acc, cost 100×) — voting agreement 99.2% dengan Fusion | `results/analysis_phase2_v2.xlsx` |
| 3 | **Anchor framing TIDAK signifikan** affect LLM correction — V1 NO_ML, V3 DEFER_ML, V4 CHALLENGE_ML semua hasilkan ~43-45% LLM correct rate | `results/anchor_bias_ablation.xlsx` |
| 4 | **Label space size IS the lever** ⭐ — top-K shortlist monotonic: K=3 → 56.3% correct, K=19 → 45.3% (-11 pp) | `results/topk_ablation.xlsx` |

Top-K ablation visualisasi: `results/figures_phase3/topk_ablation.png`

---

## Struktur Folder

```
rpl-svm1/
├── .env                          # API key + config (gitignored)
├── .env.example                  # Template config (commit-able)
├── .gitignore
├── README.md
├── requirements.txt
│
├── src/                          # Kode utama (jalankan dari project root)
│   ├── bert_classifier.py        # Wrapper BERT sklearn-compatible
│   ├── compare_svm_genai.py      # Pipeline komparasi semua skema (utama)
│   ├── visualize_results.py      # Generate heatmap dari hasil Excel
│   ├── analysis_phase1.py        # Per-class F1 + paired t-test
│   ├── analysis_phase2.py        # 3-architecture comparison + voting breakdown
│   ├── anchor_bias_ablation.py   # 5 prompt variants ablation study
│   ├── topk_ablation.py          # Label space size sweep (K=3..19)
│   ├── train_svm.py              # Standalone trainer: SVM
│   ├── train_rf.py               # Standalone trainer: Random Forest
│   ├── train_logres.py           # Standalone trainer: Logistic Regression
│   └── train_bert.py             # Standalone trainer: BERT
│
├── notebooks/                    # Jupyter notebooks (interaktif)
│   └── compare_svm_genai.ipynb   # Notebook utama komparasi semua skema
│
├── data/                         # Dataset input
│   └── cobacek_filtered.xlsx     # Dataset utama (19 kategori, default)
│
├── results/                      # Output eksperimen (current)
│   ├── cobacek_filtered_compare.xlsx       # Single split — Hybrid Fusion menang
│   ├── cobacek_filtered_kfold.xlsx         # 5-fold CV mean ± std
│   ├── cobacek_filtered_kfold_fold[0-4].xlsx
│   ├── voting_gpt41mini_v2.xlsx            # Voting Ensemble (Tahap 2)
│   ├── analysis_phase1.xlsx                # Per-class F1 + paired t-test
│   ├── analysis_phase2_v2.xlsx             # 3-architecture comparison
│   ├── anchor_bias_ablation.xlsx           # 5 prompt variants (Tahap 3)
│   ├── topk_ablation.xlsx                  # Label space size sweep (Tahap 3b)
│   ├── heatmap_filtered.png                # Heatmap single split
│   ├── heatmap_kfold.png                   # Heatmap 5-fold mean
│   ├── figures_phase1/                     # Per-class F1 plots
│   ├── figures_phase2/                     # 3-architecture comparison plot
│   ├── figures_phase3/                     # Anchor bias + top-K plots
│   └── archive/                            # Hasil eksperimen lama (14 file)
│
├── docs/                         # Dokumentasi rencana
│   ├── plan.md
│   └── phase.md
│
└── paper/                        # Paper akademik (.docx) + gambar
    ├── IT_Helpdesk_Ticket_Classifier_Paper_V2.docx   # Versi terbaru
    ├── IT_Helpdesk_Ticket_Classifier_Paper_FINAL.docx
    ├── IT_Helpdesk_Ticket_Classifier_Paper.docx
    ├── Bab_4_Metodologi.docx
    └── figures/
        ├── fig1_category_dist.png
        ├── fig2_priority_dist.png
        ├── fig3_pipeline.png
        ├── fig4_hybrid_arch.png
        ├── fig5_accuracy_compare.png
        ├── fig6_macro_f1_compare.png
        ├── fig7_time_compare.png
        ├── confusion_matrix_svm.png
        ├── confusion_matrix_rf.png
        ├── confusion_matrix_lr.png
        ├── confusion_matrix_bert.png
        ├── confusion_matrix_hybrid_gpt_5_4_mini.png
        ├── confusion_matrix_hybrid_gpt_4_1_mini.png
        └── confusion_matrix_hybrid_gpt_4o_mini.png
```

---

## Instalasi

```bash
pip install -r requirements.txt
```

Atau manual:
```bash
pip install pandas scikit-learn openpyxl openai torch transformers python-docx matplotlib
```

---

## Penggunaan

> Semua perintah dijalankan dari **project root** (`rpl-svm1/`).

### 1. Standalone Trainer (tanpa GenAI)

```bash
python src/train_svm.py
python src/train_rf.py
python src/train_logres.py
python src/train_bert.py --bert-epochs 3
```

### 2. Komparasi Lengkap (semua skema)

```bash
# Run default — pakai cobacek_filtered.xlsx + category_filtered (19 kelas)
python src/compare_svm_genai.py --skip-bert

# Aktifkan Hybrid Voting Ensemble (mahal: 1 GenAI call per test row)
python src/compare_svm_genai.py --skip-bert --enable-voting --model gpt-4.1-mini

# 5-fold Cross Validation (paper credibility)
python src/compare_svm_genai.py --skip-bert --n-folds 5
```

> **Setup `.env`:** Copy `.env.example` ke `.env`, isi `OPENAI_API_KEY`. Konfigurasi env yang dipakai:
>
> | Var | Wajib? | Fungsi |
> |---|---|---|
> | `OPENAI_API_KEY` | ✅ | API key untuk Embedding & Chat API |
> | `OPENAI_EMBED_MODEL` | Opsional | Embedding model untuk Hybrid Fusion (default `text-embedding-3-small`) |
> | `OPENAI_MODELS` | Opsional | Chat models untuk Hybrid Voting (`--enable-voting`); pisah koma untuk multi-model |
>
> Biaya tipikal: Fusion ~$0.03/run, Voting ~$2-3/run di 3268 test rows.

### 3. Visualisasi Heatmap

```bash
python src/visualize_results.py \
    --input results/cobacek_filtered_compare.xlsx \
    --output results/heatmap_filtered.png \
    --title "PERBANDINGAN HYBRID SVM-GENAI (FILTERED)"
```

### 4. Notebook Interaktif

```bash
jupyter notebook notebooks/compare_svm_genai.ipynb
```

### 5. Analysis Pipeline (untuk Paper)

```bash
# Tahap 1 — per-class F1 + paired t-test (free, ~30 detik)
python src/analysis_phase1.py

# Tahap 2 — 3-architecture comparison (butuh voting run dulu)
python src/analysis_phase2.py --input results/voting_gpt41mini_v2.xlsx

# Tahap 3 — anchor bias ablation (5 prompt variants × 300 samples, ~$2, ~1.5 jam)
python src/anchor_bias_ablation.py --n-samples 300

# Tahap 3b — top-K label space sweep (5 K values × 300 samples, ~$2, ~45 menit)
python src/topk_ablation.py --n-samples 300 --topk-list 3,5,7,10,19
```

### 6. Open-Source Embedding Alternative (SBERT)

Jalankan Hybrid Fusion dengan SBERT (sentence-transformers) sebagai alternatif gratis ke OpenAI Embedding:

```bash
python src/compare_svm_genai.py --skip-bert \
    --embed-backend sbert --embed-model all-MiniLM-L6-v2 \
    --output results/cobacek_filtered_sbert.xlsx
```

CPU-inferable, **no API key needed**. Performance comparable dengan OpenAI Embedding (Acc Cat 0.8277 vs 0.8250).

---

## Konfigurasi Utama

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `--input` | `data/cobacek_filtered.xlsx` | File dataset (.xlsx) |
| `--output` | `results/cobacek_filtered_compare.xlsx` | File hasil |
| `--category-col` | `category_filtered` | Nama kolom target (pakai `category` untuk 81 kelas asli) |
| `--model` | (auto-detect) | Paksa model GenAI tertentu untuk voting |
| `--models` | (auto-detect) | Daftar model dipisah koma (multi-model voting) |
| `--bert-model` | `distilbert-base-multilingual-cased` | HuggingFace model ID |
| `--bert-epochs` | `3` | Epoch fine-tuning BERT |
| `--skip-bert` | — | Lewati training BERT (sangat lambat di CPU) |
| `--skip-lr` | — | Lewati Logistic Regression |
| `--skip-fusion` | — | Lewati Hybrid SVM Fusion (skip embedding API) |
| `--embed-model` | `text-embedding-3-small` | OpenAI embedding model untuk Fusion |
| `--enable-voting` | — | Aktifkan Hybrid Voting Ensemble (mahal) |
| `--n-folds` | `1` | Jumlah fold (>1 = Stratified K-Fold CV) |
| `--base-seed` | `42` | Random state untuk train/test split |

---

## Dataset

`data/cobacek_filtered.xlsx` (16.338 baris, 19 kategori)

| Kolom | Keterangan |
|-------|------------|
| `subject` | Judul tiket |
| `description` | Detail masalah — **input utama model** |
| `answer` | Solusi yang diberikan |
| `type` | Tipe tiket dari sumber data |
| `category` | Label kategori asli — 81 kelas granular (tidak dipakai sebagai target) |
| `category_filtered` | **Target kategori (default)** — 19 kelas yang sudah digabung: Security, Bug, Feedback, Feature, Performance, Billing, Outage, Network, Documentation, Product, Crash, Disruption, Marketing, Login, IT, Sales, Hardware, Customer Support, Other |
| `priority` | Label priority (ground truth) — 3 kelas: low, medium, high |

---

## Metodologi

- **Stratified Train/Test Split 80/20** — by category, `random_state=42`. Train: 13.070 baris, Test: 3.268 baris.
- **Semua metrik dihitung di test set** (tidak overfit ke training).
- **Macro F1** dipakai sebagai metrik utama untuk fairness terhadap kelas minoritas.
- **5-Fold Stratified CV** tersedia via `--n-folds 5` untuk paper credibility (mean ± std).

### Output Excel (per run)

| Sheet | Isi |
|-------|------|
| `Predictions_Compare` | Tiap baris test set + prediksi per model + audit columns |
| `Metrics` | Tabel metrik lengkap (accuracy, precision, recall, F1, weighted F1) per model & label |
| `Summary` | Konfigurasi run (model, split strategy, seed, voting status) |

### Output K-Fold (`--n-folds N`)

| Sheet | Isi |
|-------|------|
| `Metrics_Aggregated` | Mean ± std per (approach, label) across folds |
| `Metrics_Per_Fold` | Detail metrik per fold |
| `Summary` | Konfigurasi K-Fold |

---

## Catatan

- **BERT di CPU** sangat lambat (~6,3 jam untuk 16K baris). Gunakan `--skip-bert` kalau tidak punya GPU.
- **Hybrid Fusion** menggunakan OpenAI Embedding API (~$0.03 per run di dataset 16K baris).
- **Hybrid Voting** sangat mahal (1 GenAI call per test row). Untuk 3268 baris ≈ ~$2-3 dan ~3 jam.
- **Hybrid Correction (decision-level)** sebelumnya dicoba tapi **tidak unggul SVM** karena prompt menyertakan "Current ML prediction" yang membuat GenAI anchored ke jawaban SVM (0/901 cat override). Diganti dengan Voting Ensemble (independen prediction).
