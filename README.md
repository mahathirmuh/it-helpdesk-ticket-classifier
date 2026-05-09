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

## Hasil — Filtered Dataset (16.338 tiket, 19 kategori, stratified 80/20 split)

| Model | Acc Cat | Acc Pri | F1 Cat (macro) | F1 Pri (macro) |
|---|---|---|---|---|
| **Hybrid SVM-GenAI (Fusion)** | **0.8250** ✅ | **0.7292** ✅ | **0.6881** ✅ | **0.7157** ✅ |
| SVM | 0.8146 | 0.7197 | 0.6698 | 0.7059 |
| Logistic Regression | 0.7723 | 0.6313 | 0.4654 | 0.5809 |
| Random Forest | 0.7619 | 0.7075 | 0.5394 | 0.6722 |

> **Hybrid SVM-GenAI (Fusion) unggul di semua metrik vs single SVM** (+1.04% Acc Cat, +0.95% Acc Pri, +1.83% F1 Cat).
> Run dengan `python src/compare_svm_genai.py --skip-bert` (default sudah pakai filtered dataset).
> Heatmap visualisasi lengkap di `results/heatmap_filtered.png`.

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
│   ├── heatmap_filtered.png                # Heatmap single split
│   ├── heatmap_kfold.png                   # Heatmap 5-fold mean
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
