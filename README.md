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
| 5 | Hybrid SVM | SVM sebagai dasar, GenAI koreksi mismatch SVM |

---

## Hasil Terbaik (16.338 tiket, Stratified K-Fold 5-fold)

| Model | Acc Category | Acc Priority |
|---|---|---|
| **SVM** | **0.8122** ✅ | 0.7250 |
| Hybrid SVM (gpt-4o-mini) | 0.7301 | **0.7503** ✅ |
| BERT | 0.7868 | 0.4940 |
| Logistic Regression | 0.7734 | 0.6422 |
| Random Forest | 0.7648 | 0.7175 |

> SVM unggul di kategori (81 kelas). Hybrid SVM + gpt-4o-mini unggul di prioritas.

---

## Struktur Folder

```
rpl-svm1/
├── .env                          # OPENAI_API_KEY
├── .gitignore
├── README.md
├── requirements.txt
│
├── src/                          # Kode utama (jalankan dari project root)
│   ├── bert_classifier.py        # Wrapper BERT sklearn-compatible
│   ├── compare_svm_genai.py      # Komparasi lengkap semua skema
│   ├── train_svm.py              # Standalone trainer: SVM
│   ├── train_rf.py               # Standalone trainer: Random Forest
│   ├── train_logres.py           # Standalone trainer: Logistic Regression
│   └── train_bert.py             # Standalone trainer: BERT
│
├── notebooks/                    # Jupyter notebooks (interaktif)
│   └── compare_svm_genai.ipynb   # Notebook utama komparasi semua skema
│
├── data/                         # Dataset input
│   ├── cobacek.xlsx              # Dataset utama (16.338 tiket)
│   ├── cobacek100data.xlsx       # Sample 100 baris
│   ├── cobacek50data.xlsx        # Sample 50 baris
│   └── cobacek20data.xlsx        # Sample 20 baris
│
├── results/                      # Output eksperimen (Excel)
│   └── hasil_final.xlsx          # Hasil final — 4 sheet: Predictions_Compare, Metrics, Summary, Category_Analysis
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
# Run default
python src/compare_svm_genai.py

# Override input/output
python src/compare_svm_genai.py \
    --input data/cobacek.xlsx \
    --output results/hasil_final.xlsx \
    --models gpt-4o-mini

# Skip BERT (lebih cepat, tanpa GPU)
python src/compare_svm_genai.py --skip-bert

# Multi-model GenAI sekaligus
python src/compare_svm_genai.py --models gpt-4.1-mini,gpt-4o-mini,gpt-5.4-mini
```

> **Catatan:** OpenAI API key harus tersedia di `.env` (`OPENAI_API_KEY=...`).

### 3. Notebook Interaktif

```bash
jupyter notebook notebooks/compare_svm_genai.ipynb
```

---

## Konfigurasi Utama

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `--input` | `data/cobacek.xlsx` | File dataset (.xlsx) |
| `--output` | `results/hasil_final.xlsx` | File hasil |
| `--models` | `gpt-4.1-mini` | ID model OpenAI (pisah koma) |
| `--bert-model` | `distilbert-base-multilingual-cased` | HuggingFace model ID |
| `--bert-epochs` | `3` | Epoch fine-tuning BERT |
| `--skip-bert` | — | Lewati training BERT |
| `--skip-lr` | — | Lewati Logistic Regression |

---

## Dataset

File `data/cobacek.xlsx` dengan kolom:

| Kolom | Keterangan |
|-------|------------|
| `subject` | Judul tiket |
| `description` | Detail masalah — **input utama model** |
| `answer` | Solusi yang diberikan |
| `type` | Tipe tiket dari sumber data |
| `category` | Label category (ground truth) — 81 kelas |
| `priority` | Label priority (ground truth) — 3 kelas: low, medium, high |

---

## Catatan

- **BERT di CPU** membutuhkan waktu sangat lama (~6,3 jam untuk 16K baris, 3-fold). Gunakan GPU (CUDA) atau `--skip-bert`.
- **Hybrid SVM** hanya mengkoreksi baris di mana SVM salah (6.607 dari 16.338 baris), bukan seluruh dataset.
- Output Excel `hasil_final.xlsx` memiliki 4 sheet: `Predictions_Compare`, `Metrics`, `Summary`, `Category_Analysis`.
