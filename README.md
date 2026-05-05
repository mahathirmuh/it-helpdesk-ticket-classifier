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
| 5 | Hybrid SVM | SVM + GenAI koreksi mismatch SVM |
| 6 | Hybrid RF | RF + GenAI koreksi mismatch RF |
| 7 | Hybrid LR | LR + GenAI koreksi mismatch LR |
| 8 | Hybrid BERT | BERT + GenAI koreksi mismatch BERT |

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
│   ├── compare_svm_genai.py      # Komparasi lengkap semua skema
│   ├── train_svm.py              # Standalone trainer: SVM
│   ├── train_rf.py               # Standalone trainer: Random Forest
│   ├── train_logres.py           # Standalone trainer: Logistic Regression
│   └── train_bert.py             # Standalone trainer: BERT
│
├── notebooks/                    # Jupyter notebooks (interaktif)
│   ├── bert_classifier.py        # Wrapper BERT sklearn-compatible
│   ├── compare_svm_genai.ipynb
│   ├── train_svm.ipynb
│   ├── train_rf.ipynb
│   ├── train_logres.ipynb
│   └── train_bert.ipynb
│
├── scripts/                      # Utilitas dan script one-off
│   ├── add_excel_notes.py        # Tambah komentar header Excel
│   ├── fix_add_timing.py
│   ├── fix_header_notes.py
│   └── paper/                    # Generator paper akademik
│       ├── create_chapter4.py
│       ├── create_full_paper.py
│       ├── create_paper_with_actual_results.py
│       └── create_paper_v2_with_figures.py   # Versi terbaru (V2 + figures)
│
├── data/                         # Dataset input
│   ├── cobacek.xlsx              # Dataset utama (16.338 tiket)
│   ├── cobacek100data.xlsx       # Sample 100 baris
│   ├── cobacek50data.xlsx        # Sample 50 baris
│   └── cobacek20data.xlsx        # Sample 20 baris
│
├── results/                      # Output eksperimen (Excel)
│   ├── cobacek_compare_final.xlsx        # Hasil final 5-fold CV
│   ├── cobacek_compare.xlsx              # Hasil run terbaru
│   ├── cobacek_compare_3models.xlsx
│   ├── cobacek_compare_3models_test.xlsx
│   ├── cobacek_compare_auto.xlsx
│   ├── cobacek_compare_gpt-4.1-mini.xlsx
│   ├── cobacek_compare_gpt-4o-mini.xlsx
│   ├── cobacek_compare_gpt-5-mini.xlsx
│   ├── cobacek_compare_20data.xlsx
│   ├── cobacek_compare_20datates.xlsx
│   ├── cobacek_compare_test20.xlsx
│   └── cobacek_pred.xlsx                 # Output train_svm.py
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
        └── fig7_time_compare.png
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
    --output results/cobacek_compare.xlsx \
    --models gpt-4.1-mini

# Skip BERT (lebih cepat, tanpa GPU)
python src/compare_svm_genai.py --skip-bert

# Multi-model GenAI sekaligus
python src/compare_svm_genai.py --models gpt-4.1-mini,gpt-4o-mini
```

> **Catatan:** OpenAI API key harus tersedia di `.env` (`OPENAI_API_KEY=...`).

### 3. Notebook Interaktif

```bash
jupyter notebook notebooks/compare_svm_genai.ipynb
```

### 4. Generate Paper

```bash
# Paper V2 (terbaru, dengan 7 gambar dan abstract)
python scripts/paper/create_paper_v2_with_figures.py
```

Output: `paper/IT_Helpdesk_Ticket_Classifier_Paper_V2.docx` + figures di `paper/figures/`.

---

## Konfigurasi Utama

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `--input` | `data/cobacek.xlsx` | File dataset (.xlsx) |
| `--output` | `results/cobacek_compare.xlsx` | File hasil |
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
| `category` | Label category (ground truth) |
| `priority` | Label priority (ground truth) |

---

## Catatan

- **BERT di CPU** membutuhkan waktu lebih lama. Gunakan GPU (CUDA) atau `--skip-bert`.
- Jika file output Excel terkunci (sedang dibuka), hasil disimpan ke `*_new.xlsx`.
