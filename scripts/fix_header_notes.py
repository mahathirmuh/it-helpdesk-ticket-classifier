import json

NB_PATH = "compare_svm_genai.ipynb"

NEW_CELL_31 = r"""import openpyxl
from openpyxl.comments import Comment

metrics_df = pd.DataFrame(metrics_rows)

summary_rows = [
    {"key": "selected_models",             "value": ", ".join(selected_models)},
    {"key": "bert_model",                  "value": BERT_MODEL_NAME if bert_available else "skipped"},
    {"key": "total_rows",                  "value": len(df)},
    {"key": "k_folds",                     "value": N_SPLITS},
    {"key": "svm_mismatch_rows_corrected", "value": len(svm_mismatch_indices)},
    {"key": "max_genai_rows",              "value": MAX_GENAI_ROWS},
]
summary_df = pd.DataFrame(summary_rows)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df.to_excel(writer,         index=False, sheet_name="Predictions_Compare")
    metrics_df.to_excel(writer, index=False, sheet_name="Metrics")
    summary_df.to_excel(writer, index=False, sheet_name="Summary")

# ── Keterangan header per sheet ──────────────────────────────────────────────
NOTES_PREDICTIONS = {
    "subject":      "Judul / subjek tiket IT helpdesk",
    "description":  "Deskripsi lengkap tiket — input utama yang digunakan semua model",
    "answer":       "Jawaban atau solusi yang diberikan untuk tiket ini",
    "type":         "Tipe tiket (contoh: Request, Incident)",
    "priority":     "Prioritas AKTUAL (ground truth) dari tiket",
    "category":     "Kategori AKTUAL (ground truth) dari tiket",
    "svm_category": "Prediksi kategori oleh SVM (TF-IDF + LinearSVC, OOF K-Fold)",
    "svm_priority": "Prediksi prioritas oleh SVM (TF-IDF + LinearSVC, OOF K-Fold)",
    "rf_category":  "Prediksi kategori oleh Random Forest (TF-IDF, OOF K-Fold)",
    "rf_priority":  "Prediksi prioritas oleh Random Forest (TF-IDF, OOF K-Fold)",
    "lr_category":  "Prediksi kategori oleh Logistic Regression (TF-IDF, OOF K-Fold)",
    "lr_priority":  "Prediksi prioritas oleh Logistic Regression (TF-IDF, OOF K-Fold)",
    "svm_match":    "TRUE jika SVM benar di category DAN priority sekaligus",
    "rf_match":     "TRUE jika Random Forest benar di category DAN priority sekaligus",
    "lr_match":     "TRUE jika Logistic Regression benar di category DAN priority sekaligus",
    "bert_category": "Prediksi kategori oleh BERT (DistilBERT multilingual, OOF K-Fold)",
    "bert_priority": "Prediksi prioritas oleh BERT (DistilBERT multilingual, OOF K-Fold)",
    "bert_match":   "TRUE jika BERT benar di category DAN priority sekaligus",
}

NOTES_METRICS = {
    "approach":         "Nama model / pendekatan yang dibandingkan (SVM, RF, LR, BERT, Hybrid SVM)",
    "label":            "Label yang dievaluasi: 'category' atau 'priority'",
    "accuracy":         "Akurasi = jumlah prediksi benar / total baris",
    "macro_precision":  "Precision rata-rata antar kelas (tidak mempertimbangkan jumlah sampel per kelas)",
    "macro_recall":     "Recall rata-rata antar kelas (tidak mempertimbangkan jumlah sampel per kelas)",
    "macro_f1":         "F1-score rata-rata antar kelas (tidak mempertimbangkan jumlah sampel per kelas)",
    "weighted_f1":      "F1-score rata-rata tertimbang — kelas dengan sampel lebih banyak diberi bobot lebih besar",
    "samples":          "Jumlah total baris yang dievaluasi",
}

NOTES_SUMMARY = {
    "key":   "Nama parameter konfigurasi yang digunakan saat menjalankan notebook",
    "value": "Nilai dari parameter konfigurasi tersebut",
}

def add_header_comments(ws, notes_dict, hybrid_prefix=False):
    for cell in ws[1]:
        col_name = cell.value
        if col_name is None:
            continue
        note = notes_dict.get(col_name)
        if note is None and hybrid_prefix:
            if col_name.startswith("hybrid_svm_category_"):
                model_id = col_name[len("hybrid_svm_category_"):]
                note = f"Prediksi kategori Hybrid SVM: SVM dikoreksi GenAI ({model_id}) untuk baris mismatch"
            elif col_name.startswith("hybrid_svm_priority_"):
                model_id = col_name[len("hybrid_svm_priority_"):]
                note = f"Prediksi prioritas Hybrid SVM: SVM dikoreksi GenAI ({model_id}) untuk baris mismatch"
        if note:
            comment = Comment(note, "System")
            comment.width = 340
            comment.height = 60
            cell.comment = comment

wb = openpyxl.load_workbook(OUTPUT_FILE)
add_header_comments(wb["Predictions_Compare"], NOTES_PREDICTIONS, hybrid_prefix=True)
add_header_comments(wb["Metrics"],             NOTES_METRICS)
add_header_comments(wb["Summary"],             NOTES_SUMMARY)
wb.save(OUTPUT_FILE)

print(f"Output tersimpan: {OUTPUT_FILE} ({len(df)} baris total)")
print("Keterangan header ditambahkan di semua sheet (hover mouse ke nama kolom untuk melihat)")
display(summary_df)
"""

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

updated = []
for cell in nb["cells"]:
    if cell.get("id") == "a1000031":
        cell["source"] = [NEW_CELL_31]
        updated.append("a1000031")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Updated cells: {updated}")
