import json

NB_PATH = "compare_svm_genai.ipynb"

NEW_CELLS = {}

# ── Cell a1000013: K-Fold loop + per-model timing ────────────────────────────
NEW_CELLS["a1000013"] = r"""lr_available = not SKIP_LR

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for col in ["svm_category", "svm_priority", "rf_category", "rf_priority"]:
    df[col] = ""
if lr_available:
    for col in ["lr_category", "lr_priority"]:
        df[col] = ""

print(f"K-Fold: {N_SPLITS} folds | {len(df)} baris total")
lbl = "+LR" if lr_available else ""
print(f"Training SVM + RF{lbl} per fold...")

svm_time = 0.0
rf_time  = 0.0
lr_time  = 0.0

for fold, (train_idx, test_idx) in enumerate(kf.split(df["description"], df["category"]), 1):
    train_desc = df["description"].iloc[train_idx]
    train_cat  = df["category"].iloc[train_idx]
    train_pri  = df["priority"].iloc[train_idx]
    test_desc  = df["description"].iloc[test_idx]
    test_index = df.index[test_idx]

    # SVM
    t0 = time.time()
    svm_c = build_svm_pipeline(); svm_p = build_svm_pipeline()
    svm_c.fit(train_desc, train_cat); svm_p.fit(train_desc, train_pri)
    df.loc[test_index, "svm_category"] = svm_c.predict(test_desc)
    df.loc[test_index, "svm_priority"]  = svm_p.predict(test_desc)
    svm_time += time.time() - t0

    # Random Forest
    t0 = time.time()
    rf_c = build_rf_pipeline(); rf_p = build_rf_pipeline()
    rf_c.fit(train_desc, train_cat); rf_p.fit(train_desc, train_pri)
    df.loc[test_index, "rf_category"] = rf_c.predict(test_desc)
    df.loc[test_index, "rf_priority"]  = rf_p.predict(test_desc)
    rf_time += time.time() - t0

    # Logistic Regression
    if lr_available:
        t0 = time.time()
        lr_c = build_lr_pipeline(); lr_p = build_lr_pipeline()
        lr_c.fit(train_desc, train_cat); lr_p.fit(train_desc, train_pri)
        df.loc[test_index, "lr_category"] = lr_c.predict(test_desc)
        df.loc[test_index, "lr_priority"]  = lr_p.predict(test_desc)
        lr_time += time.time() - t0

    print(f"  Fold {fold}/{N_SPLITS} selesai")

df["svm_match"] = (df["svm_category"] == df["category"]) & (df["svm_priority"] == df["priority"])
df["rf_match"]  = (df["rf_category"]  == df["category"]) & (df["rf_priority"]  == df["priority"])
if lr_available:
    df["lr_match"] = (df["lr_category"] == df["category"]) & (df["lr_priority"] == df["priority"])

print("\nOOF Accuracy:")
print(f"  SVM  Cat: {accuracy_score(df['category'], df['svm_category']):.4f} | Pri: {accuracy_score(df['priority'], df['svm_priority']):.4f} | Waktu: {svm_time:.1f}s")
print(f"  RF   Cat: {accuracy_score(df['category'], df['rf_category']):.4f} | Pri: {accuracy_score(df['priority'], df['rf_priority']):.4f} | Waktu: {rf_time:.1f}s")
if lr_available:
    print(f"  LR   Cat: {accuracy_score(df['category'], df['lr_category']):.4f} | Pri: {accuracy_score(df['priority'], df['lr_priority']):.4f} | Waktu: {lr_time:.1f}s")
"""

# ── Cell a1000019: BERT K-Fold + timing ──────────────────────────────────────
NEW_CELLS["a1000019"] = r"""bert_available = not SKIP_BERT

if bert_available:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {BERT_MODEL_NAME}")
    print(f"BERT K-Fold: {N_SPLITS} folds — akan cukup lama di CPU...")

    df["bert_category"] = ""
    df["bert_priority"]  = ""

    bert_time = 0.0
    for fold, (train_idx, test_idx) in enumerate(kf.split(df["description"], df["category"]), 1):
        print(f"\n[BERT] Fold {fold}/{N_SPLITS}...")
        train_desc = df["description"].iloc[train_idx].tolist()
        train_cat  = df["category"].iloc[train_idx].tolist()
        train_pri  = df["priority"].iloc[train_idx].tolist()
        test_desc  = df["description"].iloc[test_idx].tolist()
        test_index = df.index[test_idx]

        t0 = time.time()
        bert_c = BertClassifier(model_name=BERT_MODEL_NAME, epochs=BERT_EPOCHS)
        bert_p = BertClassifier(model_name=BERT_MODEL_NAME, epochs=BERT_EPOCHS)
        bert_c.fit(train_desc, train_cat)
        bert_p.fit(train_desc, train_pri)
        df.loc[test_index, "bert_category"] = bert_c.predict(test_desc)
        df.loc[test_index, "bert_priority"]  = bert_p.predict(test_desc)
        bert_time += time.time() - t0
        print(f"  Fold {fold} selesai")

    df["bert_match"] = (
        (df["bert_category"] == df["category"]) &
        (df["bert_priority"]  == df["priority"])
    )
    print(f"\nBERT Category Acc (OOF) : {accuracy_score(df['category'], df['bert_category']):.4f}")
    print(f"BERT Priority  Acc (OOF) : {accuracy_score(df['priority'], df['bert_priority']):.4f}")
    print(f"BERT Total Time          : {bert_time:.1f}s")
else:
    bert_time = 0.0
    print("BERT dilewati (SKIP_BERT=True).")
"""

# ── Cell a1000025: metrics_rows + elapsed_seconds ────────────────────────────
NEW_CELLS["a1000025"] = r"""metrics_rows = [
    {"approach": "SVM",           "elapsed_seconds": round(svm_time, 2), **metrics_dict(df["category"].tolist(), df["svm_category"].tolist(), "category")},
    {"approach": "SVM",           "elapsed_seconds": round(svm_time, 2), **metrics_dict(df["priority"].tolist(),  df["svm_priority"].tolist(),  "priority")},
    {"approach": "Random Forest", "elapsed_seconds": round(rf_time,  2), **metrics_dict(df["category"].tolist(), df["rf_category"].tolist(),  "category")},
    {"approach": "Random Forest", "elapsed_seconds": round(rf_time,  2), **metrics_dict(df["priority"].tolist(),  df["rf_priority"].tolist(),   "priority")},
]
if lr_available:
    metrics_rows += [
        {"approach": "Logistic Regression", "elapsed_seconds": round(lr_time, 2), **metrics_dict(df["category"].tolist(), df["lr_category"].tolist(), "category")},
        {"approach": "Logistic Regression", "elapsed_seconds": round(lr_time, 2), **metrics_dict(df["priority"].tolist(),  df["lr_priority"].tolist(),  "priority")},
    ]
if bert_available:
    metrics_rows += [
        {"approach": "BERT", "elapsed_seconds": round(bert_time, 2), **metrics_dict(df["category"].tolist(), df["bert_category"].tolist(), "category")},
        {"approach": "BERT", "elapsed_seconds": round(bert_time, 2), **metrics_dict(df["priority"].tolist(),  df["bert_priority"].tolist(),  "priority")},
    ]
print("Baseline metrics initialized.")
"""

# ── Cell a1000027: Hybrid SVM + timing ───────────────────────────────────────
NEW_CELLS["a1000027"] = r"""for selected_model in selected_models:
    model_key          = re.sub(r"[^a-zA-Z0-9]+", "_", selected_model).strip("_").lower()
    hybrid_svm_cat_col = f"hybrid_svm_category_{model_key}"
    hybrid_svm_pri_col = f"hybrid_svm_priority_{model_key}"

    df[hybrid_svm_cat_col] = df["svm_category"]
    df[hybrid_svm_pri_col] = df["svm_priority"]

    print(f"\n[Hybrid SVM] {selected_model}")
    hybrid_time = 0.0
    for processed, idx in enumerate(svm_mismatch_indices, start=1):
        t0 = time.time()
        new_cat, new_pri = classify_with_genai(
            client=client, model=selected_model, text=df.at[idx, "description"],
            allowed_categories=allowed_categories, allowed_priorities=allowed_priorities,
            ml_category=df.at[idx, "svm_category"], ml_priority=df.at[idx, "svm_priority"],
        )
        hybrid_time += time.time() - t0
        df.at[idx, hybrid_svm_cat_col] = new_cat
        df.at[idx, hybrid_svm_pri_col] = new_pri
        if processed % 20 == 0:
            print(f"  {processed}/{len(svm_mismatch_indices)} selesai")

    metrics_rows.append({"approach": f"Hybrid SVM ({selected_model})", "elapsed_seconds": round(hybrid_time, 2),
        **metrics_dict(df["category"].tolist(), df[hybrid_svm_cat_col].tolist(), "category")})
    metrics_rows.append({"approach": f"Hybrid SVM ({selected_model})", "elapsed_seconds": round(hybrid_time, 2),
        **metrics_dict(df["priority"].tolist(),  df[hybrid_svm_pri_col].tolist(),  "priority")})

print("\nSelesai.")
"""

# ── Cell a1000031: Export + notes (update NOTES_METRICS saja) ────────────────
NEW_CELLS["a1000031"] = r"""import openpyxl
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
    "approach":        "Nama model / pendekatan yang dibandingkan (SVM, RF, LR, BERT, Hybrid SVM)",
    "label":           "Label yang dievaluasi: 'category' atau 'priority'",
    "elapsed_seconds": "Total waktu training dan prediksi seluruh fold (dalam detik)",
    "accuracy":        "Akurasi = jumlah prediksi benar / total baris",
    "macro_precision": "Precision rata-rata antar kelas (tidak mempertimbangkan jumlah sampel per kelas)",
    "macro_recall":    "Recall rata-rata antar kelas (tidak mempertimbangkan jumlah sampel per kelas)",
    "macro_f1":        "F1-score rata-rata antar kelas (tidak mempertimbangkan jumlah sampel per kelas)",
    "weighted_f1":     "F1-score rata-rata tertimbang — kelas dengan sampel lebih banyak diberi bobot lebih besar",
    "samples":         "Jumlah total baris yang dievaluasi",
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

# ── Terapkan ke notebook ──────────────────────────────────────────────────────
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

updated = []
for cell in nb["cells"]:
    cid = cell.get("id", "")
    if cid in NEW_CELLS:
        cell["source"] = [NEW_CELLS[cid]]
        updated.append(cid)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Updated cells: {updated}")
