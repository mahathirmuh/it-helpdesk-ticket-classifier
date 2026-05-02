import pandas as pd
from openpyxl.comments import Comment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def train_and_evaluate(texts, labels):
    label_counts = labels.value_counts()
    stratify_labels = labels if label_counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=stratify_labels
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    return pipeline, report, report_dict, acc


def main():
    df = pd.read_excel("data/cobacek.xlsx")
    if "description" not in df.columns:
        raise ValueError("Kolom 'description' tidak ditemukan.")
    if "category" not in df.columns or "priority" not in df.columns:
        raise ValueError("Kolom 'category' atau 'priority' tidak ditemukan.")

    df["description"] = df["description"].fillna("").astype(str)

    category_model, category_report, category_report_dict, category_acc = train_and_evaluate(
        df["description"], df["category"]
    )
    priority_model, priority_report, priority_report_dict, priority_acc = train_and_evaluate(
        df["description"], df["priority"]
    )

    print("Category Accuracy:", round(category_acc, 4))
    print(category_report)
    print("Priority Accuracy:", round(priority_acc, 4))
    print(priority_report)

    category_model.fit(df["description"], df["category"])
    priority_model.fit(df["description"], df["priority"])

    df["predicted_category"] = category_model.predict(df["description"])
    df["predicted_priority"] = priority_model.predict(df["description"])
    df["predicted_category_match"] = df["predicted_category"] == df["category"]
    df["predicted_priority_match"] = df["predicted_priority"] == df["priority"]

    category_metrics = (
        pd.DataFrame(category_report_dict)
        .T.loc[:, ["precision", "recall", "f1-score", "support"]]
        .rename_axis("Category Name")
    )
    priority_metrics = (
        pd.DataFrame(priority_report_dict)
        .T.loc[:, ["precision", "recall", "f1-score", "support"]]
        .rename_axis("Priority Name")
    )

    prediction_notes_map = {
        "subject": "Judul ringkas tiket dari pengguna atau sistem",
        "description": "Detail masalah/kebutuhan; input utama model untuk prediksi",
        "answer": "Jawaban/solusi yang diberikan pada tiket",
        "type": "Tipe/klasifikasi tiket dari sumber data",
        "priority": "Label priority asli dari dataset (ground truth)",
        "category": "Label category asli dari dataset (ground truth)",
        "predicted_category": "Hasil prediksi category oleh Random Forest",
        "predicted_priority": "Hasil prediksi priority oleh Random Forest",
        "predicted_category_match": "True jika predicted_category sama dengan category",
        "predicted_priority_match": "True jika predicted_priority sama dengan priority",
    }
    prediction_notes = {
        index + 1: prediction_notes_map.get(col, "")
        for index, col in enumerate(df.columns)
    }
    metric_notes = {
        1: "Nama kelas/label yang dievaluasi",
        2: "Presisi per kelas: TP / (TP + FP)",
        3: "Recall per kelas: TP / (TP + FN)",
        4: "F1-score per kelas: 2 * (precision * recall) / (precision + recall)",
        5: "Jumlah data per kelas pada set evaluasi",
    }

    def write_output(file_path):
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Predictions")
            category_metrics.to_excel(writer, sheet_name="Category Accuracy")
            priority_metrics.to_excel(writer, sheet_name="Priority Accuracy")

            prediction_sheet = writer.sheets["Predictions"]
            for column_index, note in prediction_notes.items():
                if note:
                    cell = prediction_sheet.cell(row=1, column=column_index)
                    cell.comment = Comment(note, "RF")

            for sheet_name in ["Category Accuracy", "Priority Accuracy"]:
                sheet = writer.sheets[sheet_name]
                for column_index, note in metric_notes.items():
                    sheet.cell(row=1, column=column_index).comment = Comment(note, "RF")

    output_path = "cobacek_rf_pred.xlsx"
    fallback_path = "cobacek_rf_pred_new.xlsx"
    try:
        write_output(output_path)
        print(f"Output tersimpan: {output_path}")
    except PermissionError:
        write_output(fallback_path)
        print(f"File terkunci, output tersimpan ke: {fallback_path}")


if __name__ == "__main__":
    main()
