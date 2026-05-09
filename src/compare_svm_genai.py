import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

# Pastikan stdout/stderr bisa nge-print karakter non-ASCII di Windows console.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from bert_classifier import BertClassifier


def safe_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def pick_model(client: OpenAI, candidates: List[str]) -> str:
    try:
        models = client.models.list()
        available = {m.id for m in models.data}
        for model in candidates:
            if model in available:
                return model
    except Exception:
        pass
    return candidates[0]


def get_available_models(client: OpenAI) -> List[str]:
    models = client.models.list()
    return sorted({m.id for m in models.data})


def build_svm_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("svm", LinearSVC()),
    ])


class OpenAIEmbedder(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper untuk OpenAI Embedding API (GenAI)."""

    def __init__(self, model_name: str = "text-embedding-3-small",
                 batch_size: int = 128, api_key: Optional[str] = None,
                 max_chars: int = 8000):
        self.model_name = model_name
        self.batch_size = batch_size
        self.api_key = api_key
        self.max_chars = max_chars
        self._client = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        client = self._get_client()
        texts = list(X) if not isinstance(X, list) else X
        texts = [(t[: self.max_chars] if t else " ") for t in texts]

        all_embeds: List[List[float]] = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            for attempt in range(3):
                try:
                    resp = client.embeddings.create(model=self.model_name, input=batch)
                    all_embeds.extend([d.embedding for d in resp.data])
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            if (i // self.batch_size) % 10 == 0:
                print(f"  [Embed] {min(i + self.batch_size, total)}/{total}")
        return np.asarray(all_embeds, dtype=np.float32)


def build_hybrid_svm_pipeline(
    embed_model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
) -> Pipeline:
    """Hybrid SVM: TF-IDF (lexical) + OpenAI GenAI embedding (semantic) -> LinearSVC."""
    return Pipeline([
        ("features", FeatureUnion([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("embed", OpenAIEmbedder(model_name=embed_model, api_key=api_key)),
        ])),
        ("svm", LinearSVC()),
    ])


def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])


def build_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def compute_topk_and_margin(
    pipeline: Pipeline, texts: List[str], k: int = 3
) -> Tuple[List[List[str]], np.ndarray]:
    """Top-K kelas berdasarkan decision_function + margin (skor#1 - skor#2)."""
    scores = np.asarray(pipeline.decision_function(texts))
    svm_step = pipeline.named_steps["svm"]
    classes = svm_step.classes_

    if scores.ndim == 1:
        # Binary: ekspansi ke 2D supaya konsisten dengan multiclass.
        scores = np.column_stack([-scores, scores])

    n_classes = scores.shape[1]
    k_eff = max(1, min(k, n_classes))
    sorted_idx = np.argsort(-scores, axis=1)
    topk_idx = sorted_idx[:, :k_eff]
    topk_labels = [[str(classes[j]) for j in row] for row in topk_idx]

    sorted_scores = np.take_along_axis(scores, sorted_idx, axis=1)
    if n_classes >= 2:
        margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    else:
        margin = sorted_scores[:, 0]
    return topk_labels, margin


def classify_with_genai_constrained(
    client: OpenAI,
    model: str,
    text: str,
    top_categories: List[str],
    top_priorities: List[str],
    ml_category: str,
    ml_priority: str,
    retries: int = 2,
) -> Tuple[str, str]:
    """Multiple-choice validator: GenAI memilih dari top-K kandidat SVM saja.
    Kalau jawaban di luar shortlist atau gagal parse → fallback ke prediksi SVM.
    """
    default_cat = ml_category
    default_pri = ml_priority

    prompt = f"""
You are validating an IT helpdesk ticket classification.
Pick the single most appropriate category AND priority from the provided shortlists.
Return strict JSON only with keys: category, priority.

Category shortlist (pick exactly one from this list):
{json.dumps(top_categories, ensure_ascii=True)}

Priority shortlist (pick exactly one from this list):
{json.dumps(top_priorities, ensure_ascii=True)}

Current ML prediction: category='{ml_category}', priority='{ml_priority}'.
Ticket description:
\"\"\"{text[:3000]}\"\"\"
""".strip()

    for attempt in range(retries):
        try:
            response = client.responses.create(model=model, input=prompt)
            raw = (response.output_text or "").strip()
            parsed = json.loads(raw)
            cat = str(parsed.get("category", default_cat)).strip()
            pri = str(parsed.get("priority", default_pri)).strip()
            # Guard: jawaban harus dalam shortlist; kalau tidak → fallback SVM.
            if cat not in top_categories:
                cat = default_cat
            if pri not in top_priorities:
                pri = default_pri
            return cat, pri
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return default_cat, default_pri
    return default_cat, default_pri


def classify_with_genai_voter(
    client: OpenAI,
    model: str,
    text: str,
    allowed_categories: List[str],
    allowed_priorities: List[str],
    retries: int = 2,
) -> Tuple[str, str]:
    """Voter prompt: TANPA hint dari SVM. GenAI prediksi independen.
    Untuk Hybrid Voting Ensemble (Strategi 2). Hapus 'Current ML prediction'
    yang sebelumnya bikin GenAI anchored ke jawaban SVM.
    """
    default_cat = allowed_categories[0]
    default_pri = allowed_priorities[0]

    prompt = f"""
You are an IT helpdesk ticket classifier.
Read the ticket and assign the most appropriate category and priority.
Return strict JSON only with keys: category, priority. No explanation.

Allowed category values (pick exactly one):
{json.dumps(allowed_categories, ensure_ascii=True)}

Allowed priority values (pick exactly one):
{json.dumps(allowed_priorities, ensure_ascii=True)}

Ticket description:
\"\"\"{text[:3000]}\"\"\"
""".strip()

    for attempt in range(retries):
        try:
            response = client.responses.create(model=model, input=prompt)
            raw = (response.output_text or "").strip()
            parsed = json.loads(raw)
            cat = str(parsed.get("category", default_cat)).strip()
            pri = str(parsed.get("priority", default_pri)).strip()
            if cat not in allowed_categories:
                cat = default_cat
            if pri not in allowed_priorities:
                pri = default_pri
            return cat, pri
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return default_cat, default_pri
    return default_cat, default_pri


def classify_with_genai(
    client: OpenAI,
    model: str,
    text: str,
    allowed_categories: List[str],
    allowed_priorities: List[str],
    ml_category: Optional[str] = None,
    ml_priority: Optional[str] = None,
    retries: int = 2,
) -> Tuple[str, str]:
    default_category = ml_category or allowed_categories[0]
    default_priority = ml_priority or allowed_priorities[0]

    context_line = (
        f"Current ML prediction: category='{ml_category}', priority='{ml_priority}'. "
        "Improve only if clearly needed."
    )

    prompt = f"""
You validate and refine ML predictions.
Return strict JSON only with keys: category, priority.

Allowed category values:
{json.dumps(allowed_categories, ensure_ascii=True)}

Allowed priority values:
{json.dumps(allowed_priorities, ensure_ascii=True)}

{context_line}
Text:
\"\"\"{text[:3000]}\"\"\"
""".strip()

    for attempt in range(retries):
        try:
            response = client.responses.create(model=model, input=prompt)
            raw = (response.output_text or "").strip()
            parsed = json.loads(raw)
            category = str(parsed.get("category", default_category)).strip()
            priority = str(parsed.get("priority", default_priority)).strip()
            if category not in allowed_categories:
                category = default_category
            if priority not in allowed_priorities:
                priority = default_priority
            return category, priority
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return default_category, default_priority

    return default_category, default_priority


def metrics_dict(y_true: List[str], y_pred: List[str], label_name: str) -> Dict[str, float]:
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return {
        "label": label_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "samples": len(y_true),
    }


def run_pipeline(
    input_file: str = "data/cobacek.xlsx",
    output_file: str = "results/cobacek_compare.xlsx",
    max_genai_rows: int = 100,
    forced_model: Optional[str] = None,
    multi_models: Optional[List[str]] = None,
    bert_model_name: str = "distilbert-base-multilingual-cased",
    bert_epochs: int = 3,
    skip_bert: bool = False,
    skip_lr: bool = False,
    skip_fusion: bool = False,
    embed_model: str = "text-embedding-3-small",
    correction_mode: str = "confidence",
    conf_percentile: float = 30.0,
    topk: int = 3,
    random_state: int = 42,
    enable_voting: bool = False,
) -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY tidak ditemukan di .env")

    df = pd.read_excel(input_file)
    required = ["description", "category", "priority"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ada: {missing}")

    df["description"] = df["description"].apply(safe_str)
    df["category"] = df["category"].apply(safe_str)
    df["priority"] = df["priority"].apply(safe_str)

    # --- Stratified Train/Test Split (80/20) ---
    # Stratify by category karena distribusinya paling tidak seimbang.
    # Kelas yang hanya muncul 1x tidak bisa di-stratify; gabungkan ke "_rare_" untuk split.
    cat_counts = df["category"].value_counts()
    rare_cats = set(cat_counts[cat_counts < 2].index)
    if rare_cats:
        print(f"[Split] {len(rare_cats)} kategori muncul <2x, di-merge ke '_rare_' hanya untuk stratify")
    strat_key = df["category"].where(~df["category"].isin(rare_cats), "_rare_")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=strat_key
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"[Split] train={len(train_df)} test={len(test_df)} "
          f"(total={len(df)}, ratio=80/20, stratified by category, random_state={random_state})")

    # --- SVM ---
    svm_cat_model = build_svm_pipeline()
    svm_pri_model = build_svm_pipeline()
    svm_cat_model.fit(train_df["description"], train_df["category"])
    svm_pri_model.fit(train_df["description"], train_df["priority"])
    test_df["svm_category"] = svm_cat_model.predict(test_df["description"])
    test_df["svm_priority"] = svm_pri_model.predict(test_df["description"])
    test_df["svm_match"] = (
        (test_df["svm_category"] == test_df["category"]) &
        (test_df["svm_priority"] == test_df["priority"])
    )
    print(f"SVM Category Acc: {accuracy_score(test_df['category'], test_df['svm_category']):.4f} | "
          f"Priority Acc: {accuracy_score(test_df['priority'], test_df['svm_priority']):.4f}")

    # --- Random Forest ---
    rf_cat_model = build_rf_pipeline()
    rf_pri_model = build_rf_pipeline()
    rf_cat_model.fit(train_df["description"], train_df["category"])
    rf_pri_model.fit(train_df["description"], train_df["priority"])
    test_df["rf_category"] = rf_cat_model.predict(test_df["description"])
    test_df["rf_priority"] = rf_pri_model.predict(test_df["description"])
    test_df["rf_match"] = (
        (test_df["rf_category"] == test_df["category"]) &
        (test_df["rf_priority"] == test_df["priority"])
    )
    print(f"RF  Category Acc: {accuracy_score(test_df['category'], test_df['rf_category']):.4f} | "
          f"Priority Acc: {accuracy_score(test_df['priority'], test_df['rf_priority']):.4f}")

    # --- Hybrid SVM (Feature Fusion: TF-IDF + GenAI Embedding) ---
    fusion_available = not skip_fusion
    if fusion_available:
        from scipy.sparse import csr_matrix, hstack as sparse_hstack

        print(f"\n[Hybrid SVM Fusion] Embedding model: {embed_model}")
        print(f"[Fusion] Encoding {len(train_df)} train + {len(test_df)} test deskripsi via OpenAI Embedding API...")
        embedder = OpenAIEmbedder(model_name=embed_model, api_key=api_key)
        train_embeds = embedder.fit_transform(train_df["description"].tolist())
        test_embeds = embedder.transform(test_df["description"].tolist())

        print("[Fusion] Fitting TF-IDF + concat embedding...")
        tfidf = TfidfVectorizer(ngram_range=(1, 2))
        train_tfidf = tfidf.fit_transform(train_df["description"])
        test_tfidf = tfidf.transform(test_df["description"])
        train_fusion = sparse_hstack([train_tfidf, csr_matrix(train_embeds)]).tocsr()
        test_fusion = sparse_hstack([test_tfidf, csr_matrix(test_embeds)]).tocsr()

        print("[Fusion] Training SVM kategori...")
        fusion_cat_clf = LinearSVC()
        fusion_cat_clf.fit(train_fusion, train_df["category"])
        test_df["fusion_category"] = fusion_cat_clf.predict(test_fusion)

        print("[Fusion] Training SVM priority...")
        fusion_pri_clf = LinearSVC()
        fusion_pri_clf.fit(train_fusion, train_df["priority"])
        test_df["fusion_priority"] = fusion_pri_clf.predict(test_fusion)

        test_df["fusion_match"] = (
            (test_df["fusion_category"] == test_df["category"]) &
            (test_df["fusion_priority"] == test_df["priority"])
        )
        print(f"Hybrid Fusion Category Acc: {accuracy_score(test_df['category'], test_df['fusion_category']):.4f} | "
              f"Priority Acc: {accuracy_score(test_df['priority'], test_df['fusion_priority']):.4f}")

    # --- Logistic Regression ---
    lr_available = not skip_lr
    if lr_available:
        lr_cat_model = build_lr_pipeline()
        lr_pri_model = build_lr_pipeline()
        lr_cat_model.fit(train_df["description"], train_df["category"])
        lr_pri_model.fit(train_df["description"], train_df["priority"])
        test_df["lr_category"] = lr_cat_model.predict(test_df["description"])
        test_df["lr_priority"] = lr_pri_model.predict(test_df["description"])
        test_df["lr_match"] = (
            (test_df["lr_category"] == test_df["category"]) &
            (test_df["lr_priority"] == test_df["priority"])
        )
        print(f"LR  Category Acc: {accuracy_score(test_df['category'], test_df['lr_category']):.4f} | "
              f"Priority Acc: {accuracy_score(test_df['priority'], test_df['lr_priority']):.4f}")

    # --- BERT ---
    bert_available = not skip_bert
    if bert_available:
        print(f"\nTraining BERT ({bert_model_name}) - bisa memakan beberapa menit di CPU...")
        bert_cat_model = BertClassifier(model_name=bert_model_name, epochs=bert_epochs)
        bert_pri_model = BertClassifier(model_name=bert_model_name, epochs=bert_epochs)

        print("[BERT] Training category model...")
        bert_cat_model.fit(train_df["description"].tolist(), train_df["category"].tolist())
        print("[BERT] Training priority model...")
        bert_pri_model.fit(train_df["description"].tolist(), train_df["priority"].tolist())

        test_df["bert_category"] = bert_cat_model.predict(test_df["description"].tolist())
        test_df["bert_priority"] = bert_pri_model.predict(test_df["description"].tolist())
        test_df["bert_match"] = (
            (test_df["bert_category"] == test_df["category"]) &
            (test_df["bert_priority"] == test_df["priority"])
        )
        print(f"BERT Category Acc: {accuracy_score(test_df['category'], test_df['bert_category']):.4f} | "
              f"Priority Acc: {accuracy_score(test_df['priority'], test_df['bert_priority']):.4f}")

    # Allowed labels diambil dari TRAIN agar tidak bocor ke test set.
    allowed_categories = sorted(train_df["category"].dropna().unique().tolist())
    allowed_priorities = sorted(train_df["priority"].dropna().unique().tolist())

    client = OpenAI(api_key=api_key)
    available_models = get_available_models(client)
    model_candidates = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-5-mini", "gpt-4.1", "gpt-4o"]
    selected_models: List[str] = []
    if multi_models:
        selected_models = [m.strip() for m in multi_models if m.strip()]
    elif forced_model:
        selected_models = [forced_model]
    elif os.getenv("OPENAI_MODELS"):
        selected_models = [m.strip() for m in os.getenv("OPENAI_MODELS").split(",") if m.strip()]
    else:
        selected_models = [pick_model(client, model_candidates)]

    for selected_model in selected_models:
        if selected_model not in available_models:
            raise ValueError(
                f"Model '{selected_model}' tidak tersedia. "
                f"Model tersedia: {', '.join(available_models)}"
            )

    # Hitung top-K kandidat & margin SVM untuk test set (untuk Phase 3 confidence gate).
    test_texts = test_df["description"].tolist()
    cat_topk, cat_margin = compute_topk_and_margin(svm_cat_model, test_texts, k=topk)
    pri_topk, pri_margin = compute_topk_and_margin(svm_pri_model, test_texts, k=topk)
    test_df["svm_cat_margin"] = cat_margin
    test_df["svm_pri_margin"] = pri_margin

    # Trigger criteria untuk koreksi GenAI (per-kolom):
    #   - "oracle"     : ground truth (data leakage; ablation/sanity check)
    #   - "confidence" : bottom-X% margin SVM (deployable, default)
    if correction_mode == "oracle":
        cat_trigger_mask = test_df["svm_category"] != test_df["category"]
        pri_trigger_mask = test_df["svm_priority"] != test_df["priority"]
        trigger_label = "oracle (ground truth mismatch)"
    elif correction_mode == "confidence":
        cat_thr = float(np.percentile(cat_margin, conf_percentile))
        pri_thr = float(np.percentile(pri_margin, conf_percentile))
        cat_trigger_mask = pd.Series(cat_margin <= cat_thr, index=test_df.index)
        pri_trigger_mask = pd.Series(pri_margin <= pri_thr, index=test_df.index)
        trigger_label = f"confidence (bottom {conf_percentile:.0f}% margin; cat<={cat_thr:.3f}, pri<={pri_thr:.3f})"
    else:
        raise ValueError(f"correction_mode tidak dikenal: {correction_mode}")

    any_trigger_mask = cat_trigger_mask | pri_trigger_mask
    svm_mismatch_indices = test_df[any_trigger_mask].index[
        : min(max_genai_rows, any_trigger_mask.sum())
    ]
    print(f"[Hybrid trigger] mode={trigger_label}")
    print(f"[Hybrid trigger] cat_trigger={cat_trigger_mask.sum()} "
          f"pri_trigger={pri_trigger_mask.sum()} "
          f"any_trigger={any_trigger_mask.sum()} "
          f"akan dikoreksi={len(svm_mismatch_indices)}")

    # Metrik baseline ML (semua di TEST SET)
    metrics_rows: List[Dict] = [
        {"approach": "SVM",           **metrics_dict(test_df["category"].tolist(), test_df["svm_category"].tolist(), "category")},
        {"approach": "SVM",           **metrics_dict(test_df["priority"].tolist(),  test_df["svm_priority"].tolist(),  "priority")},
        {"approach": "Random Forest", **metrics_dict(test_df["category"].tolist(), test_df["rf_category"].tolist(),  "category")},
        {"approach": "Random Forest", **metrics_dict(test_df["priority"].tolist(),  test_df["rf_priority"].tolist(),   "priority")},
    ]
    if fusion_available:
        metrics_rows += [
            {"approach": "Hybrid SVM (TF-IDF + Embedding)", **metrics_dict(test_df["category"].tolist(), test_df["fusion_category"].tolist(), "category")},
            {"approach": "Hybrid SVM (TF-IDF + Embedding)", **metrics_dict(test_df["priority"].tolist(),  test_df["fusion_priority"].tolist(),  "priority")},
        ]
    if bert_available:
        metrics_rows += [
            {"approach": "BERT", **metrics_dict(test_df["category"].tolist(), test_df["bert_category"].tolist(), "category")},
            {"approach": "BERT", **metrics_dict(test_df["priority"].tolist(),  test_df["bert_priority"].tolist(),  "priority")},
        ]
    if lr_available:
        metrics_rows += [
            {"approach": "Logistic Regression", **metrics_dict(test_df["category"].tolist(), test_df["lr_category"].tolist(), "category")},
            {"approach": "Logistic Regression", **metrics_dict(test_df["priority"].tolist(),  test_df["lr_priority"].tolist(),  "priority")},
        ]

    for selected_model in selected_models:
        model_key = re.sub(r"[^a-zA-Z0-9]+", "_", selected_model).strip("_").lower()
        hybrid_svm_cat_col = f"hybrid_svm_category_{model_key}"
        hybrid_svm_pri_col = f"hybrid_svm_priority_{model_key}"

        test_df[hybrid_svm_cat_col] = test_df["svm_category"]
        test_df[hybrid_svm_pri_col] = test_df["svm_priority"]

        # Audit kolom — Phase 4: catat baris mana yang BENAR-BENAR berubah (hybrid != SVM).
        cat_audit_col = f"hybrid_svm_cat_overridden_{model_key}"
        pri_audit_col = f"hybrid_svm_pri_overridden_{model_key}"
        test_df[cat_audit_col] = False
        test_df[pri_audit_col] = False

        # Hybrid SVM: SVM prediksi, GenAI koreksi PER-KOLOM via constrained validator.
        # - Per-kolom (Phase 2): kolom dengan SVM "yakin" tidak diutak-atik
        # - Constrained (Phase 3): GenAI memilih dari top-K kandidat SVM, bukan generator bebas
        # - Guarded (Phase 4): override hanya dicatat jika nilai berbeda dari SVM top-1
        cat_consulted = pri_consulted = 0
        cat_overridden = pri_overridden = 0
        print(f"\n[Hybrid SVM] {selected_model}")
        for processed, idx in enumerate(svm_mismatch_indices, start=1):
            row_cat_topk = cat_topk[idx]
            row_pri_topk = pri_topk[idx]
            svm_cat_top1 = test_df.at[idx, "svm_category"]
            svm_pri_top1 = test_df.at[idx, "svm_priority"]

            if correction_mode == "confidence":
                new_cat, new_pri = classify_with_genai_constrained(
                    client=client, model=selected_model,
                    text=test_df.at[idx, "description"],
                    top_categories=row_cat_topk,
                    top_priorities=row_pri_topk,
                    ml_category=svm_cat_top1,
                    ml_priority=svm_pri_top1,
                )
            else:
                new_cat, new_pri = classify_with_genai(
                    client=client, model=selected_model,
                    text=test_df.at[idx, "description"],
                    allowed_categories=allowed_categories,
                    allowed_priorities=allowed_priorities,
                    ml_category=svm_cat_top1,
                    ml_priority=svm_pri_top1,
                )

            if cat_trigger_mask.at[idx]:
                cat_consulted += 1
                test_df.at[idx, hybrid_svm_cat_col] = new_cat
                if new_cat != svm_cat_top1:
                    test_df.at[idx, cat_audit_col] = True
                    cat_overridden += 1
            if pri_trigger_mask.at[idx]:
                pri_consulted += 1
                test_df.at[idx, hybrid_svm_pri_col] = new_pri
                if new_pri != svm_pri_top1:
                    test_df.at[idx, pri_audit_col] = True
                    pri_overridden += 1
            if processed % 20 == 0:
                print(f"  processed {processed}/{len(svm_mismatch_indices)}")
        print(f"  [{selected_model}] cat consulted={cat_consulted} overridden={cat_overridden} | "
              f"pri consulted={pri_consulted} overridden={pri_overridden}")

        metrics_rows.append({"approach": f"Hybrid SVM ({selected_model})",
            **metrics_dict(test_df["category"].tolist(), test_df[hybrid_svm_cat_col].tolist(), "category")})
        metrics_rows.append({"approach": f"Hybrid SVM ({selected_model})",
            **metrics_dict(test_df["priority"].tolist(),  test_df[hybrid_svm_pri_col].tolist(),  "priority")})

    # === Strategi 2: Hybrid SVM-GenAI Voting Ensemble ===
    # GenAI prediksi independen (tanpa hint SVM) di SEMUA test rows,
    # lalu majority vote dengan SVM. Tie-break: pakai Fusion kalau tersedia, else SVM.
    if enable_voting:
        from collections import Counter

        for selected_model in selected_models:
            model_key = re.sub(r"[^a-zA-Z0-9]+", "_", selected_model).strip("_").lower()
            voter_cat_col = f"genai_voter_category_{model_key}"
            voter_pri_col = f"genai_voter_priority_{model_key}"
            vote_cat_col = f"hybrid_voting_category_{model_key}"
            vote_pri_col = f"hybrid_voting_priority_{model_key}"

            print(f"\n[Voting] {selected_model} — predicting independently on {len(test_df)} test rows...")
            voter_cats: List[str] = []
            voter_pris: List[str] = []
            for processed, idx in enumerate(test_df.index, start=1):
                vc, vp = classify_with_genai_voter(
                    client=client, model=selected_model,
                    text=test_df.at[idx, "description"],
                    allowed_categories=allowed_categories,
                    allowed_priorities=allowed_priorities,
                )
                voter_cats.append(vc)
                voter_pris.append(vp)
                if processed % 50 == 0:
                    print(f"  voter processed {processed}/{len(test_df)}")
            test_df[voter_cat_col] = voter_cats
            test_df[voter_pri_col] = voter_pris

            # Majority voting per baris.
            # Voters: SVM + GenAI (+ Fusion kalau tersedia) → 3-way kalau Fusion ada, 2-way kalau tidak.
            vote_cats: List[str] = []
            vote_pris: List[str] = []
            tie_break_used = 0
            for idx in test_df.index:
                cat_votes = [test_df.at[idx, "svm_category"], test_df.at[idx, voter_cat_col]]
                pri_votes = [test_df.at[idx, "svm_priority"], test_df.at[idx, voter_pri_col]]
                if fusion_available:
                    cat_votes.append(test_df.at[idx, "fusion_category"])
                    pri_votes.append(test_df.at[idx, "fusion_priority"])

                cat_counter = Counter(cat_votes)
                pri_counter = Counter(pri_votes)
                top_cat, top_cat_count = cat_counter.most_common(1)[0]
                top_pri, top_pri_count = pri_counter.most_common(1)[0]

                # Tie-break: pakai Fusion kalau tersedia, else SVM (lebih kuat dari masing-masing voter).
                if top_cat_count == 1:
                    top_cat = test_df.at[idx, "fusion_category"] if fusion_available else test_df.at[idx, "svm_category"]
                    tie_break_used += 1
                if top_pri_count == 1:
                    top_pri = test_df.at[idx, "fusion_priority"] if fusion_available else test_df.at[idx, "svm_priority"]

                vote_cats.append(top_cat)
                vote_pris.append(top_pri)
            test_df[vote_cat_col] = vote_cats
            test_df[vote_pri_col] = vote_pris
            n_voters = 3 if fusion_available else 2
            print(f"  [Voting {selected_model}] {n_voters}-way vote done. tie_break_used (cat)={tie_break_used}/{len(test_df)}")

            metrics_rows.append({"approach": f"Hybrid Voting ({selected_model}, {n_voters}-way)",
                **metrics_dict(test_df["category"].tolist(), test_df[vote_cat_col].tolist(), "category")})
            metrics_rows.append({"approach": f"Hybrid Voting ({selected_model}, {n_voters}-way)",
                **metrics_dict(test_df["priority"].tolist(),  test_df[vote_pri_col].tolist(),  "priority")})

    metrics_df = pd.DataFrame(metrics_rows)

    summary_rows = [
        {"key": "selected_models",             "value": ", ".join(selected_models)},
        {"key": "bert_model",                  "value": bert_model_name if bert_available else "skipped"},
        {"key": "embed_model",                 "value": embed_model if fusion_available else "skipped"},
        {"key": "total_rows",                  "value": len(df)},
        {"key": "train_rows",                  "value": len(train_df)},
        {"key": "test_rows",                   "value": len(test_df)},
        {"key": "split_strategy",              "value": f"stratified 80/20 by category, random_state={random_state}"},
        {"key": "correction_mode",             "value": correction_mode},
        {"key": "conf_percentile",             "value": conf_percentile if correction_mode == "confidence" else "n/a"},
        {"key": "topk_candidates",             "value": topk if correction_mode == "confidence" else "n/a"},
        {"key": "svm_mismatch_rows_corrected", "value": len(svm_mismatch_indices)},
        {"key": "max_genai_rows",              "value": max_genai_rows},
    ]
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        test_df.to_excel(writer,    index=False, sheet_name="Predictions_Compare")
        metrics_df.to_excel(writer, index=False, sheet_name="Metrics")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

    print(f"\nDone. Output: {output_file} (test={len(test_df)} baris, train={len(train_df)} baris)")
    print(f"Models: {', '.join(selected_models)} | BERT: {bert_model_name if bert_available else 'skipped'} | LR: {'enabled' if lr_available else 'skipped'}")
    print(metrics_df.to_string(index=False))


def run_pipeline_kfold(
    input_file: str = "data/cobacek.xlsx",
    output_file: str = "results/cobacek_compare.xlsx",
    n_folds: int = 5,
    base_seed: int = 42,
    **kwargs,
) -> None:
    """Phase 5: jalankan run_pipeline N kali dengan random_state berbeda,
    agregasi metrik (mean ± std) per (approach, label) ke satu file output."""
    base, ext = os.path.splitext(output_file)
    per_fold_metrics: List[pd.DataFrame] = []

    for fold_idx in range(n_folds):
        seed = base_seed + fold_idx
        fold_output = f"{base}_fold{fold_idx}{ext}"
        print(f"\n{'=' * 70}\n[K-Fold] FOLD {fold_idx + 1}/{n_folds} (seed={seed})\n{'=' * 70}")

        run_pipeline(
            input_file=input_file,
            output_file=fold_output,
            random_state=seed,
            **kwargs,
        )

        fold_metrics = pd.read_excel(fold_output, sheet_name="Metrics")
        fold_metrics["fold"] = fold_idx
        per_fold_metrics.append(fold_metrics)

    # Agregasi: mean ± std per (approach, label)
    all_metrics = pd.concat(per_fold_metrics, ignore_index=True)
    metric_cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    agg = (
        all_metrics
        .groupby(["approach", "label"])
        .agg({col: ["mean", "std"] for col in metric_cols} | {"samples": "first"})
        .reset_index()
    )
    # Flatten kolom multi-index
    agg.columns = [
        c[0] if c[1] == "" or c[1] == "first" else f"{c[0]}_{c[1]}"
        for c in agg.columns
    ]

    # Tulis hasil agregasi
    summary_rows = [
        {"key": "n_folds",        "value": n_folds},
        {"key": "base_seed",      "value": base_seed},
        {"key": "fold_outputs",   "value": ", ".join(f"fold{i}" for i in range(n_folds))},
        {"key": "aggregation",    "value": "mean ± std across folds, grouped by (approach, label)"},
    ]
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        agg.to_excel(writer,         index=False, sheet_name="Metrics_Aggregated")
        all_metrics.to_excel(writer, index=False, sheet_name="Metrics_Per_Fold")
        summary_df.to_excel(writer,  index=False, sheet_name="Summary")

    print(f"\n{'=' * 70}\n[K-Fold] Done. Aggregated output: {output_file}\n{'=' * 70}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bandingkan SVM vs RF vs LR vs BERT vs Hybrid SVM")
    parser.add_argument("--input",          default="data/cobacek.xlsx",            help="Path file input Excel")
    parser.add_argument("--output",         default="results/cobacek_compare.xlsx", help="Path file output Excel")
    parser.add_argument("--max-genai-rows", type=int, default=100,          help="Batas baris untuk inferensi GenAI (Hybrid SVM)")
    parser.add_argument("--model",          default=None,                   help="Paksa model GenAI tertentu")
    parser.add_argument("--models",         default=None,
        help="Daftar model dipisah koma (contoh: gpt-4.1-mini,gpt-4o-mini)")
    parser.add_argument("--bert-model",     default="distilbert-base-multilingual-cased",
        help="HuggingFace model ID untuk BERT (default: distilbert-base-multilingual-cased)")
    parser.add_argument("--bert-epochs",    type=int, default=3,            help="Jumlah epoch fine-tuning BERT")
    parser.add_argument("--skip-bert",      action="store_true",            help="Lewati BERT")
    parser.add_argument("--skip-lr",        action="store_true",            help="Lewati Logistic Regression")
    parser.add_argument("--skip-fusion",    action="store_true",            help="Lewati Hybrid SVM (TF-IDF + Embedding)")
    parser.add_argument("--embed-model",    default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small)")
    parser.add_argument("--correction-mode", default="confidence", choices=["confidence", "oracle"],
        help="Trigger koreksi GenAI: 'confidence' (deployable, default) atau 'oracle' (ablation)")
    parser.add_argument("--conf-percentile", type=float, default=30.0,
        help="Bottom X%% margin SVM yang dikirim ke GenAI (default: 30)")
    parser.add_argument("--topk",           type=int, default=3,
        help="Jumlah kandidat top-K SVM untuk multiple-choice GenAI (default: 3)")
    parser.add_argument("--n-folds",        type=int, default=1,
        help="Jumlah fold untuk Stratified K-Fold CV (default: 1 = single 80/20 split)")
    parser.add_argument("--base-seed",      type=int, default=42,
        help="Base random_state; fold ke-i pakai seed = base_seed + i (default: 42)")
    parser.add_argument("--enable-voting",  action="store_true",
        help="Aktifkan Hybrid Voting Ensemble (GenAI prediksi semua test rows; mahal)")
    args = parser.parse_args()

    models_arg = None
    if args.models:
        models_arg = [m.strip() for m in args.models.split(",") if m.strip()]

    common_kwargs = dict(
        max_genai_rows=args.max_genai_rows,
        forced_model=args.model,
        multi_models=models_arg,
        bert_model_name=args.bert_model,
        bert_epochs=args.bert_epochs,
        skip_bert=args.skip_bert,
        skip_lr=args.skip_lr,
        skip_fusion=args.skip_fusion,
        embed_model=args.embed_model,
        correction_mode=args.correction_mode,
        conf_percentile=args.conf_percentile,
        topk=args.topk,
        enable_voting=args.enable_voting,
    )

    if args.n_folds > 1:
        run_pipeline_kfold(
            input_file=args.input,
            output_file=args.output,
            n_folds=args.n_folds,
            base_seed=args.base_seed,
            **common_kwargs,
        )
    else:
        run_pipeline(
            input_file=args.input,
            output_file=args.output,
            random_state=args.base_seed,
            **common_kwargs,
        )
