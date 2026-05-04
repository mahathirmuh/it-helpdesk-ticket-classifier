"""
Sklearn-compatible wrapper untuk fine-tuning BERT sequence classification.

Dependencies:
    pip install torch transformers scikit-learn
"""
from typing import List, Optional

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class _TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int,
        labels: Optional[List[int]] = None,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BertClassifier:
    """
    Fine-tuned BERT classifier dengan antarmuka sklearn (fit / predict).

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Default: distilbert-base-multilingual-cased
        (40% lebih kecil dari BERT, cocok untuk CPU & teks multibahasa).
        Ganti ke "bert-base-multilingual-cased" untuk akurasi lebih tinggi.
    epochs : int
        Jumlah epoch fine-tuning.
    batch_size : int
        Ukuran batch saat training dan inferensi.
    max_length : int
        Panjang token maksimum (teks dipotong jika lebih panjang).
    lr : float
        Learning rate AdamW.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        epochs: int = 3,
        batch_size: int = 16,
        max_length: int = 128,
        lr: float = 2e-5,
    ):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = LabelEncoder()
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None

    def fit(self, texts: List[str], labels: List[str]) -> "BertClassifier":
        labels_enc: List[int] = self.label_encoder.fit_transform(labels).tolist()
        num_labels = len(self.label_encoder.classes_)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        ).to(self.device)

        dataset = _TextDataset(list(texts), self.tokenizer, self.max_length, labels_enc)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                outputs.loss.backward()
                optimizer.step()
                total_loss += outputs.loss.item()
            print(f"  Epoch {epoch + 1}/{self.epochs} — loss: {total_loss / len(loader):.4f}")

        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model belum dilatih. Panggil fit() terlebih dahulu.")

        self.model.eval()
        dataset = _TextDataset(list(texts), self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds: List[int] = []
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())

        return self.label_encoder.inverse_transform(np.array(all_preds))
