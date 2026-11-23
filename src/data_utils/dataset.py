from src.data_utils.document import Document
import torch
class DocumentDataset:
    def __init__(self, documents: list[dict]) -> None:
        self.documents: list[Document] = self.__BuildDocuments(documents)
        self.document_embeddings: torch.Tensor | None = None
    def __BuildDocuments(self, documents):
        
        return [Document(text=text, _id=_id) for entries in documents 
                    for (text, _id) in zip(entries['text'], entries['document_id'])
                ]
        
    def GetDocuments(self) -> list[Document]:
        return self.documents

import torch
from src.data_utils.document import Document


class DPRTrainingDataset(torch.utils.data.Dataset):
    """Tokenized query-positive-passage pairs for DPR training."""

    def __init__(self, rows, tokenizer, max_length: int):
        self.rows = []

        for row in rows:
            query = row["question"]
            positive_texts = row["text"]

            # Some datasets provide one passage as a string; others provide a list.
            if isinstance(positive_texts, str):
                positive_texts = [positive_texts]

            # Use one positive passage per query for this first version.
            positive_text = next(
                (
                    text for text in positive_texts
                    if isinstance(text, str) and text.strip()
                ),
                None,
            )

            if positive_text is not None:
                self.rows.append((query, positive_text))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        query, positive_passage = self.rows[index]

        query_tokens = self.tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        passage_tokens = self.tokenizer(
            positive_passage,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return (
            query_tokens["input_ids"].squeeze(0),
            query_tokens["attention_mask"].squeeze(0),
            passage_tokens["input_ids"].squeeze(0),
            passage_tokens["attention_mask"].squeeze(0),
        )