import torch
import torch.nn as nn
from src.training.loss import DPRLoss

class Trainer:
    def __init__(self):
        self.criterion = DPRLoss()

    def train(
        self,
        query_encoder: nn.Module,
        document_encoder: nn.Module,
        train_loader,
        num_epochs: int,
        learning_rate: float,
        device: torch.device,
    ):
        query_encoder.to(device)
        document_encoder.to(device)

        optimizer = torch.optim.AdamW(
            list(query_encoder.parameters())
            + list(document_encoder.parameters()),
            lr=learning_rate,
        )

        for epoch in range(num_epochs):
            total_loss = 0.0
            n_batches = 0

            for (
                query_tokens,
                query_attention_mask,
                document_tokens,
                document_attention_mask,
            ) in train_loader:
                query_tokens = query_tokens.to(device)
                query_attention_mask = query_attention_mask.to(device)
                document_tokens = document_tokens.to(device)
                document_attention_mask = document_attention_mask.to(device)

                query_vectors = query_encoder(query_tokens, query_attention_mask)
                document_vectors = document_encoder(
                    document_tokens,
                    document_attention_mask,
                )
                # Each query is paired with the passage at the same batch position.
                per_query_losses = self.criterion(query_vectors, document_vectors)
                loss = per_query_losses.mean()
                # Clears gradients from the previous batch.
                optimizer.zero_grad()
                # Update both encoders.
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            # The average loss is the mean batch loss for that epoch.
            average_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch + 1}/{num_epochs}, loss: {average_loss:.4f}")
    