import torch
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.base_retriever import Retriever
from transformers import AutoTokenizer

class DPR(Retriever):
    def __init__(self, cfg,documents: list[dict] = None, index_path: str = None, embeddings: torch.Tensor = None)->None:
        super(DPR,self).__init__(documents, index_path)
        self.cfg = cfg
        self.embeddings = embeddings
        if embeddings is not None:
            self.index.document_embeddings = torch.as_tensor(
                embeddings,
                dtype=torch.float32,
            )
        self.__load_models()
    
    def __load_models(self):

        model_cfg = self.cfg.model
        # Use the same tokenizer that prepared the training examples.
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name)
        encoder_args = {
            "vocab_size": model_cfg.vocab_size,
            "max_len": model_cfg.max_len,
            "n_layers": model_cfg.n_layers,
            "n_heads": model_cfg.n_heads,
            "emb_dim": model_cfg.emb_dim,
            "feed_forward_dim": model_cfg.feed_forward_dim,
            "pooling_type": model_cfg.pooling_type,
        }
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim
        self.query_encoder = TransformerEncoder(**encoder_args)
        self.document_encoder = TransformerEncoder(**encoder_args)

        # Load weights if specified in the configuration
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            query_state_dict = torch.load(query_weights_path, map_location=torch.device('cpu'))
            self.query_encoder.load_state_dict(query_state_dict)

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('document_encoder_weights_path', None):
            document_state_dict = torch.load(document_weights_path, map_location=torch.device('cpu'))
            self.document_encoder.load_state_dict(document_state_dict)
    
    def CalculateScores(self, queries):

        device = next(self.query_encoder.parameters()).device

        # Lookup passes raw query strings; tensor input is also useful for direct checks.
        if torch.is_tensor(queries):
            query_embeddings = queries.to(device=device, dtype=torch.float32)
        else:
            query_texts = list(queries)
            query_embedding_batches = []
            query_batch_size = 16

            self.query_encoder.eval()
            with torch.no_grad():
                for start in range(0, len(query_texts), query_batch_size):
                    query_batch = query_texts[start : start + query_batch_size]

                    tokens = self.tokenizer(
                        query_batch,
                        max_length=self.cfg.model.max_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )

                    batch_embeddings = self.query_encoder(
                        tokens["input_ids"].to(device),
                        tokens["attention_mask"].to(device),
                    )
                    query_embedding_batches.append(batch_embeddings)

            query_embeddings = torch.cat(query_embedding_batches, dim=0)

        if self.index.document_embeddings is None:
            raise ValueError(
                "Passage embeddings are missing. Encode the corpus before retrieval."
            )

        document_embeddings = self.index.document_embeddings.to(device)
        scores = query_embeddings @ document_embeddings.T
        return scores.detach().cpu().numpy()        

