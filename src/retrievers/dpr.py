import torch
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.base_retriever import Retriever

class DPR(Retriever):
    def __init__(self, cfg,documents: list[dict] = None, index_path: str = None, embeddings: torch.Tensor = None)->None:
        super(DPR,self).__init__(documents, index_path)
        self.cfg = cfg
        self.embeddings = embeddings
        self.__load_models()
    
    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim
        query_encoder = TransformerEncoder(...)
        document_encoder = TransformerEncoder(...)
        
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            query_encoder.load_state_dict(query_weights_path)

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            document_encoder.load_state_dict(document_weights_path)
    
    def CalculateScores(self, queries):
        query_embeddings = torch.as_tensor(
            queries,
            dtype=torch.float32,
            device=self.index.document_embeddings.device,
        )
        scores = query_embeddings @ self.index.document_embeddings.T
        return scores.detach().cpu().numpy()
    

        

