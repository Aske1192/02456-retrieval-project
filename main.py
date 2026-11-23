import hydra
from src.data_utils.corpus import Corpus
from src.retrievers.tf_idf import TFIDF
from src.data_utils.query import Query
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate
from src.data_utils.dataset import DPRTrainingDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch

from src.retrievers.dpr import DPR
from src.training.trainer import Trainer
@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    ### Prepare queries
    train_queries = [Query(corpus.train['question'][idx], 
                           id=corpus.train['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.train['query_id'][idx]]
                           ) 
                        for idx in range(corpus.train.num_rows)
                    ]

    val_queries =   [Query(corpus.val['question'][idx], 
                         id=corpus.val['query_id'][idx], 
                         relevant_document_ids=corpus.relevant_ids[corpus.val['query_id'][idx]]
                           ) 
                        for idx in range(corpus.val.num_rows)
                    ]
    
    test_queries =  [Query(corpus.test['question'][idx], 
                           id=corpus.test['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.test['query_id'][idx]]
                           ) 
                        for idx in range(corpus.test.num_rows)
                    ]
    
    ### Initialize TF-IDF class
    #tf_idf = TFIDF(documents=corpus.all_documents)
    
    ### Retrieve top-k relevant queries
    #tfidf_train_relevancies = retrieveQueryAndGetRelevancies(tf_idf, train_queries, k=cfg.top_k)
    #tfidf_val_relevancies = retrieveQueryAndGetRelevancies(tf_idf, val_queries, k=cfg.top_k)
    #tfidf_test_relevancies = retrieveQueryAndGetRelevancies(tf_idf, test_queries, k=cfg.top_k)

    ### Evaluate
    #train_scores = Evaluate(tfidf_train_relevancies, train_queries)
   # val_scores = Evaluate(tfidf_val_relevancies, val_queries)
    #test_scores = Evaluate(tfidf_test_relevancies, test_queries)

    #print(f'Train results: {train_scores}')
    #print(f'Val results: {val_scores}')
    #print(f'Test results: {test_scores}')


    # Use the same tokenizer for training and passage encoding.
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)

    # Make the encoder's embedding table fit this tokenizer exactly.
    cfg.model.vocab_size = len(tokenizer)

    dpr_dataset = DPRTrainingDataset(
        rows=corpus.train,
        tokenizer=tokenizer,
        max_length=cfg.model.max_len,
    )

    train_loader = DataLoader(
        dpr_dataset,
        batch_size=32,
        shuffle=True,
    )

    # Build the DPR model with the corpus documents in its inherited index.
    dpr = DPR(
        cfg=cfg,
        documents=corpus.all_documents,
    )

    device = torch.device(cfg.device)

    # The trainer updates both encoders in place.
    trainer = Trainer()
    trainer.train(
        query_encoder=dpr.query_encoder,
        document_encoder=dpr.document_encoder,
        train_loader=train_loader,
        num_epochs=3,
        learning_rate=2e-4,
        device=device,
    )

    # Encode every indexed passage so DPR can score queries against them later.
    dpr.document_encoder.eval()
    dpr.document_encoder.to(device)

    passage_texts = [
        document.GetText()
        for document in dpr.index.GetDocuments()
    ]

    passage_embedding_batches = []
    batch_size = 32

    with torch.no_grad():
        for start in range(0, len(passage_texts), batch_size):
            passage_batch = passage_texts[start : start + batch_size]

            tokens = tokenizer(
                passage_batch,
                max_length=cfg.model.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            batch_embeddings = dpr.document_encoder(
                input_ids,
                attention_mask,
            )

            passage_embedding_batches.append(batch_embeddings.cpu())

    # Store vectors in document order so each score still matches its passage.
    dpr.index.document_embeddings = torch.cat(
        passage_embedding_batches,
        dim=0,
    )
    # Retrieve passages for each split and check them against the known relevant IDs.
    train_relevancies = retrieveQueryAndGetRelevancies(
        dpr,
        train_queries,
        k=cfg.top_k,
    )
    val_relevancies = retrieveQueryAndGetRelevancies(
        dpr,
        val_queries,
        k=cfg.top_k,
    )
    test_relevancies = retrieveQueryAndGetRelevancies(
        dpr,
        test_queries,
        k=cfg.top_k,
    )

    train_scores = Evaluate(train_relevancies, train_queries)
    val_scores = Evaluate(val_relevancies, val_queries)
    test_scores = Evaluate(test_relevancies, test_queries)

    print(f"DPR train results: {train_scores}")
    print(f"DPR validation results: {val_scores}")
    print(f"DPR test results: {test_scores}")

if __name__ == '__main__':
    main()