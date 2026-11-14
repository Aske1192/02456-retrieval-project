from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
from scipy.sparse import lil_matrix
import re


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class. This is used to make sure the child class (bm25) doesn't create the tf-idf matrix.
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted charatcters from text and lowercase
        """
        return re.sub(r"[^a-z ']+", " ", text.lower()).split() # Makes everything lower case and removes anything that is not a lowercase letter, a space, or an apostrophe with a space. Splits into list with words.
        
    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        return set(self.PreprocessText(query))

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        return set(self.PreprocessText(document.GetText()))
    
    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        # Use map function for more efficient processing
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        frequencies = {}
        for document in self.index.GetDocuments():
            vocabulary = self.GetDocumentVocabulary(document)

            for term in vocabulary:
                frequencies[term] = frequencies.get(term, 0) + 1
        
        return frequencies
       
    
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        N = len(self.index.GetDocuments())
        frequencies = self.GetDocumentFrequencies()

        IDF_dict = {term: np.log(N / nq) for term, nq in frequencies.items()}
        return IDF_dict
    
    
    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """

        term_counts = {}
        document_text = self.PreprocessText(document.GetText())

        for term in document_text:
            term_counts[term] = term_counts.get(term, 0) + 1
        
        return term_counts

    

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)} # enumerate gives each term and index value. "term: idx for idx, term in" creates a dictionary with the terms and their index values.

        # Initialize a sparse matrix
        documents = self.index.GetDocuments()
        n_documents = len(documents)
        n_terms = len(self.corpus_vocabulary)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        # TODO: Add tf-idf code
        for row_index, document in enumerate(documents):

            # Get term counts for each document
            counts = self.GetDocumentTermCounts(document)

            # Skip documents with no terms (nothing to put in the matrix)
            if not counts:
                continue

            # Find count of most frequent term
            most_frequent_term = max(counts.values())

            # Calculate TF-IDF for each term
            for term, count in counts.items():
                tf = count / most_frequent_term
                idf = self.idf[term]
                tfidf = tf * idf

                # Find column index
                column_index = self.term_to_index[term]

                # Insert TF-IDF value into matrix
                tfidf_matrix[row_index, column_index] = tfidf

        return tfidf_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)

        # TODO: add code
    
        for row_index, query in enumerate(queries):
            term_counts = {}
            terms = self.PreprocessText(query)

            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1

            for term, count in term_counts.items():

                # Find column index. If-statement makes sure that only query terms in the vocabulary are considered.
                if term not in self.term_to_index:
                    continue
                
                column_index = self.term_to_index[term]

                query_vector[row_index, column_index] = count

        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    