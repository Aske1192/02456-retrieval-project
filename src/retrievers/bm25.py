from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document
import numpy as np

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        self.bm25_matrix = self.GetDocumentsBM25Vectors() 
    
    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())
    
    
    
    # TODO: Add methods for computing self.bm25_matrix

    # Define updated IDF calculation for bm25
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        N = len(self.index.GetDocuments())
        frequencies = self.GetDocumentFrequencies()

        IDF_dict = {term: np.log((N+1) / (nq+0.5)) for term, nq in frequencies.items()}
        return IDF_dict


    def GetDocumentsBM25Vectors(self):
            """
            Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
            """
            # Create a mapping from terms to indices
            self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)} # enumerate gives each term and index value. "term: idx for idx, term in" creates a dictionary with the terms and their index values.
    
            # Initialize a sparse matrix
            documents = self.index.GetDocuments()
            n_documents = len(documents)
            n_terms = len(self.corpus_vocabulary)
            bm25_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

            # Get variables for the TF equation
            k1, b = self.k1, self.b
            document_lengths, avg_length = self.GetDocumentLengths()
    
            for row_index, document in enumerate(documents):
    
                # Get term counts for each document
                counts = self.GetDocumentTermCounts(document)
                # Define theta from the TF equation
                theta = document_lengths[document]/avg_length
    
                # Skip documents with no terms (nothing to put in the matrix)
                if not counts:
                    continue
    
                # Calculate bm25 for each term
                for term, count in counts.items():
                    idf = self.idf[term]
                    bm25 = idf * (count * (k1 + 1)) / (count + k1 * (1 - b + b * theta))
    
                    # Find column index
                    column_index = self.term_to_index[term]
    
                    # Insert bm25 value into matrix
                    bm25_matrix[row_index, column_index] = bm25
    
            return bm25_matrix.tocsr()   # Convert to CSR format for efficient row slicing




    
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores