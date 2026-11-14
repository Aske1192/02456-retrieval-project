from src.data_utils.query import Query


def calculateRecall(relevancies: list[bool], query: Query):
    return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())
    # Sum(relevancies) checks number of relevant documents in the top k
    # The denominator just has a min argument in case the number of relevant documents is smaller than k.
    # That way the score will still b 1, if all relevant documents are found in top k.




def calculatePrecisionAtK(relevancies: list[bool]):
    return sum(relevancies) / len(relevancies)

def calculateRecallAtK(relevancies: list[bool], query: Query): # The standard Recall@k method
    return sum(relevancies) / query.getNumberOfRelevantDocuments()


def calculate_MRR(relevancies: list[bool]):
    if True in relevancies:
        return 1 / (relevancies.index(True) + 1)
    return 0



def Evaluate(relevancies: list[list[bool]], queries: list[Query]):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    scores = {'recall': None, 'Precision@k': None, 'Recall@k': None, 'MRR': None}
    n_queries = len(queries)
    total_recall = 0
    total_precisionAtK = 0
    total_recallAtK = 0
    total_MRR = 0
    
    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i])
        total_precisionAtK += calculatePrecisionAtK(relevancies[i])
        total_recallAtK += calculateRecallAtK(relevancies[i], queries[i])
        total_MRR += calculate_MRR(relevancies[i])

    
    scores['recall'] = total_recall / n_queries
    scores['Precision@k'] = total_precisionAtK / n_queries
    scores['Recall@k'] = total_recallAtK / n_queries
    scores['MRR'] = total_MRR / n_queries
    
    return scores
