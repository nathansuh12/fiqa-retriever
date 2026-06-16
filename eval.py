from model import Encoder
from data import recall_at_k

def run_eval(model_name, doc_text, query_text, relevant, k=10):
    enc = Encoder(model_name)

    corpus_ids   = list(doc_text.keys())
    corpus_texts = [doc_text[i] for i in corpus_ids]
    corpus_matrix = enc.encode(corpus_texts)

    query_ids   = list(query_text.keys())
    query_texts = [query_text[i] for i in query_ids]
    query_matrix = enc.encode(query_texts)

    score = query_matrix @ corpus_matrix.T
    top_scores, top_idx = score.topk(k, dim=1)
    
    return recall_at_k(top_idx, query_ids, corpus_ids, relevant, k=k)