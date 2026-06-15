from datasets import load_dataset
from collections import defaultdict

def load_fiqa():
    corpus  = load_dataset("BeIR/fiqa", "corpus")["corpus"]     # the haystack (~57k docs)
    queries = load_dataset("BeIR/fiqa", "queries")["queries"]   # the questions
    qrels   = load_dataset("BeIR/fiqa-qrels") 
    query_text = {str(row["_id"]): row["text"] for row in queries}
    doc_text   = {str(row["_id"]): row["text"] for row in corpus}

    return query_text, doc_text, qrels

def build_train_paris(query_text, doc_text, qrels):
    train_pairs = []
    for row in qrels['train']:
        qid, did = str(row['query-id']), str(row['corpus-id'])
        if row["score"] > 0:
            train_pairs.append((query_text[qid], doc_text[did]))

    return train_pairs

def build_relevant(qrels, split):
    relevant = defaultdict(set)
    for row in qrels[split]:
        if row["score"] > 0:
            relevant[str(row["query-id"])].add(str(row["corpus-id"]))

    return relevant