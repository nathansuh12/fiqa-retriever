from datasets import load_dataset
from collections import defaultdict

def load_fiqa():
    corpus  = load_dataset("BeIR/fiqa", "corpus")["corpus"]     # the haystack (~57k docs)
    queries = load_dataset("BeIR/fiqa", "queries")["queries"]   # the questions
    qrels   = load_dataset("BeIR/fiqa-qrels") 
    query_text = {str(row["_id"]): row["text"] for row in queries}
    doc_text   = {str(row["_id"]): row["text"] for row in corpus}

    return query_text, doc_text, qrels

def build_train_pairs(query_text, doc_text, qrels):
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

def recall_at_k(top_idx, query_ids, corpus_ids, relevant, k=10):
    hits = 0
    counted = 0
    for row, qid in enumerate(query_ids):
        if qid not in relevant:          # skip queries with no known answer
            continue
        counted += 1
        retrieved_ids = {corpus_ids[pos.item()] for pos in top_idx[row][:k]}
        if retrieved_ids & relevant[qid]:   # set overlap = at least one hit
            hits += 1
    return hits / counted