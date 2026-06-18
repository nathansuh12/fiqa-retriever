# FiQA Dense Retriever — Fine-Tuning a Domain Embedding Model

Fine-tunes a general-purpose sentence embedding model into a specialized
retriever for the financial-domain FiQA dataset, and measures the change in
retrieval quality honestly: the same evaluation is run on the off-the-shelf
model and on the fine-tuned model, so any difference is attributable to the
fine-tuning and nothing else.

The question this project answers is not "can I build a retriever" but
"does fine-tuning a general embedding model actually improve retrieval on a
specialized domain, and by how much."

## Result

Metric: recall@10 on the FiQA test split (a query "hits" if at least one of
its known-relevant documents appears in the top 10 retrieved).

| Model                                   | recall@10  |
|-----------------------------------------|------------|
| Base (off-the-shelf, no fine-tuning)    | **64.81%** |
| Fine-tuned (in-batch negatives)         | **68.98%**  |
| Fine-tuned (+ hard-negative mining)     | _pending_  |

> The two fine-tuned rows are filled in after running `train.py` and
> re-running the evaluation. They are intentionally left blank until those
> numbers are real.

## Why this is non-trivial

Off-the-shelf embedding models are trained to be general-purpose. On a narrow
domain like financial QA they are mediocre at separating a query's true answer
from near-miss passages, which is exactly when retrieval fails. Fine-tuning on
in-domain (query, answer) pairs pulls each query closer to its correct answer
in vector space. The baseline of 64.81% confirms there is real headroom to
improve before any training is done.

## Dataset

[FiQA](https://huggingface.co/datasets/BeIR/fiqa) via the BeIR benchmark,
loaded from HuggingFace. Three pieces:

- **corpus** — ~57k financial passages (the searchable haystack)
- **queries** — natural-language financial questions
- **qrels** — the relevance judgments linking each query to its answer
  passage(s); `train` split is used for fine-tuning, `test` split for scoring

## Method

- **Encoder** (`model.py`): mean-pools the token embeddings of a base
  transformer (masked over padding) and L2-normalizes the result, so cosine
  similarity reduces to a dot product. Base model:
  `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
- **Retrieval** (`eval.py`): encode the whole corpus once, encode the test
  queries, score every query against every document with a single matrix
  multiply, take the top-k per query, and check the retrieved document ids
  against the relevance judgments.
- **Metric**: recall@10.
- **Training** (`train.py`): contrastive learning with the InfoNCE objective
  and in-batch negatives — for each (query, positive) pair in a batch, every
  other passage in the batch acts as a negative. Optimized with AdamW.
- **Hard-negative mining** (stage 2): use the stage-1 model to retrieve the
  highest-ranked *incorrect* passages per training query, add them as
  explicit hard negatives, and retrain.

## Setup

```bash
python -m venv retriever-env
source retriever-env/bin/activate     # Windows: retriever-env\Scripts\activate
pip install torch transformers datasets numpy
```

A CUDA GPU is used automatically if available (the encoder and training loop
both detect it); otherwise it falls back to CPU.

## Reproduce

1. `data.py` — load FiQA, build the (query, positive) training pairs and the
   per-split relevance answer key.
2. Baseline: encode the corpus and test queries with the base model, retrieve
   top-10, compute recall@10 → **64.81%**.
3. `train.py` — fine-tune the encoder on the training pairs (stage 1: in-batch
   negatives; stage 2: hard negatives).
4. Re-run step 2's evaluation on the fine-tuned model and record the new
   recall@10 in the results table above.

## Repo structure

```
fiqa-retriever/
├── README.md            # this file — the result and how to reproduce it
├── data.py              # load FiQA, build train pairs + relevance key
├── model.py             # the Encoder: text -> mean-pooled, normalized vector
├── eval.py              # encode corpus, retrieve top-k, compute recall@k
├── train.py             # contrastive fine-tuning loop (stage 1 + stage 2)
└── results/             # one record per run (base / stage-1 / stage-2)
```

## Status

- [x] M0 — data loaded, query↔qrel↔doc join verified
- [x] M1 — baseline recall@10 measured (64.81%)
- [x] M2 — training pairs and relevance key built
- [x] M3 — contrastive training loop implemented and running
- [x] M4 — fine-tuned recall@10 measured and compared to baseline
- [ ] M5 — hard-negative mining
- [ ] M6 — drift monitoring on simulated drift (stretch)
