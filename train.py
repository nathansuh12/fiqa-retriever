import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from data import load_fiqa, build_train_pairs


def train(model_name="sentence-transformers/all-MiniLM-L6-v2",
          epochs=1, batch_size=32, lr=2e-5):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data: the flat (query, positive) pairs from M2 ---
    query_text, doc_text, qrels = load_fiqa()
    pairs = build_train_pairs(query_text, doc_text, qrels)   # list of (q, pos)
    loader = DataLoader(pairs, batch_size=batch_size, shuffle=True)

    # --- model + tokenizer: trainable this time, so NO eval(), NO no_grad ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def embed(texts):
        # same masked-mean+normalize as model.py, but gradients ON
        tokens = tokenizer(list(texts), padding=True, truncation=True,
                           return_tensors="pt").to(device)
        out = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(mean, p=2, dim=1)

    for epoch in range(epochs):
        for queries, positives in loader:
            q = embed(queries)      # (batch, 384)
            p = embed(positives)    # (batch, 384)

            # similarity matrix: every query vs every positive in the batch
            scores = q @ p.T        # (batch, batch)

            # >>> YOUR PART: the InfoNCE loss <
            # the diagonal scores[i][i] is the TRUE pair; every off-diagonal
            # scores[i][j] is an in-batch negative. so this is just
            # cross-entropy where the correct "class" for row i is i.
            labels = torch.arange(len(q), device=device)
            loss = torch.nn.functional.cross_entropy(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch}: loss {loss.item():.4f}")

    return model, tokenizer