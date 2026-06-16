import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from data import load_fiqa, build_train_pairs


def train(model_name="sentence-transformers/all-MiniLM-L6-v2",
          epochs=1, batch_size=64, lr=2e-5):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    query_text, doc_text, qrels = load_fiqa()
    pairs = build_train_pairs(query_text, doc_text, qrels)
    loader = DataLoader(pairs, batch_size=batch_size, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def embed(texts):
        tokens = tokenizer(list(texts), padding=True, truncation=True,
                           return_tensors="pt").to(device)
        out = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(mean, p=2, dim=1)

    temperature = 0.05
    step = 0
    for epoch in range(epochs):
        for queries, positives in loader:
            q = embed(queries)
            p = embed(positives)

            labels = torch.arange(len(q), device=device)
            scores = (q @ p.T) / temperature
            loss = F.cross_entropy(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"step {step}: loss {loss.item():.4f}")
            step += 1

        print(f"epoch {epoch}: loss {loss.item():.4f}")

    return model, tokenizer