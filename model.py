import torch; print(torch.cuda.is_available())
from transformers import AutoModel, AutoTokenizer


class Encoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # use the GPU if one is available, otherwise fall back to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # we're not training yet, just embedding

    @torch.no_grad()  # no gradients needed for plain encoding
    def encode(self, texts, batch_size=64):
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=True,
                                    return_tensors="pt")
            # move this batch onto the same device as the model
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            output = self.model(**tokens)

            # --- masked mean pooling ---
            hidden = output.last_hidden_state           # (batch, seq_len, hidden)
            mask = tokens["attention_mask"]             # (batch, seq_len)

            # expand mask to (batch, seq_len, 1) so it lines up with hidden
            mask = mask.unsqueeze(-1).float()

            # zero out padding positions, then sum over the sequence dimension
            summed = (hidden * mask).sum(dim=1)         # (batch, hidden)

            # divide by the number of REAL tokens (clamp so we never divide by 0)
            counts = mask.sum(dim=1).clamp(min=1e-9)    # (batch, 1)
            mean = summed / counts                      # (batch, hidden)

            # L2-normalize so cosine similarity is just a dot product later
            normed = torch.nn.functional.normalize(mean, p=2, dim=1)

            # move back to CPU so the rest of your code never touches the GPU
            all_vecs.append(normed.cpu())

        return torch.cat(all_vecs)