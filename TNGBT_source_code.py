#=============================TNGBT - Transformer with N-Gram Based Training=============================

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import heapq

seed = 383904343

torch.manual_seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_file(path):
    with open (path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
 
def get_window_seq_data(data: str, win_size):
    data = data.split()
    out = []

    for i in range(win_size, len(data)):
        context = data[i - win_size:i]
        target = data[i]
        out.append((context, target))

    return out

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_size, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()

        self.context_size = context_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=layer, num_layers=n_layers)

        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        _, sequence_length = token_ids.shape

        position_start = self.context_size - sequence_length
        positions = torch.arange(position_start, self.context_size, device=token_ids.device)

        x = (self.token_embedding(token_ids) + self.position_embedding(positions))

        mask = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=token_ids.device), diagonal=1)

        x = self.transformer(x, mask=mask)
        x = self.final_norm(x)

        logits = self.output(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

        return logits, loss

def predict_word_transformer(model, tokens, token_to_id, id_to_token, max_new_tokens, temperature):
    model.eval()
    model_device = next(model.parameters()).device

    token_ids = [token_to_id[token] for token in tokens]
    out = torch.tensor([token_ids], dtype=torch.long, device=model_device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            transformer_logits, _ = model(out[:, -model.context_size:])
            next_logits = transformer_logits[0, -1]

            probs = torch.softmax(next_logits / temperature, dim=-1)

            next_tok = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_tok.reshape(1, 1)], dim=1)

    out_lst = [id_to_token[token_id] for token_id in out[0].tolist()]

    return out_lst

def train_model(data_tokens, win_size, vocab, conns_sizes, current_chunk, conns_bank_max, heap_dict, version):
    seq = get_window_seq_data(data_tokens, win_size)

    conns = conns_sizes[win_size]
    heap = heap_dict[win_size]["heap"]

    prev_vocab = set(vocab)
    tokens = set(data_tokens) - prev_vocab
    vocab = sorted(prev_vocab | tokens)

    for context_toks, target_tok in seq:
        context = tuple(context_toks)

        if context in conns:
            conns[context]["total"] += 1
            conns[context]["estimate"] += 1
        elif len(conns) < conns_bank_max[win_size]:
            conns[context] = {"total": 1, "targets": {}, "seen": current_chunk, "age": 0, "estimate": 1, "error": 0, "heap_version": 0}
        else:
            while True:
                min_estimate, min_version, min_context = heap[0]
                if min_context in conns and conns[min_context]["heap_version"] == min_version:
                    heapq.heappop(heap)
                    break
                heapq.heappop(heap)

            del conns[min_context]

            estimate = min_estimate + 1
            conns[context] = {"total": estimate, "targets": {}, "seen": current_chunk, "age": 0, "estimate": estimate, "error": min_estimate, "heap_version": 0}

        version += 1
        conns[context]["seen"] = current_chunk
        conns[context]["heap_version"] = version
        heapq.heappush(heap, (conns[context]["estimate"], version, context))

        targets = conns[context]["targets"]
        targets[target_tok] = targets.get(target_tok, 0) + 1

    return vocab, conns_sizes, heap_dict, version

def n_gram_correction(model, optimizer, vocab_size, token_to_id, conns, context_size, districts, n_gram_influence):
    batch_size = 256

    contexts_length = {}
    for context in conns:
        context_length = len(context)

        if 1 <= context_length and context_length <= context_size and conns[context]["age"] >= districts:
            contexts_length.setdefault(context_length, []).append(context)

    for context, conn in conns.items():
        conn["age"] += 1

    if not contexts_length:
        return conns

    lengths = sorted(contexts_length)

    samples_length = max(1, batch_size // len(lengths))
    samples_active = max(0, batch_size - (samples_length * len(lengths)))

    model.train()

    model_device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=model_device)

    total_data = 0
    sampled_counts = {}
    for index, context_length in enumerate(lengths):
        req_count = samples_length

        if index < samples_active:
            req_count += 1

        chosen_contexts = random.sample(contexts_length[context_length], min(req_count, len(contexts_length[context_length])))
        
        inputs = []
        probs = []
        confidences = []
        for context in chosen_contexts:
            conns[context]["age"] = 0
            target_counts = conns[context]["targets"]
            total_count = conns[context]["total"]

            context_ids = [token_to_id[token] for token in context]
            n_gram_probs = torch.zeros(vocab_size, dtype=torch.float32)

            for target_token, count in target_counts.items():
                target_id = token_to_id[target_token]
                n_gram_probs[target_id] = (count / total_count)

            inputs.append(context_ids)
            probs.append(n_gram_probs)
            confidences.append(total_count / (total_count + 2))

        if not inputs:
            continue

        input_ids = torch.tensor(inputs, dtype=torch.long, device=model_device)
        n_gram_probs = torch.stack(probs).to(device=model_device)
        model_output = model(input_ids)

        if isinstance(model_output, tuple):
            logits = model_output[0]
        elif hasattr(model_output, "logits"):
            logits = model_output.logits
        else:
            logits = model_output

        transformer_logits = logits[:, -1, :]
        transformer_log_probs = F.log_softmax(transformer_logits, dim=-1)
        transformer_probs = (transformer_log_probs.exp().detach())

        confidence = torch.tensor(confidences, dtype=torch.float32, device=model_device).unsqueeze(1)
        correction_targets = ((1 - n_gram_influence) * transformer_probs + n_gram_influence * (confidence * n_gram_probs + (1 - confidence) * transformer_probs)).detach()

        per_data_loss = -(correction_targets * transformer_log_probs).sum(dim=-1)
        total_loss = (total_loss + per_data_loss.sum())

        data_count = len(inputs)
        total_data += data_count

        sampled_counts[context_length] = data_count

    correction_loss = (total_loss / total_data)
    correction_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()

    if not model.training:
        model.eval()

    return conns

def basic_punctuation_spacer(string):
    return string.replace(".", " . ").replace("?", " ? ").replace("!", " ! ").replace(",", " , ").replace(")", " ) ").replace("(", " ( ").replace("[", " [ ").replace("]", " ] ").replace("'", " ' ").replace('"', ' " ').replace(":", " : ").replace(";", " ; ")

def main():
    conns_bank_max = {3: 1000000, 4: 1000000, 6: 1000000, 8: 1000000, 12: 1200000, 16: 1400000, 24: 1600000, 32: 1850000}
    epochs = 330

    n_gram_influence = 0.25
    districts = 10

    win_size = 32

    temperature = 1
    transformer_lr = 3e-4
    max_new_tokens = 10

    d_model = 128
    n_heads = 4
    n_layers = 4
    d_ff = 512
    dropout = 0.1

    context_windows = [3, 4, 6, 8, 12, 16, 24, 32]
    version = 0

    path = Path(rf"TRAINING DATA PATH HERE")

    file_lst = []
    for file in sorted(os.listdir(path)):
        file_path = path / file
        
        file_context = basic_punctuation_spacer(read_file(file_path)).lower()
        file_lst.append(file_context)
    
    total_files = len(file_lst)

    vocab = {"the"}
    for file_context in file_lst:
        all_tokens = file_context.split()

        for tok in all_tokens[:int(len(all_tokens) * 0.8)]:
            vocab.add(tok)

    vocab = sorted(vocab)

    token_to_id = {}
    for token_id, token in enumerate(vocab):
        token_to_id[token] = token_id

    id_to_token = {}
    for token, token_id in token_to_id.items():
        id_to_token[token_id] = token

    heap_dict = {}
    conns = {}
    for num in context_windows:
        heap_dict[num] = {"heap": [], "contexts": set()}
        conns[num] = {}

    n_gram_vocab = []

    model = Transformer(vocab_size=len(vocab), context_size=win_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=transformer_lr, weight_decay=0.01)
    
    for current_file, file_data in enumerate(file_lst, start=1):
        all_task_tokens = file_data.split()
        split_index = int(len(all_task_tokens) * 0.8)
        data2_lst = all_task_tokens[:split_index]

        print(f"Current File: {current_file}/{total_files}")

        for i in context_windows:
            n_gram_vocab, conns, heap_dict, version = train_model(data2_lst, i, n_gram_vocab, conns, current_file, conns_bank_max, heap_dict, version)

        singular_dict_conns = {}
        for dct in conns.values():
            singular_dict_conns.update(dct)

        for epoch in range(epochs):
            singular_dict_conns = n_gram_correction(model, optimizer, len(vocab), token_to_id, singular_dict_conns, win_size, districts, n_gram_influence)

    while True:
        tokens = basic_punctuation_spacer(input("input: ")).lower().split()

        output = predict_word_transformer(model, tokens, token_to_id, id_to_token, max_new_tokens, temperature)
        output = " ".join(output)

        print(output)

if __name__ == "__main__":
    main()
