# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken
from tqdm import tqdm


################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")

    # TensorBoard run name for comparison experiments
    parser.add_argument("--run_name", type=str, default="default",
                        help="Name for this training run (used in TensorBoard directory). Default='default'.")

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # fill in

        self.net = None

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        seq_len, batch_size = tokens_seq.shape
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size
                    )
                    context_flat = context_oh.flatten().float().unsqueeze(0)
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs


################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.weight

class Attention(nn.Module):
    def __init__(self, d_model=1024, n_heads=2, head_dim=128, normalization='sandwich'):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.normalization = normalization
        self.q_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, kv_cache=None, use_cache=False, return_attn: bool = False):
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, self.n_heads, self.head_dim)

        if self.normalization == 'sandwich':
            q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
            k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        else:
            q = self.q_proj(x).view(hidden_shape).transpose(1, 2)
            k = self.k_proj(x).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_cache = (k, v) if use_cache else None

        T_q = q.size(2)
        T_k = k.size(2)
        device = x.device
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        cache_len = T_k - T_q
        q_idx = torch.arange(T_q, device=device).unsqueeze(1) + cache_len
        k_idx = torch.arange(T_k, device=device).unsqueeze(0)
        causal_mask = k_idx > q_idx
        causal_mask = causal_mask.view(1, 1, T_q, T_k)

        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if return_attn:
            return attn_output, new_cache, attn_weights
        else:
            return attn_output, new_cache

class MLP(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=4096):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class TransformerBlock(nn.Module):
    def __init__(self, d_model=1024, n_heads=2, head_dim=128, normalization='sandwich'):
        super().__init__()
        self.normalization = normalization
        self.input_layernorm = RMSNorm(d_model)
        self.post_attention_layernorm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, head_dim)
        self.mlp = MLP(d_model, d_model * 4)

    def forward(
        self,
        x,
        kv_cache=None,
        use_cache: bool = False,
        output_attentions: bool = False,       # [E-added] 
    ):
        residual = x
        if self.normalization == 'pre' or self.normalization == 'sandwich':
            x = self.input_layernorm(x)

        if output_attentions:                  # [E-added]
            x_attn, new_cache, attn_weights = self.attn(
                x,
                kv_cache=kv_cache,
                use_cache=use_cache,
                return_attn=True,             # [E-added]
            )
        else:
            x_attn, new_cache = self.attn(
                x,
                kv_cache=kv_cache,
                use_cache=use_cache,
                return_attn=False,            # [E-added]
            )
            attn_weights = None               # [E-added]

        x = x_attn + residual
        if self.normalization == 'post':
            x = self.input_layernorm(x)

        residual = x
        if self.normalization == 'pre' or self.normalization == 'sandwich':
            x = self.post_attention_layernorm(x)
        x_mlp = self.mlp(x)
        x = x_mlp + residual
        if self.normalization == 'post':
            x = self.post_attention_layernorm(x)

        if output_attentions:                  # [E-added]
            return x, new_cache, attn_weights
        else:
            return x, new_cache

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4, head_dim=128, block_size=1024, normalization='sandwich'):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_embed = nn.Embedding(num_embeddings=block_size, embedding_dim=d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, head_dim=head_dim, normalization=normalization)
             for _ in range(n_blocks)]
        )
        self.rmsnorm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        x,
        kv_caches=None,
        use_cache: bool = False,
        output_attentions: bool = False,    # [E-added]
    ):
        # x: (seq_len, batch)
        x = x.transpose(0, 1)
        batch_size, seq_len = x.shape

        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos_ids)

        if kv_caches is None:
            kv_caches = [None] * self.n_blocks
        new_kv_caches = []
        attn_list = []                      # [E-added]

        for i, block in enumerate(self.blocks):
            if output_attentions:           # [E-added]
                x, new_cache, attn_weights = block(
                    x,
                    kv_cache=kv_caches[i],
                    use_cache=use_cache,
                    output_attentions=True,
                )
                attn_list.append(attn_weights)
            else:
                x, new_cache = block(
                    x,
                    kv_cache=kv_caches[i],
                    use_cache=use_cache,
                    output_attentions=False,
                )
            new_kv_caches.append(new_cache)

        x = self.rmsnorm(x)
        logits = self.lm_head(x)
        logits = logits.transpose(0, 1)

        if output_attentions:               # [E-added]
            return logits, attn_list
        else:
            return logits


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    """
    Top-p (nucleus) sampling implementation.
    
    Args:
        logits: 1D tensor of logits (vocab_size,)
        p: probability threshold (default 0.95)
    
    Returns:
        int: sampled token id
    
    Algorithm:
    1. Sort tokens by probability in descending order
    2. Find smallest k such that sum(p(1)...p(k)) >= p
    3. Sample from the truncated distribution of top k tokens
    """
    # Step 1: Convert logits to probabilities using softmax
    probs = F.softmax(logits, dim=-1)
    
    # Step 2: Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Step 3: Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Step 4: Find the cutoff point k
    mask = cumulative_probs <= p
    mask[0] = True  # Always keep at least the first token
    
    # Step 5: Truncate the distribution
    truncated_probs = sorted_probs[mask]
    truncated_indices = sorted_indices[mask]
    
    # Step 6: Renormalize the truncated probabilities
    truncated_probs = truncated_probs / truncated_probs.sum()
    
    # Step 7: Sample from the truncated distribution
    sampled_idx = torch.multinomial(truncated_probs, num_samples=1).item()
    chosen_token = truncated_indices[sampled_idx].item()
    
    return chosen_token


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


################################################################################
# 8. Training
################################################################################

def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a",
                    writer=None,
                    test_loader=None):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    Added `writer` for TensorBoard logging.
    Added `test_loader` for overfitting study (train/test gap).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    for epoch in tqdm(range(1, epochs + 1), leave=False):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            # logging loss
            loss_dict[model_name].append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            # logging gradient
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_dict[model_name].append(total_norm)

            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            # TensorBoard logging
            if writer is not None:
                writer.add_scalar(f"{model_name}/loss", loss.item(), global_step)

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    # Test multiple p values: greedy + range from 0.8 to 1.0 with step 0.05
                    p_values = [None, 0.80, 0.85, 0.90, 0.95, 1.00]
                    for p in p_values:
                        p_label = "greedy" if p is None else f"p={p:.2f}"
                        print(f"[{model_name}] Generating sample text ({p_label}) at epoch={epoch}, step={batch_idx}...")
                        text, ann = generate_text(
                            model, enc, prompt, max_new_tokens=20, device=device,
                            top_p=p,
                            monosemantic_info=monosemantic_info,
                            do_monosemantic=(monosemantic_info is not None)
                        )
                        print(f" {p_label:8} Sample: {text}")
                        print(f" Annotated: {ann}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Train Loss: {avg_loss:.4f}")

        # TensorBoard epoch average
        if writer is not None:
            writer.add_scalar(f"{model_name}/train_loss", avg_loss, epoch)

        # Evaluate on test set for overfitting study
        if test_loader is not None and len(test_loader) > 0:
            model.eval()
            test_loss = 0.0
            test_steps = 0
            with torch.no_grad():
                for batch_tokens in test_loader:
                    batch_tokens = batch_tokens.to(device)
                    logits = model(batch_tokens)
                    loss = compute_next_token_loss(logits, batch_tokens)
                    test_loss += loss.item()
                    test_steps += 1

            avg_test_loss = test_loss / test_steps if test_steps > 0 else 0.0
            gap = avg_test_loss - avg_loss  # Overfitting gap
            print(f"[{model_name}] *** Test Loss: {avg_test_loss:.4f}, Gap (test-train): {gap:.4f}")

            if writer is not None:
                writer.add_scalar(f"{model_name}/test_loss", avg_test_loss, epoch)
                writer.add_scalar(f"{model_name}/overfitting_gap", gap, epoch)

            model.train()  # Switch back to training mode

################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = 16
    num_epochs = 5
    learning_rate = 1e-3

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    # Train/Test Split for overfitting study (80/20 split)
    dataset_size = len(combined_dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # reproducible split
    )
    print(f"Train/Test split: {train_size} train, {test_size} test sequences")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    transformer_pre_norm = TransformerModel(
        vocab_size=vocab_size,
        d_model=embed_size,
        n_heads=4,
        n_blocks=4,
        head_dim=128,
        block_size=block_size,
        normalization='pre'
    ).to(device)

    transformer_post_norm = TransformerModel(
        vocab_size=vocab_size,
        d_model=embed_size,
        n_heads=4,
        n_blocks=4,
        head_dim=128,
        block_size=block_size,
        normalization='post'
    ).to(device)

    transformer_sandwich_norm = TransformerModel(
        vocab_size=vocab_size,
        d_model=embed_size,
        n_heads=4,
        n_blocks=4,
        head_dim=128,
        block_size=block_size,
        normalization='sandwich'
    ).to(device)

    models = {
        # "kgram_mlp_seq": kgram_model,
        # "lstm_seq": lstm_model,
        # "kvcache_transformer": transformer,
        "pre_norm": transformer_pre_norm,
        "post_norm": transformer_post_norm,
        "sandwich_norm": transformer_sandwich_norm
    }


    ############################################################################
    # Train each model
    ############################################################################
    # Create TensorBoard writer with custom run name
    writer = SummaryWriter(f'./runs/{args.run_name}')

    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,  # <--- Pass the user-specified prompt here
            writer=writer,  # <--- Pass TensorBoard writer
            test_loader=test_loader  # <--- Pass test loader for overfitting study
        )

        ckpt_path = f"{model_name}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[{model_name}] Saved checkpoint to {ckpt_path}")
        # =========================================================================================
        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            p_values = [None, 0.80, 0.85, 0.90, 0.95, 1.00]
            for p in p_values:
                p_label = "greedy" if p is None else f"top-p={p:.2f}"
                text, ann = generate_text(
                    model, enc, args.prompt, max_new_tokens=20, device=device,
                    top_p=p,
                )
                print(f"[{model_name}] Final sample ({p_label}) from prompt: '{args.prompt}'")
                print(text)
                print(f"Annotated:\n{ann}\n")
        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    loss_dict = {
        "pre_norm": [],
        "post_norm": [],
        "sandwich_norm": []
    }
    grad_dict = {
        "pre_norm": [],
        "post_norm": [],
        "sandwich_norm": []
    }

    main()

    torch.save(loss_dict, "loss_dict.pt")
    torch.save(grad_dict, "grad_dict.pt")
