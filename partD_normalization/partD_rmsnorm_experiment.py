import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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


class NoNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x


class SimpleBlock(nn.Module):
    def __init__(self, dim, hidden_dim, norm_type="rms"):
        super().__init__()
        if norm_type == "rms":
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        elif norm_type == "ln":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        elif norm_type == "nonorm":
            self.norm1 = NoNorm(dim)
            self.norm2 = NoNorm(dim)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = x + self.fc2(F.gelu(self.fc1(self.norm1(x))))
        x = self.norm2(x)
        return x


class SimpleModel(nn.Module):
    def __init__(self, dim, hidden_dim, depth, num_classes, norm_type):
        super().__init__()
        self.input_proj = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList(
            [SimpleBlock(dim, hidden_dim, norm_type=norm_type) for _ in range(depth)]
        )
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(x)
        return logits


def make_dummy_data(n_samples, dim, num_classes, device):
    x = torch.randn(n_samples, dim, device=device)
    w_true = torch.randn(dim, num_classes, device=device)
    logits = x @ w_true
    y = torch.argmax(logits, dim=-1)
    return x, y


def train_one_model(norm_type, device="cpu", steps=80, batch_size=64, dim=256, hidden_dim=1024, depth=12, num_classes=10, lr=5e-3):
    torch.manual_seed(0)
    model = SimpleModel(dim, hidden_dim, depth, num_classes, norm_type=norm_type).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    x_all, y_all = make_dummy_data(8192, dim, num_classes, device)
    n = x_all.size(0)
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        x = x_all[idx]
        y = y_all[idx]
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps = 80
    losses_rms = train_one_model("rms", device=device, steps=steps)
    losses_ln = train_one_model("ln", device=device, steps=steps)
    losses_nonorm = train_one_model("nonorm", device=device, steps=steps)
    x_axis = list(range(1, steps + 1))
    plt.figure()
    plt.plot(x_axis, losses_rms, label="RMSNorm")
    plt.plot(x_axis, losses_ln, label="LayerNorm")
    plt.plot(x_axis, losses_nonorm, label="NoNorm")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("RMSNorm vs LayerNorm vs NoNorm (dummy data)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("partD_rmsnorm_vs_layernorm_loss.png")


if __name__ == "__main__":
    main()
