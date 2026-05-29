import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MLPSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        self.features = torch.tensor(
            np.stack([s["features"] for s in sequences]),
            dtype=torch.float32,
        )
        self.labels = torch.tensor(
            np.array([s["label"] for s in sequences], dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MLPStockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class StockSequenceDataset(Dataset):
    def __init__(self, sequences, code_to_id):
        self.sequences = sequences
        self.features = torch.tensor(
            np.stack([s["features"] for s in sequences]),
            dtype=torch.float32,
        )
        self.labels = torch.tensor(
            np.array([s["label"] for s in sequences], dtype=np.float32),
            dtype=torch.float32,
        )
        self.stock_ids = torch.tensor(
            [code_to_id.get(s["ts_code"], 0) for s in sequences],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.stock_ids[idx], idx


class StockGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout, num_stocks, embed_dim):
        super().__init__()
        self.stock_embedding = nn.Embedding(num_stocks, embed_dim)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.gru_norm = nn.LayerNorm(hidden_size * 2)
        self.attn1 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.attn2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4 + embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "embedding" in name:
                nn.init.normal_(param, mean=0, std=0.01)
            elif "gru" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
            elif "norm" in name:
                continue
            elif "weight" in name and param.ndimension() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x, stock_ids):
        x_proj = self.input_proj(x)
        x_proj = self.input_norm(x_proj)
        x_proj = torch.nn.functional.gelu(x_proj)
        gru_out, _ = self.gru(x_proj)
        gru_out = self.gru_norm(gru_out)

        attn1_w = torch.softmax(self.attn1(gru_out), dim=1)
        ctx1 = torch.sum(attn1_w * gru_out, dim=1)
        attn2_w = torch.softmax(self.attn2(gru_out), dim=1)
        ctx2 = torch.sum(attn2_w * gru_out, dim=1)

        multi_ctx = torch.cat([ctx1, ctx2], dim=1)
        stock_emb = self.stock_embedding(stock_ids)
        combined = torch.cat([multi_ctx, stock_emb], dim=1)
        output = self.fc(combined)
        return output.squeeze(-1)

