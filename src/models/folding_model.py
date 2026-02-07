import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinFoldingModel(nn.Module):
    def __init__(self, vocab_size=25, d_model=256, nhead=8, num_layers=6):
        super(ProteinFoldingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        # Predict 3D coordinates (x, y, z) for each residue
        self.fc = nn.Linear(d_model, 3)

    def forward(self, x):
        # x: [seq_len, batch_size]
        x = self.embedding(x)
        x = self.encoder(x)
        coords = self.fc(x)
        return coords

def load_folding_model(weights_path=None):
    model = ProteinFoldingModel()
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model
