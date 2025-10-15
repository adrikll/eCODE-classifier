import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ECG_Transformer(nn.Module):
    
    def __init__(self, num_classes=1, num_leads=8, d_model=96, nhead=6, num_layers=3, dropout=0.1):
        super(ECG_Transformer, self).__init__()
        
        # Downsampling: reduz o comprimento da sequência pela metade (4096 -> 2048)
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.input_projection = nn.Conv1d(num_leads, d_model, kernel_size=1)
        # Ajusta o max_len para o novo tamanho da sequência
        self.pos_encoder = PositionalEncoding(d_model, max_len=2500) 
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x):
        # x shape: (batch, leads, length=4096)
        
        #aplica o downsampling
        x = self.downsample(x) # -> (batch, leads, length=2048)

        x = self.input_projection(x) # -> (batch, d_model, length=2048)
        x = x.permute(2, 0, 1) # -> (length=2048, batch, d_model) para o pos_encoder
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2) # -> (batch, length=2048, d_model) para o transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) 
        x = self.fc(x)
        return x