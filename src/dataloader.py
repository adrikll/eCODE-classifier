import torch
import pandas as pd
import numpy as np
import wfdb
from torch.utils.data import Dataset
import os

class ECGDataset(Dataset):
    """
    Dataset customizado para carregar sinais de ECG sob demanda.
    Otimizado para não carregar todos os dados na RAM e pular amostras corrompidas.
    """
    def __init__(self, csv_file, data_root_dir, signal_len=4096):
        self.df = pd.read_csv(csv_file)
        self.data_root_dir = data_root_dir
        self.signal_len = signal_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal_relative_path = row['file_path']
        signal_path = os.path.join(self.data_root_dir, signal_relative_path)
        
        try:
            record = wfdb.rdrecord(signal_path)
            signal = record.p_signal

            # --- CORREÇÃO PRINCIPAL ESTÁ AQUI ---
            # Garante que o sinal tem 8 derivações (canais)
            if signal.shape[1] != 8:
                # A mensagem de aviso também foi corrigida para refletir a expectativa de 8.
                print(f"AVISO: Sinal {signal_path} tem {signal.shape[1]} derivações. Esperado 8. Pulando.")
                return None
            # --- FIM DA CORREÇÃO ---

        except Exception as e:
            # Se houver qualquer erro de leitura (arquivo não encontrado, etc.), retorna None.
            print(f"AVISO: Erro ao carregar {signal_path}: {e}. Pulando amostra.")
            return None

        # Processamento do sinal (se ele for válido)
        signal = np.nan_to_num(signal)
        signal = torch.from_numpy(signal).float().T # Transpõe para (8, length)

        if signal.shape[1] > self.signal_len:
            # Trunca se for maior
            signal = signal[:, :self.signal_len]
        elif signal.shape[1] < self.signal_len:
            # Adiciona padding de zeros se for menor
            padding = torch.zeros(signal.shape[0], self.signal_len - signal.shape[1])
            signal = torch.cat([signal, padding], dim=1)

        # Pega o label (Chagas) e converte para tensor
        label = torch.tensor(float(row['chagas']))

        return signal, label