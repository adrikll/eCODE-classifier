import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import argparse
import random
import os
import json
from tqdm import tqdm
from torch.amp import GradScaler, autocast

from .dataloader import ECGDataset
from .utils import get_metrics
from models.cnn import ECG_CNN
from models.transformer import ECG_Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torch.utils.data.dataloader import default_collate
def collate_fn_skip_corrupted(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return torch.tensor([]), torch.tensor([])
    return default_collate(batch)

SEARCH_SPACE = {
    'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
    'batch_size': [16, 32], 
    'optimizer': ['AdamW'],
    'weight_decay': [0.0, 0.01]
}
EPOCHS_PER_FOLD = 3 

def run_training_fold(train_loader, val_loader, params, model_name, device, pos_weight):
    if model_name == 'cnn': model = ECG_CNN().to(device)
    else: model = ECG_Transformer().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    best_gmean = 0.0

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        for data, targets in train_loader:
            if data.nelement() == 0: continue
            data, targets = data.to(device), targets.to(device).unsqueeze(1)
            
            with autocast(device_type=device):
                scores = model(data)
                loss = criterion(scores, targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        all_targets, all_preds = [], []
        with torch.no_grad():
            for data, targets in val_loader:
                if data.nelement() == 0: continue
                data, targets = data.to(device), targets.to(device)
                # --- MUDANÇA 3: autocast com o tipo de dispositivo ---
                with autocast(device_type=device):
                    scores = model(data)
                preds = torch.sigmoid(scores)
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy().flatten())

        if all_targets:
            metrics, _ = get_metrics(all_targets, all_preds)
            current_gmean = metrics['g_mean']
            if current_gmean > best_gmean:
                best_gmean = current_gmean
    return best_gmean

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    train_df_path = os.path.join(args.data_dir, 'train_split.csv')
    if not os.path.exists(train_df_path):
        print(f"Arquivo '{train_df_path}' não encontrado. Execute o split_data.py primeiro.")
        return
        
    full_train_df = pd.read_csv(train_df_path)
    labels = full_train_df['chagas'].values
    
    print("Calculando o peso para a classe positiva (pos_weight)...")
    neg_count = full_train_df['chagas'].value_counts()[False]
    pos_count = full_train_df['chagas'].value_counts()[True]
    pos_weight = torch.tensor([neg_count / pos_count], device=device)
    print(f"Peso calculado: {pos_weight.item():.2f}")
    
    full_dataset = ECGDataset(
        csv_file=train_df_path,
        data_root_dir=os.path.join(args.data_dir, 'eCODE')
    )

    #CHECKPOINT E RESUMO
    checkpoint_path = os.path.join("outputs", "optimization_checkpoint.json")
    start_trial = 0
    results = []
    best_gmean = -1
    best_params = None

    if os.path.exists(checkpoint_path):
        print(f"Encontrado arquivo de checkpoint: '{checkpoint_path}'. Carregando progresso...")
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        start_trial = checkpoint_data['next_trial']
        results = checkpoint_data['results']
        best_gmean = checkpoint_data['best_gmean']
        best_params = checkpoint_data['best_params']
        print(f"Resumindo a partir da tentativa {start_trial + 1}/{args.n_trials}")

    print(f"--- Iniciando Random Search ---")

    for i in range(start_trial, args.n_trials):
        params = {key: random.choice(value) for key, value in SEARCH_SPACE.items()}
        print(f"\n[Tentativa {i+1}/{args.n_trials}] Testando: {params}")

        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        fold_gmeans = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, labels)):
            print(f"  -> Fold {fold+1}/{args.n_folds}")
            
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn_skip_corrupted)
            val_loader = DataLoader(val_subset, batch_size=params['batch_size'], num_workers=4, pin_memory=True, collate_fn=collate_fn_skip_corrupted)

            gmean_fold = run_training_fold(train_loader, val_loader, params, args.model, device, pos_weight)
            
            fold_gmeans.append(gmean_fold)
            print(f"     G-Mean do Fold: {gmean_fold:.4f}")
        
        avg_gmean = np.mean(fold_gmeans)
        print(f"  -> Média G-Mean da Tentativa: {avg_gmean:.4f}")
        results.append({'params': str(params), 'g_mean': avg_gmean}) # Converte params para string para JSON

        if avg_gmean > best_gmean:
            best_gmean = avg_gmean
            best_params = params
            print(f"  *** Novo melhor resultado encontrado! ***")

        #salva o progresso após cada tentativa concluída
        checkpoint_data = {
            'next_trial': i + 1,
            'results': results,
            'best_gmean': best_gmean,
            'best_params': best_params
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=4)
        print(f"Progresso salvo. Próxima execução começará da tentativa {i + 2}.")


    print("\n--- Otimização Concluída ---")
    if best_params:
        print(f"Melhor Média Geométrica (G-Mean) encontrada: {best_gmean:.4f}")
        print("Melhores Hiperparâmetros:")
        print(best_params)

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        params_path = os.path.join(output_dir, "best_params.json")
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"\nMelhores parâmetros salvos em '{params_path}'")
    else:
        print("Nenhum resultado válido foi obtido na otimização.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Otimização de Hiperparâmetros para Chagas")
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'transformer'], help='Modelo a ser otimizado')
    parser.add_argument('--n-trials', type=int, default=20, help='Número de combinações de hiperparâmetros a testar')
    parser.add_argument('--n-folds', type=int, default=5, help='Número de folds para a validação cruzada')
    parser.add_argument('--data-dir', type=str, default='data/', help='Diretório dos dados')
    args = parser.parse_args()
    main(args)
