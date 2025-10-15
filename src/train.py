import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import default_collate
from torch.amp import GradScaler, autocast
import argparse
from tqdm import tqdm
import os
import numpy as np
import json
import pandas as pd
from datetime import datetime

from .dataloader import ECGDataset
from .utils import (save_checkpoint, load_checkpoint, get_metrics,
                    find_best_threshold, plot_learning_curve,
                    plot_confusion_matrix, plot_roc_curve)

# Importações de modelo a partir da pasta raiz
from models.cnn import ECG_CNN
from models.transformer import ECG_Transformer


def collate_fn_skip_corrupted(batch):
    """
    Função collate que filtra amostras corrompidas (que retornaram None do Dataset).
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return default_collate(batch)


def main(args):
    # --- 1. CARREGAR PARÂMETROS E CONFIGURAR O EXPERIMENTO ---
    params_path = os.path.join("outputs", "best_params.json")
    if not os.path.exists(params_path):
        print(f"Erro: Arquivo de parâmetros '{params_path}' não foi encontrado.")
        print("Por favor, execute o script 'optimize.py' primeiro.")
        return

    with open(params_path, 'r') as f:
        params = json.load(f)
    print("Parâmetros de otimização carregados:")
    print(json.dumps(params, indent=4))

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"{args.model}_final_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Resultados deste experimento serão salvos em: '{output_dir}'")
    print(f"Usando dispositivo: {DEVICE}")

    # --- 2. PREPARAÇÃO DOS DADOS ---
    data_root_dir = os.path.join(args.data_dir, 'eCODE')
    train_csv_path = os.path.join(args.data_dir, 'train_split.csv')

    print("Calculando o peso para a classe positiva (pos_weight)...")
    train_df = pd.read_csv(train_csv_path)
    if True in train_df['chagas'].value_counts() and train_df['chagas'].value_counts()[True] > 0:
        neg_count = train_df['chagas'].value_counts()[False]
        pos_count = train_df['chagas'].value_counts()[True]
        pos_weight = torch.tensor([neg_count / pos_count], device=DEVICE)
    else:
        pos_weight = torch.tensor([1.0], device=DEVICE)
    print(f"Peso calculado: {pos_weight.item():.2f}")

    train_dataset = ECGDataset(csv_file=train_csv_path, data_root_dir=data_root_dir)
    val_dataset = ECGDataset(csv_file=os.path.join(args.data_dir, 'val_split.csv'), data_root_dir=data_root_dir)
    test_dataset = ECGDataset(csv_file=os.path.join(args.data_dir, 'test_split.csv'), data_root_dir=data_root_dir)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn_skip_corrupted)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_skip_corrupted)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_skip_corrupted)

    # --- 3. INICIALIZAÇÃO DO MODELO, OTIMIZADOR E LOSS ---
    if args.model == 'cnn': model = ECG_CNN().to(DEVICE)
    else: model = ECG_Transformer().to(DEVICE)
    
    if params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9, weight_decay=params['weight_decay'])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    writer = SummaryWriter(os.path.join("runs", f"chagas_{args.model}_final_{timestamp}"))
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # --- 4. LÓGICA DE CHECKPOINT E RESUMO ---
    start_epoch = 0
    best_gmean = 0.0
    history = {'train_loss': [], 'val_loss': []}
    checkpoint_path = os.path.join(output_dir, "latest_checkpoint.pth.tar")

    if args.resume:
        if os.path.exists(checkpoint_path):
            print(f"Resumindo treinamento a partir de '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_gmean = checkpoint.get('best_gmean', 0.0)
            history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        else:
            print(f"Aviso: --resume foi especificado, mas o checkpoint não foi encontrado. Iniciando do zero.")

    # --- 5. LOOP DE TREINAMENTO E VALIDAÇÃO ---
    for epoch in range(start_epoch, args.epochs):
        # --- Loop de Treino ---
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Treino")
        optimizer.zero_grad()
        
        for batch_idx, (data, targets) in enumerate(loop):
            if data.nelement() == 0: continue
            data = data.to(DEVICE)
            targets = targets.to(DEVICE).unsqueeze(1)

            with autocast(device_type=DEVICE):
                scores = model(data)
                loss = criterion(scores, targets)
                if args.accumulation_steps > 1:
                    loss = loss / args.accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loop.set_postfix(loss=loss.item() * args.accumulation_steps)
        
        # --- Loop de Validação (executado ao final de cada época) ---
        model.eval()
        epoch_val_loss = 0.0
        all_targets, all_preds = [], []
        with torch.no_grad():
            for data, targets in val_loader:
                if data.nelement() == 0: continue
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                with autocast(device_type=DEVICE):
                    scores = model(data)
                epoch_val_loss += criterion(scores, targets.unsqueeze(1)).item()
                preds = torch.sigmoid(scores)
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy().flatten())
        
        history['train_loss'].append(np.nan) # Placeholder, a loss de treino é por passo
        history['val_loss'].append(epoch_val_loss / len(val_loader))
        
        if all_targets:
            metrics, _ = get_metrics(all_targets, all_preds, threshold=0.5)
            current_gmean = metrics['g_mean']
            print(f"Val G-Mean (limiar 0.5) - Epoch {epoch+1}: {current_gmean:.4f}")
            writer.add_scalar("G-Mean/val", current_gmean, epoch)

            if current_gmean > best_gmean:
                best_gmean = current_gmean
                best_model_path = os.path.join(output_dir, "best_model.pth.tar")
                # CORREÇÃO: Passa o estado completo do otimizador
                save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, filename=best_model_path)
        
        # Salva o checkpoint mais recente a cada época
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_gmean': best_gmean,
            'history': history
        }, filename=checkpoint_path)
        print(f"Checkpoint da época {epoch+1} salvo em '{checkpoint_path}'")

    writer.close()
    
    # --- 6. OTIMIZAÇÃO DO LIMIAR ---
    print("\n--- Carregando o melhor modelo para otimização de limiar ---")
    best_model_path = os.path.join(output_dir, "best_model.pth.tar")
    if not os.path.exists(best_model_path):
        print("ERRO: Nenhum checkpoint do melhor modelo foi salvo.")
        return
        
    # Re-inicializa o modelo para carregar os pesos
    if args.model == 'cnn': best_model = ECG_CNN().to(DEVICE)
    else: best_model = ECG_Transformer().to(DEVICE)
    best_model, _ = load_checkpoint(best_model_path, best_model, optimizer)
    best_threshold = find_best_threshold(best_model, val_loader, DEVICE)

    # --- 7. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ---
    print("\n--- Iniciando Avaliação Final no Conjunto de Teste ---")
    best_model.eval()
    test_targets, test_preds_proba = [], []
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Avaliando no Teste"):
            if data.nelement() == 0: continue
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            with autocast(device_type=DEVICE):
                scores = best_model(data)
            preds = torch.sigmoid(scores)
            test_targets.extend(targets.cpu().numpy())
            test_preds_proba.extend(preds.cpu().numpy().flatten())

    final_metrics, report_str = get_metrics(test_targets, test_preds_proba, threshold=best_threshold)
    print("\nMétricas Finais no Conjunto de Teste (com limiar otimizado):")
    print(json.dumps(final_metrics, indent=4))

    # --- 8. SALVAR TODOS OS ARTEFATOS DO EXPERIMENTO ---
    print("\nSalvando relatórios e gráficos...")
    with open(os.path.join(output_dir, "classification_report.txt"), 'w') as f:
        f.write(f"Limiar Otimizado: {best_threshold:.4f}\n\n")
        f.write(report_str)

    results_summary = {
        'experiment_parameters': vars(args),
        'best_hyperparameters': params,
        'best_threshold': best_threshold,
        'test_set_metrics': final_metrics
    }
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    plot_roc_curve(test_targets, test_preds_proba, os.path.join(output_dir, "roc_curve.png"))
    plot_confusion_matrix(test_targets, test_preds_proba, best_threshold, os.path.join(output_dir, "confusion_matrix.png"))

    print(f"\nTreinamento e avaliação concluídos! Todos os resultados foram salvos em '{output_dir}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treinamento Final com Parâmetros Otimizados")
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'transformer'], help='Modelo a ser treinado')
    parser.add_argument('--epochs', type=int, default=20, help='Número de épocas para o treino final')
    parser.add_argument('--data-dir', type=str, default='data/', help='Diretório dos dados')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='Número de passos para acumulação de gradiente')
    parser.add_argument('--resume', action='store_true', help='Resume o treinamento a partir do último checkpoint')
    args = parser.parse_args()
    main(args)