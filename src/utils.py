import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from torch.utils.data.dataloader import default_collate
def collate_fn_skip_corrupted(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return torch.tensor([]), torch.tensor([])
    return default_collate(batch)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Salvando checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    print("=> Carregando checkpoint")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def calculate_gmean(y_true, y_pred_binary):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sensitivity * specificity)

def get_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred_binary = (np.array(y_pred_proba) > threshold).astype(int)
    y_true_array = np.array(y_true)
    metrics = {
        'accuracy': accuracy_score(y_true_array, y_pred_binary),
        'precision': precision_score(y_true_array, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true_array, y_pred_binary, zero_division=0),
        'f1_score': f1_score(y_true_array, y_pred_binary, zero_division=0),
        'roc_auc': roc_auc_score(y_true_array, y_pred_proba),
        'g_mean': calculate_gmean(y_true_array, y_pred_binary)
    }
    return metrics, classification_report(y_true_array, y_pred_binary, zero_division=0)

def find_best_threshold(model, val_loader, device):
    print("Otimizando limiar de classificação no conjunto de validação...")
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc="Otimizando Limiar"):
            if data.nelement() == 0: continue
            data, targets = data.to(device), targets.to(device)
            scores = model(data)
            preds = torch.sigmoid(scores)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy().flatten())
    
    thresholds = np.arange(0.01, 1.0, 0.01)
    gmeans = [calculate_gmean(all_targets, (np.array(all_preds) > t).astype(int)) for t in thresholds]
    
    best_t_idx = np.argmax(gmeans)
    best_threshold = thresholds[best_t_idx]
    print(f"Melhor limiar encontrado: {best_threshold:.2f} (G-Mean: {gmeans[best_t_idx]:.4f})")
    return best_threshold

def plot_learning_curve(history, path):
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Curva de Aprendizado')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred_proba, threshold, path):
    plt.figure()
    y_pred_binary = (np.array(y_pred_proba) > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.savefig(path)
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, path):
    plt.figure()
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('Curva ROC')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.legend()
    plt.savefig(path)
    plt.close()