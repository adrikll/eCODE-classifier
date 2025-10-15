import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import wfdb

def get_exam_path(exam_id: int) -> str:
    """
    Gera o caminho relativo para um arquivo de exame (_N1) baseado no seu ID.
    """
    folder_prefix = exam_id // 10000
    folder_name = f"S{folder_prefix:03d}0000"
    file_name = f"TNMG{exam_id}_N1"
    return os.path.join(folder_name, file_name)

def pre_validate_signals(df, data_root_dir):
    """
    Verifica cada arquivo de sinal para garantir que AMBOS .hea e .dat existem
    e que o cabeçalho reporta 8 derivações.
    Retorna uma lista de exam_ids que são válidos.
    """
    print("\nIniciando Pré-Validação dos Sinais (verificando arquivos .hea e .dat)...")
    valid_exam_ids = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Validando Sinais"):
        exam_id = row['exam_id']
        relative_path = row['file_path']
        # Constrói o caminho base, sem extensão
        full_path_base = os.path.join(data_root_dir, relative_path)
        
        # --- LÓGICA DE VALIDAÇÃO CORRIGIDA ---
        path_hea = full_path_base + ".hea"
        path_dat = full_path_base + ".dat"
        
        # 1. Verifica se AMBOS os arquivos existem primeiro
        if os.path.exists(path_hea) and os.path.exists(path_dat):
            try:
                # 2. Se existem, tenta ler o cabeçalho para checar as derivações
                header = wfdb.rdheader(full_path_base)
                if header.n_sig == 8:
                    valid_exam_ids.append(exam_id)
            except Exception:
                # Ignora se o arquivo .hea estiver corrompido
                continue
    
    return valid_exam_ids

# --- Início do Script Principal ---
# O resto do script permanece o mesmo.
path_annotations = 'data/annotations.csv'
path_chagas = 'data/chagas_code.csv'
path_comorbities = 'data/comorbities.csv'
path_samitrop = 'data/samitrop_patients.txt'
DATA_ROOT_DIR = 'data/eCODE'

print("Iniciando o pré-processamento...")

print("Carregando arquivos de metadados...")
df_annotations = pd.read_csv(path_annotations)
df_chagas = pd.read_csv(path_chagas)
df_comorbities = pd.read_csv(path_comorbities)

df_annotations.rename(columns={'id_exam': 'exam_id'}, inplace=True)
df_comorbities.rename(columns={'id_exam': 'exam_id'}, inplace=True)
df_chagas.rename(columns={'patient_id': 'id_patient'}, inplace=True)

print("Unificando dados tabulares...")
df_master = pd.merge(df_annotations, df_comorbities, on=['exam_id', 'id_patient'], how='left')
df_master = pd.merge(df_master, df_chagas, on=['exam_id', 'id_patient'], how='left')

print("Consolidando labels de Chagas de ambas as fontes...")
df_master['chagas_final'] = df_master['chagas'].combine_first(df_master['doencadechagas'])
df_master.drop(columns=['chagas', 'doencadechagas'], inplace=True)
df_master.dropna(subset=['chagas_final'], inplace=True)
df_master.rename(columns={'chagas_final': 'chagas'}, inplace=True)

df_master['exam_id'] = df_master['exam_id'].astype(int)
df_master['id_patient'] = df_master['id_patient'].astype(int)
df_master['chagas'] = df_master['chagas'].astype(bool)

positivos_antes = df_master['chagas'].sum()
df_master['chagas'] = df_master.groupby('id_patient')['chagas'].transform('any')
positivos_depois = df_master['chagas'].sum()
print(f"Rótulos positivos alterados: {positivos_depois - positivos_antes} exames foram atualizados.\n")

print("Gerando caminhos de arquivo dinamicamente a partir dos IDs dos exames...")
df_master['file_path'] = df_master['exam_id'].apply(get_exam_path)

valid_ids = pre_validate_signals(df_master, DATA_ROOT_DIR)
num_sinais_validados = len(valid_ids)
num_total_sinais_mapeados = len(df_master)

print(f"Validação concluída: {num_sinais_validados} de {num_total_sinais_mapeados} sinais mapeados são válidos (8 derivações, .hea + .dat).")

df_final = df_master[df_master['exam_id'].isin(valid_ids)].copy()

if os.path.exists(path_samitrop):
    print("Removendo pacientes do dataset SaMi-Trop...")
    samitrop_patients = np.loadtxt(path_samitrop, dtype=int)
    df_final = df_final[~df_final['id_patient'].isin(samitrop_patients)]

output_filename = 'master_dataset.csv'
print(f"Salvando o dataset final pré-processado e VALIDADO em '{output_filename}'...")
df_final.to_csv(output_filename, index=False)

print("\n------------------------------------------------------------")
print("---               Pré-processamento Concluído              ---")
print("------------------------------------------------------------")
print(f"Total de exames com metadados e rótulo:      {num_total_sinais_mapeados}")
print(f"Total de sinais validados (8 derivações):      {num_sinais_validados}")
print(f"Total de registros no master_dataset.csv final:  {len(df_final)}")
print("------------------------------------------------------------\n")