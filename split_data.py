import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import os

def run_split(base_csv_path='data/full_dataset.csv', data_dir='data/'):
    """
    Divide o dataset de forma AGRUPADA (por paciente) e ESTRATIFICADA (pelo label 'chagas').
    Proporção: 80% treino, 10% validação, 10% teste.
    """
    train_path = os.path.join(data_dir, 'train_split.csv')
    val_path = os.path.join(data_dir, 'val_split.csv')
    test_path = os.path.join(data_dir, 'test_split.csv')

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("Arquivos de split (agrupado e estratificado) já existem. Pulando a etapa de divisão.")
        return

    print("Criando novos splits de dados (Agrupado por Paciente e Estratificado por Chagas)...")
    if not os.path.exists(base_csv_path):
        print(f"Erro: O arquivo base '{base_csv_path}' não foi encontrado. Execute o preprocess.py primeiro.")
        return
        
    # Carregamento padrão sem otimização de tipo de dados
    df = pd.read_csv(base_csv_path)
    
    groups = df['id_patient']
    y = df['chagas']

    # Primeiro Split: 80% para Treino vs. 20% para (Val + Teste)
    sgkf_train_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, temp_indices = next(sgkf_train_test.split(df, y, groups))
    df_train = df.iloc[train_indices]
    df_temp = df.iloc[temp_indices]
    
    # Segundo Split: Dividir o df_temp (20% do total) ao meio (10% val, 10% test)
    temp_groups = df_temp['id_patient']
    temp_y = df_temp['chagas']
    
    sgkf_val_test = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    val_indices_rel, test_indices_rel = next(sgkf_val_test.split(df_temp, temp_y, temp_groups))
    
    df_val = df_temp.iloc[val_indices_rel]
    df_test = df_temp.iloc[test_indices_rel]

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"Dados divididos e salvos: {len(df_train)} treino, {len(df_val)} validação, {len(df_test)} teste.")
    return

if __name__ == '__main__':
    run_split()