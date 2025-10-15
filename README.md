# ECODE Classifier: Detector de Doença de Chagas com ECG e Deep Learning

Este projeto implementa um pipeline completo e robusto para treinar e avaliar modelos de Deep Learning para a detecção da Doença de Chagas a partir de sinais de ECG de 8 derivações da base de dados eCODE.

O fluxo de trabalho inclui pré-processamento de dados, otimização de hiperparâmetros, treinamento, otimização de limiar de classificação e avaliação final com geração de relatórios e gráficos.

## Configuração do Ambiente

1.  **Estrutura de Pastas:** Certifique-se de que a pasta `eCODE`, contendo todos os subdiretórios de sinais (`S0000000`, `S0010000`, etc.), esteja localizada dentro da pasta `data/`.

2.  **Ambiente Virtual:** É altamente recomendado criar um ambiente virtual para isolar as dependências do projeto.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instalar Dependências:** Instale todas as bibliotecas necessárias.
    ```bash
    pip install -r requirements.txt
    ```

## Execução

1.  **Pré-processar os Dados (apenas uma vez):**
    Cria o `master_dataset.csv` a partir dos dados brutos.
    ```bash
    python3 preprocess.py
    ```

2.  **Dividir os Dados (apenas uma vez):**
    Cria os arquivos `train_split.csv`, `val_split.csv` e `test_split.csv`.
    ```bash
    python3 split_data.py
    ```

3.  **Otimizar Hiperparâmetros:**
    Encontra os melhores parâmetros e os salva em `outputs/best_params.json`.
    ```bash
    python3 -m src.optimize --model transformer --n-trials 20 --n-folds 5
    ```

4.  **Treinar e Avaliar o Modelo Final:**
    Lê os melhores parâmetros do passo anterior e executa o treinamento final e a avaliação completa.
    ```bash
    python3 -m src.train --model cnn --epochs 20
    ```

5.  **Monitorar o treinamento:**
    ```bash
    tensorboard --logdir=runs
    ```
