import os
import shutil
import tarfile

pasta_origem_nome = "CODE"
pasta_destino_nome = "eCODE"

diretorio_atual = os.getcwd()

pasta_origem = os.path.join(diretorio_atual, pasta_origem_nome)
pasta_destino = os.path.join(diretorio_atual, pasta_destino_nome)

if not os.path.isdir(pasta_origem):
    print(f"ERRO: A pasta de origem '{pasta_origem_nome}' não foi encontrada.")
    exit()

if not os.path.exists(pasta_destino):
    print(f"Criando a pasta de destino: '{pasta_destino_nome}'")
    os.makedirs(pasta_destino)
else:
    print(f"A pasta de destino '{pasta_destino_nome}' já existe.")

print("\n--- Iniciando o processo de extração e movimentação ---")

for nome_arquivo in os.listdir(pasta_origem):
    
        if nome_arquivo.startswith('S') and nome_arquivo.endswith('.tar.gz'):
        
            caminho_completo_arquivo = os.path.join(pasta_origem, nome_arquivo)
        
        try:
            print(f"\n[PROCESSANDO]: {nome_arquivo}")

            conteudo_antes = set(os.listdir(pasta_origem))
            
            print(f"  -> Extraindo...")
            with tarfile.open(caminho_completo_arquivo, 'r:gz') as tar:
                tar.extractall(path=pasta_origem)
            
            conteudo_depois = set(os.listdir(pasta_origem))
            
            itens_extraidos = conteudo_depois - conteudo_antes
            
            if not itens_extraidos:
                print(f"  -> AVISO: Nenhum arquivo novo encontrado após extrair {nome_arquivo}. O arquivo pode estar vazio.")
                continue

            for item in itens_extraidos:
                caminho_origem_item = os.path.join(pasta_origem, item)
                caminho_destino_item = os.path.join(pasta_destino, item)
                
                print(f"  -> Movendo '{item}' para '{pasta_destino_nome}'...")
                shutil.move(caminho_origem_item, caminho_destino_item)
            
            print(f"[SUCESSO]: {nome_arquivo} processado com êxito.")

        except tarfile.ReadError:
            print(f"  -> ERRO: O arquivo {nome_arquivo} parece estar corrompido ou não é um .tar.gz válido.")
        except Exception as e:
            print(f"  -> ERRO inesperado ao processar {nome_arquivo}: {e}")

print("\nProcesso finalizado!")