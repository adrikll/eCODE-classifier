import os
import shutil
import zipfile

# --- Configuração das pastas e prefixo ---
pasta_downloads = os.path.join(os.path.expanduser('~'), 'Downloads')
pasta_destino = os.path.join(pasta_downloads, 'eCODE')
# !!!!! ATENÇÃO: AJUSTE ESTE PREFIXO COM BASE NO NOME REAL DO SEU ARQUIVO !!!!!
prefixo_arquivo = "drive-download-20251002" # <--- TENTE USAR UM PREFIXO MAIS CURTO

# --- Criação da pasta de destino ---
if not os.path.exists(pasta_destino):
    print(f"Criando a pasta de destino: {pasta_destino}")
    os.makedirs(pasta_destino)
else:
    print(f"A pasta de destino '{pasta_destino}' já existe.")

print("\nIniciando a busca e processamento dos arquivos...")

# --- Busca dos arquivos na pasta de downloads ---
try:
    arquivos_na_pasta = os.listdir(pasta_downloads)
except FileNotFoundError:
    print(f"ERRO: A pasta de Downloads não foi encontrada em '{pasta_downloads}'")
    exit()

# --- Processamento dos arquivos ZIP ---
arquivos_encontrados = 0
for nome_arquivo in arquivos_na_pasta:
    # --- LINHA DE DEPURAÇÃO ---
    # A linha abaixo vai imprimir cada arquivo que o script está analisando.
    # print(f"Verificando: {nome_arquivo}") 

    # Verifica se o arquivo corresponde ao padrão (prefixo e extensão .zip)
    if nome_arquivo.startswith(prefixo_arquivo) and nome_arquivo.endswith(".zip"):
        arquivos_encontrados += 1 # Conta os arquivos que correspondem
        caminho_original = os.path.join(pasta_downloads, nome_arquivo)
        caminho_novo_zip = os.path.join(pasta_destino, nome_arquivo)

        try:
            print(f"\n- Processando arquivo: '{nome_arquivo}'")
            shutil.move(caminho_original, caminho_novo_zip)
            print(f"  -> Movido para '{pasta_destino}'")

            print(f"  -> Descompactando '{nome_arquivo}'...")
            with zipfile.ZipFile(caminho_novo_zip, 'r') as zip_ref:
                for info_arquivo in zip_ref.infolist():
                    if info_arquivo.is_dir():
                        continue
                    
                    caminho_extracao = os.path.join(pasta_destino, info_arquivo.filename)
                    
                    contador = 1
                    nome, extensao = os.path.splitext(caminho_extracao)
                    while os.path.exists(caminho_extracao):
                        caminho_extracao = f"{nome} ({contador}){extensao}"
                        contador += 1
                    
                    with open(caminho_extracao, "wb") as f_out:
                        f_out.write(zip_ref.read(info_arquivo.filename))

            print(f"  -> '{nome_arquivo}' descompactado com sucesso!")

        except FileNotFoundError:
            print(f"  -> AVISO: O arquivo '{nome_arquivo}' não foi encontrado para mover. Pode já ter sido movido.")
        except Exception as e:
            print(f"  -> ERRO ao processar o arquivo '{nome_arquivo}': {e}")

# Mensagem final informando se algum arquivo foi processado
if arquivos_encontrados == 0:
    print("\nAVISO: Nenhum arquivo com o prefixo especificado foi encontrado na pasta Downloads.")

print("\nProcesso finalizado com sucesso!")