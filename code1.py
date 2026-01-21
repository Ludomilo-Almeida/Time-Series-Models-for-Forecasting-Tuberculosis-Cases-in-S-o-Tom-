
import pandas as pd

# Ler o Excel SEM tentar definir datas ou índices ainda
df_temp = pd.read_excel('tuberculose_stp.xlsx')

# Imprimir o nome de todas as colunas que o Python encontrou
print("As colunas encontradas foram:")
print(df_temp.columns.tolist())

# Imprimir as primeiras 5 linhas para ver o aspeto dos dados
print("\nPrimeiras linhas dos dados:")
print(df_temp.head())


import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Carregar os dados ---
# Como o seu ficheiro parece ser CSV separado por vírgulas
try:
    df = pd.read_csv('tuberculose_stp.csv')
except:
    # Caso seja Excel, tenta ler como Excel
    df = pd.read_excel('tuberculose_stp.xlsx')

print("Dados carregados. Linhas totais:", len(df))

# --- 2. Limpeza da coluna "Mês" ---
# Vamos converter tudo para minúsculas e tirar espaços vazios para evitar erros
df['Mês'] = df['Mês'].astype(str).str.lower().str.strip()

# Criar um mapa ROBUSTO baseado no que vi no seu ficheiro
mapa_meses = {
    'jan': 1, 'janeiro': 1,
    'fev': 2, 'fevereiro': 2,
    'mar': 3, 'março': 3,
    'abr': 4, 'abril': 4,
    'mai': 5, 'maio': 5,
    'jun': 6, 'junho': 6,
    'jul': 7, 'julho': 7,
    'ago': 8, 'agosto': 8, 'agost': 8, # 'agost' apareceu no seu ficheiro
    'set': 9, 'setembro': 9,
    'out': 10, 'outubro': 10,
    'nov': 11, 'novembro': 11,
    'dez': 12, 'dezembro': 12
}

# Aplicar o mapa
df['Mes_Num'] = df['Mês'].map(mapa_meses)

# Verificar se sobrou algum mês sem número (NaN)
erros = df[df['Mes_Num'].isna()]
if not erros.empty:
    print("Atenção: Algumas linhas foram ignoradas por erro no mês:")
    print(erros['Mês'].unique())
    df = df.dropna(subset=['Mes_Num']) # Remove linhas com erro

# Garantir que o mês é inteiro
df['Mes_Num'] = df['Mes_Num'].astype(int)

# --- 3. Criar a Data ---
# Dia 1 de cada mês
df['Data_Index'] = pd.to_datetime(
    df['Ano'].astype(str) + '-' + df['Mes_Num'].astype(str) + '-01'
)

# --- 4. Agregação (Transformar lista de pacientes em Série Temporal) ---
# Conta quantas linhas existem por cada data
df_final = df.groupby('Data_Index').size().reset_index(name='Casos')

# Definir índice temporal
df_final.set_index('Data_Index', inplace=True)

# Preencher buracos: Se houver meses sem casos, coloca 0
df_final = df_final.asfreq('MS', fill_value=0)

# --- 5. Resultados e Guardar CSV Limpo ---
print("\nSérie Temporal Gerada com Sucesso!")
print(df_final.head())
print(f"Total de casos contabilizados: {df_final['Casos'].sum()}")

# Guardar este ficheiro limpo para usar depois (opcional)
df_final.to_csv('tuberculose_stp_processado.csv')

# --- 6. Gráfico Profissional ---
plt.figure(figsize=(8, 4))
plt.plot(df_final.index, df_final['Casos'], marker='o', markersize=4, linestyle='-', color='#2c3e50', linewidth=1.5)

# Títulos
plt.title('Evolution (2011–2023)', fontsize=14, fontweight='bold')
plt.ylabel('Reported Cases', fontsize=12)
plt.xlabel('Year', fontsize=12)

# Adicionar uma média móvel (linha vermelha) para suavizar e ver melhor a tendência
plt.plot(df_final.index, df_final['Casos'].rolling(window=12).mean(), color='red', linewidth=2, label='Média Móvel (12 meses)')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Salvar gráfico
plt.savefig('grafico_tuberculose_final.png')
plt.show()