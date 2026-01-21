import matplotlib.pyplot as plt

# 1. Carregar os dados brutos (Assumindo que o ficheiro é o mesmo)
df = pd.read_excel('tuberculose_stp.xlsx')

# 2. Criar um "Dicionário" para traduzir os meses de Texto para Número
# Ajuste as abreviaturas se no seu Excel estiverem diferentes (ex: 'Mai' ou 'Maio')
meses_map = {
    'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4, 'Mai': 5, 'Jun': 6,
    'Jul': 7, 'Ago': 8, 'Set': 9, 'Out': 10, 'Nov': 11, 'Dez': 12
}

# 3. Criar uma coluna numérica para o mês
df['Mes_Num'] = df['Mês'].map(meses_map)

# 4. Criar uma coluna de Data válida (Ano-Mês-Dia)
# O Pandas precisa de um dia, por isso vamos assumir sempre o dia 1 de cada mês
df['Data_Index'] = pd.to_datetime(
    df['Ano'].astype(str) + '-' + df['Mes_Num'].astype(str) + '-01'
)

# 5. A MÁGICA: Agrupar por data e contar as linhas (cada linha é 1 caso)
df_final = df.groupby('Data_Index').size().reset_index(name='Casos')

# 6. Definir a Data como Índice (Requisito para Séries Temporais)
df_final.set_index('Data_Index', inplace=True)

# 7. Verificar se há meses sem casos (Buracos na série temporal)
# Isto garante que se um mês teve 0 casos, ele aparece como 0 e não desaparece
df_final = df_final.asfreq('MS', fill_value=0) # 'MS' significa Month Start

print("Série Temporal Criada com Sucesso:")
print(df_final.head())

# 8. Plotar o gráfico final
plt.figure(figsize=(12, 6))
plt.plot(df_final.index, df_final['Casos'], marker='o', linestyle='-', color='#007acc')
plt.title('Incidência Mensal de Tuberculose em STP (2011-2023)')
plt.xlabel('Data')
plt.ylabel('Número de Casos')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() # Vai abrir a janela com o gráfico