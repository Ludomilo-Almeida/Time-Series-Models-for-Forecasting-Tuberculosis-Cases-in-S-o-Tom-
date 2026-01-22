import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# %%
# ==========================================
# CÉLULA 1: CARREGAR E LIMPAR (Corra esta apenas uma vez)
# ==========================================
print("--- A CARREGAR DADOS ---")
try:
    df = pd.read_csv('tuberculose_stp.csv')
except:
    df = pd.read_excel('tuberculose_stp.xlsx')

print(f"Dados carregados. Total de linhas brutas: {len(df)}")

# Normalizar a coluna "Mês"
df['Mês'] = df['Mês'].astype(str).str.lower().str.strip()

mapa_meses = {
    'jan': 1, 'janeiro': 1, 'fev': 2, 'fevereiro': 2, 'mar': 3, 'março': 3,
    'abr': 4, 'abril': 4, 'mai': 5, 'maio': 5, 'jun': 6, 'junho': 6,
    'jul': 7, 'julho': 7, 'ago': 8, 'agosto': 8, 'agost': 8,
    'set': 9, 'setembro': 9, 'out': 10, 'outubro': 10,
    'nov': 11, 'novembro': 11, 'dez': 12, 'dezembro': 12
}

df['Mes_Num'] = df['Mês'].map(mapa_meses)
df = df.dropna(subset=['Mes_Num'])
df['Mes_Num'] = df['Mes_Num'].astype(int)

# Criar Data e Série Temporal
df['Data_Index'] = pd.to_datetime(df['Ano'].astype(str) + '-' + df['Mes_Num'].astype(str) + '-01')
df_final = df.groupby('Data_Index').size().reset_index(name='Casos')
df_final.set_index('Data_Index', inplace=True)
df_final = df_final.asfreq('MS', fill_value=0)

print("Série Temporal pronta! (Célula 1 concluída)")


# %%
# ==========================================
# CÉLULA 2: ESTATÍSTICA (Teste ADF)
# ==========================================
def realizar_teste_adf(serie, nome="Série"):
    print(f"\n--- TESTE DICKEY-FULLER: {nome} ---")
    serie_limpa = serie.dropna()
    resultado = adfuller(serie_limpa)
    p_value = resultado[1]
    
    print(f'Estatística: {resultado[0]:.4f}')
    print(f'Valor-p:     {p_value:.4f}')
    
    if p_value < 0.05:
        print(">> CONCLUSÃO: A série é ESTACIONÁRIA (p < 0.05).")
    else:
        print(">> CONCLUSÃO: A série NÃO é estacionária (p > 0.05).")

# Executar Testes
realizar_teste_adf(df_final['Casos'], nome="Original")
diff_1 = df_final['Casos'].diff()
realizar_teste_adf(diff_1, nome="1ª Diferença (d=1)")


# %%
# ==========================================
# CÉLULA 3: GRÁFICO (Pode correr esta muitas vezes para ajustar visual)
# ==========================================
plt.figure(figsize=(10, 5))

# Apenas os dados reais
plt.plot(df_final.index, df_final['Casos'], 
         marker='o', markersize=4, linestyle='-', 
         color='#2c3e50', linewidth=1.5, label='Casos Observados')

plt.title('Evolution of Tuberculosis Cases (2011–2023)', fontsize=14, fontweight='bold')
plt.ylabel('Reported Cases', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("A guardar gráfico...")
plt.savefig('grafico_tuberculose_final.png')
plt.show()

# %%
# ==========================================
# CÉLULA 4: APLICAR 1ª DIFERENÇA (d=1)
# ==========================================
# 1. Calcular a Diferença (Mês atual - Mês anterior)
df_final['Diff_1'] = df_final['Casos'].diff()

# 2. Testar se a Diferença tornou a série Estacionária
print("\n--- TESTE ADF: 1ª Diferença (d=1) ---")
# O primeiro valor fica NaN na diferença, temos de o ignorar no teste
diff_clean = df_final['Diff_1'].dropna() 
result_diff = adfuller(diff_clean)

print(f'Estatística: {result_diff[0]:.4f}')
print(f'Valor-p:     {result_diff[1]:.4f}')

if result_diff[1] < 0.05:
    print(">> CONCLUSÃO: A série diferenciada é ESTACIONÁRIA. Podemos seguir para ACF/PACF.")
else:
    print(">> CONCLUSÃO: A série ainda não é estacionária.")

# 3. Gráfico da Diferença
plt.figure(figsize=(10, 5))
plt.plot(df_final.index, df_final['Diff_1'], 
         marker='o', markersize=4, linestyle='-', 
         color='#e74c3c', linewidth=1.5, label='1ª Diferença')

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5) # Linha de referência no zero
plt.title('Stationarity Check: 1st Difference', fontsize=14, fontweight='bold')
plt.ylabel('Change in Cases', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# ==========================================
# CÉLULA 5: IDENTIFICAÇÃO DO MODELO (ACF e PACF)
# ==========================================
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Configurar a figura com 2 gráficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 1. Plotar ACF (Autocorrelação) -> Ajuda a escolher o 'q' (Moving Average)
# Usamos o .dropna() porque a diferenciação cria valores vazios no início
plot_acf(df_final['Diff_1'].dropna(), lags=20, ax=ax1, 
         title='Autocorrelation (ACF) - Indica o valor de q (MA)')

# 2. Plotar PACF (Autocorrelação Parcial) -> Ajuda a escolher o 'p' (AutoRegressive)
plot_pacf(df_final['Diff_1'].dropna(), lags=20, ax=ax2, 
          title='Partial Autocorrelation (PACF) - Indica o valor de p (AR)')

plt.tight_layout()
plt.show()