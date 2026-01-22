import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX # Agora usamos SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# %%
# ==========================================
# CÉLULA 1: CARREGAR E LIMPAR DADOS
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
# CÉLULA 2: FUNÇÃO DE TESTE ADF (Estacionaridade)
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

# %%
# ==========================================
# CÉLULA 3: VERIFICAÇÃO DE SAZONALIDADE (NOVO)
# ==========================================
print("\n--- A VERIFICAR SAZONALIDADE (Decomposição) ---")

# Decompor a série em: Tendência, Sazonalidade e Resíduos
decomposicao = seasonal_decompose(df_final['Casos'], model='additive')

# Plotar a decomposição para análise visual
fig = decomposicao.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.show()

# Teste ADF na série original
realizar_teste_adf(df_final['Casos'], nome="Original")

# %%
# ==========================================
# CÉLULA 4: APLICAR DIFERENÇAS (Normal e Sazonal)
# ==========================================
# 1. Diferença Normal (d=1) -> Remove tendência
df_final['Diff_1'] = df_final['Casos'].diff()

# 2. Diferença Sazonal (D=1, s=12) -> Remove padrão anual
# Subtrai o Janeiro deste ano pelo Janeiro do ano passado
df_final['Diff_Sazonal'] = df_final['Casos'].diff(12)

# 3. Diferença Combinada (Primeiro faz sazonal, depois normal)
df_final['Diff_Total'] = df_final['Diff_Sazonal'].diff()

print("\n--- TESTES NAS DIFERENÇAS ---")
realizar_teste_adf(df_final['Diff_1'], nome="1ª Diferença (Tendência)")
realizar_teste_adf(df_final['Diff_Sazonal'], nome="Diferença Sazonal (12 meses)")
realizar_teste_adf(df_final['Diff_Total'], nome="Diferença Total (Tendência + Sazonal)")

# Gráfico da Diferença Sazonal
plt.figure(figsize=(10, 4))
plt.plot(df_final.index, df_final['Diff_Sazonal'], color='#e67e22', label='Diferença Sazonal (12 meses)')
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title('Série após Diferenciação Sazonal')
plt.legend()
plt.show()

# %%
# ==========================================
# CÉLULA 5: IDENTIFICAÇÃO (ACF/PACF na Série Sazonal)
# ==========================================
# Vamos olhar para a série diferenciada sazonalmente
series_to_plot = df_final['Diff_Sazonal'].dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# ACF
plot_acf(series_to_plot, lags=36, ax=ax1, title='ACF (Autocorrelação) - Série Sazonal')
# PACF
plot_pacf(series_to_plot, lags=36, ax=ax2, title='PACF (Autocorrelação Parcial) - Série Sazonal')

plt.tight_layout()
plt.show()

# %%
# ==========================================
# CÉLULA 6: TREINAR MODELO SARIMA (Seasonal ARIMA)
# ==========================================
# SARIMA requer: (p,d,q) normais X (P,D,Q,s) sazonais
# Sazonalidade (s) = 12 (Mensal)
print("\n--- A TREINAR MODELO SARIMA (1,1,1)x(1,1,1,12) ---")

model_sarima = SARIMAX(df_final['Casos'], 
                       order=(1, 1, 1),              # Parte não sazonal (p,d,q)
                       seasonal_order=(1, 1, 1, 12), # Parte sazonal (P,D,Q,s)
                       enforce_stationarity=False,
                       enforce_invertibility=False)

model_fit = model_sarima.fit(disp=False)

print(model_fit.summary())

# Diagnóstico
model_fit.plot_diagnostics(figsize=(10, 8))
plt.tight_layout()
plt.show()

# %%
# ==========================================
# CÉLULA 7: AVALIAÇÃO DE DESEMPENHO (SARIMA)
# ==========================================
print("\n--- PERFORMANCE DO MODELO SARIMA ---")

predictions = model_fit.fittedvalues
observed = df_final['Casos']

# Métricas
rmse = np.sqrt(mean_squared_error(observed, predictions))
mae = mean_absolute_error(observed, predictions)

mask = observed != 0
mape = np.mean(np.abs((observed[mask] - predictions[mask]) / observed[mask])) * 100

print(f"RMSE (Erro Quadrático Médio): {rmse:.4f}")
print(f"MAE  (Erro Absoluto Médio):   {mae:.4f}")
print(f"MAPE (Erro Percentual):       {mape:.2f}%")
print(f"AIC do Modelo: {model_fit.aic:.4f}")

# Gráfico de Ajuste
plt.figure(figsize=(12, 6))
plt.plot(observed.index, observed, 'o', color='gray', alpha=0.4, label='Real')
# Começamos o plot mais à frente porque o SARIMA perde os primeiros 12 meses
plt.plot(predictions.index[13:], predictions[13:], color='#2980b9', linewidth=2, label='Ajuste SARIMA')
plt.legend()
plt.title(f'Ajuste do Modelo Sazonal (RMSE={rmse:.2f})')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# ==========================================
# CÉLULA 8: PREVISÃO FINAL COM CURVAS (3 ANOS)
# ==========================================
print("\n--- A CALCULAR PREVISÃO SAZONAL (2024-2026) ---")

# Previsão para 36 meses (3 anos)
meses_futuros = 36
forecast_res = model_fit.get_forecast(steps=meses_futuros)
forecast_vals = forecast_res.predicted_mean
conf_int = forecast_res.conf_int(alpha=0.05)

# Setup gráfico completo
plt.figure(figsize=(14, 7))

# 1. Histórico
plt.plot(df_final.index, df_final['Casos'], 
         label='Histórico', color='#2c3e50', alpha=0.7)

# 2. Ajuste (Treino)
plt.plot(model_fit.fittedvalues.index, model_fit.fittedvalues, 
         label='Ajuste Modelo', color='#2980b9', alpha=0.6, linewidth=1)

# 3. Previsão Futura (SARIMA acompanha as curvas!)
plt.plot(forecast_vals.index, forecast_vals, 
         label='Previsão (3 Anos)', color='#c0392b', linewidth=2.5)

# 4. Intervalo de Confiança
plt.fill_between(forecast_vals.index, 
                 conf_int.iloc[:, 0], 
                 conf_int.iloc[:, 1], 
                 color='#c0392b', alpha=0.15)

plt.title('Previsão Sazonal de Tuberculose (SARIMA) - Acompanhando as Curvas', fontsize=16, fontweight='bold')
plt.xlabel('Ano')
plt.ylabel('Casos')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("A guardar 'grafico_sarima_final.png'...")
plt.savefig('grafico_sarima_final.png')
plt.show()

print("\nPrimeiros 6 meses previstos:")
print(forecast_vals.head(6))