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

# %%
# ==========================================
# CÉLULA 6: COMPARAÇÃO ROBUSTA (AIC & BIC)
# ==========================================
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

print("\n--- COMPARAÇÃO DE MODELOS: ARIMA(2,1,1) vs ARIMA(1,1,1) ---")

# 1. Ajustar Modelo A: ARIMA(2, 1, 1)
model_211 = ARIMA(df_final['Casos'], order=(2, 1, 1))
res_211 = model_211.fit()

# 2. Ajustar Modelo B: ARIMA(1, 1, 1)
model_111 = ARIMA(df_final['Casos'], order=(1, 1, 1))
res_111 = model_111.fit()

# 3. Tabela Comparativa
resultados = pd.DataFrame({
    'Modelo': ['ARIMA(2,1,1)', 'ARIMA(1,1,1)'],
    'AIC': [res_211.aic, res_111.aic],
    'BIC': [res_211.bic, res_111.bic]
})

print(resultados)

# 4. Decisão Automática
# Vamos ver qual ganha no AIC (que é o mais importante para previsão)
melhor_aic = resultados.loc[resultados['AIC'].idxmin()]

print("\n------------------------------------------------")
print(f">> VENCEDOR PELO AIC: {melhor_aic['Modelo']}")
print(f"   (AIC: {melhor_aic['AIC']:.4f})")
print("------------------------------------------------")

# Diagnóstico visual do vencedor
if melhor_aic['Modelo'] == 'ARIMA(2,1,1)':
    best_model = res_211
else:
    best_model = res_111

print(f"\nResumo estatístico do vencedor ({melhor_aic['Modelo']}):")
print(best_model.summary())

# Gráfico de diagnóstico
best_model.plot_diagnostics(figsize=(10, 8))
plt.tight_layout()
plt.show()

# %%
# ==========================================
# CÉLULA 7: ESTIMAÇÃO DE PARÂMETROS (ARIMA 1,1,1)
# ==========================================
from statsmodels.tsa.arima.model import ARIMA

print("\n--- RESUMO DO MODELO ARIMA(1,1,1) ---")

# Treinar o modelo
model_final = ARIMA(df_final['Casos'], order=(1, 1, 1))
model_fit = model_final.fit()

# Mostrar a tabela completa com os coeficientes
print(model_fit.summary())

# Se quiser ver os coeficientes isolados:
print("\nParâmetros Isolados:")
print(f"AR(1): {model_fit.params['ar.L1']:.4f}")
print(f"MA(1): {model_fit.params['ma.L1']:.4f}")
print(f"Sigma2: {model_fit.params['sigma2']:.4f}")

# %%
# ==========================================
# CÉLULA 8: AVALIAÇÃO DE DESEMPENHO (Performance)
# ==========================================
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("\n--- MODEL PERFORMANCE (IN-SAMPLE) ---")

# 1. Obter valores previstos pelo modelo (Fitted Values)
# O modelo tenta 'prever' o passado para ver se aprendeu bem
predictions = model_fit.fittedvalues
observed = df_final['Casos']

# 2. Calcular Métricas de Erro
# RMSE: Raiz do Erro Quadrático Médio (Penaliza grandes erros)
rmse = np.sqrt(mean_squared_error(observed, predictions))

# MAE: Erro Médio Absoluto (Média simples dos erros)
mae = mean_absolute_error(observed, predictions)

# MAPE: Erro Percentual (Cuidado com divisões por zero!)
# Vamos calcular apenas onde os casos reais > 0 para não dar erro
mask = observed != 0
mape = np.mean(np.abs((observed[mask] - predictions[mask]) / observed[mask])) * 100

print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
print(f"MAE  (Mean Absolute Error):    {mae:.4f}")
print(f"MAPE (Mean Abs. Perc. Error):  {mape:.2f}%")

# 3. Gráfico: Real vs Modelo
plt.figure(figsize=(12, 6))

# Dados Reais (Pontos cinzentos)
plt.plot(observed.index, observed, 'o', color='gray', alpha=0.5, label='Casos Reais')

# Linha do Modelo (Azul)
plt.plot(predictions.index, predictions, color='#2980b9', linewidth=2, label='Ajuste do Modelo (ARIMA)')

plt.title(f'Model Performance: Real vs Fitted (RMSE={rmse:.2f})', fontsize=14, fontweight='bold')
plt.legend()
plt.ylabel('Casos')
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("A guardar gráfico de performance...")
plt.savefig('grafico_performance_modelo.png')
plt.show()


# %%
# ==========================================
# CÉLULA 9: PREVISÃO FUTURA (FORECASTING)
# ==========================================
print("\n--- A CALCULAR PREVISÕES (2024-2025) ---")

# 1. Definir quantos meses queremos prever para a frente
# Vamos prever 24 meses (2 anos)
meses_futuros = 36
forecast_object = model_fit.get_forecast(steps=meses_futuros)

# 2. Extrair os dados da previsão
forecast_mean = forecast_object.predicted_mean
conf_int = forecast_object.conf_int(alpha=0.05) # Intervalo de Confiança de 95%

# Criar datas futuras para o eixo X
last_date = df_final.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                               periods=meses_futuros, freq='MS')

# Atribuir as datas à série de previsão
forecast_mean.index = forecast_dates
conf_int.index = forecast_dates

# 3. Gráfico Final: Passado + Futuro
plt.figure(figsize=(12, 6))

# A) Dados Históricos (Passado)
plt.plot(df_final.index, df_final['Casos'], label='Histórico (2011-2023)', color='#2c3e50')

# B) Previsão (Futuro)
plt.plot(forecast_mean.index, forecast_mean, label='Previsão (2024-2025)', color='#e74c3c', linestyle='--', linewidth=2)

# C) Intervalo de Confiança (Sombra)
# Mostra a margem de erro (onde o valor real provavelmente vai cair)
plt.fill_between(conf_int.index, 
                 conf_int.iloc[:, 0], 
                 conf_int.iloc[:, 1], 
                 color='#e74c3c', alpha=0.2, label='Intervalo de Confiança (95%)')

plt.title('Tuberculosis Forecast in São Tomé and Príncipe (2024-2026)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Projected Cases', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("A guardar gráfico de previsão...")
plt.savefig('grafico_previsao_futura.png')
plt.show()

# 4. Mostrar os números exatos dos primeiros 6 meses previstos
print("\nPrevisão para os próximos 6 meses:")
print(forecast_mean.head(6))