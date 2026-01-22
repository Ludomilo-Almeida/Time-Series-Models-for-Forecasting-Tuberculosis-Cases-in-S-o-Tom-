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