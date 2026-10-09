"""
Gera a imagem do post: taxa de inadimplência por faixa de uso do limite.
Saída: imagens/calote_por_uso_do_limite.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dados/cs-training.csv')

# 1. Separar os clientes em faixas de uso do limite (1 = 100% do limite)
limites = [-0.001, 0.5, 1, 2, 10]
nomes = ['até 50%', '50% a 100%', '100% a 200%', '200% a 1000%']
df['faixa'] = pd.cut(df['RevolvingUtilizationOfUnsecuredLines'], bins=limites, labels=nomes)

# 2. Taxa de inadimplência (%) e número de clientes em cada faixa
taxa = df.groupby('faixa', observed=True)['SeriousDlqin2yrs'].mean() * 100
clientes = df.groupby('faixa', observed=True)['SeriousDlqin2yrs'].count()
media_geral = df['SeriousDlqin2yrs'].mean() * 100

# 3. Cores: uma faixa em destaque, as outras em cinza
azul = '#2a78d6'
cinza = '#c3c2b7'
cores = [cinza, cinza, azul, cinza]

# 4. Gráfico
fig, ax = plt.subplots(figsize=(8, 6.4), dpi=150)
fig.patch.set_facecolor('#fcfcfb')
ax.set_facecolor('#fcfcfb')

barras = ax.bar(taxa.index, taxa.values, color=cores, width=0.5)

# valor em cima de cada barra
ax.bar_label(barras, labels=[f'{v:.1f}%'.replace('.', ',') for v in taxa.values],
             padding=4, fontsize=12, color='#0b0b0b')

# linha da média geral
ax.axhline(media_geral, color='#898781', linewidth=1, linestyle='--')
ax.text(3.42, media_geral + 0.8, f'média geral: {media_geral:.1f}%'.replace('.', ','),
        ha='right', fontsize=10, color='#52514e')

# rótulos do eixo x com o número de clientes embaixo
rotulos = [f'{nome}\n{n:,} clientes'.replace(',', '.') for nome, n in zip(nomes, clientes.values)]
ax.set_xticks(range(len(nomes)))
ax.set_xticklabels(rotulos, fontsize=10, color='#52514e')

ax.set_xlabel('Quanto do limite de crédito o cliente está usando', fontsize=10.5, color='#52514e', labelpad=10)
ax.set_yticks([])
ax.set_ylim(0, 48)
for lado in ['top', 'right', 'left']:
    ax.spines[lado].set_visible(False)
ax.spines['bottom'].set_color('#c3c2b7')
ax.tick_params(axis='x', length=0)

fig.suptitle('Quem estoura o limite do cartão é quem mais dá calote',
             x=0.06, y=0.965, ha='left', fontsize=15, fontweight='bold', color='#0b0b0b')
fig.text(0.06, 0.895, '% de clientes que atrasaram 90 dias ou mais nos 2 anos seguintes',
         ha='left', fontsize=10.5, color='#52514e')
fig.text(0.06, 0.03, 'Dados: Give Me Some Credit (Kaggle), 150 mil clientes. '
         'Valores acima de 1000% foram tratados como erro de cadastro.',
         ha='left', fontsize=8.5, color='#898781')

fig.subplots_adjust(top=0.84, bottom=0.2, left=0.06, right=0.96)

os.makedirs('imagens', exist_ok=True)
fig.savefig('imagens/calote_por_uso_do_limite.png', facecolor=fig.get_facecolor())
print(taxa.round(1))
print('Imagem salva em imagens/calote_por_uso_do_limite.png')
