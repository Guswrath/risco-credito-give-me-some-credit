# Carregar Bibliotecas
import pandas as pd
import numpy as np


# Carregar Dataset

df = pd.read_csv('dados/cs-training.csv')
print(df.head())

# Analise Inicial 

print('Shape Bruto:\n', df.shape)
print('Formato das Colunas:\n', df.dtypes)
print('Descrição:\n', df.describe().T)
print('Valores Faltantes:\n', df.isnull().sum())
print('Proporção Inadimplentes:\n', df['SeriousDlqin2yrs'].value_counts(normalize=True))
print(df[df['MonthlyIncome'].isna()]['DebtRatio'].median())    # quem NÃO tem renda informada
print(df[df['MonthlyIncome'].notna()]['DebtRatio'].median())   # quem tem renda informada


# O que o diagnóstico mostrou:
#
# - Target: SeriousDlqin2yrs. 6,7% de inadimplentes, base bem desbalanceada.
# - Valores faltantes: MonthlyIncome (29.731) e NumberOfDependents (3.924).
# - Unnamed: 0 é só o número da linha. Sai na hora de separar X e y.
# - age tem mínimo 0.
# - RevolvingUtilizationOfUnsecuredLines chega a 50.708, e deveria ficar perto de 0 a 1.
# - As três colunas de atraso têm máximo 98, o que não é possível em 2 anos.
# - DebtRatio e MonthlyIncome têm máximos muito altos (329 mil e 3 milhões).
# - A mediana do DebtRatio é 1.159 para quem não tem renda informada e 0,29 para
#   quem tem. Sem a renda não dá para calcular a proporção, então nessas linhas a
#   coluna guarda o valor da dívida.
#
# O tratamento de cada um desses pontos está no pipeline.py.