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


# Quais colunas têm valores faltando? MonthlyIncome e NumberOfDependents
# Qual a proporção de inadimplentes? Não sei nem qual é a coluna de inadimplente, seria a SeriousDlqin2yrs?
# Você achou algum valor estranho no describe()? (olhe as linhas min e max) Sim, RevolvingUtilizationOfUnsecuredLines tem um valor muito alto 50708.000000, coluna age tem uma idade muita alta e um NumberOfDependents muita alta com 20 


# Não consegui ver nada no value_counts(normalize=True)
# Achei o maximo do DebtRation muito alto, acho que é um outlier 
# MonthlyIncome com 3 milhão é outlier tbm
# NumberOfTimes90DaysLate tem um outiler de 98 sendo que a coluna fala que é 90 dias, NumberOfTime60-89DaysPastDueNotWorse tbm 
# O que fazer com a coluna Unnamed? Eu droparia ela na hora de separa o X e y