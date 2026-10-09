# Carregar Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Carregar Dataset

df = pd.read_csv('dados/cs-training.csv')

# Tratar outliers
print('Valores Faltantes:\n', df.isnull().sum())
print('Idade < 15\n', df[~df['age'].between(15, 100)])
df.loc[df['age'] < 15, 'age'] = np.nan
print('Valores Faltantes:\n', df.isnull().sum())

# printei entre 15 e 100, mas dei o loc apenas em quem tem menos que 15 anos pois pode ter adolecente que já trabalhe.

print('Valores de Number Of Time 30-59:\n', sorted(df['NumberOfTime30-59DaysPastDueNotWorse'].unique()))
df.loc[df['NumberOfTime30-59DaysPastDueNotWorse'] >= 96, 'NumberOfTime30-59DaysPastDueNotWorse'] = np.nan
print('Valores de Number Of Time 30-59:\n', sorted(df['NumberOfTime30-59DaysPastDueNotWorse'].unique()))

# Usei a mesma logica que eu já conheço do loc, e os valores 96 e 98 sumiram, e apareceu um nan no meio que eu vou ter que tratar.

print('Valores de Number Of Times 90:\n', sorted(df['NumberOfTimes90DaysLate'].unique()))
df.loc[df['NumberOfTimes90DaysLate'] >= 96, 'NumberOfTimes90DaysLate'] = np.nan
print('Valores de Number Of Times 90:\n', sorted(df['NumberOfTimes90DaysLate'].unique()))

# Usei a mesma logica que eu já conheço do loc, e os valores 96 e 98 sumiram.

print('Number Of Time 60-89:\n',sorted(df['NumberOfTime60-89DaysPastDueNotWorse'].unique()))
df.loc[df['NumberOfTime60-89DaysPastDueNotWorse'] >= 96, 'NumberOfTime60-89DaysPastDueNotWorse'] = np.nan
print('Number Of Time 60-89:\n',sorted(df['NumberOfTime60-89DaysPastDueNotWorse'].unique()))


# DebtRatio

df.loc[df['MonthlyIncome'] <= 1, 'MonthlyIncome'] = np.nan


df.loc[df['MonthlyIncome'].isna(), 'DebtRatio'] = np.nan
print('Valores Faltantes:\n', df.isnull().sum())

# a condição procura quem tem a renda faltando. Se você imputar a renda primeiro, não sobra nenhuma renda faltando, 
# a condição não encontra ninguém, e os 29.731 valores errados de DebtRatio ficam lá para sempre. A renda vazia é a única pista de quais linhas estão erradas, 
# e imputar apaga essa pista.


print((df['RevolvingUtilizationOfUnsecuredLines'] > 1).sum())
print((df['RevolvingUtilizationOfUnsecuredLines'] > 2).sum())
print((df['RevolvingUtilizationOfUnsecuredLines'] > 10).sum())

df.loc[df['RevolvingUtilizationOfUnsecuredLines'] > 10, 'RevolvingUtilizationOfUnsecuredLines'] = np.nan

print((df['RevolvingUtilizationOfUnsecuredLines'] > 1).sum())
print((df['RevolvingUtilizationOfUnsecuredLines'] > 2).sum())
print((df['RevolvingUtilizationOfUnsecuredLines'] > 10).sum())

# escolhi > 10 pois ainda acha viavel ter o limite de credito usado até 2, agora 10 eu acho impossivel.

print(df['MonthlyIncome'].quantile([0.01, 0.5, 0.99, 0.999]))
print((df['MonthlyIncome'] > 100000).sum())

print((df['MonthlyIncome'] == 0).sum())
print((df['MonthlyIncome'] == 1).sum())
print(df[df['MonthlyIncome'] == 0]['DebtRatio'].median())
print(df[df['MonthlyIncome'] == 1]['DebtRatio'].median())



# Separar X e y

X = df.drop(columns=['SeriousDlqin2yrs', 'Unnamed: 0'])
y = df['SeriousDlqin2yrs']
print(X)
print(y)

# Treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(X_train.shape)
print(X_test.shape)

# imputar 

print('Valores Faltantes:\n', df.isnull().sum())

imputer = SimpleImputer(strategy='median')
colunas = X_train.columns
X_train[colunas] = imputer.fit_transform(X_train[colunas])
X_test[colunas] = imputer.transform(X_test[colunas])

print(X_train.isna().sum().sum())
print(X_test.isna().sum().sum())


# Scaler

scaler = StandardScaler()
X_train[colunas] = scaler.fit_transform(X_train[colunas])
X_test[colunas] = scaler.transform(X_test[colunas])