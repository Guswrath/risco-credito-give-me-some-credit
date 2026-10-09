# Carregar Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Carregar Dataset

df = pd.read_csv('dados/cs-training.csv')

# Valores errados viram NaN (a linha continua na base, só a célula é apagada)
print('Valores Faltantes:\n', df.isnull().sum())
print('Idade < 15\n', df[~df['age'].between(15, 100)])
df.loc[df['age'] < 15, 'age'] = np.nan
print('Valores Faltantes:\n', df.isnull().sum())

# Idade: um registro com idade 0. Cortei abaixo de 15, e não de 18, porque pode haver adolescente que já trabalha.

print('Valores de Number Of Time 30-59:\n', sorted(df['NumberOfTime30-59DaysPastDueNotWorse'].unique()))
df.loc[df['NumberOfTime30-59DaysPastDueNotWorse'] >= 96, 'NumberOfTime30-59DaysPastDueNotWorse'] = np.nan
print('Valores de Number Of Time 30-59:\n', sorted(df['NumberOfTime30-59DaysPastDueNotWorse'].unique()))

# Atrasos: 96 e 98 são códigos do sistema, não quantidade de atrasos. Viram NaN e são imputados depois.

print('Valores de Number Of Times 90:\n', sorted(df['NumberOfTimes90DaysLate'].unique()))
df.loc[df['NumberOfTimes90DaysLate'] >= 96, 'NumberOfTimes90DaysLate'] = np.nan
print('Valores de Number Of Times 90:\n', sorted(df['NumberOfTimes90DaysLate'].unique()))

# Mesma regra para as outras duas colunas de atraso.

print('Number Of Time 60-89:\n',sorted(df['NumberOfTime60-89DaysPastDueNotWorse'].unique()))
df.loc[df['NumberOfTime60-89DaysPastDueNotWorse'] >= 96, 'NumberOfTime60-89DaysPastDueNotWorse'] = np.nan
print('Number Of Time 60-89:\n',sorted(df['NumberOfTime60-89DaysPastDueNotWorse'].unique()))


# Renda e DebtRatio

df.loc[df['MonthlyIncome'] <= 1, 'MonthlyIncome'] = np.nan


df.loc[df['MonthlyIncome'].isna(), 'DebtRatio'] = np.nan
print('Valores Faltantes:\n', df.isnull().sum())

# Renda 0 e 1 foram tratadas como renda não informada.
# Para quem não tem renda, o DebtRatio guarda o valor da dívida e não a proporção, então também vira NaN.
# Isso precisa vir antes da imputação: a renda vazia é a única pista de quais linhas estão erradas,
# e imputar a renda primeiro apagaria essa pista.


print((df['RevolvingUtilizationOfUnsecuredLines'] > 1).sum())
print((df['RevolvingUtilizationOfUnsecuredLines'] > 2).sum())
print((df['RevolvingUtilizationOfUnsecuredLines'] > 10).sum())

df.loc[df['RevolvingUtilizationOfUnsecuredLines'] > 10, 'RevolvingUtilizationOfUnsecuredLines'] = np.nan

print((df['RevolvingUtilizationOfUnsecuredLines'] > 1).sum())
print((df['RevolvingUtilizationOfUnsecuredLines'] > 2).sum())
print((df['RevolvingUtilizationOfUnsecuredLines'] > 10).sum())

# Uso do limite: escolhi > 10 porque usar até 200% do limite é viável (juros, tarifas), e esses clientes
# são os de maior inadimplência. Acima de 1000% considerei erro de cadastro.

print(df['MonthlyIncome'].quantile([0.01, 0.5, 0.99, 0.999]))
print((df['MonthlyIncome'] > 100000).sum())

print((df['MonthlyIncome'] == 0).sum())
print((df['MonthlyIncome'] == 1).sum())
print(df[df['MonthlyIncome'] == 0]['DebtRatio'].median())
print(df[df['MonthlyIncome'] == 1]['DebtRatio'].median())



# Separar X e y ('Unnamed: 0' é só o número da linha)

X = df.drop(columns=['SeriousDlqin2yrs', 'Unnamed: 0'])
y = df['SeriousDlqin2yrs']
print(X)
print(y)

# Treino e teste (antes de qualquer fit, para não vazar informação do teste)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(X_train.shape)
print(X_test.shape)

# Imputar com a mediana: fit só no treino, transform nos dois

print('Valores Faltantes:\n', df.isnull().sum())

imputer = SimpleImputer(strategy='median')
colunas = X_train.columns
X_train[colunas] = imputer.fit_transform(X_train[colunas])
X_test[colunas] = imputer.transform(X_test[colunas])

print(X_train.isna().sum().sum())
print(X_test.isna().sum().sum())


# Padronizar: fit só no treino, transform nos dois

scaler = StandardScaler()
X_train[colunas] = scaler.fit_transform(X_train[colunas])
X_test[colunas] = scaler.transform(X_test[colunas])