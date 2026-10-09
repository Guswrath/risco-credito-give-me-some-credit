from pipeline import X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



# Logistic Regression

model_reg = LogisticRegression(random_state=42, max_iter=1000)
model_reg.fit(X_train, y_train)
proba = model_reg.predict_proba(X_test)[:, 1]
print(proba)

auc = roc_auc_score(y_test, proba)
print(auc)

gini = 2 * auc - 1

fpr, tpr, cortes = roc_curve(y_test, proba)
ks = (tpr - fpr).max()
print('AUC:', auc)
print('GINI:', gini)
print('KS:', ks)

# Random Forest

model_rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
proba_rf = model_rf.predict_proba(X_test)[:, 1]
print(proba_rf)

auc_rf = roc_auc_score(y_test, proba_rf)

gini_rf = 2 * auc_rf - 1

fpr, tpr, cortes = roc_curve(y_test, proba_rf)
ks_rf = (tpr - fpr).max()

print('AUC_rf:', auc_rf)
print('GINI RF:', gini_rf)
print('KS_rf:', ks_rf)

# XGBoost

model_xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
model_xgb.fit(X_train, y_train)
proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
print(proba_xgb)

auc_xgb = roc_auc_score(y_test, proba_xgb)

gini_xgb = 2 * auc_xgb - 1

fpr, tpr, cortes = roc_curve(y_test, proba_xgb)
ks_xgb = (tpr - fpr).max()

print('AUC_xgb:', auc_xgb)
print('GINI xgb:', gini_xgb)
print('KS_xgb:', ks_xgb)


# Tabela 

import pandas as pd

tabela = pd.DataFrame({
    'modelo': ['Regressao Logistica', 'Random Forest', 'XGBoost'],
    'auc':    [auc, auc_rf, auc_xgb],
    'gini':    [gini, gini_rf, gini_xgb],
    'ks':   [ks, ks_rf, ks_xgb]
})
print(tabela.round(3))

# Decisão: eu recomendaria a Regressão Logística. O XGBoost foi o melhor modelo,
# mas ganha por pouco (0,021 de Gini) e é uma caixa preta. O regulador ou o cliente
# podem querer saber o motivo de uma negativa, e a Regressão Logística é caixa
# branca: dá para explicar quanto cada informação pesou.

# Quais informações do cliente mais pesam na decisão do modelo?
# Peso positivo aumenta o risco, peso negativo reduz.

pesos = pd.Series(model_reg.coef_[0], index=X_train.columns)
print(pesos.sort_values())

# RevolvingUtilizationOfUnsecuredLines (uso do limite) é a que mais aumenta o risco: +0,657.
# MonthlyIncome (renda) é a que mais reduz: -0,327.
