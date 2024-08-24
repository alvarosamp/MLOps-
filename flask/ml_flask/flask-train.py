# Importando os pacotes necessários
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

# Carregar o dataset
dataset = pd.read_csv("train.csv")

# Identificar colunas numéricas e categóricas
numerical_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Loan_Status')
categorical_cols.remove('Loan_ID')

# Preencher colunas categóricas com a moda
for col in categorical_cols:
    mode_value = dataset[col].mode()[0]
    dataset[col] = dataset[col].fillna(mode_value)

# Preencher colunas numéricas com a mediana
for col in numerical_cols:
    median_value = dataset[col].median()
    dataset[col] = dataset[col].fillna(median_value)

# Lidar com outliers
dataset[numerical_cols] = dataset[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))

# Transformações logarítmicas e processamento de domínio
dataset['LoanAmount'] = np.log(dataset['LoanAmount'])
dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome'] = np.log(dataset['TotalIncome'])

# Remover ApplicantIncome e CoapplicantIncome
dataset = dataset.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

# Codificação das variáveis categóricas
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    label_encoders[col] = le

# Codificar a variável alvo
le_status = LabelEncoder()
dataset['Loan_Status'] = le_status.fit_transform(dataset['Loan_Status'])

# Dividir o dataset
X = dataset.drop(columns=['Loan_Status', 'Loan_ID'])
y = dataset['Loan_Status']
RANDOM_SEED = 6

# RandomForest
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    'n_estimators': [200, 400, 700],
    'max_depth': [10, 20, 30],
    'criterion': ["gini", "entropy"],
    'max_leaf_nodes': [50, 100]
}

grid_forest = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_forest,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)

model_forest = grid_forest.fit(X, y)

# Salvar e carregar o modelo
joblib.dump(model_forest, 'RF_Loan_model.joblib')
loaded_model = joblib.load('RF_Loan_model.joblib')

# Dados para previsão
data = pd.DataFrame([[
    1.0,  # Exemplo de dados, ajuste conforme a ordem das colunas
    0.0,
    0.0,
    0.0,
    0.0,
    4.98745,
    360.0,
    1.0,
    2.0,
    8.698
]], columns=X.columns)  # Garantir que as colunas coincidam

print(f"Prediction is: {loaded_model.predict(data)}")
