import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Завантаження даних
data = pd.read_csv('Loan_default.csv')

# Підготовка даних
non_numeric_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}

for column in non_numeric_columns:
    if column != 'Default':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

X = data.drop(columns=['Default', 'LoanID'])
y = data['Default']

# Поділ датасету навчальний, валідаційний та тестувальний набір
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

results = []

for warm_start in [True, False]:
    for C in [0.1, 1, 10]:
        for penalty in ['l1', 'l2']:
            for max_iter in [100, 500, 1000]:
                model = LogisticRegression(
                    warm_start=warm_start, C=C, penalty=penalty, solver="liblinear",  max_iter=max_iter, random_state=42
                )
                model.fit(X_train_scaled, y_train)

                y_val_pred = model.predict(X_val_scaled)
                accuracy = accuracy_score(y_val, y_val_pred)

                results.append({
                    "Warm_start": warm_start,
                    "C": C,
                    "Penalty": penalty,
                    "Solver": "liblinear",
                    "Max_Iter": max_iter,
                    "Accuracy": accuracy
                })

# Вивід результатів
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

print("Результати експериментів:")
print(results_df)
