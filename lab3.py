import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Поділ датасету навчальний, валідаційний та тестувальний набір
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

# Гіперпараметри градієнтного бустингу
learning_rates = [0.01, 0.05, 0.1]
n_estimators_list = [10, 50, 100]
max_depth_list = [3, 5, 7]

results = []

for lr in learning_rates:
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            model = GradientBoostingClassifier(
                learning_rate=lr, n_estimators=n_estimators, max_depth=max_depth, random_state=42, verbose=0
            )
            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_val_pred)

            results.append({
                "Learning Rate": lr,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "Accuracy": accuracy
            })

# Вивід результатів
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

print("Результати експериментів:")
print(results_df)

