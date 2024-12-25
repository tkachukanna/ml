import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# 1. Завантаження даних
data = pd.read_csv('Loan_default.csv')

# 2. Підготовка даних
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

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

# 3. Власна реалізація логістичної регресії
class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, epochs=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(model)
        return [1 if i > self.threshold else 0 for i in y_pred]

# 4. Навчання моделі
results = []
learning_rates = [0.01, 0.1, 1]
epochs_list = [100, 200, 300]
thresholds = [0.4, 0.5, 0.6]

for lr in learning_rates:
    for epochs in epochs_list:
        for threshold in thresholds:
            model = LogisticRegressionCustom(learning_rate=lr, epochs=epochs, threshold=threshold)
            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_val_pred)

            results.append({
                "Learning Rate": lr,
                "Epochs": epochs,
                "Threshold": threshold,
                "Accuracy": accuracy
            })

# 5. Вивід результатів
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

print("Результати експериментів:")
print(results_df)
