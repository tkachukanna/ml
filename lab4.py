import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

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

# Перетворення даних у тензори
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Гіперпараметри нейронної мережі
learning_rates = [0.001, 0.01]
architectures = [(64, 32), (64, 64, 32)]
batch_sizes = [32, 64]
epochs = 10
results = []

# Реалізація нейронної мережі
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, architecture):
        super(NeuralNetwork, self).__init__()
        layers = []
        for i, neurons in enumerate(architecture):
            if i == 0:
                layers.append(nn.Linear(input_size, neurons))
            else:
                layers.append(nn.Linear(architecture[i - 1], neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(architecture[-1], 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

for lr in learning_rates:
    for architecture in architectures:
        for batch_size in batch_sizes:
            model = NeuralNetwork(input_size=X_train.shape[1], architecture=architecture)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Навчання моделі
            for epoch in range(epochs):
                permutation = torch.randperm(X_train_tensor.size(0))
                for i in range(0, X_train_tensor.size(0), batch_size):
                    indices = permutation[i:i + batch_size]
                    batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Оцінка на валідаційній вибірці
            with torch.no_grad():
                y_val_pred = model(X_val_tensor)
                y_val_pred_class = (y_val_pred > 0.5).float()
                accuracy = accuracy_score(y_val_tensor.numpy(), y_val_pred_class.numpy())

            # Збереження результатів
            results.append({
                "Learning Rate": lr,
                "Architecture": architecture,
                "Batch Size": batch_size,
                "Epochs": epochs,
                "Accuracy": accuracy
            })


# Вивід результатів
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

print("Результати експериментів:")
print(results_df)
