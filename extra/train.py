# ================================================
# Train + Save Model (Deep Learning Version)
# ================================================
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "dataset.csv")
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# ----------------------------
# 1. Load dataset
# ----------------------------
print(f"Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

# Encode final_grade (A=1, B=0)
df["final_grade"] = df["final_grade"].map({"A": 1, "B": 0}).fillna(0)

# Encode mineral code (target variable)
le = LabelEncoder()
df["mineral"] = le.fit_transform(df["commod"])

print(f"Total samples: {len(df)}")
print(f"Target classes ({len(le.classes_)}): {le.classes_}")

# Features and target
X = df[["wgs84_lat", "wgs84_lon", "final_grade"]].values
y = df["mineral"].values

# Scale features for stability
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 2. Define PyTorch dataset
# ----------------------------
class MineralDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = MineralDataset(X_train, y_train)
test_ds = MineralDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512)

# ----------------------------
# 3. Define neural network
# ----------------------------
class MineralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MineralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_dim = X.shape[1]
hidden_dim = 64
output_dim = len(le.classes_)

model = MineralNet(input_dim, hidden_dim, output_dim)

# ----------------------------
# 4. Training setup
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# ----------------------------
# 5. Training loop
# ----------------------------
epochs = 30
print("\nStarting training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate on test set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    acc = correct / total

    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss:.4f} | Test Acc: {acc*100:.2f}%")

# ----------------------------
# 6. Save model + encoder + scaler
# ----------------------------
model_path = os.path.join(models_dir, "mineral_dl_model.pth")
le_path = os.path.join(models_dir, "label_encoder.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")

torch.save(model.state_dict(), model_path)
joblib.dump(le, le_path)
joblib.dump(scaler, scaler_path)

print(f"\n[OK] DL model saved to: {model_path}")
print(f"[OK] Label encoder saved to: {le_path}")
print(f"[OK] Scaler saved to: {scaler_path}")
