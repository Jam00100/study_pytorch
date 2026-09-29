import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

INPUT_DIR = Path("/kaggle/input/competitions/titanic")
# INPUT_DIR = Path("./")
OUTPUT_DIR = Path("/kaggle/working")
# OUTPUT_DIR = Path("./")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# 選擇運算裝置
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Usage device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif device.type == "mps":
    print("GPU: Apple Silicon GPU(MPS)")
else:
    print("USE CPU")

# check data
train_df = pd.read_csv(INPUT_DIR/"train.csv")
print(train_df.head())

# 分析資料
# 1. train_df information
print("train_df shape", train_df.shape)
# print(train_df.dtypes)
# print(train_df.describe())
# print(train_df.describe(include="object"))

# 2. 檢查Survived (0, 1) 分佈
print(train_df["Survived"].value_counts())

ax = sns.countplot(data=train_df, x="Survived")
ax.bar_label(ax.containers[0])

plt.title("Survival count")
plt.show()

# 3. 分析存活與以下欄位的分布情況
columns = ["Sex", "Pclass", "SibSp", "Parch", "Embarked"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, column in enumerate(columns):
    ax = axes[i]

    sns.countplot(
        data=train_df,
        x=column,
        hue="Survived",
        hue_order=[0, 1],
        ax=ax
    )

    # 在每根柱子上方標記人數
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=2)

    ax.set_title(column)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.legend(title="Survived", labels=["No (0)", "Yes (1)"])

axes[-1].axis("off")

plt.tight_layout()
plt.show()

# 男女之間的存活率差異
train_df.groupby('Sex')[['Survived']].mean()
train_df.pivot_table('Survived',index = 'Sex', columns = 'Pclass')

age = pd.cut(train_df['Age'],[0, 18, 80])
train_df.pivot_table('Survived',['Sex',age], 'Pclass')

# 檢查missing value
missing = pd.DataFrame({
    "Number of missing values": train_df.isna().sum(),
    "Missing value ratio (%)": train_df.isna().mean() * 100
})

print(missing.sort_values("Missing value ratio (%)", ascending=False))
# 資料清理
train_data = train_df.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis=1)
train_data = train_data.dropna(subset=["Embarked", "Age"])

# 處理input dtype
# 處理Sex and Embarked
labelencoder = LabelEncoder()
train_data["Sex"] = labelencoder.fit_transform(train_data["Sex"])
train_data["Embarked"] = labelencoder.fit_transform(train_data["Embarked"])
# 檢查Label Encoder轉換情形
print("Sex:",  train_data['Sex'].unique())
print("Embarked:", train_data['Embarked'].unique())

X = train_data.drop("Survived", axis=1)
y = train_data[["Survived"]]

x_train, x_validate, y_train, y_validate = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_validate = scaler.transform(x_validate)

class TitaincNN(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_features, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.network(x)  # shape: [batch_size, 2]

model = TitaincNN(input_features=x_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999), # default
    eps = 1e-8, # default
    weight_decay=0, # default
    amsgrad=False # default
)

# %%
x_train = torch.tensor(x_train).to(torch.float)
y_train = torch.tensor(y_train.values).to(torch.long).squeeze(1)
x_validate = torch.tensor(x_validate).to(torch.float)
y_validate = torch.tensor(y_validate.values).to(torch.long).squeeze(1)

model = model.to(device)

train_features = x_train.to(device)
train_labels = y_train.to(device)
val_features = x_validate.to(device)
val_labels = y_validate.to(device)

Number_of_Epoch = 100
for epoch in range(Number_of_Epoch):
    # 訓練
    model.train()
    optimizer.zero_grad()

    train_logits = model(train_features)
    train_loss = criterion(train_logits, train_labels)

    train_loss.backward()
    optimizer.step()

    train_predictions = train_logits.argmax(dim=1)
    train_accuracy = (
        (train_predictions == train_labels).float().mean().item()
    )

    # 驗證
    model.eval()
    with torch.no_grad():
        val_logits = model(val_features)
        val_loss = criterion(val_logits, val_labels)

        val_predictions = val_logits.argmax(dim=1)
        val_accuracy = (
            (val_predictions == val_labels).float().mean().item()
        )

    print(
        f"Epoch {epoch + 1:03d} | "
        f"Train loss: {train_loss.item():.4f}, "
        f"accuracy: {train_accuracy:.4f} | "
        f"Val loss: {val_loss.item():.4f}, "
        f"accuracy: {val_accuracy:.4f}"
    )

# Independent test：先只預測資料完整的列
test_df = pd.read_csv(INPUT_DIR / "test.csv")

test_data = test_df.drop(
    ["Cabin", "Name", "Ticket", "PassengerId"],
    axis=1
).copy()

# 使用訓練資料的統計值補缺值，保留 test.csv 的所有乘客
test_data["Age"] = test_data["Age"].fillna(train_data["Age"].median())
test_data["Fare"] = test_data["Fare"].fillna(train_data["Fare"].median())

# 編碼須與訓練時的 LabelEncoder 一致
test_data["Sex"] = test_data["Sex"].map({"female": 0, "male": 1})
test_data["Embarked"] = test_data["Embarked"].map({
    "C": 0, "Q": 1, "S": 2
})

# 保持與訓練時相同的欄位順序
test_data = test_data[X.columns]
test_data = scaler.transform(test_data)

test_features = torch.tensor(
    test_data,
    dtype=torch.float32
).to(device)

model.eval()
with torch.no_grad():
    logits = model(test_features)
    predictions = logits.argmax(dim=1).cpu().numpy()

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": predictions
})

submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

print(submission.head())
print("提交：", len(submission))