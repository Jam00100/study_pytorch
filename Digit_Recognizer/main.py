import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from zipfile import ZipFile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
BATCH_SIZE = 32

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

# import training data
df = pd.read_csv("./train.csv")
features = df.drop("label", axis=1).values
labels = df["label"].values

# Seperate training and validation dataset
train_features, val_features, train_labels, val_labels = train_test_split(
    features, labels, test_size=0.2, shuffle=True, random_state=RANDOM_SEED, stratify=labels
)

print(f"training features {train_features.shape} | training labels {train_labels.shape}")
print(f"validation features {val_features.shape} | validation labels {val_labels.shape}")

# transform np.array to torch.tensor
train_features = torch.tensor(train_features, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
val_features = torch.tensor(val_features, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0

train_labels = torch.tensor(train_labels, dtype=torch.long)
val_labels = torch.tensor(val_labels, dtype=torch.long)

# Build dataset
train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)

PIN_MEMORY = device.type == "cuda"
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY
)

# Build Digit CNN
class DigitCNN(nn.Module):
    def __init__ (self):
        super().__init__()
        
        # extract features, 資料是灰階沒有RGB，所以in channel = 1
        self.features = nn.Sequential(
            # First layer
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # Second layer
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # Third layer
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10)
        )
    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

model = DigitCNN().to(device)
# Gradient descent
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    # train mode
    model.train()

   # 記錄一個 epoch 的 loss、正確數量與樣本數
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 清除上一個 batch 留下來的梯度
        optimizer.zero_grad(set_to_none=True)

        # Forward：取得模型輸出的 logits
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward (修正錯誤結果提高學習能力)
        loss.backward()

        # 更新模型参数
        optimizer.step()

        batch_size = images.size(0)

        #累計所有樣本的 loss
        running_loss += loss.item() * batch_size

        # 找出分數最高的類别
        predictions = logits.argmax(dim=1)

        correct_predictions += (predictions == labels).sum().item()

        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_accuracy = (correct_predictions / total_samples)

    return epoch_loss, epoch_accuracy

def val_one_epoch(model, data_loader, criterion, device):
    # 切換成驗證模式
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward
            logits = model(images)

            # 計算 loss
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            # 找出分數最高的類别
            predictions = logits.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()

            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_accuracy = (correct_predictions / total_samples)

    return epoch_loss, epoch_accuracy

# 準備訓練模型
NUM_EPOCHS = 1 # 訓練多少輪
PATIENCE = 10 # 設定多少次模型沒變好就停止訓練
MIN_DELTA = 0.001  # 這次與上一次訓練結果的loss差距

BEST_MODEL_PATH = Path(
    # "/kaggle/working/best_cnn.pth"
    "./best_cnn.pth"
)

# 記錄模型結果
history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    # Training
    train_loss, train_accuracy = train_one_epoch(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    # validation
    val_loss, val_accuracy = val_one_epoch(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        device=device
    )
    # 紀錄結果
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(f"Train Loss: {train_loss:.4f} | "f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f} | "f"Val Accuracy: {val_accuracy:.4f}")

    # 保存最佳模型
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(model.state_dict(), BEST_MODEL_PATH)

        print(f"已儲存最佳模型，" f"Val Loss：{best_val_loss:.4f}")

    else:
        epochs_without_improvement += 1
        print("驗證 Loss 未改善："f"{epochs_without_improvement}/{PATIENCE}")

    # Early Stopping
    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping："
            f"驗證 Loss 已連續 {PATIENCE} 個 epoch 未改善。"
        )
        break

# epochs = range(1, NUM_EPOCHS + 1)
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# # Loss
# axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
# axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation Loss")

# axes[0].set_title("Training and Validation Loss")
# axes[0].set_xlabel("Epoch")
# axes[0].set_ylabel("Loss")
# axes[0].legend()
# axes[0].grid(True)


# Accuracy
# axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy")
# axes[1].plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")

# axes[1].set_title("Training and Validation Accuracy")
# axes[1].set_xlabel("Epoch")
# axes[1].set_ylabel("Accuracy")
# axes[1].legend()
# axes[1].grid(True)


# plt.tight_layout()
# plt.show()

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

best_val_loss, best_val_accuracy = val_one_epoch(
    model=model,
    data_loader=val_loader,
    criterion=criterion,
    device=device
)

print(f"最佳驗證 Loss：{best_val_loss:.4f}")
print(f"最佳驗證 Accuracy：{best_val_accuracy:.4f}")

BATCH_SIZE=32
df_test = pd.read_csv("./test.csv")
df_test = df_test.values
test_features = torch.tensor(df_test, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0

test_loader = DataLoader(
    test_features,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model.eval()

test_predictions = []

with torch.inference_mode():
    for images in test_loader:
        images = images.to(device)

        # 
        logits = model(images)

        #
        predictions = logits.argmax(dim=1)

        # 將預測結果送到CPU保存
        test_predictions.extend(predictions.cpu().tolist())

print(f"預測總數: {len(test_predictions)}")
print(f"前 20 個預測結果: {test_predictions[:20]}")

submission = pd.DataFrame({
    "ImageId": range(1, len(test_predictions) + 1),
    "Label": test_predictions
})

SUBMISSION_PATH = "./submission.csv"

submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission: {SUBMISSION_PATH}")
print(submission.head())
print(submission.shape)