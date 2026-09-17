#%%
import pandas as pd
import numpy as np
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 基本設定
INPUT_DIR = Path("/kaggle/input/competitions/dogs-vs-cats-redux-kernels-edition")
WORK_DIR = Path("/kaggle/working")

TRAIN_ZIP = INPUT_DIR / "train.zip"
TEST_ZIP = INPUT_DIR / "test.zip"

TRAIN_DIR = WORK_DIR / "train"
TEST_DIR = WORK_DIR / "test"

BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_WORKERS = 2
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

# 選擇運算裝置
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"使用裝置：{device}")
if device.type == "cuda":
    print(f"GPU：{torch.cuda.get_device_name(0)}")

# 建立圖片路徑與標籤
image_paths = sorted(TRAIN_DIR.glob("*.jpg"))

if not TRAIN_ZIP.exists():
    raise FileNotFoundError(
        f"找不到訓練資料：{TRAIN_ZIP}\n" "請確認已透過 Add Input 加入資料。")

if not TEST_ZIP.exists():
    raise FileNotFoundError(f"找不到測試資料：{TEST_ZIP}\n" "請確認已透過 Add Input 加入資料。")

# 資料夾不存在或裡面沒有 JPG 時才重新解壓縮
if not TRAIN_DIR.exists() or not any(TRAIN_DIR.glob("*.jpg")):
    print("正在解壓縮 train.zip...")

    with ZipFile(TRAIN_ZIP, "r") as zip_file:
        zip_file.extractall(WORK_DIR)

if not TEST_DIR.exists() or not any(TEST_DIR.glob("*.jpg")):
    print("正在解壓縮 test.zip...")

    with ZipFile(TEST_ZIP, "r") as zip_file:
        zip_file.extractall(WORK_DIR)

train_images = sorted(TRAIN_DIR.glob("*.jpg"))
test_images = sorted(TEST_DIR.glob("*.jpg"))

# print(f"訓練圖片：{len(train_images)}")
# print(f"測試圖片：{len(test_images)}")

if len(train_images) == 0:
    raise RuntimeError("train 資料夾內沒有找到 JPG 圖片。")

if len(test_images) == 0:
    raise RuntimeError("test 資料夾內沒有找到 JPG 圖片。")

image_paths = train_images


def get_label(image_path):
    if image_path.name.startswith("cat."):
        return 0

    if image_path.name.startswith("dog."):
        return 1

    raise ValueError(
        f"無法從檔名判斷標籤：{image_path.name}"
    )

labels = [
    get_label(image_path)
    for image_path in image_paths
]

# print(f"圖片數量：{len(image_paths)} | 標籤數量：{len(labels)}")
# print(f"貓數量：{labels.count(0)}")
# print(f"狗數量：{labels.count(1)}")

# 切分訓練集與驗證集
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=labels
)

print(f"訓練集數量：{len(train_paths)}")
print(f"驗證集數量：{len(val_paths)}")

# 圖片轉換
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        size=IMAGE_SIZE,
        scale=(0.7, 1.0)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 自訂 Class Dataset
class CatDogDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# 建立 Dataset
train_dataset = CatDogDataset(
    image_paths=train_paths,
    labels=train_labels,
    transform=train_transform
)

val_dataset = CatDogDataset(
    image_paths=val_paths,
    labels=val_labels,
    transform=val_transform
)

# print(f"train_dataset：{len(train_dataset)}")
# print(f"val_dataset：{len(val_dataset)}")

# 建立 DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# 取得一個 batch
images, batch_labels = next(iter(train_loader))

# print(f"圖片 batch shape：{images.shape}")
# print(f"標籤 batch shape：{batch_labels.shape}")
# print(f"前十個標籤：{batch_labels[:10]}")

# 顯示圖片
def denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    return image * std + mean


class_names = {
    0: "Cat",
    1: "Dog"
}

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for index, axis in enumerate(axes.flat):
    image = denormalize(images[index])
    image = image.clamp(0, 1)

    # PyTorch: [C, H, W]
    # Matplotlib: [H, W, C]
    image = image.permute(1, 2, 0)

    label = batch_labels[index].item()

    axis.imshow(image)
    axis.set_title(class_names[label])
    axis.axis("off")

plt.tight_layout()
# plt.show()
# plt.savefig("./plot.png")

# 用class建立 cat and dog CNN
class CatDogCNN(nn.Module):
    def __init__ (self):
        super().__init__()

        # 提取特徵
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # [batch, 256, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        # [batch, 1] → [batch]
        return x.squeeze(dim=1)

# 將model送至GPU
model = CatDogCNN().to(device)
# print(model)

sample_images, sample_labels = next(iter(train_loader))

sample_images = sample_images.to(device)

with torch.inference_mode():
    sample_outputs = model(sample_images)

# print("輸入形狀：", sample_images.shape)
# print("輸出形狀：", sample_outputs.shape)
# print("前五個輸出：", sample_outputs[:5])

# 計算損失函數
criterion = nn.BCEWithLogitsLoss()

# 更新與調整神經網路中權重（Weights）和偏差（Biases), 使用預設
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps = 1e-8,
    weight_decay=0,
    amsgrad=False
)

# 訓練模型需要：
# model、data_loader、criterion、optimizer、device
def train_one_epoch(model, data_loader, criterion, optimizer, device):
    # 切換成訓練模式
    model.train()

    # 記錄一個 epoch 的 loss、正確數量與樣本數
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 逐個 batch 讀取資料
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.float32, non_blocking=True)

        # 清除上一個 batch 留下來的梯度
        optimizer.zero_grad(set_to_none=True)

        # Forward：取得模型輸出的 logits
        logits = model(images)

        # 計算 loss
        loss = criterion(logits, labels)

        # Backward：計算每個參數的梯度
        loss.backward()

        # 根據梯度更新模型參數
        optimizer.step()

        # 取得這個 batch 的圖片數量
        batch_size = images.size(0)

        # 累計所有樣本的 loss
        running_loss += loss.item() * batch_size

        # Logits 轉換成機率
        probabilities = torch.sigmoid(logits)

        # 機率大於或等於 0.5 時判斷為類別 1
        predictions = (probabilities >= 0.5).float()

        # 累計預測正確的數量
        correct_predictions += (predictions == labels).sum().item()

        # 累計樣本數量
        total_samples += batch_size

    # 計算整個 epoch 的平均 loss
    epoch_loss = running_loss / total_samples

    # 計算整個 epoch 的準確率
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, data_loader, criterion, device):
    # 切換成驗證模式
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 關閉梯度計算，降低記憶體與運算量
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)

            # Forward
            logits = model(images)

            # 計算 loss
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            # Logits 轉換成機率
            probabilities = torch.sigmoid(logits)

            # 機率大於或等於 0.5 時判斷為類別 1
            predictions = (probabilities >= 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy

# 準備訓練模型
NUM_EPOCHS = 30 # 訓練多少輪
PATIENCE = 10 # 設定多少次模型沒變好就停止訓練
MIN_DELTA = 0.001  # 這次與上一次訓練結果的loss差距

# 保存模型路徑
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

    # Validation
    val_loss, val_accuracy = validate_one_epoch(
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


epochs = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation Loss")

axes[0].set_title("Training and Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)


# Accuracy
axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy")
axes[1].plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")

axes[1].set_title("Training and Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)


plt.tight_layout()
plt.show()

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

best_val_loss, best_val_accuracy = validate_one_epoch(
    model=model,
    data_loader=val_loader,
    criterion=criterion,
    device=device
)

print(f"最佳驗證 Loss：{best_val_loss:.4f}")
print(f"最佳驗證 Accuracy：{best_val_accuracy:.4f}")

class TestDataset(Dataset):

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        # 例如 123.jpg → 123
        image_id = int(image_path.stem)

        return image, image_id

test_paths = sorted(TEST_DIR.glob("*.jpg"), key=lambda path: int(path.stem))
# print(f"測試圖片數量：{len(test_paths)}")
# print("前五張：", [path.name for path in test_paths[:5]])
# print("最後五張：", [path.name for path in test_paths[-5:]])

test_dataset = TestDataset(
    image_paths=test_paths,
    transform=val_transform
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print(f"測試資料數量：{len(test_dataset)}")
print(f"測試 batch 數量：{len(test_loader)}")

from tqdm.auto import tqdm

model.eval()

test_ids_list = []
test_probabilities = []

with torch.inference_mode():
    for images, image_ids in tqdm(test_loader, desc="Predicting"):
        images = images.to(device, non_blocking=True)

        # 模型輸出 logits
        logits = model(images)

        # logits → 狗的機率
        probabilities = torch.sigmoid(logits)

        # 移回 CPU 並存入 list
        test_ids_list.extend(image_ids.cpu().numpy().tolist())

        test_probabilities.extend(probabilities.cpu().numpy().tolist())

# print(f"圖片 ID 數量：{len(test_ids_list)}")
# print(f"預測機率數量：{len(test_probabilities)}")
# print("\n前十筆預測：")

for image_id, probability in zip(test_ids_list[:10], test_probabilities[:10]):
    print(f"ID：{image_id:5d} | " f"狗的機率：{probability:.4f}")

print(f"最小機率：{min(test_probabilities):.6f}")
print(f"最大機率：{max(test_probabilities):.6f}")

submission = pd.DataFrame({
    "id": test_ids_list,
    "label": test_probabilities
})

# 按圖片 ID 排序
submission = submission.sort_values("id").reset_index(drop=True)

# 避免機率剛好等於 0 或 1
submission["label"] = np.clip(submission["label"], 0.005,0.995)

SUBMISSION_PATH = Path("/kaggle/working/submission.csv")

submission.to_csv(SUBMISSION_PATH, index=False)

print(f"提交檔案位置：{SUBMISSION_PATH}")
print(f"提交資料數量：{len(submission)}")

submission.head(10)