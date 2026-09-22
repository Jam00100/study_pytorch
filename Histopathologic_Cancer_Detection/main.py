# %% [code] {"jupyter":{"outputs_hidden":false}}
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# %% [code]
INPUT_DIR = Path("/kaggle/input/competitions/histopathologic-cancer-detection")
OUTPUT_DIR = Path("/kaggle/working")

LABEL_PATH = Path(f"{INPUT_DIR}/train_labels.csv")
TRAIN_DIR = Path(f"{INPUT_DIR}/train")
TEST_DIR = Path(f"{INPUT_DIR}/test")

RANDOM_SEED = 42
BATCH_SIZE = 64
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

df = pd.read_csv(LABEL_PATH)
print(df.head())
print(df.shape)

# train_labels 0 and distrubution
count_df = df["label"].value_counts().reset_index(name="count")

plt.figure(figsize=(6, 4))
ax = sns.barplot(data=count_df, x="label", y="count", legend=False)
ax.bar_label(ax.containers[0])

plt.title("Label Counts")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# %% [code]
# 先選取一張樣本圖片檢查各屬性
sample_id = df.iloc[0]["id"]
sample_label = df.iloc[0]["label"]

sample_path = TRAIN_DIR / f"{sample_id}.tif"
sample_image = Image.open(sample_path)

print(f"IMAGE path：{sample_path}")
print(f"IMAGE size：{sample_image.size}")
print(f"IMAGE mode：{sample_image.mode}")
print(f"Label：{sample_label}")

# Randomly selected 5 images from each group (tumor and normal)
n = 5
selected_samples = df.groupby("label", as_index=False).sample(n=n, random_state=RANDOM_SEED)

fig, axes = plt.subplots(2, n, figsize=(12, 6))

for ax, row in zip(axes.flat, selected_samples.itertuples()):
    image = Image.open(TRAIN_DIR / f"{row.id}.tif")
    ax.imshow(image)
    ax.set_title(f"Label: {row.label}", color="red" if row.label == 1 else "blue")
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [code]
# A positive label indicates that the center 32x32px
# region of a patch contains at least one pixel of tumor tissue
fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(12, 6))

for ax, (_, row) in zip(axes.flat, selected_samples.iterrows()):
    image_path = TRAIN_DIR / f"{row['id']}.tif"
    image = Image.open(image_path)

    ax.imshow(image)

    center_region = Rectangle(
        xy=(32, 32), width=32, height=32,
        linewidth=2, edgecolor="yellow", facecolor="none"
    )
    ax.add_patch(center_region)

    ax.set_title(f"Label: {row['label']}", color="red" if row["label"] == 1 else "blue")
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [code]
# split train and validation df
train_df, validation_df = train_test_split(
    df, test_size=0.3, random_state=RANDOM_SEED, stratify=df["label"]
)

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)

# %% [code]
# 設定image transforms方法與順序
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0, 360)), # 生醫影像常用

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5], #先隨便設定，但之後可以參考ImageNet或是自已算
        std=[0.5, 0.5, 0.5]
    )
])

validation_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

class CancerDataset(Dataset):
    def __init__(self, dataframe,image_dir,transform=None):
        
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        
        # 從df中的id column讀取該樣本id並用來讀取圖片
        image_id = row["id"]
        image_path = self.image_dir / f"{image_id}.tif"

        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(
            row["label"],
            dtype=torch.float32
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, label

train_dataset = CancerDataset(
    dataframe=train_df,
    image_dir=TRAIN_DIR,
    transform=train_transform
)

validation_dataset = CancerDataset(
    dataframe=validation_df,
    image_dir=TRAIN_DIR,
    transform=validation_transform
)

print(f"Training dataset: {len(train_dataset)}")
print(f"Validation dataset: {len(validation_dataset)}")

# %% [code]
image, label = train_dataset[0]

print(f"Image shape: {image.shape}")
print(f"Label: {label}")
print(f"Image dtype: {image.dtype}")
print(f"Label dtype: {label.dtype}")

def denormalize(image):
    image = image * 0.5 + 0.5
    return image.clamp(0, 1)

fig, axes = plt.subplots(
    nrows=2,
    ncols=5,
    figsize=(12, 6)
)

# 看一下transform後的訓練圖片結果
for index, ax in enumerate(axes.flat):
    image, label = train_dataset[index]

    image = denormalize(image)
    
    # pytorch: [channel, height, width]
    # matplotlib: [channel, height, width]
    image = image.permute(1, 2, 0)

    ax.imshow(image)
    ax.set_title(f"Label: {int(label.item())}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [code]
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

validate_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# %% [code]
# 搭建CNN
class HistopathologicCancerDetectionCNN(nn.Module):
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
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        # [batch, 1] → [batch]
        return x.squeeze(dim=1)


model = HistopathologicCancerDetectionCNN().to(device)
sample_images, sample_labels = next(iter(train_loader))
sample_images = sample_images.to(device)

with torch.inference_mode():
    sample_outputs = model(sample_images)

# 檢查一個Batch
images, labels = next(iter(train_loader))

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Labels: {labels[:5]}")

# 計算損失函數
criterion = nn.BCEWithLogitsLoss()

# 更新與調整神經網路中權重（Weights）和偏差（Biases), 使用預設
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999), # default
    eps = 1e-8, # default
    weight_decay=0, # default
    amsgrad=False # default
)

# %% [code]
# train one epoch
def train_one_epoch(model, data_loader, criterion, optimizer, device):
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

# validation one epoch
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




# %% [code]
# 準備訓練模型
NUM_EPOCHS = 30 # 訓練多少輪
PATIENCE = 10 # 設定多少次模型沒變好就停止訓練
MIN_DELTA = 0.001  # 這次與上一次訓練結果的loss差距

# 保存模型
BEST_MODEL_PATH = Path(
    "/kaggle/working/best_cnn.pth"
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
        data_loader=validate_loader,
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
    data_loader=validate_loader,
    criterion=criterion,
    device=device
)

print(f"最佳驗證 Loss：{best_val_loss:.4f}")
print(f"最佳驗證 Accuracy：{best_val_accuracy:.4f}")

# %% [code]
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

        # 例如 123.tif → 123
        image_id = int(image_path.stem)

        return image, image_id

test_paths = sorted(TEST_DIR.glob("*.tif"), key=lambda path: int(path.stem))
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
    print(f"ID：{image_id:5d} | " f"cancer的機率：{probability:.4f}")

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