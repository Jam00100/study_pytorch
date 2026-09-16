# Basic PyTorch Knowledge

## Device

選擇模型訓練時使用的運算裝置：

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

- `mps`：使用 Apple Silicon GPU
- `cuda`：使用 NVIDIA GPU
- `cpu`：使用 CPU

模型和資料必須放在相同裝置：

```python
model = model.to(device)
images = images.to(device)
labels = labels.to(device)
```

---

## Tensor

### 將資料轉換成 Tensor

PyTorch 模型使用 Tensor 進行運算。

```python
tensor = torch.tensor(要轉換的資料)
```

例如：

```python
x = torch.tensor([1, 2, 3])
```

### 轉換資料型別

可以透過 `dtype` 指定資料型別：

```python
x = torch.tensor([1, 2, 3], dtype=torch.float32)
```

也可以在建立 Tensor 後進行轉換：

```python
x = x.to(torch.float32)
y = y.to(torch.int64)
```

常見的簡寫方式：

```python
x = x.float()
y = y.long()
```

> 建議使用 `torch.float32`、`torch.int64` 等 PyTorch dtype，不要寫成 `.to(float or int)`。

### 將 Tensor 送到運算裝置

```python
x = x.to(device)
```

也可以同時轉換資料型別和裝置：

```python
x = x.to(device=device, dtype=torch.float32)
```

---

# CNN（Convolutional Neural Network）

## 圖片轉換

### `transforms.Compose()`

- 將多個圖片轉換操作組合起來
- 按照程式碼中設定的順序處理圖片

使用方式：

```python
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
```

---

### `transforms.RandomResizedCrop()`

- 從圖片中隨機裁切一個區域
- 再將裁切區域縮放成指定大小

主要參數：

- `size`：最後輸出的圖片大小
- `scale`：裁切區域占原圖面積的比例範圍
- `ratio`：裁切區域的寬高比，通常可以使用預設值

```python
transforms.RandomResizedCrop(
    size=IMAGE_SIZE,
    scale=(0.7, 1.0)
)
```

#### 目的：增加圖片構圖的變化

對貓狗分類而言，可以讓模型學習：

1. 貓狗可能出現在圖片中的不同位置。
2. 主體距離鏡頭可能不同。
3. 圖片可能只拍到部分身體。
4. 不依賴固定的背景與構圖。

在不同 epoch 中讀取同一張圖片時，可能得到：

```text
第 1 次：包含整隻貓
第 2 次：主要看到貓的臉
第 3 次：貓出現在畫面左側
```

---

### `transforms.RandomHorizontalFlip(p=0.5)`

- 隨機水平翻轉圖片
- `p` 是 probability，代表執行翻轉的機率
- `p=0.5` 表示 50% 機率翻轉，50% 機率保持原圖

#### 目的：資料增強（Data Augmentation）

1. 增加資料多樣性：同一張圖片可以呈現不同方向。
2. 降低過擬合：避免模型記住特定圖片或方向。
3. 增強方向不變性：貓狗朝左或朝右都不影響分類。

不適合水平翻轉的任務包括：

- 文字辨識
- 判斷左手或右手
- 交通標誌方向辨識
- 醫學影像的左側或右側病灶
- 判斷車輛行駛方向

---

### `transforms.ColorJitter()`

- 隨機調整圖片的亮度、對比度和飽和度
- 屬於資料增強的一種
- 通常只用於訓練集，不用於驗證集和測試集

```python
transforms.ColorJitter(
    brightness=0.1,  # 亮度在原本的 0.9～1.1 倍之間變化
    contrast=0.1,    # 對比度在原本的 0.9～1.1 倍之間變化
    saturation=0.1   # 飽和度在原本的 0.9～1.1 倍之間變化
)
```

#### 目的：讓模型適應不同的光線與拍攝環境

避免模型過度依賴圖片的亮度、色調或背景顏色判斷貓狗。

---

### `transforms.ToTensor()`

#### 目的：將圖片轉換成 CNN 可以處理的 Tensor

```python
transforms.ToTensor()
```

主要進行兩項轉換：

1. 將圖片排列從 `HWC` 改成 `CHW`。
2. 將像素值從 `0～255` 縮放到 `0～1`。

假設圖片大小為 `224 × 224`：

```text
轉換前：[Height=224, Width=224, Channel=3]
轉換後：[Channel=3, Height=224, Width=224]
```

---

### `transforms.Normalize()`

#### 目的：調整各色彩通道的數值分布，讓模型訓練更加穩定

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

標準化公式：

$$
x' = \frac{x - mean}{std}
$$

- `mean`：RGB 三個通道的平均值
- `std`：RGB 三個通道的標準差
- 這組數值來自 ImageNet，常用於搭配 ImageNet 預訓練模型
- `Normalize()` 不會移除離群值
- 必須放在 `ToTensor()` 後面

---

## CNN模型建立

### class CatDogCNN(nn.Module)

- 使用 `class` 建立自己的 CNN 模型
- 必須繼承 `nn.Module`
- `nn.Module` 是所有 PyTorch 神經網路模型的基礎類別

```python
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
```

### nn.Sequential()

- 將多個神經網路層按照順序組合
- 資料會按照放入 `nn.Sequential()` 的順序進行處理
- `nn.Sequential()` 不一定是全連接網路，也可以用來建立卷積網路

```python
self.features = nn.Sequential(
    nn.Conv2d(...),
    nn.BatchNorm2d(...),
    nn.ReLU(),
    nn.MaxPool2d(...)
)
```

---

### nn.Conv2d()

#### 目的：從圖片中提取特徵

可以提取的特徵包括：

1. 圖片邊緣
2. 線條和紋理
3. 貓狗的眼睛、鼻子和耳朵
4. 貓狗的身體輪廓

使用方式：

```python
nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    padding=1
)
```

參數說明：

- `in_channels=3`：輸入圖片有 RGB 三個通道
- `out_channels=32`：使用 32 個卷積核，產生 32 張特徵圖
- `kernel_size=3`：卷積核大小為 `3 × 3`
- `padding=1`：在圖片周圍補一圈，使圖片的高度和寬度保持不變

資料形狀變化：

```text
[batch, 3, 224, 224]
→ [batch, 32, 224, 224]
```

此模型的通道數變化：

```text
3 → 32 → 64 → 128 → 256
```

- 前面的卷積層通常學習邊緣、線條等簡單特徵
- 後面的卷積層會組合成耳朵、臉部和身體輪廓等複雜特徵

---

### nn.BatchNorm2d()

#### 目的：讓模型中間特徵的數值分布更加穩定

使用方式：

```python
nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    padding=1
),
nn.BatchNorm2d(32)
```

- `BatchNorm2d` 的參數必須與前一層的 `out_channels` 相同
- `out_channels=32`，因此使用 `BatchNorm2d(32)`

主要作用：

1. 讓模型訓練更加穩定
2. 幫助模型更快收斂
3. 降低不同 batch 之間的特徵分布變化
4. 具有些微降低過擬合的效果

不會改變資料形狀：

```text
[batch, 32, 224, 224]
→ [batch, 32, 224, 224]
```

`transforms.Normalize()` 與 `BatchNorm2d()` 的差別：

- `transforms.Normalize()`：標準化輸入模型前的圖片
- `nn.BatchNorm2d()`：標準化模型中間產生的特徵圖

---

### nn.ReLU()

#### 目的：加入非線性，讓模型能學習更複雜的特徵

使用方式：

```python
nn.ReLU()
```

ReLU 的計算方式：

```text
輸入小於 0 → 輸出 0
輸入大於 0 → 保留原本數值
```

公式：

```text
ReLU(x) = max(0, x)
```

例如：

```text
輸入：[-2, -1, 0, 1, 3]
輸出：[ 0,  0, 0, 1, 3]
```

ReLU 不會改變 Tensor 的形狀。

---

### nn.MaxPool2d()

#### 目的：縮小特徵圖並保留較明顯的特徵

使用方式：

```python
nn.MaxPool2d(kernel_size=2)
```

- 將特徵圖分成多個 `2 × 2` 區域
- 保留每個區域中最大的數值
- 預設的 `stride` 與 `kernel_size` 相同，因此高度和寬度減半

例如：

```text
1  5
2  3
```

經過最大池化後：

```text
5
```

資料形狀變化：

```text
[batch, 32, 224, 224]
→ [batch, 32, 112, 112]
```

主要作用：

1. 縮小特徵圖
2. 減少模型的運算量
3. 保留較明顯的特徵
4. 讓模型對特徵的小幅位置變化較不敏感

此模型共有三次最大池化：

```text
224 × 224
→ 112 × 112
→ 56 × 56
→ 28 × 28
```

`MaxPool2d` 只會改變 Height 和 Width，不會改變 Channel。

---

### nn.AdaptiveAvgPool2d()

#### 目的：將每個通道的特徵圖壓縮成一個代表值

使用方式：

```python
nn.AdaptiveAvgPool2d((1, 1))
```

資料形狀變化：

```text
[batch, 256, 28, 28]
→ [batch, 256, 1, 1]
```

- 對每個通道的特徵圖計算平均值
- 每個通道最後只保留一個數值
- 不論輸入圖片大小如何，都會輸出指定的 `(1, 1)`

如果沒有使用 `AdaptiveAvgPool2d()`：

```text
256 × 28 × 28 = 200704 個特徵
```

使用後：

```text
256 × 1 × 1 = 256 個特徵
```

可以大幅減少後面全連接層的參數數量。

---

## 分類器

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 1)
)
```

- `self.features`：負責提取圖片特徵
- `self.classifier`：根據特徵判斷圖片是貓或狗

### nn.Flatten()

#### 目的：將多維特徵圖攤平成一維資料

使用方式：

```python
nn.Flatten()
```

資料形狀變化：

```text
[batch, 256, 1, 1]
→ [batch, 256]
```

- 只改變 Tensor 的形狀
- 不會改變裡面的數值
- 沒有需要訓練的參數

---

### nn.Dropout()

#### 目的：降低模型過擬合

使用方式：

```python
nn.Dropout(p=0.3)
```

- `p=0.3` 表示訓練時隨機關閉 30% 的特徵
- 避免模型過度依賴某些特定特徵
- 迫使模型同時學習更多不同的特徵

不同模式下的行為：

```python
model.train()  # Dropout 啟用
model.eval()   # Dropout 關閉
```

Dropout 只會在訓練模型時使用，驗證和預測時會自動關閉。

---

### nn.Linear()

#### 目的：根據提取出的特徵進行分類

使用方式：

```python
nn.Linear(256, 1)
```

參數說明：

- `256`：輸入的特徵數量
- `1`：輸出一個二元分類分數

資料形狀變化：

```text
[batch, 256]
→ [batch, 1]
```

`Linear` 是此模型中真正的全連接層。

輸出的數值稱為 `logit`，還不是機率。

如果使用：

```python
criterion = nn.BCEWithLogitsLoss()
```

模型最後不需要加入 `Sigmoid`，因為 `BCEWithLogitsLoss()` 已經包含 Sigmoid。

預測時可以轉換成機率：

```python
probability = torch.sigmoid(logits)
```

---

## forward()

#### 目的：定義資料通過模型的順序

```python
def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)

    return x.squeeze(dim=1)
```

### 提取特徵

```python
x = self.features(x)
```

讓圖片通過卷積網路：

```text
Conv2d
→ BatchNorm2d
→ ReLU
→ MaxPool2d
→ AdaptiveAvgPool2d
```

輸出形狀：

```text
[batch, 256, 1, 1]
```

### 進行分類

```python
x = self.classifier(x)
```

讓提取出的特徵通過分類器：

```text
Flatten
→ Dropout
→ Linear
```

輸出形狀：

```text
[batch, 1]
```

### squeeze(dim=1)

```python
return x.squeeze(dim=1)
```

移除大小為 1 的第 1 維：

```text
[batch, 1]
→ [batch]
```

例如：

```text
轉換前：[32, 1]
轉換後：[32]
```

指定 `dim=1` 可以避免在 `batch_size=1` 時，連 batch 維度一起被移除。

---

## 完整Tensor形狀變化

假設輸入圖片大小為 `224 × 224`：

| 階段 | 輸出形狀 |
|---|---|
| 輸入圖片 | `[batch, 3, 224, 224]` |
| Conv1 | `[batch, 32, 224, 224]` |
| MaxPool1 | `[batch, 32, 112, 112]` |
| Conv2 | `[batch, 64, 112, 112]` |
| MaxPool2 | `[batch, 64, 56, 56]` |
| Conv3 | `[batch, 128, 56, 56]` |
| MaxPool3 | `[batch, 128, 28, 28]` |
| Conv4 | `[batch, 256, 28, 28]` |
| AdaptiveAvgPool | `[batch, 256, 1, 1]` |
| Flatten | `[batch, 256]` |
| Linear | `[batch, 1]` |
| Squeeze | `[batch]` |

## 模型總結

這個 CNN 模型分成兩個主要部分：

1. `self.features`：從圖片中提取貓狗的特徵
2. `self.classifier`：根據提取出的特徵判斷是貓或狗

模型總共有：

- 4 個 `Conv2d` 卷積層
- 4 個 `BatchNorm2d`
- 4 個 `ReLU`
- 3 個 `MaxPool2d`
- 1 個 `AdaptiveAvgPool2d`
- 1 個 `Linear` 全連接層