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

> 注意：應該使用 `nn.Module`，不是 `nn.modules`

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