# Dogs vs. Cats CNN

使用 PyTorch 建立 CNN，完成 Kaggle Dogs vs. Cats 二元圖片分類。

## 使用工具

- Python
- PyTorch
- Torchvision
- Pandas
- Scikit-learn
- Pillow
- Matplotlib
- numpy

## 資料集

Kaggle：

[Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition)

標籤設定：

```text
cat = 0
dog = 1
```

資料數量：

```text
訓練圖片：25,000
測試圖片：12,500
```

## 程式流程

```text
1. 載入套件與設定參數
2. 解壓縮 Kaggle 資料
3. 建立圖片路徑與標籤
4. 切分訓練集與驗證集
5. 設定圖片 transforms
6. 建立 Dataset 與 DataLoader
7. 建立 CNN 模型
8. 訓練與驗證模型
9. 預測測試圖片
10. 產生 submission.csv
```

## 模型架構

```text
Input
→ Conv2d：3 → 32
→ MaxPool2d
→ Conv2d：32 → 64
→ MaxPool2d
→ Conv2d：64 → 128
→ MaxPool2d
→ Conv2d：128 → 256
→ AdaptiveAvgPool2d
→ Flatten
→ Dropout
→ Linear：256 → 1
```

- Loss：`BCEWithLogitsLoss`
- Optimizer：`Adam`
- Batch size：`32`
- Image size：`224 × 224`
- Epochs: 30
- Early stopping: 10
- Minimum Delta: 0.001

## 訓練結果

```text
Best epoch：29
Best validation loss：0.1653
Best validation accuracy：0.9350
Kaggle score：0.18369
```