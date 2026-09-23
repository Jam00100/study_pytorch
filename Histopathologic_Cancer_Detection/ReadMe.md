# Histopathologic Cancer Detection

使用 PyTorch 建立 CNN，完成 Histopathologic Cancer Detection

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

[Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection)

1. train_label.csv
2. train images
3. test images

標籤設定：

```text
沒癌症 = 0
有癌症 = 1
```

資料數量：

```text
訓練圖片：220,025
測試圖片：57,458
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

## 圖片transform
```text
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
可以參考ImageNet數值，或是自己計算
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
→ MaxPool2d
→ Conv2d：256 → 512
→ AdaptiveAvgPool2d
→ Flatten
→ Dropout
→ Linear：512 → 1
```

- Loss：`BCEWithLogitsLoss`
- Optimizer：`Adam`
- Batch size：`64`
- Image size：`96 × 96`
- Epochs: 30
- Early stopping: 10
- Minimum Delta: 0.001

## 訓練結果

```text
Best epoch：27
Best validation loss：0.0887
Best validation accuracy：0.9697
Kaggle score：0.9420
```