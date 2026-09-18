# Digit Recognizer

使用 PyTorch 建立 CNN，完成 Digit Recognizer

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

[Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)

標籤設定：

```text
labels = 0 ~ 9
```

資料數量：

```text
training images：42,000
testing images：28,000
```

## 程式流程

```text
1. 載入套件與設定參數
2. 解壓縮 Kaggle 資料
3. 建立圖片路徑與標籤
4. 切分訓練集與驗證集
5. 將灰階圖片資料轉換成tensor
6. 建立 Dataset 與 DataLoader
7. 建立 CNN 模型
8. 訓練與驗證模型
9. 預測測試圖片
10. 產生 submission.csv
```

## 模型架構

```text
Input
→ Conv2d：1 → 32 (由於是灰階圖片所以in channel = 1)
→ MaxPool2d
→ Conv2d：32 → 64
→ MaxPool2d
→ Conv2d：64 → 128
→ MaxPool2d
→ AdaptiveAvgPool2d
→ Flatten
→ Dropout
→ Linear：128 → 1
```

- Loss：`CrossEntropyLoss`
- Optimizer：`Adam`
- Batch size：`32`
- Image size：`28 × 28`
- Epochs: `30`
- Early stopping: `10`
- Minimum Delta: `0.001`
- features = `784` pixels

## 訓練結果

```text
Best epoch：17
Best validation loss：0.0239
Best validation accuracy：0.9921
```

## 獨立測試結果

```text
Kaggle sore: 0.99178
```


