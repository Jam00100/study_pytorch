# Titanic

使用 PyTorch 建立 CNN，完成存活率預測
使用 Skelarn 中的模型，完成存活率預測

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

[Titanic](https://www.kaggle.com/competitions/titanic)

1. train.csv
2. test.csv

train.csv (n = 891)

test.csv (n = 418)

文件標籤

| 欄位         | 說明             | 數值／代碼                                        |
| ---------- | -------------- | -------------------------------------------- |
| `survived` | 是否生還           | `0`＝否；`1`＝是                                  |
| `pclass`   | 船票艙等           | `1`＝頭等艙；`2`＝二等艙；`3`＝三等艙                      |
| `sex`      | 性別             |                                              |
| `age`      | 年齡（歲）          |                                              |
| `sibSp`    | 船上同行的兄弟姊妹與配偶人數 |                                              |
| `parch`    | 船上同行的父母與子女人數   |                                              |
| `ticket`   | 船票號碼           |                                              |
| `fare`     | 票價             |                                              |
| `cabin`    | 艙房號碼           |                                              |
| `embarked` | 登船港口           | `C`＝Cherbourg；`Q`＝Queenstown；`S`＝Southampton |

Variable Notes

pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

## 測試模型 (Sklearn)

1. Random Forest Classifier
2. Support Vector Classifier
3. LogisticRegression
4. Gaussian Naive Bayes
5. Decision Tree Classifier

## Kaggle Test score
1. Use Pytorch NN (main.py): 0.77033
2. Use Skelarn RF (main_v2.py): 0.78468

推測NN不如RF的原因以及提升模型表現的想法：

1. 樣本數量不夠NN無法完整學習特徵
2. 資料處理方式可以優化？說不定能提升兩者表現
3. 對於資料缺漏數值的取捨（age之類的特徵如果缺失是要補值還是去除）