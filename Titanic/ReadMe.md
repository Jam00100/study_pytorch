# Titanic


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