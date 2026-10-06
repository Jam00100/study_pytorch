import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RANDOM_SEED = 43
np.random.seed(seed=RANDOM_SEED)

# INPUT_DIR = Path("/kaggle/input/competitions/titanic")
INPUT_DIR = Path("./")
# OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR = Path("./")

# check data
train_df = pd.read_csv(INPUT_DIR/"train.csv")
print(train_df.head())

# 分析資料
# 1. train_df information
print("train_df shape", train_df.shape)
# print(train_df.dtypes)
# print(train_df.describe())
# print(train_df.describe(include="object"))

# 2. 檢查Survived (0, 1) 分佈
# print(train_df["Survived"].value_counts())

# ax = sns.countplot(data=train_df, x="Survived")
# ax.bar_label(ax.containers[0])

# plt.title("Survival count")
# plt.show()

# # 3. 分析存活與以下欄位的分布情況
# columns = ["Sex", "Pclass", "SibSp", "Parch", "Embarked"]

# fig, axes = plt.subplots(2, 3, figsize=(16, 9))
# axes = axes.flatten()

# for i, column in enumerate(columns):
#     ax = axes[i]

#     sns.countplot(
#         data=train_df,
#         x=column,
#         hue="Survived",
#         hue_order=[0, 1],
#         ax=ax
#     )

#     # 在每根柱子上方標記人數
#     for container in ax.containers:
#         ax.bar_label(container, fmt="%d", padding=2)

#     ax.set_title(column)
#     ax.set_xlabel(column)
#     ax.set_ylabel("Count")
#     ax.legend(title="Survived", labels=["No (0)", "Yes (1)"])

# axes[-1].axis("off")

# plt.tight_layout()
# plt.show()

# 男女之間的存活率差異
train_df.groupby('Sex')[['Survived']].mean()
train_df.pivot_table('Survived',index = 'Sex', columns = 'Pclass')

age = pd.cut(train_df['Age'],[0, 18, 80])
train_df.pivot_table('Survived',['Sex',age], 'Pclass')

# 檢查missing value
missing = pd.DataFrame({
    "Number of missing values": train_df.isna().sum(),
    "Missing value ratio (%)": train_df.isna().mean() * 100
})

print(missing.sort_values("Missing value ratio (%)", ascending=False))
# 資料清理
train_data = train_df.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis=1)
# train_data = train_data.dropna(subset=["Embarked", "Age"])
# 只刪除 Embarked 缺值的資料列
train_data = train_data.dropna(subset=["Embarked"]).copy()

# Age 缺值使用中位數填補
train_data["Age"] = train_data["Age"].fillna(
    train_data["Age"].median()
)

# 處理input dtype
# 處理Sex and Embarked
labelencoder = LabelEncoder()
train_data["Sex"] = labelencoder.fit_transform(train_data["Sex"])
train_data["Embarked"] = labelencoder.fit_transform(train_data["Embarked"])
# 檢查Label Encoder轉換情形
print("Sex:",  train_data['Sex'].unique())
print("Embarked:", train_data['Embarked'].unique())

X = train_data.drop("Survived", axis=1)
y = train_data[["Survived"]]

x_train, x_validate, y_train, y_validate = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_validate = scaler.transform(x_validate)

def primary_train(ml, X_train, y_train, random_seed=None):
    if ml == "RF":
        model = RandomForestClassifier(random_state=random_seed)
    elif ml == "LR":
        model = LogisticRegression()
    elif ml == "SVC":
        model = SVC()
    elif ml == "DC":
        model = DecisionTreeClassifier(random_state=random_seed)
    else:
        model = GaussianNB()
    
    model = model.fit(X_train, y_train)
    
    return model

primary_result = {}
ML_list = ["RF", "LR", "SVC", "DC", "GNB"]
for ml in ML_list:
    print(ml)
    model = primary_train(ml, x_train, y_train.values.ravel(), random_seed=RANDOM_SEED)
    
    y_predict = model.predict(x_validate)
    
    cm = confusion_matrix(y_validate, y_predict)
    accuracy = accuracy_score(y_validate, y_predict)
    F1 = f1_score(y_validate, y_predict)
    primary_result[ml] = {"accuracy":accuracy, "f1":F1}

primary_df = pd.DataFrame(primary_result)

# 選排名前三的model進行後續調參數訓練
def final_training(ml, X_train, y_train, random_seed=None):
    if ml == "RF":
        model = RandomForestClassifier(n_jobs=-1, random_state=random_seed)
        parameters = {
            "n_estimators": [50, 100, 300],
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [1, 2, 5, 10]
        }
    elif ml == "LR":
        model = LogisticRegression(penalty="l2", max_iter=10000)
        parameters = {
            "C": [0.01, 0.1, 1, 10, 100],
        }
    elif ml == "SVC":
        model = SVC()
        parameters = [
            {
                "kernel": ["linear"],
                "C": [0.1, 1, 10, 100]
            },
            {
                "kernel": ["rbf"],
                "C": [0.01, 0.1, 1, 10, 100],
                "gamma": ["scale", 0.01, 0.1, 1]
            }
        ]
        
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    return best_model

final_result = {}
final_models = {}
ML_list = ["RF", "LR", "SVC"]
for ml in ML_list:
    print(ml)
    model = final_training(ml, x_train, y_train.values.ravel(), random_seed=RANDOM_SEED)
    
    y_predict = model.predict(x_validate)
    
    cm = confusion_matrix(y_validate, y_predict)
    accuracy = accuracy_score(y_validate, y_predict)
    F1 = f1_score(y_validate, y_predict)
    final_result[ml] = {"accuracy":accuracy, "f1":F1}
    final_models[ml] = model

final_df = pd.DataFrame(final_result)
print(primary_df)
print(final_df)

# 選擇調參後驗證 f1 最高的模型
best_ml = final_df.loc["f1"].idxmax()
best_model = final_models[best_ml]
print("提交使用的模型：", best_ml)

# 讀取 Kaggle 測試資料
test_df = pd.read_csv(INPUT_DIR / "test.csv")

test_data = test_df.drop(
    ["Cabin", "Name", "Ticket", "PassengerId"],
    axis=1
).copy()

# 保留所有測試乘客，填補缺值
test_data["Age"] = test_data["Age"].fillna(train_data["Age"].median())
test_data["Fare"] = test_data["Fare"].fillna(train_data["Fare"].median())

# 與訓練時 LabelEncoder 的編碼一致
test_data["Sex"] = test_data["Sex"].map({
    "female": 0,
    "male": 1
})
test_data["Embarked"] = test_data["Embarked"].map({
    "C": 0,
    "Q": 1,
    "S": 2
})
test_data["Embarked"] = test_data["Embarked"].fillna(
    train_data["Embarked"].mode()[0]
)

# 欄位順序與訓練一致，套用原本的 scaler
test_data = test_data[X.columns]
test_features = scaler.transform(test_data)

# Predict
predictions = best_model.predict(test_features)

# 建立 Kaggle 檔案格式
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": predictions.astype(int)
})

submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

print(submission.head())
print("Len of data", len(submission))
print("Output path：", OUTPUT_DIR / "submission.csv")