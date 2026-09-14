import torch
import torch.nn as nn
import pandas as pd

device = torch.device(
	"mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)

raw_df = pd.read_csv("train.csv")

# Label
label = raw_df["label"].values
# Features
raw_df = raw_df.drop(["label"], axis=1).values

# Seperate df into 2 part: training set and testing set
train_feature = raw_df[:int(len(raw_df) * 0.8)]
train_lable = label[:int(len(label) * 0.8)]

test_feature = raw_df[int(len(raw_df) * 0.8):]
test_lable = label[int(len(label) * 0.8):]

# print(len(train_feature), len(train_lable), len(test_feature), len(test_lable))
# to tensor
train_feature = torch.tensor(train_feature).to(torch.float).to(device)
train_lable = torch.tensor(train_lable).to(device)
test_feature = torch.tensor(test_feature).to(torch.float).to(device)
test_lable = torch.tensor(test_lable).to(device)


# Build NN
# 784 pixel --> model --> 10 number (0~9)
model = nn.Sequential(
	nn.Linear(784, 444),
	nn.ReLU(),
	nn.Linear(444, 512),
	nn.ReLU(),
	nn.Linear(512, 512),
	nn.ReLU(),
	nn.Linear(512, 10),
	nn.Softmax()
).to(device)

# Gradient descent
lossfunction = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

# training iteration
for i in range(100):
	# reset optimizer
	optimizer.zero_grad()
	predict = model(train_feature)
	result = torch.argmax(predict, axis=1)
	train_acc = torch.mean((result == train_lable).to(torch.float))
	loss = lossfunction(predict, train_lable)
	loss.backward()
	optimizer.step()

	# print(f"train loss: {loss}\t train accuracy: {train_acc.item()}")
	# print(loss, train_acc)

	optimizer.zero_grad()
	predict = model(test_feature)
	result = torch.argmax(predict, axis=1)
	test_acc = torch.mean((result == test_lable).to(torch.float))
	loss = lossfunction(predict, test_lable)
	print(f"train loss: {loss}\ttrain accuracy: {train_acc.item()}\ttest accuracy: {test_acc.item()}")

torch.save(model.state_dict(), "./Handwriting_Classification_model.pt")

