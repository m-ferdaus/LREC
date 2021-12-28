import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset,DataLoader
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import random as r
np.random.seed(1)
r.seed(1)
torch.manual_seed(1)

# hyper parameters
input_size = 590
batch_size = 1
hidden_size = 100
num_classes = 2
num_epochs = 10
learning_rate = 1e-3

# read csv file and load the row data into input and target variables
dfData = pd.read_csv('secom.data.txt', delimiter=' ', header=None)
dfLabels = pd.read_csv('secom_labels.data.txt', delimiter=' ', header=None)


# # ==> KNN imputation for the missing samples ==<
X = dfData.values
cols = dfData.columns
imp = KNNImputer(n_neighbors=5)
X = imp.fit_transform(X)
X = pd.DataFrame(X, columns=cols)

dfLabels = dfLabels.replace(-1, 0)
y = dfLabels[0].values


scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# print(y)
train_X, test_X, train_y, test_y = train_test_split(X,
                                                    y, test_size=0.2, random_state=0, stratify=y)

train_X = torch.tensor(train_X, dtype=torch.float32)
test_X = torch.tensor(test_X, dtype=torch.float32)
train_y = torch.tensor(train_y)
train_y = train_y.type(torch.LongTensor)
test_y = torch.tensor(test_y)
test_y = test_y.type(torch.LongTensor)

train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        # out = F.relu(out) tanh  leaky_relu sigmoid relu
        out = F.tanh(out)
        out = self.fc2(out)
        # out = F.relu(out)
        out = F.tanh(out)
        out = self.fc3(out)
        return out

model = FFNN(input_size, hidden_size, num_classes)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for steps, (data, labels) in enumerate(train_loader):
        # forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (steps + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {steps + 1}/{n_total_steps}, loss = {loss.item():.3f}')

# testing loop
def check_accuracy(loader, model):
    # if train_loader:
    #     print("Checking accuracy on training data")
    # else:
    #     print("Checking accuracy on test data")
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data, labels in loader:
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

    # Overall testing accuracy
        acc = 100.0 * n_correct / n_samples
        print(
            f"Got {n_correct} / {n_samples} with accuracy {acc:.2f}"
        )

check_accuracy(test_loader, model)

def get_all_eval_Mat(label_all, preds):
    conf_matrix = confusion_matrix(label_all, preds)
    report_com = classification_report(label_all, preds, digits=4)
    g_mean_score_hp = geometric_mean_score(label_all, preds)
    return conf_matrix, g_mean_score_hp, report_com

def get_all_data(model, loader):
    with torch.no_grad():
        preds = torch.tensor([]).long()
        label_all = torch.tensor([]).long()
        for batch in loader:
            data, labels = batch
            outputs2 = model(data)
            _, predictions = torch.max(outputs2, 1)
            batch_predictions = predictions
            preds = torch.cat((preds, batch_predictions), dim=0)
            label_all = torch.cat((label_all, labels), dim=0)

        conf_matrix, g_mean_score_hp, report_com = get_all_eval_Mat(label_all, preds)
        print("Confusion matrix testing:\n", conf_matrix)
        print("Classification report:\n", report_com)
    return label_all, preds

get_all_data(model, test_loader)
