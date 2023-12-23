# %%
import torch
from torch import nn
from torch import Tensor
from torch import optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# %%
EPOCH = 100

batch_size = 2048

# %%
import pandas as pd

dataset = pd.read_csv("../archive/data.csv")

dataset.drop(columns=["title", "movieId", "year"], inplace=True)
dataset.head()

# %%
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

data_train, data_val = train_test_split(dataset, test_size=0.2, random_state=42)

# 训练集特征和标签
X_train = torch.FloatTensor(data_train.iloc[:, :20].values)
y_train = torch.FloatTensor(data_train['rating'].values)

# 验证集特征和标签
X_val = torch.FloatTensor(data_val.iloc[:, :20].values)
y_val = torch.FloatTensor(data_val['rating'].values)

# 将数据转换为 DataLoader
train_dataset = TensorDataset(X_train, y_train)

val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# train_loader

# %%
class MyModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyModule, self).__init__()
        self.fc1= nn.Linear(input_size, hidden_size)
        self.fc2= nn.Linear(hidden_size, out_features=1)

    def forward(self, input: Tensor):
        x = self.fc1(input)
        x = self.fc2(x)
        x = torch.sigmoid(x) * 5
        return x

# %%
model = MyModule(input_size=20, hidden_size=50)

# %%
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(0, EPOCH):
    model.train()
    for inputs, labels in train_loader:
        # print(type(batch))
        # (x, y) = batch[:, :20], batch[:, 20:]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
    
    #eval
    model.eval()
    val_predictions=[]
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_predictions.extend(outputs.squeeze().tolist())
    
    val_predictions = torch.FloatTensor(val_predictions)
    val_loss = mean_squared_error(y_val.numpy(), val_predictions.numpy())
    val_abs_loss = mean_absolute_error(y_val.numpy(), val_predictions.numpy())
    rmse = np.sqrt(((y_val.numpy() - val_predictions.numpy())**2).mean())
    print(f'Epoch {epoch+1}/{EPOCH}, Validation Loss: {val_loss:.4f}, abs Loss:{val_abs_loss:.4f}, RMSE: {rmse:.4f}')




