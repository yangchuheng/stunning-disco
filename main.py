import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 创建一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(1, 1)  # 一个输入特征，一个输出特征

    def forward(self, x):
        x = self.fc(x)
        return x


# 创建一些虚拟数据
X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
y = np.array([[2.0], [4.0], [6.0], [8.0]], dtype=np.float32)

X_train = torch.from_numpy(X)
y_train = torch.from_numpy(y)

# 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# 测试模型
X_test = torch.tensor([[5.0], [6.0]], dtype=torch.float32)
predicted = model(X_test)
print(f'Predictions: {predicted.detach().numpy()}')
