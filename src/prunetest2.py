import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

model = MyModel()

# 定义需要剪枝的层
prunable_modules = [
    model.conv1, model.conv2, model.fc1
]


# 获取剪枝前的模型参数

with open('before_model_params.txt', 'w') as f:
    for name, params in model.state_dict().items():
        for param in params:
            f.write(str(param))
# 执行剪枝

for module in prunable_modules:
    prune.ln_structured(module, name="weight", amount=0.2, n=2, dim=0)
# 获取剪枝后的模型参数

for module in prunable_modules:
    prune.remove(module, name='weight')

with open('after_model_params.txt', 'w') as f:
    for name, params in model.state_dict().items():
        print(name, ':')
        print(params.shape)
        for param in params:
            f.write(str(name))
            f.write(str(param))

prune_dict={}


# 获得被剪枝的通道
for name, params in model.state_dict().items():
    if 'weight' in str(name) and str(name)!='fc2.weight':
        prune_dict[name]=[]
        idx=0
        for param in params:
            idx+=1
            if torch.allclose(param, torch.zeros_like(param)):
                prune_dict[name].append(idx)

print(prune_dict)
