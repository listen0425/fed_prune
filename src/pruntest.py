import torch.nn as nn

# # 定义一个包含卷积层的模型
# model = nn.Sequential(
#     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2),
#     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2),
#     nn.Flatten(),
#     nn.Linear(in_features=128 * 8 * 8, out_features=10)
# )
#
# # 获取模型的参数字典
# state_dict = model.state_dict()
#
# # 打印参数字典中卷积层的权重和偏置
# for name, params in state_dict.items():
#     print(name,':')
#     for param in params:
#         print(name, param)

list1=[1,2,3,4,5,5,5,2,2,3,3,4]
list2=[3,4,5,6,7,5,5,4,3,3,7,7,6]
list3=[5,6,7,8,9,5,5,6,6,7,8,9,9,9]
set1=set(list1)
set2=set(list2)
set3=set(list3)
merged_set=set1.union(set2,set3)
print(merged_set)