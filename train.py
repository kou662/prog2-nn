import time

import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import models

# データセットの前処理を定義
ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

#データセットの読み込み
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,  # 訓練用を指定
    download=True,
    transform=ds_transform
)

ds_test = datasets.FashionMNIST(
    root='data',
    train=False,  # テスト用を指定
    download=True,
    transform=ds_transform
)

#ミニバッチに分割する DAtaLoder を作る
batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

# バッチを取り出す実験
# この後の処理では不要なので、確認したら削除してよい

# for image_batch, label_batch in dataloader_test:
#     print(image_batch.shape)
#     print(label_batch.shape)
#     break  # 1つ目で終了

model = models.Mymodel()

# 精度を計算する
acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.2f}%')
acc_test = models.test_accuracy(model, dataloader_train)
print(f'train accuracy: {acc_test*100:.2f}%')

# ロス関数の選択
loss_fn = torch.nn.CrossEntropyLoss()

#最適化手法の選択
learning_rate = 0.003
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#criterion(規準)とも呼ぶ

n_epoch = 5 

loss_train_histry = []
loss_test_histry = []
acc_train_histry = []
acc_test_histry = []

for k in range(n_epoch):
    print(f'epoch{k+1}/{n_epoch}')

    loss_train = models.train(model,dataloader_train, loss_fn,optimizer)
    print(f'train loss: {loss_train}')
    loss_train_histry.append(loss_train)

    loss_test = models.test(model, dataloader_test,loss_fn)
    loss_test_histry.append(loss_test)
    
    acc_test = models.test_accuracy(model,dataloader_test)
    print(f'test accuracy: {acc_test*100:.2f}%')
    acc_test_histry.append(acc_test)

    acc_train = models.test_accuracy(model,dataloader_train)
    acc_train_histry.append(acc_train)
    print(f'train accuracy: {acc_test*100:.2f}%')

plt.plot(acc_train_histry,label='train')
plt.plot(acc_test_histry,label='test')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.show()

plt.plot(loss_train_histry,label='train')
plt.plot(loss_test_histry,label='test')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.show()
# # 精度を計算
# acc_test = models.test_accuracy(model, dataloader_test)
# print(f'test accuracy: {acc_test*100:.2f}%')

# # 学習
# models.train(model, dataloader_test, loss_fn, optimizer)

# # もう一度精度を計算
# acc_test = models.test_accuracy(model, dataloader_test)
# print(f'test accuracy: {acc_test*100:.2f}%')